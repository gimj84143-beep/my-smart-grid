import os
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import platform
import warnings
import streamlit as st

warnings.filterwarnings('ignore')

# ===== 0. 기본 환경 및 폰트 설정 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEN_CSV_PATH = os.path.join(BASE_DIR, "한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_20241231.csv")
WEATHER_XLSX_PATH = os.path.join(BASE_DIR, "2024년 제주도 날씨 정보.xlsx")
CONSUME_CSV_PATH = os.path.join(BASE_DIR, "가구당_시간별_전력소비량.csv")

if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# ===== 1. AI 모델 학습 (캐싱 적용) =====
@st.cache_resource
def load_models():
    try:
        # 1) 전력 소비량 모델
        df_consume = pd.read_csv(CONSUME_CSV_PATH, encoding='cp949')
        df_consume['날짜'] = pd.to_datetime(df_consume['날짜'])
        X_con = pd.DataFrame({'월': df_consume['날짜'].dt.month, '일': df_consume['날짜'].dt.day, '요일': df_consume['날짜'].dt.dayofweek})
        hours_cols = [f'{i}시' for i in range(1, 25)]
        y_con = df_consume[hours_cols] * 0.174
        consume_model = RandomForestRegressor(n_estimators=100, random_state=42)
        consume_model.fit(X_con, y_con)

        # 2) 태양광 발전량 모델
        gen = pd.read_csv(GEN_CSV_PATH, encoding="cp949")
        gen = gen[gen['지역'].str.contains("제주")].copy()
        gen['datetime'] = pd.to_datetime(gen['거래일자']) + pd.to_timedelta(gen['거래시간'] - 1, unit='h')
        solar = gen[gen['연료원'] == '태양광'][['datetime', '전력거래량(MWh)']].rename(columns={'전력거래량(MWh)': 'solar_gen'})

        weather = pd.read_excel(WEATHER_XLSX_PATH, engine="openpyxl")
        weather = weather[["일시", "기온(°C)", "풍속(m/s)", "전운량(10분위)"]]
        weather["일시"] = pd.to_datetime(weather["일시"])
        weather = weather.set_index("일시").sort_index()
        weather.columns = ["temp", "wind_speed", "cloud"]
        
        df_gen = solar.set_index('datetime').join(weather, how="inner").interpolate(method='linear').fillna(0)
        df_gen['month'] = df_gen.index.month
        df_gen['hour'] = df_gen.index.hour
        
        X_gen = df_gen[['temp', 'wind_speed', 'cloud', 'month', 'hour']]
        y_gen = df_gen['solar_gen']
        gen_model = RandomForestRegressor(n_estimators=100, random_state=42)
        gen_model.fit(X_gen, y_gen)
        
        return consume_model, gen_model, df_gen['solar_gen'].max()
    except Exception as e:
        raise Exception(f"데이터 로드 중 오류 발생: {e}")

# ===== 2. 실시간 날씨 데이터 수신 =====
@st.cache_data
def fetch_weather_data(api_key):
    lat, lon = 33.4996, 126.5312
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    weather_list = []
    for item in data['list']:
        kst_time = pd.to_datetime(item['dt_txt']) + pd.Timedelta(hours=9)
        weather_list.append({
            'datetime': kst_time, 'temp': item['main']['temp'],
            'wind_speed': item['wind']['speed'], 'cloud': item['clouds']['all'] / 10.0 
        })
    df = pd.DataFrame(weather_list).set_index('datetime')
    return df.resample('1h').interpolate(method='linear')

# ===== 3. 시간대별 탄력 요금제 적용 함수 =====
def get_dynamic_price(hour):
    if 13 <= hour <= 17: return 250
    elif 18 <= hour <= 22: return 200
    elif hour >= 23 or hour <= 8: return 80
    else: return 130

# ===== 4. 웹 화면 구성 (Streamlit) =====
st.set_page_config(page_title="AI 스마트 그리드 통합 분석", page_icon="⚡", layout="wide")

st.title("⚡ AI 기반 가정용 스마트 그리드 & 경제성 분석 대시보드")
st.markdown("실시간 일기예보와 **시간대별 탄력 요금제(TOU)**를 반영하여 향후 5일간의 전력 시뮬레이션 및 경제적 가치를 분석합니다.")

# API 키 자동 설정
USER_API_KEY = "c836bff1b19e7105c684199643d71474"

with st.sidebar:
    st.header("⚙️ 시스템 상태")
    st.success("✅ 시뮬레이션 자동 실행 중...")

# 버튼 조건문(if run_button:)을 없애고 바로 실행되도록 수정
with st.spinner('AI 모델 분석 및 경제성 지표를 계산 중입니다...'):
    try:
        # 데이터 준비 및 예측
        consume_model, gen_model, max_gen_hist = load_models()
        future_weather = fetch_weather_data(USER_API_KEY)
        
        if future_weather is None:
            st.error("❌ 자동 입력된 API 키로 날씨 데이터를 가져오지 못했습니다. 키를 확인해 주세요.")
            st.stop()
        
        future_weather['month'] = future_weather.index.month
        future_weather['hour'] = future_weather.index.hour
        
        # 태양광 발전량 예측
        X_future_gen = future_weather[['temp', 'wind_speed', 'cloud', 'month', 'hour']]
        pred_solar = gen_model.predict(X_future_gen)
        max_gen = max_gen_hist if max_gen_hist > 0 else 1
        future_weather['home_gen_kwh'] = (np.maximum(pred_solar, 0) / max_gen) * 3.0
        
        # 가정 전력 소비량 예측
        unique_dates = pd.Series(future_weather.index.date).unique()
        home_use_list = []
        for d in unique_dates:
            target_date = pd.to_datetime(d)
            X_consume = pd.DataFrame({'월': [target_date.month], '일': [target_date.day], '요일': [target_date.dayofweek]})
            home_use_list.extend(consume_model.predict(X_consume).flatten())
        future_weather['home_use_kwh'] = home_use_list[:len(future_weather)]
        
        # 경제성 지표 계산 로직
        future_weather['net_power_kwh'] = future_weather['home_gen_kwh'] - future_weather['home_use_kwh']
        prices = [get_dynamic_price(h) for h in future_weather['hour']]
        
        future_weather['normal_cost'] = future_weather['home_use_kwh'] * prices 
        future_weather['trade_profit_krw'] = future_weather['net_power_kwh'] * prices 
        future_weather['cumulative_profit'] = future_weather['trade_profit_krw'].cumsum() 
        
        total_normal_pay = future_weather['normal_cost'].sum()
        total_solar_result = future_weather['trade_profit_krw'].sum()
        total_savings = total_normal_pay + total_solar_result 
        
        total_gen = future_weather['home_gen_kwh'].sum()
        total_use = future_weather['home_use_kwh'].sum()
        
        # [UI 파트 1] 경제적 가치 요약
        st.markdown("---")
        st.subheader("💰 5일간 경제적 가치 요약")
        st.success(f"✅ 이번 기간 동안 일반 가정 대비 총 **{total_savings:,.0f}원**의 가치를 창출했습니다.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("태양광 미설치 시 예상 지출", f"{total_normal_pay:,.0f}원", delta="-지출 예정", delta_color="inverse")
        with col2:
            st.metric("현재 시스템 순 손익", f"{total_solar_result:,.0f}원", delta="판매/절약 합계")
        with col3:
            st.metric("최종 창출 경제적 이득", f"{total_savings:,.0f}원", delta="총 절감액", delta_color="normal")

        # 막대 그래프
        fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
        categories = ['일반 가정 (예상 지출)', '태양광 스마트 그리드 (경제적 가치)']
        values = [total_normal_pay, total_savings]
        colors = ['#ff9999', '#66b3ff']
        
        bars = ax_bar.bar(categories, values, color=colors, alpha=0.8)
        ax_bar.set_title("시스템 도입 전후 경제성 비교", fontsize=15)
        ax_bar.set_ylabel("금액 (원)")
        
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height + 50, f'{int(height):,}원', ha='center', va='bottom', fontweight='bold')
        st.pyplot(fig_bar)

        # [UI 파트 2] 상세 시뮬레이션
        st.markdown("---")
        st.subheader("📈 실시간 발전/소비 시뮬레이션 상세")
        
        fig_ts, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        x_data = future_weather.index

        axes[0].plot(x_data, future_weather['home_gen_kwh'], color='orange', label='태양광 발전량', linewidth=2)
        axes[0].plot(x_data, future_weather['home_use_kwh'], color='gray', label='가정 소비량', linestyle='--', linewidth=2)
        axes[0].set_ylabel('전력량 (kWh)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, linestyle='--', alpha=0.7)

        colors_net = ['green' if val > 0 else 'red' for val in future_weather['net_power_kwh']]
        axes[1].bar(x_data, future_weather['net_power_kwh'], width=0.03, color=colors_net, alpha=0.7)
        axes[1].axhline(0, color='black', linewidth=1)
        axes[1].set_ylabel('순 전력량 (kWh)')
        axes[1].grid(True, linestyle='--', alpha=0.7)

        axes[2].plot(x_data, future_weather['cumulative_profit'], color='blue', linewidth=2)
        axes[2].fill_between(x_data, future_weather['cumulative_profit'], 0, color='blue', alpha=0.1)
        axes[2].set_ylabel('누적 수익 (원)')
        axes[2].grid(True, linestyle='--', alpha=0.7)
        axes[2].xaxis.set_major_locator(mdates.DayLocator())
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_ts)
        
    except Exception as e:
        st.error(f"⚠️ 시스템 실행 중 오류가 발생했습니다: {e}")