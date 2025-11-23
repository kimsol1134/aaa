"""
Streamlit 앱 설정 파일

환경 변수, 상수, 경로 관리
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# === API 설정 ===
DART_API_KEY = os.getenv('DART_API_KEY', '')

# === 경로 설정 ===
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = DATA_DIR / 'processed'

# 모델 파일 경로
MODEL_PATH = MODEL_DIR / '발표_Part3_v3_최종모델.pkl'
THRESHOLDS_PATH = MODEL_DIR / '발표_Part3_v3_임계값.pkl'
SCALER_PATH = MODEL_DIR / 'scaler.pkl'
FEATURES_PATH = MODEL_DIR / 'selected_features.csv'

# === 앱 설정 ===
APP_TITLE = "한국 기업 부도 예측 시스템"
APP_ICON = "📊"
PAGE_CONFIG = {
    'page_title': APP_TITLE,
    'page_icon': APP_ICON,
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# === 임계값 설정 ===
# Part 3 노트북의 최적 임계값 (Recall 80% 기준)
TRAFFIC_LIGHT_THRESHOLDS = {
    'green': 0.0168,   # < 1.68%: Safe
    'yellow': 0.0468,  # < 4.68%: Potential Risk
    'red': 1.0         # >= 4.68%: High Risk
}

# === 한글 폰트 설정 ===
import platform

if platform.system() == 'Darwin':  # macOS
    KOREAN_FONT = 'AppleGothic'
elif platform.system() == 'Windows':
    KOREAN_FONT = 'Malgun Gothic'
else:  # Linux
    KOREAN_FONT = 'NanumGothic'
