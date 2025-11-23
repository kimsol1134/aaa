# 📊 한국 기업 부도 예측 시스템 (Korean Corporate Bankruptcy Prediction)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B.svg)](https://streamlit.io/)

도메인 지식 기반 AI를 활용한 한국 기업 부도 위험 실시간 예측 시스템

> **⚠️ 중요**: 이 프로젝트는 학습 및 연구 목적의 프로토타입입니다. 현재 모델은 **높은 False Positive(25%)와 낮은 Precision(5%)** 으로 인해 실무 적용 시 개선이 필요합니다. 자세한 내용은 [모델 성능](#모델-성능-test-set) 및 [알려진 제한사항](#-알려진-제한사항-및-개선-과제)을 참조하세요.

## 🎯 프로젝트 개요

이 프로젝트는 한국 기업의 재무 데이터를 분석하여 부도 위험을 예측하는 머신러닝 시스템입니다. 단순한 통계적 접근이 아닌 **도메인 지식 기반의 특성 공학**을 통해 "왜 기업이 부도가 나는가?"에 대한 깊은 이해를 반영합니다.

### 핵심 특징

- ✅ **도메인 기반 Feature Engineering**: 65개의 전문가 지식 기반 특성 (유동성 위기, 지급불능, 재무조작 탐지 등)
- ✅ **불균형 데이터 처리**: SMOTE + Tomek Links를 활용한 최적화된 샘플링
- ✅ **앙상블 모델**: LightGBM, XGBoost, CatBoost 스태킹으로 높은 예측 정확도
- ✅ **실시간 웹 대시보드**: Streamlit 기반 대화형 분석 인터페이스
- ✅ **DART API 연동**: 한국 상장기업 실시간 재무제표 조회
- ✅ **설명 가능한 AI**: SHAP 분석을 통한 예측 근거 제시

### 데이터셋

- **기업 수**: 50,000+ 개 한국 기업
- **변수**: 170+ 개 재무 및 신용 변수
- **부도율**: ~1.5% (불균형 비율 1:20)
- **기준일**: 2021년 8월 (횡단면 데이터)

### 모델 성능 (Test Set)

**부도 탐지 능력 (강점)**
- **Recall**: 86.84% (부도 기업의 86.8%를 정확히 탐지)
- **Type II Error**: 13.16% (부도 미탐지율 < 20% 목표 달성)
- **ROC-AUC**: 0.8847 (높은 구분 능력)

**불균형 데이터 지표 (개선 필요)**
- **PR-AUC**: 0.1602 (불균형 데이터 핵심 지표)
- **F2-Score**: 0.2046 (재현율 중시)
- **Precision**: 5.04% (오탐 개선 필요)

**비즈니스 영향**
- **손실 감소**: +4.62억원 (부도 손실 86.8% 감소)
- **기회 손실**: -12.43억원 (정상 기업 2,486개 오탐)
- **순 효과**: -7.81억원 (임계값 조정으로 개선 가능)

---

## 📁 프로젝트 구조

```
junwoo/
├── data/                                        # 데이터 디렉토리
│   ├── 기업신용평가정보_210801.csv            # 원본 데이터 (50,105 기업, 159 변수)
│   ├── ksic_mapping.csv                         # 한국표준산업분류 매핑
│   ├── features/                                # 엔지니어링된 특성
│   │   ├── domain_based_features.csv            # 65개 도메인 특성
│   │   └── feature_metadata.csv                 # 특성 메타데이터
│   └── processed/                               # 전처리 및 모델 파일
│       ├── 발표_Part3_v3_최종모델.pkl          # 최종 학습 모델
│       ├── 발표_Part3_v3_임계값.pkl            # 최적 임계값
│       ├── scaler.pkl                           # StandardScaler
│       └── selected_features.csv                # 선택된 특성 데이터
│
├── notebooks/                                   # Jupyter 노트북
│   ├── 발표_Part1_문제정의_및_핵심발견_executed.ipynb     # Part 1: 문제 정의
│   ├── 발표_Part2_도메인_특성_공학_완전판_executed.ipynb  # Part 2: 특성 공학
│   ├── 발표_Part3_모델링_및_최적화_완전판_executed.ipynb  # Part 3: 모델링
│   ├── 발표_Part4_결과_및_비즈니스_가치.ipynb             # Part 4: 비즈니스 가치
│   └── backup/                                  # 이전 버전 노트북
│
├── src/                                         # 소스 코드 모듈
│   ├── domain_features/                         # 도메인 특성 생성
│   │   ├── feature_generator.py                 # 메인 특성 생성기
│   │   ├── liquidity_features.py                # 유동성 위기 특성 (10개)
│   │   ├── insolvency_features.py               # 지급불능 패턴 특성 (8개)
│   │   ├── manipulation_features.py             # 재무조작 탐지 특성 (15개)
│   │   ├── korea_market_features.py             # 한국 시장 특화 특성 (13개)
│   │   ├── stakeholder_features.py              # 이해관계자 행동 특성 (9개)
│   │   └── composite_features.py                # 복합 리스크 특성 (10개)
│   ├── models/                                  # 모델 관련
│   │   └── predictor.py                         # 부도 예측 모델 클래스
│   ├── dart_api/                                # DART API 연동
│   │   ├── client.py                            # API 클라이언트
│   │   └── parser.py                            # 재무제표 파싱
│   ├── visualization/                           # 시각화
│   │   └── charts.py                            # 대시보드 차트 생성
│   └── utils/                                   # 유틸리티
│       ├── helpers.py                           # 헬퍼 함수
│       └── business_value.py                    # 비즈니스 가치 계산
│
├── streamlit_app/                               # Streamlit 웹 앱
│   ├── app.py                                   # 메인 애플리케이션
│   ├── config.py                                # 설정 파일
│   ├── requirements.txt                         # 앱 의존성
│   └── .streamlit/
│       └── config.toml                          # Streamlit 설정
│
├── scripts/                                     # 유틸리티 스크립트
│   ├── create_industry_analysis.py             # 업종별 분석 생성
│   ├── create_notebook_summary.py               # 노트북 요약 문서 생성
│   ├── fix_financial_calculations.py           # 재무 계산 수정
│   └── parse_ksic.py                            # KSIC 코드 파싱
│
├── docs/                                        # 프로젝트 문서
│   ├── bankruptcy_prediction_plan.md            # 프로젝트 계획서
│   ├── notebook_summaries/                      # 노트북 요약 (Claude Code용)
│   │   ├── README.md                            # 요약 사용 가이드
│   │   └── 02_특성공학/                         # 특성공학 카테고리별 분할
│   │       ├── 00_개요_및_목차.md
│   │       ├── 01_유동성위기_특성.md
│   │       ├── 02_지급불능패턴_특성.md
│   │       ├── 03_재무조작탐지_특성.md
│   │       ├── 04_한국시장특화_특성.md
│   │       ├── 05_이해관계자행동_특성.md
│   │       ├── 06_복합리스크_특성.md
│   │       └── 07_상호작용비선형_특성.md
│   └── streamlit_prompts/                       # Streamlit 개발 가이드
│
├── CLAUDE.md                                    # Claude Code 사용 가이드
├── requirements.txt                             # Python 의존성
└── README.md                                    # 프로젝트 개요 (이 파일)
```

---

## 🚀 빠른 시작 (Quick Start)

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd junwoo

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. Streamlit 웹 앱 실행

```bash
cd streamlit_app
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

### 3. DART API 설정 (선택사항)

실제 상장기업 데이터를 조회하려면 DART API 키가 필요합니다:

```bash
# .env 파일 생성
echo "DART_API_KEY=your_api_key_here" > .env
```

[DART API 키 발급받기](https://opendart.fss.or.kr/intro/main.do)

---

## 📊 도메인 기반 특성 공학 (65개 특성)

이 프로젝트의 핵심은 **"왜 기업이 부도가 나는가?"**에 대한 도메인 지식을 반영한 특성 공학입니다.

### 특성 카테고리 (7가지)

#### 1. 유동성 위기 특성 (10개)
- **즉각지급능력**: 현금 / 유동부채
- **현금소진일수**: 현금 / 일평균 영업비용
- **운전자본건전성**: (유동자산 - 유동부채) / 매출액
- **긴급유동성비율**: (현금 + 단기금융상품) / 유동부채

#### 2. 지급불능 패턴 특성 (8개)
- **자본잠식도**: 1 - (자본총계 / 납입자본금)
- **이자보상배율**: EBIT / 이자비용
- **부채상환년수**: 총차입금 / 영업현금흐름
- **재무레버리지**: 부채총계 / 자본총계

#### 3. 재무조작 탐지 특성 (15개)
- **한국형 M-Score**: Beneish M-Score 개량
- **매출채권 이상 비율**: 매출채권 / 매출액 (업종 대비)
- **재고자산 적체율**: 재고자산 / 매출원가
- **발생액 품질**: (당기순이익 - 영업현금흐름) / 자산총계

#### 4. 한국 시장 특화 특성 (13개)
- **대기업 의존도**: 상위 5개 거래처 매출 비중
- **제조업 리스크**: 업종별 평균 부도율 대비
- **외감 여부**: 외부감사 실시 여부 (0/1)
- **업력**: 설립연도부터 경과 년수

#### 5. 이해관계자 행동 특성 (9개)
- **연체 점수**: 30일/60일/90일 연체 가중 합산
- **세금체납 건수**: 국세/지방세 체납 건수
- **신용등급 위험도**: 신용평가등급 점수화
- **이해관계자 불신지수**: 연체 + 체납 + 소송 종합

#### 6. 복합 리스크 지표 (7개)
- **종합부도위험스코어**: 다차원 리스크 통합 점수
- **조기경보신호수**: 임계값 초과 지표 개수
- **재무건전성지수**: 재무 안정성 종합 평가 (0-100점)
- **유동성스트레스지수**: 유동성 관련 지표 통합

#### 7. 상호작용/비선형 특성 (3개)
- **레버리지×수익성**: 부채비율 × ROA
- **부채비율 제곱**: 비선형 관계 포착
- **임계값 기반 특성**: 위험 구간 더미 변수

---

## 🤖 모델링 및 최적화

### 불균형 데이터 처리 전략

```python
# SMOTE + Tomek Links
from imblearn.combine import SMOTETomek

smote_tomek = SMOTETomek(
    smote=SMOTE(sampling_strategy=0.3, k_neighbors=5),
    tomek=TomekLinks(sampling_strategy='majority')
)

X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
```

### 앙상블 모델 구조

```
Level 1 (Base Models):
├── LightGBM    (빠른 학습, 높은 정확도)
├── XGBoost     (강력한 성능)
└── CatBoost    (범주형 변수 처리 우수)

Level 2 (Meta Learner):
└── Logistic Regression (앙상블 결과 통합)
```

### 평가 메트릭 우선순위

1. **PR-AUC** (Precision-Recall AUC) - 불균형 데이터 핵심 지표
2. **F2-Score** - 재현율 중시 (부도 미탐지 최소화)
3. **Type II Error** - 부도 기업을 정상으로 잘못 분류한 비율 (< 20% 목표)
4. **ROC-AUC** - 참고용 (불균형 데이터에서 과대평가 가능)

---

## 🌐 Streamlit 웹 애플리케이션

### 주요 기능

#### 1. 데이터 입력 방식 (3가지)
- **DART API 검색**: 상장기업 실시간 재무제표 조회
- **재무제표 직접 입력**: 비상장기업 또는 가상 데이터
- **샘플 데이터**: 정상/주의/위험 기업 샘플로 테스트

#### 2. 부도 위험 평가
- **신호등 시스템**: 🟢 안전 / 🟡 주의 / 🔴 위험
- **부도 확률**: 실시간 확률 계산 (0-100%)
- **핵심 지표**: 재무건전성, 조기경보신호, 종합위험스코어

#### 3. 위험 요인 분석
- **Critical 리스크**: 즉시 조치 필요한 항목
- **Warning**: 개선 권장 항목
- **SHAP Waterfall 차트**: 각 특성의 부도 확률 기여도 시각화

#### 4. 비즈니스 가치 분석
- **예상 손실/수익**: 대출 조건별 기대값 계산
- **ROI 계산기**: 모델 도입 효과 측정
- **시나리오 분석**: 승인/거절 시나리오 비교

#### 5. 실행 가능한 개선 권장사항
- 재무 상태 기반 맞춤형 개선 방안 제시
- 우선순위별 실행 계획 제공

### 사용 예시

```
┌─────────────────────────────────────────┐
│  🟢 안전                                │
│  부도 확률: 0.85% (기준: < 1.68%)       │
│  모든 핵심 지표가 정상 범위입니다       │
└─────────────────────────────────────────┘

📊 핵심 지표
┌──────────┬──────────┬──────────┬──────────┐
│부도 확률 │재무 건전성│조기경보  │종합위험  │
│  0.85%   │   78점   │   0개    │  25점    │
└──────────┴──────────┴──────────┴──────────┘
```

---

## 📈 노트북 실행 순서

**반드시 순서대로 실행해야 합니다** (각 노트북이 이전 단계의 출력 파일에 의존):

1. **Part 1: 문제 정의 및 핵심 발견**
   - `발표_Part1_문제정의_및_핵심발견_executed.ipynb`
   - 데이터 탐색, 부도 원인 분석, 업종별 특성 파악

2. **Part 2: 도메인 특성 공학**
   - `발표_Part2_도메인_특성_공학_완전판_executed.ipynb`
   - 65개 도메인 특성 생성 및 검증
   - 출력: `data/features/domain_based_features.csv`

3. **Part 3: 모델링 및 최적화**
   - `발표_Part3_모델링_및_최적화_완전판_executed.ipynb`
   - SMOTE 적용, 앙상블 모델 학습, 하이퍼파라미터 튜닝
   - 출력: `data/processed/발표_Part3_v3_최종모델.pkl`

4. **Part 4: 결과 및 비즈니스 가치**
   - `발표_Part4_결과_및_비즈니스_가치.ipynb`
   - SHAP 분석, 성능 평가, ROI 계산

---

## 🛠️ 개발 환경 설정

### Python 패키지

```txt
# 핵심 라이브러리
pandas >= 2.1.3
numpy >= 1.24.3
scikit-learn >= 1.3.2

# 머신러닝 모델
xgboost >= 2.0.2
lightgbm >= 4.1.0
catboost >= 1.2.2

# 불균형 데이터 처리
imbalanced-learn >= 0.9.0

# 모델 해석
shap >= 0.41.0

# 시각화
matplotlib >= 3.8.2
seaborn >= 0.11.0
plotly >= 5.18.0

# 웹 프레임워크
streamlit >= 1.29.0

# API & 유틸리티
requests >= 2.31.0
python-dotenv >= 1.0.0
joblib >= 1.3.2
```

### 한글 폰트 설정

```python
import matplotlib.pyplot as plt
import platform

if platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:  # Linux
    plt.rc('font', family='NanumGothic')

plt.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지
```

---

## 📝 주요 코딩 규칙 및 주의사항

### 1. 절대 하드 코딩 금지
```python
# ❌ 잘못된 예
MODEL_PATH = '/Users/john/project/model.pkl'

# ✅ 올바른 예
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / 'data' / 'processed' / 'model.pkl'
```

### 2. 범주형 변수 처리
```python
# Category dtype은 수치 계산 전에 변환 필수
if '위험경보등급' in X.columns:
    X['위험경보등급'] = X['위험경보등급'].cat.codes

X_filled = X.fillna(X.median())
```

### 3. 데이터 특성
- **시계열 데이터가 아님** - 2021년 8월 기준 횡단면(cross-sectional) 데이터
- 시간 순서 의존적 로직 사용 금지
- 독립적인 데이터 포인트로 처리

### 4. 재무 비율 계산
```python
# Division by zero 방지
부채비율 = 부채총계 / (자본총계 + 0.1)
유동비율 = 유동자산 / (유동부채 + 0.1)
```

---

## 🎓 학습 리소스

### 이론적 배경
- **Altman Z-Score (1968)**: 전통적 부도 예측 모델
- **Beneish M-Score (1999)**: 재무조작 탐지 모델
- **SMOTE (2002)**: 불균형 데이터 처리 기법
- **SHAP (2017)**: 모델 해석 프레임워크

### 프로젝트 문서
- [프로젝트 계획서](docs/bankruptcy_prediction_plan.md)
- [Claude Code 사용 가이드](CLAUDE.md)
- [노트북 요약 문서](docs/notebook_summaries/README.md)
- [Streamlit 배포 가이드](docs/streamlit_prompts/README.md)

---

## 🚧 알려진 제한사항 및 개선 과제

### 데이터 및 시스템
1. **데이터 범위**: 2021년 8월 기준 (최신 데이터 업데이트 필요)
2. **시계열 미지원**: 현재는 횡단면 분석만 지원 (시계열 예측 미구현)
3. **DART API 제한**: 상장기업만 조회 가능 (비상장기업은 직접 입력)
4. **메모리 사용량**: 앙상블 모델 학습 시 8GB+ RAM 권장

### 모델 성능 개선 과제
5. **높은 False Positive 비율 (25%)**
   - 현재: 정상 기업 2,486개를 부도로 잘못 분류
   - 원인: Recall 우선 전략으로 임계값(0.0497)을 매우 낮게 설정
   - 해결 방안: 임계값 조정, Cost-sensitive learning, 앙상블 가중치 최적화

6. **낮은 Precision (5.04%)**
   - 모델이 "부도"라고 예측한 것 중 실제 부도는 5%만
   - 불균형 비율 1:65의 구조적 한계
   - 개선: Focal Loss, 업종별 모델, 더 많은 도메인 특성

7. **비즈니스 효율성 개선 필요**
   - 현재: 순 효과 -7.81억원 (기회 손실 > 손실 감소)
   - 목표: 임계값 조정으로 Precision ↑, FP ↓
   - 예상: 임계값 0.1로 조정 시 FP 50% 감소 가능

---

## 🤝 기여 방법

이 프로젝트는 학습 및 연구 목적으로 개발되었습니다. 기여를 환영합니다!

### 우선순위 개선 과제

**🔥 높은 우선순위 (성능 개선)**
1. **임계값 최적화**: 현재 0.0497 → 0.1~0.15로 조정하여 FP 감소
2. **Cost-sensitive Learning**: 비즈니스 비용 기반 손실 함수 설계
3. **업종별 모델**: 제조업, 서비스업, 건설업별 개별 모델 개발
4. **Focal Loss 적용**: 불균형 데이터 특화 손실 함수

**📊 중간 우선순위 (기능 추가)**
5. **실시간 시계열 분석**: 재무제표 변화 추적 및 추세 분석
6. **앙상블 가중치 재조정**: Precision 우선 모델 추가
7. **더 많은 도메인 특성**: 현금흐름 패턴, 시장 점유율 등
8. **REST API 서버**: 외부 시스템 연동

**🔧 낮은 우선순위 (편의성)**
9. **대시보드 개선**: 임계값 조정 UI 추가
10. **배치 예측**: 여러 기업 동시 분석

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

## 📧 문의

프로젝트 관련 문의사항은 이슈를 통해 남겨주세요.

---

## 🙏 감사의 말

- **금융감독원 전자공시시스템 (DART)**: 공개 재무 데이터 제공
- **Scikit-learn, XGBoost, LightGBM, CatBoost**: 머신러닝 프레임워크
- **Streamlit**: 빠른 웹 앱 프로토타이핑
- **SHAP**: 모델 해석 도구

---

**마지막 업데이트**: 2025-11-23
