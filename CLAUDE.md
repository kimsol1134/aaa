# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

한국 기업 부도 예측 모델 프로젝트: 도메인 지식 기반 AI를 활용한 기업 부도 위험 실시간 예측 시스템

**핵심 특징:**
- 도메인 기반 Feature Engineering (유동성 위기, 지급불능, 재무조작 탐지 등 7개 카테고리)
- 불균형 데이터 처리 (SMOTE + Tomek Links)
- 앙상블 모델 (LightGBM, XGBoost, CatBoost 스태킹)
- Streamlit 기반 실시간 웹 대시보드

**데이터셋:** 한국 기업 50,000+ 개, 170+ 재무/신용 변수, 부도율 ~1.5% (불균형 비율 1:20)

## 필수 준수 사항

### 1. 절대 하드 코딩 금지
- 모든 설정값은 환경 변수나 설정 파일로 관리
- 상수는 별도 constants 파일에 정의
- API 엔드포인트, 파일 경로 등 모두 변수화

### 2. 한글 폰트 깨짐 방지
- UTF-8 인코딩 확인
- 한글 폰트 설정 시 폴백 폰트 지정 (macOS: AppleGothic, Windows: Malgun Gothic, Linux: NanumGothic)
- 차트/그래프 생성 시 한글 폰트 명시적 설정
- `plt.rc('axes', unicode_minus=False)` 설정으로 마이너스 기호 깨짐 방지

### 3. 데이터 특성
- **시계열 데이터가 아님** - 데이터 처리 시 시간 순서 의존적 로직 사용 금지
- 독립적인 데이터 포인트로 처리 (횡단면 데이터)
- 2021년 8월 기준 스냅샷 데이터

## 개발 환경 설정

### 패키지 설치
```bash
pip install -r requirements.txt
```

### Jupyter Notebook 실행
```bash
jupyter notebook
```

### Streamlit 앱 실행
```bash
cd streamlit_app
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

## 노트북 실행 순서

**반드시 순서대로 실행:**

1. `01_도메인_기반_부도원인_분석.ipynb` - 데이터 탐색 및 도메인 분석
2. `02_고급_도메인_특성공학.ipynb` - 도메인 지식 기반 특성 생성 (65개 특성)
3. `03_상관관계_및_리스크_패턴_분석.ipynb` - 특성 선택 및 상관관계 분석
4. `04_불균형_분류_모델링_final.ipynb` - 모델 학습 및 최적화
5. `05_모델_평가_및_해석.ipynb` - SHAP 분석 및 성능 평가

**주의:** 각 노트북은 이전 단계의 출력 파일에 의존하므로 순서를 지켜야 함

## 프로젝트 구조

```
junwoo/
├── data/
│   ├── 기업신용평가정보_210801.csv         # 원본 데이터 (50,105 기업, 159 변수)
│   ├── ksic_mapping.csv                    # 한국표준산업분류 매핑
│   ├── features/                           # 엔지니어링된 특성
│   │   ├── domain_based_features.csv       # 65개 도메인 특성
│   │   └── feature_metadata.csv            # 특성 메타데이터
│   └── processed/                          # 전처리 및 모델 파일
│       ├── best_model_*.pkl                # 학습된 모델들
│       ├── scaler.pkl                      # StandardScaler
│       ├── selected_features.csv           # 선택된 특성 데이터
│       └── preprocessing_pipeline.pkl      # 전처리 파이프라인
├── notebooks/                              # Jupyter 노트북 (순서대로 실행)
├── streamlit_app/
│   └── app.py                             # Streamlit 웹 대시보드
├── scripts/                                # 유틸리티 스크립트
│   ├── create_industry_analysis.py         # 업종별 분석
│   ├── fix_financial_calculations.py       # 재무 계산 수정
│   ├── fix_heatmap.py                     # 히트맵 수정
│   └── parse_ksic.py                      # KSIC 코드 파싱
├── src/                                    # 소스 코드 (현재 비어있음, 향후 모듈화 예정)
│   ├── domain_features/                    # 도메인 특성 생성 로직
│   ├── models/                            # 모델 클래스
│   ├── evaluation/                        # 평가 메트릭
│   └── utils/                             # 유틸리티 함수
└── docs/                                   # 프로젝트 문서
    └── bankruptcy_prediction_plan.md       # 프로젝트 계획서
```

## 핵심 아키텍처 및 개념

### 1. 도메인 기반 특성 공학 (총 65개 특성)

**특성 카테고리별 구성:**
- **유동성 위기 (10개)**: 즉각지급능력, 현금소진일수, 운전자본 건전성, 긴급유동성
- **지급불능 패턴 (8개)**: 자본잠식도, 이자보상배율, 부채상환년수, 재무레버리지
- **재무조작 탐지 (15개)**: 한국형 M-Score, 매출채권/재고 이상지표, 발생액 품질
- **한국 시장 특화 (13개)**: 대기업 의존도, 제조업 리스크, 외감 여부, 업력
- **이해관계자 행동 (9개)**: 연체/세금체납, 신용등급, 이해관계자 불신지수
- **복합 리스크 지표 (7개)**: 종합부도위험스코어, 조기경보신호수, 재무건전성지수
- **상호작용/비선형 (3개)**: 레버리지×수익성, 부채비율 제곱, 임계값 기반 특성

**특성 생성 핵심 원칙:**
- "왜 기업이 부도가 나는가?"에 대한 도메인 지식 반영
- 통계적 접근이 아닌 실무적/이론적 근거 기반
- 한국 시장 특성 반영 (외감 여부, 제조업 중심, 대기업 의존도)

### 2. 불균형 데이터 처리

**핵심 전략:**
- SMOTE + Tomek Links: 소수 클래스 오버샘플링 + 경계 정리
- BorderlineSMOT: 경계선 샘플 중심 생성
- 클래스 가중치 (class_weight='balanced')
- 평가 메트릭: PR-AUC (핵심), F2-Score, Type II Error < 20%

**주의사항:**
- ROC-AUC는 불균형 데이터에서 오해의 소지가 있으므로 PR-AUC를 주요 지표로 사용
- F2-Score는 재현율 중시 (부도 미탐지 최소화)

### 3. 앙상블 모델 구조

```python
# 스태킹 앙상블 구조
Level 1 (Base Models):
  - LightGBM (빠른 학습, 높은 정확도)
  - XGBoost (강력한 성능)
  - CatBoost (범주형 변수 처리 우수)

Level 2 (Meta Learner):
  - Logistic Regression (앙상블 결과 통합)
```

**모델 저장 위치:** `data/processed/best_model_*.pkl`

### 4. 데이터 파일 경로 규칙

**원본 데이터:**
- `data/기업신용평가정보_210801.csv` (노트북에서 `../data/` 경로 사용)

**생성된 특성:**
- `data/features/domain_based_features.csv`
- `data/features/feature_metadata.csv`

**모델 파일:**
- `data/processed/best_model_XGBoost.pkl`
- `data/processed/scaler.pkl`
- `data/processed/selected_features.csv`

**노트북 내 경로 패턴:**
- 데이터 로딩: `pd.read_csv('../data/기업신용평가정보_210801.csv', encoding='utf-8')`
- 특성 저장: `df.to_csv('../data/features/domain_based_features.csv', encoding='utf-8-sig')`
- 모델 저장: `joblib.dump(model, '../data/processed/best_model_XGBoost.pkl')`

## 코딩 규칙

### 1. 범주형 변수 처리

범주형 변수(Category dtype)는 수치 계산 전에 반드시 숫자로 변환:

```python
# 잘못된 예 (에러 발생)
X_filled = X.fillna(X.median())  # Category dtype에서 median() 에러

# 올바른 예
if '위험경보등급' in X.columns:
    X['위험경보등급'] = X['위험경보등급'].cat.codes
X_filled = X.fillna(X.median())
```

### 2. 결측치 및 무한대 처리

```python
# 결측치 처리
X_filled = X.fillna(X.median())

# 무한대 값 처리 (재무 비율 계산 시 발생 가능)
X_filled = X_filled.replace([np.inf, -np.inf], 0)
```

### 3. 한글 폰트 설정 표준 코드

```python
import matplotlib.pyplot as plt
import platform

if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)
```

### 4. 모델 로딩 (Streamlit 앱)

```python
@st.cache_resource
def load_models():
    model_dir = '../data/processed/'
    model = joblib.load(os.path.join(model_dir, 'best_model_XGBoost.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    return model, scaler
```

## 일반적인 작업 패턴

### 노트북에서 새로운 특성 추가

1. `02_고급_도메인_특성공학.ipynb` 열기
2. 새로운 특성 생성 함수 작성 (예: `create_xxx_features(df)`)
3. 특성 생성 및 검증
4. `all_features`에 concat 추가
5. `data/features/domain_based_features.csv` 재저장
6. 이후 노트북 순서대로 재실행

### 모델 재학습

1. `04_불균형_분류_모델링_final.ipynb` 실행
2. `data/processed/selected_features.csv` 로드
3. SMOTE 적용 및 모델 학습
4. `data/processed/best_model_*.pkl` 저장
5. `05_모델_평가_및_해석.ipynb`로 성능 검증

### 스크립트 실행

```bash
# 업종별 분석 생성
python scripts/create_industry_analysis.py

# 재무 계산 수정
python scripts/fix_financial_calculations.py

# KSIC 코드 파싱
python scripts/parse_ksic.py
```

## 평가 메트릭 우선순위

1. **PR-AUC** (Precision-Recall AUC) - 불균형 데이터 핵심 지표
2. **F2-Score** - 재현율 중시 (부도 미탐지 최소화)
3. **Type II Error** - 부도 기업을 정상으로 잘못 분류한 비율 (< 20% 목표)
4. ROC-AUC - 참고용 (불균형 데이터에서 과대평가 가능)

## 주의사항

1. **타겟 변수명:** `모형개발용Performance(향후1년내부도여부)` (긴 컬럼명 주의)
2. **데이터 인코딩:** CSV 읽기 시 `encoding='utf-8'`, 쓰기 시 `encoding='utf-8-sig'` 사용
3. **클래스 불균형:** 항상 `stratify=y` 옵션으로 train/test split
4. **재무 비율 계산:** 분모에 항상 `+ 1` 또는 `+ 0.1` 추가하여 division by zero 방지
5. **모델 학습 시간:** CatBoost와 XGBoost는 학습 시간이 길 수 있음 (수 분 ~ 십수 분)
6. **메모리 사용:** 스태킹 앙상블 모델은 메모리를 많이 사용 (8GB+ 권장)

## 참고 자료

- Altman Z-Score (1968) - 전통적 부도 예측 모델
- Beneish M-Score (1999) - 재무조작 탐지
- SMOTE (2002) - 불균형 데이터 처리
- SHAP (2017) - 모델 해석
- Kaggle Imbalanced Classification Best Practices
