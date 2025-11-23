# Part3 파이프라인 통합 가이드

## 개요

Streamlit 앱이 이제 Part3 노트북과 동일한 전처리 파이프라인을 사용합니다.

## 주요 변경사항

### 1. 새로운 전처리 모듈 추가

**파일:** `src/preprocessing/transformers.py`

Part3 노트북의 커스텀 transformer 클래스들:

- **InfiniteHandler**: 무한대 값을 0으로 변환
- **LogTransformer**: 양수 값에 대해 log1p 변환 적용
- **Winsorizer**: 이상치를 특정 백분위수로 클리핑
- **create_preprocessing_pipeline()**: 전체 전처리 파이프라인 생성 함수

```python
from src.preprocessing.transformers import create_preprocessing_pipeline

# 파이프라인 생성
pipeline = create_preprocessing_pipeline(
    use_log_transform=True,
    use_winsorizer=False,
    scaler_type='robust'
)
```

### 2. Predictor 업데이트

**파일:** `src/models/predictor.py`

BankruptcyPredictor가 이제 3가지 모드를 지원합니다:

#### 모드 1: Part3 전체 파이프라인 (우선)
```python
predictor = BankruptcyPredictor(
    pipeline_path=Path('data/processed/발표_Part3_v3_최종모델.pkl'),
    use_pipeline=True
)
```

- 전처리 + 모델이 하나의 파이프라인으로 통합
- InfiniteHandler → Imputer → LogTransformer → RobustScaler → SMOTE → Classifier
- Part3 노트북과 100% 동일한 전처리

#### 모드 2: 전처리 파이프라인 + 모델 분리
```python
predictor = BankruptcyPredictor(
    model_path=Path('model.pkl'),
    use_pipeline=True
)
```

- 모델은 별도 pkl 파일
- 전처리는 create_preprocessing_pipeline()으로 생성

#### 모드 3: 레거시 모드 (기존)
```python
predictor = BankruptcyPredictor(
    model_path=Path('model.pkl'),
    scaler_path=Path('scaler.pkl'),
    use_pipeline=False
)
```

- 기존 방식 (모델 + 스케일러 분리)
- 후방 호환성 유지

### 3. Config 업데이트

**파일:** `config.py`

```python
# Part3 파이프라인 모델 (우선 사용)
PIPELINE_PATH = MODEL_DIR / '발표_Part3_v3_최종모델.pkl'

# 레거시 모델 (파이프라인 없을 경우)
MODEL_PATH = MODEL_DIR / 'best_model.pkl'
SCALER_PATH = MODEL_DIR / 'scaler.pkl'
```

### 4. App.py 업데이트

**파일:** `app.py`

```python
@st.cache_resource
def load_predictor():
    """모델 로딩 (캐시) - Part3 파이프라인 우선 사용"""
    predictor = BankruptcyPredictor(
        pipeline_path=PIPELINE_PATH,  # Part3 전체 파이프라인
        model_path=MODEL_PATH,        # 레거시 모델 (fallback)
        scaler_path=SCALER_PATH,      # 레거시 스케일러 (fallback)
        use_pipeline=True             # 파이프라인 우선 사용
    )
    predictor.load_model()
    return predictor
```

## 모델 학습 방법

### Part3 파이프라인 모델 생성

**스크립트:** `train_final_model.py`

```bash
python train_final_model.py
```

**출력 파일:**
- `data/processed/발표_Part3_v3_최종모델.pkl` - 전체 파이프라인 (전처리 + 모델)
- `data/processed/발표_Part3_v3_임계값.pkl` - 최적 임계값
- `data/processed/preprocessing_pipeline.pkl` - 전처리만 분리

**파이프라인 구조:**
```
Pipeline([
    ('inf_handler', InfiniteHandler()),
    ('imputer', SimpleImputer(strategy='median')),
    ('log_transform', LogTransformer()),
    ('scaler', RobustScaler()),
    ('smote', SMOTE(sampling_strategy=0.2, random_state=42)),
    ('classifier', LogisticRegression(...))
])
```

## 테스트 방법

### 전체 파이프라인 테스트

**스크립트:** `test_dart_pipeline.py`

```bash
python test_dart_pipeline.py
```

**테스트 내용:**
1. DART API로 기업 데이터 조회 (또는 더미 데이터 사용)
2. 도메인 특성 65개 생성
3. Part3 파이프라인으로 예측
4. 휴리스틱 모델과 비교

**출력 예시:**
```
🔧 더미 데이터로 특성 생성 중...
   ✅ 특성 생성 완료: 65개

🎯 예측 테스트...

   [A] Part3 파이프라인 모델 테스트
   📊 테스트 기업 부도 예측 결과:
   - 부도 확률: 15.23%
   - 위험 등급: 🟡 주의
   - 신뢰도: 85.43%
   - 모델: Pipeline(LogisticRegression)
   - 사용 특성 수: 27개
   - SHAP 분석: ✅ 완료

   [B] 휴리스틱 모델 테스트 (모델 없음)
   📊 테스트 기업 부도 예측 결과 (휴리스틱):
   - 부도 확률: 11.08%
   - 위험 등급: 🔴 고위험
   - 신뢰도: 70.00%
   - 모델: Heuristic
```

## 폴더 구조

```
deployment/
├── src/
│   ├── preprocessing/          # 🆕 Part3 전처리 모듈
│   │   ├── __init__.py
│   │   └── transformers.py    # InfiniteHandler, LogTransformer, Winsorizer
│   ├── models/
│   │   └── predictor.py       # ✏️ 파이프라인 지원으로 업데이트
│   └── ...
├── data/
│   └── processed/
│       ├── 발표_Part3_v3_최종모델.pkl  # 🆕 Part3 전체 파이프라인
│       ├── 발표_Part3_v3_임계값.pkl
│       ├── preprocessing_pipeline.pkl  # 전처리만 분리
│       ├── best_model.pkl              # 레거시 모델 (fallback)
│       └── scaler.pkl                  # 레거시 스케일러 (fallback)
├── config.py                   # ✏️ PIPELINE_PATH 추가
├── app.py                      # ✏️ 파이프라인 우선 사용
├── train_final_model.py        # 🆕 Part3 모델 학습 스크립트
└── test_dart_pipeline.py       # 🆕 파이프라인 테스트 스크립트
```

## 후방 호환성

**모델 파일이 없어도 앱은 정상 작동합니다:**

1. **Part3 파이프라인 모델 있을 때** → 파이프라인 사용 (최적)
2. **레거시 모델 있을 때** → 레거시 모델 + 스케일러 사용
3. **모델이 전혀 없을 때** → 휴리스틱 모델 사용 (도메인 지식 기반)

## 다음 단계

### Streamlit Cloud 배포 전 체크리스트

- [ ] `train_final_model.py` 실행하여 Part3 모델 생성
- [ ] Git LFS로 모델 파일 추적 (`*.pkl` 파일)
- [ ] `.env` 파일에 DART_API_KEY 설정 (로컬 테스트용)
- [ ] Streamlit Cloud에서 Secret으로 DART_API_KEY 설정
- [ ] `requirements.txt`에 `python-dotenv` 추가됨 확인
- [ ] `packages.txt`에 한글 폰트 패키지 추가됨 확인

### 로컬 테스트

```bash
# 1. 환경 변수 설정
cp .env.example .env
# .env 파일에 실제 DART_API_KEY 입력

# 2. 모델 학습 (선택)
python train_final_model.py

# 3. 파이프라인 테스트
python test_dart_pipeline.py

# 4. Streamlit 앱 실행
streamlit run app.py
```

## 참고사항

- Part3 파이프라인은 노트북의 최고 성능 모델과 동일한 전처리를 보장합니다
- SMOTE는 학습 시에만 적용되며, 예측 시에는 적용되지 않습니다
- LogTransformer는 양수 값에만 적용되므로 음수 값이 있는 특성은 변환되지 않습니다
- 휴리스틱 모델은 모델 파일 없이도 합리적인 예측을 제공합니다 (신뢰도는 낮음)
