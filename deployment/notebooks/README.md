# 📓 Jupyter Notebooks - 프로젝트 진행 과정

이 디렉토리에는 한국 기업 부도 예측 모델의 **전체 개발 과정**이 담긴 Jupyter 노트북이 포함되어 있습니다.

---

## 📊 발표용 노트북 (메인)

### Part 1: 문제 정의 및 핵심 발견
**파일**: `발표_Part1_문제정의_및_핵심발견_executed.ipynb` (1.1MB)

**내용**:
- 📈 데이터셋 개요 (50,000+ 기업, 170+ 변수)
- 🎯 비즈니스 문제 정의
- 📉 불균형 데이터 분석 (부도율 1.5%)
- 🔍 핵심 발견 사항
- 💡 도메인 지식 기반 접근법 제시

**주요 발견**:
- 부도 기업은 유동성, 지급불능, 재무조작의 3가지 패턴으로 구분
- 한국 시장 특화 요인 (외감 여부, 제조업 중심, 대기업 의존도)

---

### Part 2: 도메인 특성 공학
**파일**: `발표_Part2_도메인_특성_공학_완전판_executed.ipynb` (116KB)

**내용**:
- 🏗️ 65개 도메인 기반 특성 생성
- 📋 7개 카테고리 특성 설계:
  1. **유동성 위기** (10개): 즉각지급능력, 현금소진일수 등
  2. **지급불능 패턴** (8개): 자본잠식도, 이자보상배율 등
  3. **재무조작 탐지** (15개): 한국형 M-Score, 발생액 품질 등
  4. **한국 시장 특화** (13개): 대기업 의존도, 제조업 리스크 등
  5. **이해관계자 행동** (9개): 연체/세금체납, 신용등급 등
  6. **복합 리스크** (7개): 종합부도위험스코어 등
  7. **상호작용/비선형** (3개): 레버리지×수익성 등

**핵심 원칙**:
- "왜 기업이 부도가 나는가?"에 대한 도메인 지식 반영
- 통계적 접근이 아닌 실무적/이론적 근거 기반

---

### Part 3: 모델링 및 최적화
**파일**: `발표_Part3_모델링_및_최적화_완전판_executed.ipynb` (3.3MB)

**내용**:
- 🔄 불균형 데이터 처리 (SMOTE, BorderlineSMOTE, Tomek Links)
- 🤖 앙상블 모델 학습:
  - LightGBM
  - XGBoost
  - CatBoost
  - 스태킹 앙상블 (Logistic Regression 메타 모델)
- 🎯 하이퍼파라미터 최적화 (Optuna)
- 📊 모델 평가 (PR-AUC, F2-Score, Type II Error)
- 🔍 특성 중요도 분석

**최종 성능**:
- **PR-AUC**: 0.85+
- **F2-Score**: 0.75+
- **Type II Error**: < 20%

---

### Part 4: 결과 및 비즈니스 가치
**파일**: `발표_Part4_결과_및_비즈니스_가치.ipynb` (872KB)

**내용**:
- 📈 모델 성능 종합 분석
- 💰 비즈니스 가치 계산:
  - **ROI**: 920% (투자 대비 9배 수익)
  - **Payback**: 1.3개월
  - **연간 절감**: 460M KRW
- 🎯 임계값 최적화 (Traffic Light System)
- 🔍 SHAP 분석 및 모델 해석
- 💡 실무 적용 방안

---

## 🗂️ backup/ 디렉토리

### 개발 과정 노트북

**파일 목록**:
1. `01_도메인_기반_부도원인_분석.ipynb` - 초기 데이터 탐색
2. `01_심화_재무_분석.ipynb` - 재무 지표 심화 분석
3. `02_고급_도메인_특성공학.ipynb` - 특성 생성 상세 과정
4. `03_상관관계_및_리스크_패턴_분석.ipynb` - 특성 선택 및 상관관계
5. `04_불균형_분류_모델링_final.ipynb` - 모델 학습 및 최적화
6. `05_모델_평가_및_해석.ipynb` - SHAP 분석 및 성능 평가

**분할 버전**:
- `01_도메인_기반_부도원인_분석_Part1_데이터로딩_및_기본분석.ipynb`
- `01_도메인_기반_부도원인_분석_Part2_한국시장_특화패턴.ipynb`
- `01_도메인_기반_부도원인_분석_Part3_업종별_리스크_및_종합.ipynb`

> 💡 **참고**: backup/ 디렉토리는 프로젝트 개발 중간 과정을 보여주는 노트북들입니다. 최종 결과는 발표용 노트북(Part 1-4)을 참조하세요.

---

## 🚀 노트북 실행 방법

### 1. 환경 설정

```bash
# 루트 디렉토리에서 패키지 설치
pip install -r requirements.txt

# Jupyter 설치 확인
jupyter --version
```

### 2. Jupyter Notebook 실행

```bash
# 노트북 서버 시작
jupyter notebook

# 또는 JupyterLab 사용
jupyter lab
```

브라우저에서 `http://localhost:8888` 접속

### 3. 노트북 실행 순서 (권장)

**발표용 노트북**:
1. Part 1: 문제정의 및 핵심발견
2. Part 2: 도메인 특성 공학
3. Part 3: 모델링 및 최적화
4. Part 4: 결과 및 비즈니스 가치

**개발 과정 노트북 (backup/)**:
1. `01_도메인_기반_부도원인_분석.ipynb`
2. `02_고급_도메인_특성공학.ipynb`
3. `03_상관관계_및_리스크_패턴_분석.ipynb`
4. `04_불균형_분류_모델링_final.ipynb`
5. `05_모델_평가_및_해석.ipynb`

---

## 📋 필요한 데이터 파일

노트북 실행에는 다음 데이터 파일이 필요합니다:

### 원본 데이터 (필수)
- `../data/기업신용평가정보_210801.csv` - 메인 데이터셋
- `../data/ksic_mapping.csv` - 한국표준산업분류 매핑

### 생성된 특성 (Part 2 실행 후 생성)
- `../data/features/domain_based_features.csv` - 65개 도메인 특성
- `../data/features/feature_metadata.csv` - 특성 메타데이터

### 모델 파일 (Part 3 실행 후 생성)
- `../data/processed/best_model_*.pkl` - 학습된 모델들
- `../data/processed/scaler.pkl` - StandardScaler
- `../data/processed/selected_features.csv` - 선택된 특성

> ⚠️ **주의**: 원본 데이터 파일(`기업신용평가정보_210801.csv`)은 크기가 크므로 Git에 포함되지 않습니다. 프로젝트 관리자에게 별도로 요청하세요.

---

## 🔍 주요 기술 및 라이브러리

### 데이터 처리
- pandas, numpy
- scikit-learn

### 불균형 데이터 처리
- imbalanced-learn (SMOTE, BorderlineSMOTE, Tomek Links)

### 머신러닝 모델
- LightGBM, XGBoost, CatBoost
- scikit-learn (Logistic Regression, Random Forest)

### 모델 해석
- SHAP (SHapley Additive exPlanations)

### 시각화
- matplotlib, seaborn, plotly

### 통계
- scipy, statsmodels

---

## 💡 노트북 사용 팁

### 1. 메모리 부족 시
```python
# 데이터 샘플링
df_sample = df.sample(n=10000, random_state=42)

# 불필요한 변수 삭제
del large_dataframe
import gc
gc.collect()
```

### 2. 실행 시간 단축
```python
# 모델 학습 시 n_jobs 활용
model = LGBMClassifier(n_jobs=-1)  # 모든 CPU 코어 사용

# 파라미터 최적화 시 trials 줄이기
study.optimize(objective, n_trials=50)  # 100 → 50
```

### 3. 한글 폰트 깨짐 방지
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

---

## 📊 예상 실행 시간

| 노트북 | 실행 시간 | 비고 |
|-------|----------|------|
| Part 1 | ~5분 | 데이터 로딩 및 탐색 |
| Part 2 | ~3분 | 특성 생성 |
| Part 3 | ~30분 | 모델 학습 (CPU 기준) |
| Part 4 | ~5분 | 평가 및 시각화 |

> 💡 **팁**: Part 3의 모델 학습은 GPU 사용 시 10-15분으로 단축 가능합니다.

---

## 🆘 문제 해결

### 문제 1: 모듈 import 에러
```bash
# 패키지 재설치
pip install -r requirements.txt --upgrade
```

### 문제 2: 데이터 파일 없음
```python
# 에러 메시지:
# FileNotFoundError: ../data/기업신용평가정보_210801.csv

# 해결: 프로젝트 루트에서 실행하거나 경로 수정
```

### 문제 3: 메모리 부족
```python
# Jupyter 메모리 증가 (환경 변수 설정)
export JUPYTER_MEMORY_LIMIT=8G
```

---

## 📞 문의

노트북 관련 문의사항:
- **GitHub Issues**: 레포지토리에 이슈 등록
- **Email**: your.email@example.com

---

## 📄 라이센스

MIT License - 자유롭게 사용, 수정, 배포 가능

---

**⭐ 이 노트북들이 도움이 되셨다면 Star를 눌러주세요!**
