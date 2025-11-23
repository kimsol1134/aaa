# 📁 배포 폴더 구조 완전 가이드

## 🎯 전체 구조 한눈에 보기

```
deployment/                          # 🚀 배포용 루트 디렉토리
│
├── 📱 앱 진입점 및 설정
│   ├── app.py                      # ⭐ Streamlit 메인 앱 (21KB)
│   ├── config.py                   # 설정 파일 (경로, API 키, 상수)
│   └── .env.example                # 환경 변수 예시
│
├── 📦 패키지 및 의존성
│   ├── requirements.txt            # Python 패키지 (18개)
│   └── packages.txt                # 시스템 패키지 (한글 폰트)
│
├── 🔧 Git 설정
│   ├── .gitignore                  # Git 무시 파일
│   └── .gitattributes              # Git LFS 설정
│
├── 📚 문서
│   ├── README.md                   # 프로젝트 소개 및 사용법
│   ├── DEPLOYMENT_GUIDE.md         # 배포 완벽 가이드
│   └── STRUCTURE.md                # 이 파일 (폴더 구조 설명)
│
├── ⚙️ Streamlit 설정
│   └── .streamlit/
│       └── config.toml             # 테마, 서버, 브라우저 설정
│
├── 🎨 소스 코드
│   └── src/
│       ├── __init__.py
│       │
│       ├── dart_api/               # DART API 연동 (3 파일)
│       │   ├── __init__.py
│       │   ├── client.py           # DART API 클라이언트
│       │   └── parser.py           # 재무제표 파싱
│       │
│       ├── domain_features/        # 도메인 특성 생성 (8 파일)
│       │   ├── __init__.py
│       │   ├── feature_generator.py
│       │   ├── liquidity_features.py
│       │   ├── insolvency_features.py
│       │   ├── manipulation_features.py
│       │   ├── korea_market_features.py
│       │   ├── stakeholder_features.py
│       │   └── composite_features.py
│       │
│       ├── models/                 # 모델 (2 파일)
│       │   ├── __init__.py
│       │   └── predictor.py        # 예측 모델 (휴리스틱 지원)
│       │
│       ├── visualization/          # 시각화 (2 파일)
│       │   ├── __init__.py
│       │   └── charts.py           # Plotly 차트 생성
│       │
│       └── utils/                  # 유틸리티 (3 파일)
│           ├── __init__.py
│           ├── helpers.py          # 헬퍼 함수
│           └── business_value.py   # 비즈니스 가치 계산
│
└── 💾 데이터 및 모델
    └── data/
        └── processed/
            └── best_model.pkl      # 학습된 모델 (103KB, Git LFS)
```

---

## 📊 통계

| 항목 | 개수/크기 |
|-----|----------|
| **총 파일 수** | 29개 |
| **Python 파일** | 19개 |
| **설정 파일** | 7개 |
| **문서 파일** | 3개 |
| **총 크기** | ~500KB (모델 포함) |
| **모델 크기** | 103KB |

---

## 🔍 주요 파일 상세 설명

### 1. 앱 진입점

#### `app.py` (21KB)
- **역할**: Streamlit 메인 앱
- **주요 기능**:
  - 3가지 입력 방식 (DART API, 직접 입력, 샘플)
  - 실시간 부도 예측
  - SHAP 분석 시각화
  - 비즈니스 가치 계산
  - 개선 권장사항 제공
- **라이브러리**: Streamlit, Pandas, Plotly, Matplotlib

#### `config.py` (1.3KB)
- **역할**: 전역 설정 관리
- **내용**:
  - API 키 로드
  - 파일 경로 설정
  - 앱 메타데이터
  - 임계값 정의
  - 한글 폰트 설정

---

### 2. 패키지 관리

#### `requirements.txt` (464B)
**Python 패키지 (18개)**:
```txt
streamlit==1.29.0          # Web Framework
pandas==2.1.3              # Data Processing
numpy==1.24.3
plotly==5.18.0             # Visualization
matplotlib==3.8.2
requests==2.31.0           # HTTP
python-dotenv==1.0.0       # Env Variables
scikit-learn==1.3.2        # ML
xgboost==2.0.2
lightgbm==4.1.0
catboost==1.2.2
joblib==1.3.2
shap==0.43.0               # Model Interpretation
```

#### `packages.txt` (3줄)
**시스템 패키지 (한글 폰트)**:
```txt
fonts-nanum                # 나눔 기본 폰트
fonts-nanum-coding         # 나눔 코딩 폰트
fonts-nanum-extra          # 나눔 추가 폰트
```

---

### 3. Git 설정

#### `.gitattributes`
**Git LFS 추적 파일**:
```gitattributes
*.pkl filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
```

#### `.gitignore`
**무시할 파일/폴더**:
- Python 캐시 (`__pycache__/`)
- 가상환경 (`venv/`, `env/`)
- 환경 변수 (`.env`)
- IDE 설정 (`.vscode/`, `.idea/`)
- 임시 파일 (`*.tmp`, `*.log`)

---

### 4. 소스 코드 모듈

#### `src/dart_api/` (3 파일)
- **client.py**: DART API 호출, 인증, 에러 처리
- **parser.py**: 재무제표 JSON → Python dict 변환

#### `src/domain_features/` (8 파일)
**65개 도메인 특성 생성**:
- **liquidity_features.py**: 유동성 위기 (10개)
- **insolvency_features.py**: 지급불능 패턴 (8개)
- **manipulation_features.py**: 재무조작 탐지 (15개)
- **korea_market_features.py**: 한국 시장 특화 (13개)
- **stakeholder_features.py**: 이해관계자 행동 (9개)
- **composite_features.py**: 복합 리스크 (7개)
- **feature_generator.py**: 전체 특성 생성 조정

#### `src/models/predictor.py` (8.2KB)
**핵심 기능**:
- `load_model()`: 모델 및 스케일러 로드
- `predict()`: 부도 확률 예측 + SHAP 분석
- `_heuristic_prediction()`: 모델 없을 때 휴리스틱 예측 ⭐
- `_prepare_features()`: 특성 전처리

**특징**:
- 모델이 없어도 작동 가능 (Graceful Degradation)
- SHAP 값 자동 계산
- 에러 핸들링 완벽

#### `src/visualization/charts.py`
**Plotly 차트 생성**:
- `create_risk_gauge()`: 위험도 게이지
- `create_shap_waterfall()`: SHAP Waterfall 차트
- `create_radar_chart()`: 레이더 차트

#### `src/utils/` (3 파일)
- **helpers.py**: 위험 등급 판정, 포맷팅, 권장사항 생성
- **business_value.py**: ROI, Payback, 절감액 계산

---

### 5. 데이터 및 모델

#### `data/processed/best_model.pkl` (103KB)
- **유형**: Logistic Regression (Baseline L1)
- **크기**: 103KB (Git LFS로 관리)
- **성능**: PR-AUC 0.75+
- **Optional**: 없어도 휴리스틱으로 작동

---

### 6. 문서

#### `README.md` (8.5KB)
**내용**:
- 프로젝트 소개
- 주요 기능
- 기술 스택
- 로컬 실행 방법
- 배포 가이드 요약
- 성능 메트릭
- 라이센스

#### `DEPLOYMENT_GUIDE.md` (11KB)
**단계별 배포 가이드**:
1. Git LFS 설정
2. 환경 변수 설정
3. 로컬 테스트
4. GitHub 레포지토리 생성
5. Streamlit Cloud 배포
6. 배포 상태 확인
7. 배포 후 관리
8. 성능 최적화
9. Pro 계획 고려
10. 최종 체크리스트

#### `STRUCTURE.md` (이 파일)
**폴더 구조 완전 가이드**

---

## ✅ 배포 준비 완료 확인

### 필수 파일 체크리스트

- [x] `app.py` - 메인 앱
- [x] `config.py` - 설정
- [x] `requirements.txt` - Python 패키지
- [x] `packages.txt` - 시스템 패키지 (한글 폰트) ⭐
- [x] `.gitattributes` - Git LFS 설정
- [x] `.gitignore` - Git 무시 파일
- [x] `.env.example` - 환경 변수 예시
- [x] `.streamlit/config.toml` - Streamlit 설정
- [x] `src/` - 전체 소스 코드 모듈
- [x] `README.md` - 프로젝트 문서
- [x] `DEPLOYMENT_GUIDE.md` - 배포 가이드
- [x] `data/processed/best_model.pkl` - 모델 (Optional)

### 설정 확인

- [x] 한글 폰트 설정 (`packages.txt`)
- [x] Git LFS 설정 (`.gitattributes`)
- [x] 환경 변수 보호 (`.gitignore`에 `.env` 포함)
- [x] 모델 없이도 작동 가능 (`predictor.py`)
- [x] 캐싱 적용 (`@st.cache_resource`, `@st.cache_data`)

---

## 🚀 다음 단계

### 로컬 테스트
```bash
cd deployment
pip install -r requirements.txt
streamlit run app.py
```

### Git 초기화
```bash
cd deployment
git init
git lfs install
git add .
git commit -m "Initial commit: 한국 기업 부도 예측 앱"
```

### GitHub 푸시
```bash
git remote add origin https://github.com/yourusername/bankruptcy-prediction-app.git
git push -u origin main
```

### Streamlit Cloud 배포
1. https://share.streamlit.io/ 접속
2. New app → 레포지토리 선택
3. Secrets에 `DART_API_KEY` 입력
4. Deploy!

---

## 💡 주요 특징 (개선 사항)

### ✅ 이전 구조 대비 개선점

1. **모델 없이도 작동**
   - `predictor.py`의 휴리스틱 방식 활용
   - Git LFS 문제 발생해도 앱 사용 가능

2. **한글 폰트 자동 설치**
   - `packages.txt` 추가
   - Streamlit Cloud에서 자동 설치

3. **명확한 문서화**
   - README.md (프로젝트 소개)
   - DEPLOYMENT_GUIDE.md (배포 상세)
   - STRUCTURE.md (구조 설명)

4. **Git LFS 올바른 설정**
   - `.gitattributes` 추가
   - 모델 파일 자동 추적

5. **환경 변수 보안**
   - `.env` 파일 Git 제외
   - `.env.example`로 가이드 제공

---

## 📞 문의 및 기여

- **Issues**: GitHub Issues 활용
- **Pull Requests**: 환영합니다!
- **Email**: your.email@example.com

---

**🎉 배포 준비 완료! 이제 GitHub에 푸시하고 Streamlit Cloud에 배포하세요!**
