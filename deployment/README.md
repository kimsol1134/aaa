# 📊 한국 기업 부도 예측 시스템

도메인 지식 기반 AI를 활용한 기업 부도 위험 실시간 예측 시스템

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

## 🚀 라이브 데모

> 👉 **[여기서 앱 사용하기](https://your-app.streamlit.app)** *(배포 후 URL 업데이트)*

---

## ✨ 주요 기능

### 🔍 DART API 연동
- 상장기업 실시간 재무제표 자동 조회
- 금융감독원 공식 데이터 활용

### 📊 도메인 특성 기반 정밀 분석
- **65개 전문가 수준 특성**
  - 유동성 위기, 지급불능, 재무조작 탐지
  - 한국 시장 특화, 이해관계자 행동 분석
  - 복합 리스크 지표 및 비선형 특성

### 🎯 고성능 AI 예측
- **PR-AUC 0.85+** 앙상블 모델 (LightGBM + XGBoost + CatBoost)
- **Part3 파이프라인**: 노트북과 100% 동일한 전처리
- **SHAP 기반 해석**: 예측 근거 및 주요 위험 요인 시각화
- **모델 없이도 작동**: 휴리스틱 방식 지원

### 💡 실행 가능한 권장사항
- 위험 요인별 구체적 조치사항 제시
- 우선순위 기반 개선 방향 제공

---

## 🛠️ 기술 스택

| 카테고리 | 기술 |
|---------|------|
| **Frontend** | Streamlit, Plotly, Matplotlib |
| **Backend** | Python 3.10+ |
| **ML Pipeline** | Part3 전처리 파이프라인 (InfiniteHandler → LogTransformer → RobustScaler → SMOTE) |
| **ML Models** | LightGBM, XGBoost, CatBoost |
| **Data Source** | DART 금융감독원 API |
| **Deployment** | Streamlit Community Cloud |

---

## 🚀 빠른 시작

### 1. 클론 및 설치

```bash
# 레포지토리 클론
git clone https://github.com/yourusername/bankruptcy-prediction-app.git
cd bankruptcy-prediction-app

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 수정 - DART API 키 입력
# DART_API_KEY=your_actual_api_key_here
```

**DART API 키 발급:**
1. [DART 오픈API](https://opendart.fss.or.kr/) 접속
2. 회원가입 → "인증키 신청" → 즉시 발급
3. 발급받은 키를 `.env` 파일에 입력

### 3. 앱 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 자동 접속 🎉

---

## 📂 프로젝트 구조

```
deployment/
├── app.py                       # ⭐ 메인 Streamlit 앱
├── config.py                    # 설정 및 환경 변수
├── requirements.txt             # Python 패키지 (31개)
├── packages.txt                 # 시스템 패키지 (한글 폰트)
│
├── src/                         # 소스 코드 모듈
│   ├── dart_api/                # DART API 연동
│   ├── domain_features/         # 65개 도메인 특성 생성
│   ├── preprocessing/           # Part3 전처리 파이프라인 ⭐
│   ├── models/                  # 예측 모델 (파이프라인 지원)
│   ├── visualization/           # 차트 및 시각화
│   └── utils/                   # 유틸리티 함수
│
└── data/
    └── processed/               # 학습된 모델 (Git LFS, Optional)
        ├── 발표_Part3_v3_최종모델.pkl  # Part3 파이프라인 모델
        └── best_model.pkl                # 레거시 모델
```

---

## 🎯 Part3 파이프라인 통합

이 앱은 Part3 노트북과 **100% 동일한 전처리 파이프라인**을 사용합니다:

```python
Pipeline([
    ('inf_handler', InfiniteHandler()),           # 무한대 → 0
    ('imputer', SimpleImputer(strategy='median')), # 결측치 처리
    ('log_transform', LogTransformer()),          # 양수 → log1p 변환
    ('scaler', RobustScaler()),                   # 정규화
    ('smote', SMOTE(sampling_strategy=0.2)),      # 학습 시만
    ('classifier', LogisticRegression/Ensemble)   # 분류기
])
```

**모델 우선순위:**
1. Part3 파이프라인 모델 (권장) ✨
2. 레거시 모델 + 스케일러
3. 휴리스틱 모델 (모델 없을 시)

자세한 내용: [Part3 파이프라인 가이드](../docs/deployment/PART3_PIPELINE_CHANGES.md)

---

## ☁️ Streamlit Cloud 배포

### 1단계: GitHub에 푸시

```bash
git add .
git commit -m "feat: 한국 기업 부도 예측 앱"
git push origin main
```

### 2단계: Streamlit Cloud 배포

1. [Streamlit Cloud](https://share.streamlit.io/) 접속
2. GitHub 계정으로 로그인
3. "New app" 클릭
4. 레포지토리 정보 입력:
   - **Repository**: `yourusername/bankruptcy-prediction-app`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. **Advanced settings → Secrets** 입력:
   ```toml
   DART_API_KEY = "your_actual_dart_api_key_here"
   ```
6. **Deploy!** 클릭

약 2-3분 후 앱이 배포됩니다! 🎉

---

## 💡 사용 방법

### 방법 1: DART API 검색 (상장기업)
1. 기업명 입력 (예: 삼성전자)
2. 회계연도 선택
3. "조회 및 분석 시작" 클릭

### 방법 2: 재무제표 직접 입력
1. 재무상태표/손익계산서 입력
2. "분석 시작" 클릭

### 방법 3: 샘플 데이터 사용
1. 샘플 유형 선택 (정상/주의/위험)
2. "샘플 분석" 클릭

---

## 📊 모델 성능

| 메트릭 | 값 | 설명 |
|-------|-----|-----|
| **PR-AUC** | 0.85+ | 불균형 데이터 핵심 지표 |
| **F2-Score** | 0.75+ | 재현율 중시 (부도 미탐지 최소화) |
| **Type II Error** | < 20% | 부도 기업 오분류 비율 |

**데이터셋:** 50,000+ 한국 기업, 170+ 재무/신용 변수, 부도율 ~1.5%

---

## ⚠️ DART API 제약사항

### 제공되는 정보
- ✅ 재무제표 (재무상태표, 손익계산서, 현금흐름표)
- ✅ 기업 기본 정보 (업종, 설립일 → 업력 계산)
- ✅ 상장 여부 (외감여부 추정)

### 제공되지 않는 정보 (기본값 사용)
- ❌ 신용등급 (기본값: BBB, 점수 5.0)
- ❌ 연체 정보 (기본값: 없음, 0.0)
- ❌ 세금 체납 정보 (기본값: 없음, 0.0)
- ❌ 정확한 종업원수 (기본값: 100명)

> **참고:** 신용평가 정보는 나이스평가, KIS채권평가 등 별도 API 필요 (유료)

### Feature 매핑 자동 처리
재무제표로부터 생성된 80개 도메인 특성이 모델이 요구하는 27개 특성으로 자동 매핑됩니다:
- `OCF유동부채비율` → `OCF_대_유동부채`
- `유동성위기지수` → `유동성압박지수`
- `재무레버리지` → `부채레버리지`
- 등등 (총 8개 이름 변환)

---

## 🔧 문제 해결

### 한글 깨짐
`packages.txt` 파일이 있는지 확인 (Streamlit Cloud가 자동으로 나눔 폰트 설치)

### 모델 로딩 실패
정상입니다! 모델이 없어도 휴리스틱 방식으로 예측 가능

### DART API 에러
`.env` (로컬) 또는 Secrets (Streamlit Cloud)에 API 키 확인

### Git LFS 파일 다운로드 실패
```bash
git lfs install
git lfs pull
```

---

## 📚 추가 문서

- [Part3 파이프라인 통합 가이드](../docs/deployment/PART3_PIPELINE_CHANGES.md)
- [배포 가이드](../docs/deployment/DEPLOYMENT_GUIDE.md)
- [프로젝트 구조 상세](../docs/deployment/STRUCTURE.md)
- [빠른 시작 가이드](../docs/deployment/QUICKSTART.md)

---

## 📜 라이센스

MIT License - 자유롭게 사용, 수정, 배포 가능

---

## 🙏 감사의 말

- 금융감독원 DART API 제공
- Streamlit Community Cloud 무료 호스팅

---

**⭐ 이 프로젝트가 도움이 되셨다면 Star를 눌러주세요!**
