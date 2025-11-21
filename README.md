# 🚨 한국 기업 부도 예측 모델

도메인 지식 기반의 AI를 활용한 기업 부도 위험 실시간 예측 시스템

## 📌 프로젝트 개요

이 프로젝트는 한국 기업의 재무 데이터를 기반으로 부도 위험을 예측하는 AI 시스템입니다.
단순한 통계적 접근이 아닌, **"왜 기업이 부도가 나는가?"**에 대한 도메인 전문 지식을 활용한 특성 공학과
Kaggle의 불균형 분류 베스트 프랙티스를 결합하여 높은 예측 정확도를 달성합니다.

### 🎯 핵심 특징

- ✅ **도메인 기반 Feature Engineering**: 유동성 위기, 지급불능, 재무조작 탐지 등 7개 카테고리 특성
- ✅ **불균형 데이터 처리**: SMOTE + Tomek Links 최적화
- ✅ **앙상블 모델**: LightGBM, XGBoost, CatBoost 스태킹
- ✅ **설명 가능한 AI**: SHAP 분석으로 예측 근거 제공
- ✅ **실시간 웹 배포**: Streamlit 기반 대화형 대시보드

## 📊 데이터

- **데이터셋**: 한국 기업 신용평가 정보 (2021년 8월 기준)
- **샘플 수**: 50,000+ 기업
- **변수 수**: 170+ 재무/신용 변수
- **타겟**: 향후 1년 내 부도 여부
- **불균형 비율**: 약 1:20 (부도:정상)

## 🗂 프로젝트 구조

```
junwoo/
├── data/                           # 데이터 디렉토리
│   ├── raw/                        # 원본 데이터
│   ├── processed/                  # 전처리 데이터
│   └── features/                   # 특성 공학 결과
├── notebooks/                      # Jupyter 노트북
│   ├── 01_도메인_기반_부도원인_분석.ipynb
│   ├── 02_고급_도메인_특성공학.ipynb
│   ├── 03_상관관계_및_리스크_패턴_분석.ipynb
│   ├── 04_불균형_분류_모델링.ipynb
│   └── 05_모델_평가_및_해석.ipynb
├── streamlit_app/                 # Streamlit 웹 앱
│   └── app.py
├── docs/                           # 문서
│   └── bankruptcy_prediction_plan.md
├── requirements.txt                # 패키지 의존성
└── README.md                       # 프로젝트 설명
```

## 🚀 시작하기

### 1. 환경 설정

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 노트북 실행

Jupyter Notebook을 실행하고 순서대로 노트북을 실행합니다:

```bash
jupyter notebook
```

**실행 순서:**
1. `01_도메인_기반_부도원인_분석.ipynb` - 데이터 탐색 및 도메인 분석
2. `02_고급_도메인_특성공학.ipynb` - 도메인 지식 기반 특성 생성
3. `03_상관관계_및_리스크_패턴_분석.ipynb` - 특성 선택 및 상관관계 분석
4. `04_불균형_분류_모델링.ipynb` - 모델 학습 및 최적화
5. `05_모델_평가_및_해석.ipynb` - SHAP 분석 및 성능 평가

### 3. Streamlit 앱 실행

```bash
cd streamlit_app
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속하면 부도 위험 예측 대시보드를 사용할 수 있습니다.

## 🔧 주요 기능

### 1. 도메인 기반 Feature Engineering

**7개 카테고리 특성 생성:**
- 유동성 위기 조기 감지 (현금 소진 일수, 운전자본 건전성 등)
- 지급불능 패턴 (자본 잠식도, 부채 상환 능력 등)
- 재무조작 탐지 (Beneish M-Score 한국형)
- 한국 시장 특화 (대기업 의존도, 수출 민감도 등)
- 이해관계자 신뢰도 (연체, 세금체납, 신용등급 등)
- 복합 리스크 지표
- 비선형 관계 및 상호작용

### 2. 불균형 데이터 처리

- **SMOTE + Tomek Links**: 소수 클래스 오버샘플링 + 경계 정리
- **BorderlineSMOTE**: 경계선 샘플 중심 생성
- **비용 민감 학습**: 클래스 가중치 및 Focal Loss

### 3. 앙상블 모델

```python
# 스태킹 앙상블
Level 1: LightGBM, XGBoost, CatBoost
Level 2: Logistic Regression (Meta Learner)
```

### 4. 평가 메트릭

- **PR-AUC**: 불균형 데이터 핵심 지표
- **F2-Score**: 재현율 중시 (부도 미탐지 최소화)
- **Type II Error**: 부도 미탐지율 < 20%

## 📈 모델 성능

| 모델 | PR-AUC | F2-Score | Type II Error |
|------|--------|----------|---------------|
| **Best Model** | **0.65+** | **0.70+** | **< 20%** |

## 💡 사용 예시

### Streamlit 대시보드

1. **수동 입력**: 재무 지표 직접 입력하여 단일 기업 분석
2. **CSV 업로드**: 여러 기업 데이터 일괄 분석
3. **샘플 데이터**: 테스트용 데이터로 기능 확인

### Python API

```python
import joblib
import pandas as pd

# 모델 로딩
model = joblib.load('data/processed/best_model_XGBoost.pkl')
scaler = joblib.load('data/processed/scaler.pkl')

# 예측
input_data = pd.DataFrame({...})  # 재무 데이터
input_scaled = scaler.transform(input_data)
risk_proba = model.predict_proba(input_scaled)[:, 1]

print(f"부도 확률: {risk_proba[0]*100:.1f}%")
```

## 📚 문서

- [프로젝트 계획서](docs/bankruptcy_prediction_plan.md)
- 각 노트북에 상세한 주석 및 설명 포함

## ⚠️ 주의사항

1. **참고용 시스템**: 최종 의사결정은 전문가와 상담 필요
2. **데이터 품질**: 입력 데이터의 정확성이 예측 성능에 영향
3. **모델 업데이트**: 정기적인 재학습 권장 (시장 환경 변화 반영)
4. **윤리적 사용**: AI 모델의 한계를 인지하고 책임감 있게 사용

## 🤝 기여

이슈 및 풀 리퀘스트를 환영합니다!

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 사용할 수 있습니다.

## 🔗 참고 자료

- Kaggle Imbalanced Classification Competition Best Practices
- Altman Z-Score (1968)
- Beneish M-Score (1999)
- SMOTE: Synthetic Minority Over-sampling Technique (2002)
- SHAP: SHapley Additive exPlanations (2017)

---

🤖 **Powered by AI** | 도메인 지식 + 데이터 사이언스 = 정확한 부도 예측
