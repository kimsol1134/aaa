# 🚨 한국 기업 부도 예측 시스템

**DART API 연동 기반 실시간 부도 위험 분석 및 예측 시스템**

도메인 지식 기반의 AI를 활용한 기업 부도 위험 실시간 예측 시스템

## 📌 프로젝트 개요

이 프로젝트는 한국 기업의 재무 데이터를 기반으로 부도 위험을 예측하는 AI 시스템입니다.
단순한 통계적 접근이 아닌, **"왜 기업이 부도가 나는가?"**에 대한 도메인 전문 지식을 활용한 특성 공학과
Kaggle의 불균형 분류 베스트 프랙티스를 결합하여 높은 예측 정확도를 달성합니다.

### 🎯 핵심 특징

- ✅ **DART API 연동**: 상장기업 재무제표 자동 조회 및 분석
- ✅ **도메인 기반 Feature Engineering**: 유동성 위기, 지급불능, 재무조작 탐지 등 7개 카테고리 65개 특성
- ✅ **불균형 데이터 처리**: SMOTE + Tomek Links 최적화
- ✅ **앙상블 모델**: LightGBM, XGBoost, CatBoost 스태킹
- ✅ **설명 가능한 AI**: 위험 요인 분석 및 개선 권장사항 제공
- ✅ **실시간 웹 배포**: Streamlit 기반 인터랙티브 대시보드

## 🆕 최신 업데이트 (2024)

### Phase 1: DART API 연동
- 전자공시시스템 API를 통한 실시간 재무제표 조회
- 기업명/종목코드 검색 기능
- 자동 재무제표 파싱 및 표준화

### Phase 2: 도메인 특성 자동 생성
- 65개 도메인 특성 자동 생성 모듈
- 카테고리별 특성 생성 (유동성, 지급불능, 재무조작 등)
- 무한대/결측치 자동 처리

### Phase 3: Streamlit 앱 대폭 개선
- 3가지 입력 모드 (DART API, 직접 입력, 샘플 데이터)
- 종합 위험 평가 대시보드
- Critical/Warning 위험 요인 분석
- 구체적 개선 권장사항 제공
- 인터랙티브 차트 (게이지, Waterfall, 레이더)

### Phase 4: 배포 준비
- 환경 설정 자동화 (.env, config.toml)
- Git LFS 설정 (대용량 모델 파일)
- Streamlit Cloud 배포 가능

## 📊 데이터

- **데이터셋**: 한국 기업 신용평가 정보 (2021년 8월 기준)
- **샘플 수**: 50,000+ 기업
- **변수 수**: 170+ 재무/신용 변수
- **생성 특성**: 65개 도메인 특성
- **타겟**: 향후 1년 내 부도 여부
- **불균형 비율**: 약 1:20 (부도:정상)

## 🗂 프로젝트 구조

```
junwoo/
├── src/                            # 소스 코드
│   ├── dart_api/                   # DART API 연동 모듈
│   │   ├── client.py               # API 클라이언트
│   │   └── parser.py               # 재무제표 파싱
│   ├── domain_features/            # 도메인 특성 생성 모듈
│   │   ├── feature_generator.py    # 통합 생성기
│   │   ├── liquidity_features.py   # 유동성 특성
│   │   ├── insolvency_features.py  # 지급불능 특성
│   │   ├── manipulation_features.py # 재무조작 특성
│   │   ├── korea_market_features.py # 한국 시장 특성
│   │   ├── stakeholder_features.py  # 이해관계자 특성
│   │   └── composite_features.py    # 복합 특성
│   ├── models/                     # 모델 예측 모듈
│   │   └── predictor.py            # 부도 예측
│   ├── visualization/              # 시각화 모듈
│   │   └── charts.py               # Plotly 차트
│   └── utils/                      # 유틸리티
│       └── helpers.py              # 헬퍼 함수
├── streamlit_app/                  # Streamlit 웹 앱
│   ├── app.py                      # 메인 앱
│   ├── config.py                   # 설정
│   ├── requirements.txt            # 앱 의존성
│   └── .streamlit/
│       └── config.toml             # Streamlit 설정
├── data/                           # 데이터 디렉토리
│   ├── processed/                  # 모델 파일
│   │   ├── best_model_XGBoost.pkl  # 학습된 모델
│   │   └── scaler.pkl              # 스케일러
│   └── features/                   # 특성 공학 결과
├── notebooks/                      # Jupyter 노트북
│   ├── 01_도메인_기반_부도원인_분석.ipynb
│   ├── 02_고급_도메인_특성공학.ipynb
│   ├── 03_상관관계_및_리스크_패턴_분석.ipynb
│   ├── 04_불균형_분류_모델링.ipynb
│   └── 05_모델_평가_및_해석.ipynb
├── tests/                          # 단위 테스트
│   ├── test_dart_api.py
│   └── test_feature_generator.py
├── docs/                           # 문서
│   ├── bankruptcy_prediction_plan.md
│   └── streamlit_deployment_plan.md
├── .env.example                    # 환경 변수 템플릿
├── .gitattributes                  # Git LFS 설정
├── requirements.txt                # 패키지 의존성
├── CLAUDE.md                       # 프로젝트 가이드
└── README.md                       # 프로젝트 설명
```

## 🚀 시작하기

### 1. 저장소 클론 및 환경 설정

```bash
# 저장소 클론
git clone https://github.com/your-username/junwoo.git
cd junwoo

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. DART API 키 발급 및 설정

```bash
# 1. https://opendart.fss.or.kr/ 접속
# 2. 인증키 신청/관리 메뉴에서 API 키 발급 (무료)
# 3. .env 파일 생성
cp .env.example .env

# 4. .env 파일에 API 키 입력
# DART_API_KEY=your_actual_api_key_here
```

### 3. Streamlit 앱 실행

```bash
cd streamlit_app
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속하면 부도 위험 예측 대시보드를 사용할 수 있습니다.

### 4. 노트북 실행 (선택)

데이터 분석 및 모델 학습 과정을 확인하려면:

```bash
jupyter notebook
```

**실행 순서:**
1. `01_도메인_기반_부도원인_분석.ipynb` - 데이터 탐색
2. `02_고급_도메인_특성공학.ipynb` - 특성 생성
3. `03_상관관계_및_리스크_패턴_분석.ipynb` - 특성 선택
4. `04_불균형_분류_모델링.ipynb` - 모델 학습
5. `05_모델_평가_및_해석.ipynb` - SHAP 분석

## 🔧 주요 기능

### 1. DART API 연동

```python
from src.dart_api import DartAPIClient, FinancialStatementParser

# 기업 검색 및 재무제표 조회
client = DartAPIClient(api_key="your_api_key")
company = client.search_company("삼성전자")
statements = client.get_financial_statements(
    corp_code=company['corp_code'],
    bsns_year="2023"
)

# 재무제표 파싱
parser = FinancialStatementParser()
financial_data = parser.parse(statements)
```

### 2. 도메인 특성 자동 생성

```python
from src.domain_features import DomainFeatureGenerator

# 65개 도메인 특성 자동 생성
generator = DomainFeatureGenerator()
features_df = generator.generate_all_features(financial_data)

# 생성된 특성 확인
print(f"생성된 특성 수: {len(features_df.columns)}")
print(features_df.head())
```

**특성 카테고리 (총 65개):**
- 유동성 위기 (10개): 현금소진일수, 유동비율 등
- 지급불능 (8개): 이자보상배율, 부채비율 등
- 재무조작 탐지 (15개): 발생액비율, 이익의질 등
- 한국 시장 특화 (13개): 제조업여부, 대기업여부 등
- 이해관계자 행동 (9개): 신용등급, 연체여부 등
- 복합 리스크 (10개): 종합부도위험스코어 등

### 3. 부도 위험 예측

```python
from src.models import BankruptcyPredictor

# 모델 로딩 및 예측
predictor = BankruptcyPredictor(
    model_path="data/processed/best_model_XGBoost.pkl",
    scaler_path="data/processed/scaler.pkl"
)
predictor.load_model()

# 예측
result = predictor.predict(features_df)

print(f"부도 확률: {result['bankruptcy_probability']*100:.1f}%")
print(f"위험 등급: {result['risk_level']}")
```

### 4. Streamlit 대시보드

#### 입력 모드

1. **DART API 검색**: 기업명 입력 → 자동 조회 및 분석
2. **재무제표 직접 입력**: 주요 재무 항목 수동 입력
3. **샘플 데이터**: 정상/주의/위험 기업 샘플로 테스트

#### 주요 섹션

- **종합 평가**: 부도 확률, 위험 등급, 재무 건전성, 조기경보신호
- **위험 요인 분석**: Critical/Warning 리스크, SHAP-style Waterfall 차트
- **개선 권장사항**: 우선순위별 구체적 액션 플랜 제공
- **상세 특성**: 카테고리별 특성 값 확인
- **재무제표 원본**: 입력 데이터 검증

## 📈 모델 성능

| 메트릭 | 값 |
|--------|-----|
| **PR-AUC** | 0.65+ |
| **F2-Score** | 0.70+ |
| **Type II Error** | < 20% |

## 🧪 테스트

```bash
# 단위 테스트 실행
pytest tests/ -v

# DART API 테스트
pytest tests/test_dart_api.py -v -s

# 특성 생성 테스트
pytest tests/test_feature_generator.py -v -s
```

## 🌐 Streamlit Cloud 배포

1. GitHub에 코드 푸시

```bash
git add .
git commit -m "feat: DART API 연동 및 Streamlit 앱 개선"
git push -u origin claude/streamlit-deployment-phases-01MjMUwJ9hXJtW9uFLuiERDy
```

2. Streamlit Cloud 설정
   - https://share.streamlit.io/ 접속
   - GitHub 저장소 연결
   - Main file path: `streamlit_app/app.py`
   - Secrets에 DART_API_KEY 추가

3. 배포 완료!

## ⚠️ 주의사항

1. **DART API 제한**: 초당 1회 호출 제한 (자동 처리됨)
2. **참고용 시스템**: 최종 의사결정은 전문가와 상담 필요
3. **데이터 품질**: 입력 데이터의 정확성이 예측 성능에 영향
4. **모델 업데이트**: 정기적인 재학습 권장 (시장 환경 변화 반영)
5. **윤리적 사용**: AI 모델의 한계를 인지하고 책임감 있게 사용

## 📚 문서

- [프로젝트 계획서](docs/bankruptcy_prediction_plan.md)
- [Streamlit 배포 계획](docs/streamlit_deployment_plan.md)
- [CLAUDE.md](CLAUDE.md) - Claude Code용 프로젝트 가이드
- 각 노트북에 상세한 주석 및 설명 포함

## 🤝 기여

이슈 및 풀 리퀘스트를 환영합니다!

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 사용할 수 있습니다.

## 🔗 참고 자료

- [DART Open API](https://opendart.fss.or.kr/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- Kaggle Imbalanced Classification Best Practices
- Altman Z-Score (1968)
- Beneish M-Score (1999)
- SMOTE: Synthetic Minority Over-sampling Technique (2002)
- SHAP: SHapley Additive exPlanations (2017)

---

🤖 **Powered by AI + DART API** | 도메인 지식 + 데이터 사이언스 = 정확한 부도 예측
