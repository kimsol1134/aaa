# 🚀 한국 기업 부도 예측 시스템 - DART API 연동 및 Streamlit 배포 프로젝트

## 📋 프로젝트 개요

**목표:** DART(전자공시시스템) API를 활용하여 상장기업 재무제표를 자동 조회하고, 도메인 특성을 자동 생성하여 부도 위험을 예측하는 웹 애플리케이션을 Streamlit Cloud에 배포

**핵심 플로우:**
```
사용자 입력 (회사명/종목코드)
    ↓
DART API 재무제표 자동 조회
    ↓
65개 도메인 특성 자동 생성
    ↓
ML 모델 예측 (부도 확률)
    ↓
실무 활용 가능한 리포트 제공
```

**기술 스택:**
- Backend: Python 3.9+
- Frontend: Streamlit
- API: DART Open API
- ML: XGBoost/LightGBM (기존 학습 모델)
- Deployment: Streamlit Cloud
- Version Control: Git + Git LFS

---

## 🎯 핵심 요구사항

### 1. DART API 연동 기능
- [x] 기업명/종목코드로 상장기업 검색
- [x] 최신 재무제표 자동 조회 (재무상태표, 손익계산서, 현금흐름표)
- [x] API 응답 파싱 및 데이터 정제
- [x] 에러 핸들링 (기업 미존재, API 오류, Rate Limit 등)

### 2. 도메인 특성 자동 생성 모듈
- [x] 재무제표 원본 데이터 → 65개 도메인 특성 변환
- [x] 7개 카테고리 특성 생성 (유동성 위기, 지급불능, 재무조작 탐지 등)
- [x] 결측치 및 이상치 처리
- [x] 특성 메타데이터 관리

### 3. Streamlit 앱 개선
- [x] 사용자 친화적 UI/UX
- [x] 실시간 예측 결과 시각화
- [x] 위험 요인 분석 및 설명
- [x] 구체적 개선 권장사항 제공
- [x] 동종업계 벤치마크 비교
- [x] PDF 리포트 다운로드

### 4. Streamlit Cloud 배포
- [x] 환경 변수 설정
- [x] 대용량 파일 관리 (Git LFS)
- [x] 배포 최적화
- [x] 에러 모니터링

---

## 📁 프로젝트 파일 구조

```
junwoo/
├── streamlit_app/
│   ├── app.py                          # 메인 Streamlit 앱 (개선)
│   ├── config.py                       # 설정 파일 (API Key, 상수 등)
│   ├── requirements.txt                # Python 패키지 의존성
│   └── .streamlit/
│       └── config.toml                 # Streamlit 설정
├── src/
│   ├── dart_api/
│   │   ├── __init__.py
│   │   ├── client.py                   # DART API 클라이언트
│   │   ├── parser.py                   # 재무제표 파싱
│   │   └── utils.py                    # 유틸리티 함수
│   ├── domain_features/
│   │   ├── __init__.py
│   │   ├── feature_generator.py        # 도메인 특성 자동 생성
│   │   ├── liquidity_features.py       # 유동성 위기 특성
│   │   ├── insolvency_features.py      # 지급불능 특성
│   │   ├── manipulation_features.py    # 재무조작 탐지 특성
│   │   ├── korea_market_features.py    # 한국 시장 특화 특성
│   │   ├── stakeholder_features.py     # 이해관계자 행동 특성
│   │   └── composite_features.py       # 복합 리스크 특성
│   ├── models/
│   │   ├── __init__.py
│   │   └── predictor.py                # 모델 로딩 및 예측
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── dashboard.py                # 대시보드 컴포넌트
│   │   ├── charts.py                   # 차트 생성
│   │   └── report.py                   # PDF 리포트 생성
│   └── utils/
│       ├── __init__.py
│       ├── validation.py               # 데이터 검증
│       └── constants.py                # 상수 정의
├── data/
│   ├── processed/
│   │   ├── best_model_XGBoost.pkl      # 학습된 모델
│   │   ├── scaler.pkl                  # 스케일러
│   │   └── selected_features.csv       # 선택된 특성 (샘플)
│   ├── industry_benchmarks/
│   │   └── industry_stats.csv          # 산업별 벤치마크 (생성 필요)
│   └── templates/
│       └── 재무제표_입력_템플릿.xlsx   # 수동 입력용 템플릿
├── tests/
│   ├── test_dart_api.py                # DART API 테스트
│   ├── test_feature_generator.py       # 특성 생성 테스트
│   └── test_predictor.py               # 예측 테스트
├── docs/
│   ├── streamlit_deployment_plan.md    # 이 파일
│   ├── DART_API_Guide.md               # DART API 사용 가이드
│   └── Feature_Engineering_Spec.md     # 특성 공학 명세서
├── .env.example                        # 환경 변수 템플릿
├── .gitignore
├── .gitattributes                      # Git LFS 설정
└── README.md                           # 프로젝트 README 업데이트
```

---

## 🔧 단계별 구현 계획

### Phase 1: DART API 연동 모듈 개발 (1단계 - 최우선)

#### Task 1.1: DART API 클라이언트 구현

**파일:** `src/dart_api/client.py`

**구현 내용:**
```python
"""
DART Open API 클라이언트

주요 기능:
1. 기업 검색 (회사명 → 종목코드 변환)
2. 재무제표 조회 (단일 회계연도)
3. 다년도 재무제표 조회 (추이 분석용)
4. API Rate Limit 관리
5. 에러 핸들링 및 재시도 로직

필수 API 엔드포인트:
- 공시검색: /api/company.json (기업 기본정보)
- 재무제표: /api/fnlttSinglAcntAll.json (단일회사 전체 재무제표)
"""

class DartAPIClient:
    def __init__(self, api_key: str):
        """
        Args:
            api_key: DART API 키 (환경 변수에서 로드)
        """
        pass

    def search_company(self, company_name: str) -> dict:
        """기업명으로 종목코드 검색"""
        pass

    def get_financial_statements(
        self,
        corp_code: str,
        bsns_year: str,
        reprt_code: str = "11011"  # 사업보고서
    ) -> dict:
        """재무제표 조회

        Returns:
            {
                'balance_sheet': {...},      # 재무상태표
                'income_statement': {...},   # 손익계산서
                'cash_flow': {...}           # 현금흐름표
            }
        """
        pass

    def _handle_rate_limit(self):
        """API Rate Limit 관리 (1초당 1회 제한)"""
        pass
```

**API Key 발급:**
1. https://opendart.fss.or.kr/ 접속
2. 인증키 신청/관리 메뉴
3. 신청 후 즉시 발급 (무료)
4. `.env` 파일에 저장: `DART_API_KEY=your_key_here`

**테스트 방법:**
```python
# tests/test_dart_api.py
def test_search_company():
    client = DartAPIClient(api_key=os.getenv('DART_API_KEY'))
    result = client.search_company("삼성전자")
    assert result['corp_code'] is not None
    assert '005930' in result['stock_code']

def test_get_financial_statements():
    client = DartAPIClient(api_key=os.getenv('DART_API_KEY'))
    fs = client.get_financial_statements(
        corp_code="00126380",  # 삼성전자
        bsns_year="2023"
    )
    assert 'balance_sheet' in fs
    assert '자산총계' in fs['balance_sheet']
```

---

#### Task 1.2: 재무제표 파싱 모듈 구현

**파일:** `src/dart_api/parser.py`

**구현 내용:**
```python
"""
DART API 응답을 표준 재무제표 포맷으로 변환

입력: DART API JSON 응답
출력: 표준화된 재무제표 딕셔너리

주요 기능:
1. 계정과목 매핑 (DART 표준 → 프로젝트 표준)
2. 금액 단위 변환 (원 → 백만원)
3. 결측 항목 처리
4. 데이터 검증 (음수 자산 등 이상치 탐지)
"""

class FinancialStatementParser:
    # 계정과목 매핑 테이블
    ACCOUNT_MAPPING = {
        # 재무상태표
        '유동자산': ['유동자산', '당좌자산'],
        '현금및현금성자산': ['현금및현금성자산', '현금및현금성자산(유동)'],
        '매출채권': ['매출채권', '매출채권 및 기타채권'],
        '재고자산': ['재고자산'],
        '유동부채': ['유동부채'],
        '자산총계': ['자산총계'],
        '부채총계': ['부채총계'],
        '자본총계': ['자본총계'],

        # 손익계산서
        '매출액': ['매출액', '수익(매출액)'],
        '매출원가': ['매출원가'],
        '영업이익': ['영업이익(손실)', '영업이익'],
        '당기순이익': ['당기순이익(손실)', '당기순이익'],
        '이자비용': ['이자비용', '금융원가'],

        # 현금흐름표
        '영업활동현금흐름': ['영업활동으로인한현금흐름', '영업활동현금흐름'],
        '투자활동현금흐름': ['투자활동으로인한현금흐름'],
    }

    def parse(self, dart_response: dict) -> dict:
        """
        DART API 응답 파싱

        Returns:
            {
                '유동자산': 1000000,  # 백만원 단위
                '유동부채': 500000,
                '매출액': 3000000,
                ...
            }
        """
        pass

    def validate(self, financial_data: dict) -> tuple[bool, list]:
        """
        재무제표 검증

        Returns:
            (is_valid, error_messages)
        """
        pass
```

---

### Phase 2: 도메인 특성 자동 생성 모듈 개발 (2단계)

#### Task 2.1: 특성 생성 프레임워크 구현

**파일:** `src/domain_features/feature_generator.py`

**구현 내용:**
```python
"""
재무제표 → 65개 도메인 특성 자동 생성

입력: 파싱된 재무제표 딕셔너리
출력: 65개 특성을 포함한 DataFrame (1행)

특성 카테고리:
1. 유동성 위기 (10개) - liquidity_features.py
2. 지급불능 (8개) - insolvency_features.py
3. 재무조작 탐지 (15개) - manipulation_features.py
4. 한국 시장 특화 (13개) - korea_market_features.py
5. 이해관계자 행동 (9개) - stakeholder_features.py
6. 복합 리스크 (7개) - composite_features.py
7. 비선형/상호작용 (3개) - composite_features.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

class DomainFeatureGenerator:
    def __init__(self):
        """특성 생성기 초기화"""
        self.feature_metadata = self._load_feature_metadata()

    def generate_all_features(
        self,
        financial_data: dict,
        company_info: dict = None
    ) -> pd.DataFrame:
        """
        모든 도메인 특성 생성

        Args:
            financial_data: 재무제표 데이터
            company_info: 기업 추가 정보 (업력, 외감여부 등)

        Returns:
            65개 특성을 포함한 DataFrame
        """
        features = {}

        # 1. 유동성 위기 특성 (10개)
        liquidity = self._generate_liquidity_features(financial_data)
        features.update(liquidity)

        # 2. 지급불능 특성 (8개)
        insolvency = self._generate_insolvency_features(financial_data)
        features.update(insolvency)

        # 3. 재무조작 탐지 특성 (15개)
        manipulation = self._generate_manipulation_features(financial_data)
        features.update(manipulation)

        # 4. 한국 시장 특화 특성 (13개)
        korea_market = self._generate_korea_market_features(
            financial_data, company_info
        )
        features.update(korea_market)

        # 5. 이해관계자 행동 특성 (9개)
        stakeholder = self._generate_stakeholder_features(company_info)
        features.update(stakeholder)

        # 6. 복합 리스크 특성 (7개)
        composite = self._generate_composite_features(features)
        features.update(composite)

        # 7. 비선형/상호작용 특성 (3개)
        nonlinear = self._generate_nonlinear_features(features)
        features.update(nonlinear)

        # DataFrame 변환 및 검증
        df = pd.DataFrame([features])
        df = self._validate_and_clean(df)

        return df

    def _generate_liquidity_features(self, data: dict) -> dict:
        """유동성 위기 특성 10개 생성"""
        features = {}

        # 1. 유동비율
        features['유동비율'] = data.get('유동자산', 0) / (data.get('유동부채', 1) + 1)

        # 2. 당좌비율
        당좌자산 = (
            data.get('유동자산', 0) -
            data.get('재고자산', 0)
        )
        features['당좌비율'] = 당좌자산 / (data.get('유동부채', 1) + 1)

        # 3. 현금비율
        features['현금비율'] = (
            data.get('현금및현금성자산', 0) /
            (data.get('유동부채', 1) + 1)
        )

        # 4. 현금소진일수 (중요!)
        일평균영업비용 = (
            (data.get('매출원가', 0) + data.get('판매비와관리비', 0)) / 365
        )
        features['현금소진일수'] = (
            data.get('현금및현금성자산', 0) / (일평균영업비용 + 1)
        )

        # 5. 운전자본비율
        운전자본 = data.get('유동자산', 0) - data.get('유동부채', 0)
        features['운전자본비율'] = 운전자본 / (data.get('자산총계', 1) + 1)

        # 6. 긴급유동성비율
        features['긴급유동성비율'] = (
            (data.get('현금및현금성자산', 0) + data.get('단기금융상품', 0)) /
            (data.get('유동부채', 1) + 1)
        )

        # 7. OCF 유동부채 비율
        features['OCF유동부채비율'] = (
            data.get('영업활동현금흐름', 0) /
            (data.get('유동부채', 1) + 1)
        )

        # 8. 현금흐름 적정성
        features['현금흐름적정성'] = (
            data.get('영업활동현금흐름', 0) /
            (data.get('당기순이익', 1) + 1)
        )

        # 9. 유동성위기지수 (복합)
        features['유동성위기지수'] = (
            (1 - features['유동비율'] / 1.5) * 0.3 +
            (1 - features['현금비율'] / 0.5) * 0.3 +
            (1 - features['현금소진일수'] / 90) * 0.4
        )

        # 10. 단기지급능력
        features['단기지급능력'] = (
            (data.get('현금및현금성자산', 0) +
             data.get('영업활동현금흐름', 0) / 12) /
            (data.get('유동부채', 1) / 12 + 1)
        )

        return features

    def _generate_insolvency_features(self, data: dict) -> dict:
        """지급불능 특성 8개 생성"""
        features = {}

        # 1. 부채비율
        features['부채비율'] = (
            data.get('부채총계', 0) /
            (data.get('자본총계', 1) + 1) * 100
        )

        # 2. 자본잠식도
        features['자본잠식도'] = (
            max(0, -data.get('자본총계', 0)) /
            (data.get('자산총계', 1) + 1) * 100
        )

        # 3. 이자보상배율 (중요!)
        features['이자보상배율'] = (
            data.get('영업이익', 0) /
            (data.get('이자비용', 1) + 1)
        )

        # 4. 부채상환년수
        features['부채상환년수'] = (
            data.get('부채총계', 0) /
            (data.get('영업활동현금흐름', 1) + 1)
        )

        # 5. 재무레버리지
        features['재무레버리지'] = (
            data.get('자산총계', 0) /
            (data.get('자본총계', 1) + 1)
        )

        # 6. 고정장기적합률
        features['고정장기적합률'] = (
            data.get('비유동자산', 0) /
            (data.get('자본총계', 0) + data.get('비유동부채', 0) + 1)
        )

        # 7. 순차입금의존도
        순차입금 = (
            data.get('단기차입금', 0) +
            data.get('장기차입금', 0) -
            data.get('현금및현금성자산', 0)
        )
        features['순차입금의존도'] = 순차입금 / (data.get('자산총계', 1) + 1)

        # 8. 지급불능위험지수 (복합)
        features['지급불능위험지수'] = (
            (features['부채비율'] / 200) * 0.3 +
            (features['자본잠식도'] / 50) * 0.3 +
            (1 - features['이자보상배율'] / 2) * 0.4
        )

        return features

    def _generate_manipulation_features(self, data: dict) -> dict:
        """재무조작 탐지 특성 15개 생성 (한국형 M-Score 포함)"""
        features = {}

        # Beneish M-Score 기반 지표들
        # (실제 구현 시 전년도 데이터 필요 - 여기서는 단순화)

        # 1. 매출채권증가율 (DSRI)
        매출채권회전일수 = (
            data.get('매출채권', 0) /
            (data.get('매출액', 1) / 365 + 1)
        )
        features['매출채권회전일수'] = 매출채권회전일수

        # 2. 매출총이익률 (GMI)
        features['매출총이익률'] = (
            (data.get('매출액', 0) - data.get('매출원가', 0)) /
            (data.get('매출액', 1) + 1)
        )

        # 3. 자산질지수 (AQI)
        features['자산질지수'] = (
            (data.get('자산총계', 0) -
             data.get('유동자산', 0) -
             data.get('유형자산', 0)) /
            (data.get('자산총계', 1) + 1)
        )

        # 4. 발생액 비율
        발생액 = (
            data.get('당기순이익', 0) -
            data.get('영업활동현금흐름', 0)
        )
        features['발생액비율'] = 발생액 / (data.get('자산총계', 1) + 1)

        # 5-15. 추가 재무조작 탐지 지표
        # (상세 구현은 manipulation_features.py에서)

        return features

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        특성 검증 및 정제

        - 무한대 값 → 0 변환
        - 결측치 → 중앙값/0 변환
        - 이상치 클리핑 (예: 비율은 -10 ~ 10 범위)
        """
        # 무한대 처리
        df = df.replace([np.inf, -np.inf], 0)

        # 결측치 처리
        df = df.fillna(0)

        # 이상치 클리핑
        for col in df.columns:
            if '비율' in col or '배율' in col:
                df[col] = df[col].clip(-10, 10)

        return df
```

**중요:** 위 코드는 핵심 로직만 포함한 템플릿입니다. 실제 구현 시:
1. 각 카테고리별 특성을 별도 파일로 분리 (`liquidity_features.py` 등)
2. 전년도 데이터 비교가 필요한 특성은 다년도 재무제표 조회 필요
3. 한국 시장 특화 특성은 DART에서 제공하지 않는 정보 포함 (외감여부 등) → 별도 DB 필요

---

#### Task 2.2: 카테고리별 특성 구현

**파일들:**
- `src/domain_features/liquidity_features.py`
- `src/domain_features/insolvency_features.py`
- `src/domain_features/manipulation_features.py`
- `src/domain_features/korea_market_features.py`
- `src/domain_features/stakeholder_features.py`
- `src/domain_features/composite_features.py`

**지침:**
1. 각 파일은 해당 카테고리의 특성만 생성
2. `notebooks/02_고급_도메인_특성공학.ipynb`의 로직을 모듈화
3. 단위 테스트 작성 (`tests/test_feature_generator.py`)
4. docstring에 각 특성의 의미와 임계값 설명

**예시 구조:**
```python
# src/domain_features/liquidity_features.py
def create_liquidity_features(financial_data: dict) -> dict:
    """
    유동성 위기 특성 10개 생성

    Args:
        financial_data: 재무제표 딕셔너리

    Returns:
        {
            '유동비율': 1.5,
            '현금소진일수': 45,
            ...
        }

    특성 설명:
    - 유동비율: 단기 지급 능력 (정상: > 100%)
    - 현금소진일수: 현금 고갈까지 일수 (위험: < 30일)
    """
    features = {}
    # 구현...
    return features
```

---

### Phase 3: Streamlit 앱 고도화 (3단계)

#### Task 3.1: 메인 앱 리팩토링

**파일:** `streamlit_app/app.py`

**개선 사항:**

1. **입력 방식 다양화**
```python
# 3가지 입력 모드
input_method = st.sidebar.radio(
    "입력 방식 선택",
    [
        "🔍 DART API 검색 (상장기업)",
        "📁 재무제표 직접 입력",
        "📂 CSV 업로드"
    ]
)

if input_method == "🔍 DART API 검색 (상장기업)":
    company_name = st.text_input("기업명 또는 종목코드", "삼성전자")
    year = st.selectbox("회계연도", ["2023", "2022", "2021"])

    if st.button("조회 및 분석"):
        with st.spinner("재무제표 조회 중..."):
            # DART API 호출
            dart_client = DartAPIClient(api_key=DART_API_KEY)
            company = dart_client.search_company(company_name)
            fs = dart_client.get_financial_statements(
                corp_code=company['corp_code'],
                bsns_year=year
            )

        with st.spinner("도메인 특성 생성 중..."):
            # 특성 생성
            generator = DomainFeatureGenerator()
            features_df = generator.generate_all_features(fs)

        with st.spinner("부도 위험 예측 중..."):
            # 모델 예측
            predictor = BankruptcyPredictor()
            result = predictor.predict(features_df)

        # 결과 표시
        display_results(result, features_df, fs)
```

2. **대시보드 레이아웃 개선**
```python
def display_results(result, features_df, financial_data):
    """결과 표시 - 실무 활용 강화"""

    # === 섹션 1: 종합 평가 ===
    st.markdown("## 🚨 종합 부도 위험 평가")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        risk_score = result['bankruptcy_probability']
        risk_level, risk_icon, risk_msg = get_risk_level(risk_score)

        st.metric(
            label="부도 확률",
            value=f"{risk_score*100:.1f}%",
            delta=f"{risk_level} {risk_icon}"
        )

    with col2:
        st.metric(
            label="위험 등급",
            value=risk_level,
            delta=risk_icon
        )

    with col3:
        # 동종업계 대비 순위
        industry_rank = calculate_industry_rank(
            financial_data,
            industry='제조업'
        )
        st.metric(
            label="업계 내 순위",
            value=f"하위 {industry_rank}%",
            delta="업계평균 대비"
        )

    with col4:
        # 개선 가능성
        improvement_potential = calculate_improvement_potential(features_df)
        st.metric(
            label="개선 가능성",
            value=improvement_potential,
            delta="즉시 조치 시"
        )

    # 게이지 차트
    fig_gauge = create_risk_gauge(risk_score)
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    # === 섹션 2: 위험 요인 분석 (SHAP 기반) ===
    st.markdown("## 🔍 위험 요인 상세 분석")

    # SHAP value 계산
    shap_values = result.get('shap_values', None)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔴 Critical 리스크 (즉시 조치 필요)")

        critical_risks = identify_critical_risks(features_df)

        for risk in critical_risks:
            st.error(
                f"**{risk['name']}**: {risk['value']:.2f} "
                f"(기준: {risk['threshold']:.2f})\n\n"
                f"→ {risk['explanation']}"
            )

    with col2:
        st.markdown("### 🟡 Warning (개선 권장)")

        warnings = identify_warnings(features_df)

        for warning in warnings:
            st.warning(
                f"**{warning['name']}**: {warning['value']:.2f} "
                f"(권장: {warning['threshold']:.2f})"
            )

    # SHAP Waterfall 차트
    if shap_values is not None:
        fig_shap = create_shap_waterfall(shap_values, features_df)
        st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("---")

    # === 섹션 3: 구체적 개선 권장사항 ===
    st.markdown("## 💡 실행 가능한 개선 권장사항")

    recommendations = generate_recommendations(features_df, financial_data)

    for i, rec in enumerate(recommendations, 1):
        with st.expander(
            f"권장사항 {i}: {rec['title']} (우선순위: {rec['priority']})",
            expanded=(i == 1)
        ):
            st.markdown(f"**현재 상태:**\n{rec['current_status']}")
            st.markdown(f"**문제점:**\n{rec['problem']}")
            st.markdown(f"**개선 방안:**\n{rec['solution']}")
            st.markdown(f"**예상 효과:**\n{rec['expected_impact']}")

            # 시뮬레이션
            if rec.get('simulation'):
                st.markdown("**시뮬레이션:**")
                simulate_improvement(rec['simulation'], features_df)

    st.markdown("---")

    # === 섹션 4: 동종업계 벤치마크 ===
    st.markdown("## 📊 동종업계 비교")

    industry = get_industry(financial_data)
    benchmarks = load_industry_benchmarks(industry)

    comparison_df = create_comparison_table(
        features_df,
        benchmarks
    )

    st.dataframe(
        comparison_df.style.apply(
            highlight_comparison,
            axis=1
        ),
        use_container_width=True
    )

    # 레이더 차트
    fig_radar = create_radar_chart(features_df, benchmarks)
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # === 섹션 5: 재무제표 원본 데이터 ===
    with st.expander("📋 재무제표 원본 데이터 보기"):
        tab1, tab2, tab3 = st.tabs([
            "재무상태표",
            "손익계산서",
            "현금흐름표"
        ])

        with tab1:
            st.dataframe(
                format_balance_sheet(financial_data),
                use_container_width=True
            )

        with tab2:
            st.dataframe(
                format_income_statement(financial_data),
                use_container_width=True
            )

        with tab3:
            st.dataframe(
                format_cash_flow(financial_data),
                use_container_width=True
            )

    # === 섹션 6: PDF 리포트 다운로드 ===
    st.markdown("---")
    st.markdown("## 📥 리포트 다운로드")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 PDF 리포트 생성"):
            pdf_bytes = generate_pdf_report(
                result,
                features_df,
                financial_data
            )

            st.download_button(
                label="PDF 다운로드",
                data=pdf_bytes,
                file_name=f"부도위험평가_{company_name}_{year}.pdf",
                mime="application/pdf"
            )

    with col2:
        if st.button("📊 Excel 상세 데이터"):
            excel_bytes = generate_excel_report(features_df)

            st.download_button(
                label="Excel 다운로드",
                data=excel_bytes,
                file_name=f"상세데이터_{company_name}_{year}.xlsx",
                mime="application/vnd.ms-excel"
            )
```

---

#### Task 3.2: 시각화 컴포넌트 개발

**파일:** `src/visualization/charts.py`

**구현할 차트:**

1. **위험 신호등 (Traffic Light)**
```python
def create_traffic_light(risk_score: float) -> go.Figure:
    """
    신호등 스타일 위험 표시
    - 🟢 안전 (0-30%)
    - 🟡 주의 (30-60%)
    - 🟠 경고 (60-80%)
    - 🔴 위험 (80-100%)
    """
    pass
```

2. **SHAP Waterfall 차트**
```python
def create_shap_waterfall(
    shap_values: np.ndarray,
    features: pd.DataFrame
) -> go.Figure:
    """
    어떤 특성이 부도 확률을 높이는지 시각화
    """
    pass
```

3. **업계 비교 레이더 차트**
```python
def create_radar_chart(
    company_features: pd.DataFrame,
    industry_benchmarks: dict
) -> go.Figure:
    """
    5대 재무 지표 업계 비교
    - 유동성
    - 수익성
    - 안정성
    - 활동성
    - 성장성
    """
    pass
```

4. **시나리오 시뮬레이션 차트**
```python
def create_simulation_chart(
    current_score: float,
    scenarios: list
) -> go.Figure:
    """
    "만약 ~하면?" 시뮬레이션 결과
    """
    pass
```

---

#### Task 3.3: PDF 리포트 생성

**파일:** `src/visualization/report.py`

**라이브러리:** `reportlab` 또는 `weasyprint`

**리포트 구성:**
```
페이지 1: 표지
  - 기업명, 분석일자
  - 종합 위험 등급

페이지 2: 종합 평가
  - 부도 확률 게이지
  - 5대 지표 요약
  - 업계 순위

페이지 3-4: 위험 요인 분석
  - Critical/Warning 리스트
  - SHAP 분석 차트
  - 특성별 상세 설명

페이지 5: 개선 권장사항
  - 우선순위별 액션 플랜
  - 예상 효과

페이지 6: 동종업계 비교
  - 벤치마크 테이블
  - 레이더 차트

페이지 7: 재무제표 원본
```

---

### Phase 4: 배포 준비 (4단계)

#### Task 4.1: 환경 설정

**파일:** `.env.example`
```bash
# DART API
DART_API_KEY=your_dart_api_key_here

# 모델 설정
MODEL_PATH=data/processed/best_model_XGBoost.pkl
SCALER_PATH=data/processed/scaler.pkl

# 로깅
LOG_LEVEL=INFO
```

**파일:** `streamlit_app/.streamlit/config.toml`
```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

**파일:** `streamlit_app/requirements.txt`
```txt
streamlit==1.29.0
pandas==2.1.3
numpy==1.24.3
plotly==5.18.0
requests==2.31.0
python-dotenv==1.0.0
scikit-learn==1.3.2
xgboost==2.0.2
lightgbm==4.1.0
catboost==1.2.2
shap==0.43.0
openpyxl==3.1.2
reportlab==4.0.7
joblib==1.3.2
```

---

#### Task 4.2: Git LFS 설정

**문제:** 모델 파일이 크면 GitHub에 업로드 불가

**해결:** Git LFS (Large File Storage)

```bash
# Git LFS 설치
git lfs install

# 대용량 파일 추적
git lfs track "*.pkl"
git lfs track "*.csv"
git lfs track "data/processed/*"

# .gitattributes 자동 생성됨
cat .gitattributes
# *.pkl filter=lfs diff=lfs merge=lfs -text
# *.csv filter=lfs diff=lfs merge=lfs -text
```

**주의:** Streamlit Cloud는 Git LFS를 지원하므로 추가 설정 불필요

---

#### Task 4.3: Streamlit Cloud 배포

**배포 단계:**

1. **GitHub에 코드 푸시**
```bash
git add .
git commit -m "feat: DART API 연동 및 Streamlit 앱 개선"
git push origin main
```

2. **Streamlit Cloud 설정**
   - https://share.streamlit.io/ 접속
   - GitHub 계정 연동
   - "New app" 클릭
   - Repository: `your-username/junwoo`
   - Branch: `main`
   - Main file path: `streamlit_app/app.py`

3. **환경 변수 설정 (중요!)**
   - Streamlit Cloud 대시보드 → Settings → Secrets
   - TOML 형식으로 입력:
   ```toml
   DART_API_KEY = "your_actual_api_key_here"
   ```

   - 앱 코드에서 사용:
   ```python
   import streamlit as st
   DART_API_KEY = st.secrets["DART_API_KEY"]
   ```

4. **배포 확인**
   - 자동으로 앱 빌드 시작
   - 로그 확인하여 에러 체크
   - 배포 완료 시 공개 URL 제공: `https://your-app.streamlit.app`

---

#### Task 4.4: 에러 모니터링 및 최적화

**성능 최적화:**

1. **캐싱 활용**
```python
@st.cache_resource
def load_model():
    """모델은 한 번만 로딩"""
    return joblib.load(MODEL_PATH)

@st.cache_data(ttl=3600)  # 1시간 캐시
def get_financial_statements(corp_code, year):
    """같은 기업/연도는 재조회 안 함"""
    return dart_client.get_financial_statements(corp_code, year)
```

2. **에러 핸들링**
```python
try:
    result = dart_client.search_company(company_name)
except requests.exceptions.Timeout:
    st.error("⏰ DART API 응답 시간 초과. 잠시 후 다시 시도해주세요.")
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 429:
        st.error("⚠️ API 호출 한도 초과. 1분 후 다시 시도해주세요.")
    else:
        st.error(f"❌ API 오류: {str(e)}")
except Exception as e:
    st.error(f"❌ 예상치 못한 오류: {str(e)}")
    st.exception(e)  # 개발 모드에서만
```

3. **로깅**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 사용
logger.info(f"DART API 조회: {company_name}")
logger.error(f"예측 실패: {error_msg}")
```

---

## ✅ 구현 검증 체크리스트

### Phase 1: DART API 연동
- [ ] DART API Key 발급 완료
- [ ] `src/dart_api/client.py` 구현 완료
- [ ] 기업 검색 기능 테스트 (삼성전자, SK하이닉스 등)
- [ ] 재무제표 조회 기능 테스트
- [ ] 파싱 모듈 단위 테스트 통과
- [ ] Rate Limit 핸들링 확인
- [ ] 에러 케이스 처리 (존재하지 않는 기업, 네트워크 오류 등)

### Phase 2: 도메인 특성 생성
- [ ] `DomainFeatureGenerator` 클래스 구현 완료
- [ ] 유동성 위기 특성 10개 생성 확인
- [ ] 지급불능 특성 8개 생성 확인
- [ ] 재무조작 탐지 특성 15개 생성 확인
- [ ] 총 65개 특성 자동 생성 검증
- [ ] 결측치/무한대 처리 확인
- [ ] 샘플 기업으로 End-to-End 테스트

### Phase 3: Streamlit 앱 개선
- [ ] DART API 검색 입력 모드 구현
- [ ] 실시간 예측 파이프라인 구축
- [ ] 위험 요인 분석 섹션 추가
- [ ] 개선 권장사항 알고리즘 구현
- [ ] 동종업계 벤치마크 비교 기능
- [ ] PDF 리포트 생성 기능
- [ ] 로컬에서 `streamlit run streamlit_app/app.py` 정상 작동

### Phase 4: 배포
- [ ] `requirements.txt` 정리
- [ ] `.env.example` 생성
- [ ] Git LFS 설정
- [ ] GitHub에 코드 푸시
- [ ] Streamlit Cloud 배포 완료
- [ ] 환경 변수 설정 (DART_API_KEY)
- [ ] 배포 URL 접속 확인
- [ ] 실제 기업 조회 테스트 (최소 5개 기업)
- [ ] 모바일 반응형 확인

---

## 🔍 테스트 시나리오

### 시나리오 1: 정상 상장기업 조회
```
입력: "삼성전자", 2023년
예상 결과:
- 재무제표 정상 조회
- 65개 특성 생성
- 부도 확률 < 5% (안전 등급)
- 벤치마크 상위 10%
```

### 시나리오 2: 재무 위기 기업
```
입력: 부도 위험이 높은 기업
예상 결과:
- 부도 확률 > 70%
- Critical 리스크 3개 이상
- 유동성/지급불능 경고
- 구체적 개선안 제시
```

### 시나리오 3: 비상장기업 (수동 입력)
```
입력: 재무제표 직접 입력
예상 결과:
- 특성 생성 정상 작동
- DART 데이터 없음 안내
- 벤치마크는 산업 평균과만 비교
```

### 시나리오 4: 에러 케이스
```
입력: 존재하지 않는 기업명
예상 결과:
- 친절한 오류 메시지
- 재입력 유도
- 앱 크래시 없음
```

---

## 📚 추가 개발 가이드

### DART API 상세 가이드

**파일 생성:** `docs/DART_API_Guide.md`

**내용:**
1. API 엔드포인트 목록
2. 응답 JSON 구조 예시
3. 계정과목 매핑 테이블
4. Rate Limit 정책
5. 에러 코드 목록

### 특성 공학 명세서

**파일 생성:** `docs/Feature_Engineering_Spec.md`

**내용:**
1. 65개 특성 전체 목록
2. 각 특성의 계산 공식
3. 임계값 및 해석 기준
4. 업계별 평균값 참고표

---

## 🚀 실행 프롬프트 (Claude Code에 제공)

### 프롬프트 1: DART API 모듈 구현

```
다음 요구사항에 따라 DART API 연동 모듈을 구현해주세요:

1. src/dart_api/client.py 생성
   - DartAPIClient 클래스 구현
   - search_company() 메서드: 기업명 → 종목코드 변환
   - get_financial_statements() 메서드: 재무제표 조회 (재무상태표, 손익계산서, 현금흐름표)
   - Rate Limit 처리: 1초당 1회 제한
   - 에러 핸들링: Timeout, HTTPError, 존재하지 않는 기업

2. src/dart_api/parser.py 생성
   - FinancialStatementParser 클래스 구현
   - DART API JSON 응답 → 표준 재무제표 딕셔너리 변환
   - 계정과목 매핑 (DART 표준 → 프로젝트 표준)
   - 금액 단위 변환 (원 → 백만원)
   - 데이터 검증 (음수 자산 등 이상치 탐지)

3. tests/test_dart_api.py 생성
   - 기업 검색 테스트 (삼성전자)
   - 재무제표 조회 테스트
   - 에러 케이스 테스트

4. 환경 설정
   - .env.example 생성 (DART_API_KEY 포함)
   - src/dart_api/__init__.py 생성

참고 문서:
- DART API: https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019001
- 기존 코드: docs/notebook_summaries/02_고급_도메인_특성공학_summary.md

중요 제약사항:
- 절대 하드코딩 금지 (모든 값은 환경 변수 또는 상수)
- 한글 UTF-8 인코딩 필수
- docstring 한글로 작성
- 타입 힌트 사용 (typing 모듈)

구현 완료 후:
1. 단위 테스트 실행 (pytest tests/test_dart_api.py)
2. 삼성전자 재무제표 조회 예시 실행
3. 결과 출력하여 확인
```

### 프롬프트 2: 도메인 특성 자동 생성 모듈

```
재무제표 데이터를 입력받아 65개 도메인 특성을 자동 생성하는 모듈을 구현해주세요:

1. src/domain_features/feature_generator.py 생성
   - DomainFeatureGenerator 클래스 구현
   - generate_all_features() 메서드: 재무제표 → 65개 특성 DataFrame
   - 7개 카테고리 특성 생성 메서드 구현

2. 카테고리별 특성 생성 파일들:
   - src/domain_features/liquidity_features.py (유동성 위기 10개)
   - src/domain_features/insolvency_features.py (지급불능 8개)
   - src/domain_features/manipulation_features.py (재무조작 15개)
   - src/domain_features/korea_market_features.py (한국 시장 13개)
   - src/domain_features/stakeholder_features.py (이해관계자 9개)
   - src/domain_features/composite_features.py (복합 리스크 10개)

3. 특성 생성 로직:
   - notebooks/02_고급_도메인_특성공학.ipynb의 로직을 모듈화
   - 각 특성의 계산 공식, 의미, 임계값을 docstring에 명시
   - 무한대/결측치 처리 (np.inf → 0, NaN → 0)
   - 이상치 클리핑 (비율은 -10 ~ 10)

4. tests/test_feature_generator.py 생성
   - 샘플 재무제표로 특성 생성 테스트
   - 65개 특성 모두 생성 확인
   - 데이터 타입 및 범위 검증

참고 자료:
- docs/notebook_summaries/02_고급_도메인_특성공학_summary.md
- CLAUDE.md의 "도메인 기반 특성 공학" 섹션

핵심 특성 예시:
- 현금소진일수 = 현금및현금성자산 / (일평균영업비용 + 1)
- 이자보상배율 = 영업이익 / (이자비용 + 1)
- 유동비율 = 유동자산 / (유동부채 + 1)

구현 완료 후:
1. 테스트 실행
2. 샘플 재무제표로 특성 생성 시연
3. 생성된 특성 DataFrame 출력
```

### 프롬프트 3: Streamlit 앱 고도화

```
기존 streamlit_app/app.py를 대폭 개선하여 실무 활용 가능한 앱으로 업그레이드해주세요:

1. 입력 방식 다양화
   - DART API 검색 모드 추가 (기업명 → 자동 조회)
   - 재무제표 직접 입력 모드 (기존 개선)
   - CSV 업로드 모드 (기존 유지)

2. 대시보드 개선
   - 섹션 1: 종합 평가 (부도 확률, 위험 등급, 업계 순위)
   - 섹션 2: 위험 요인 분석 (Critical/Warning 리스트, SHAP 차트)
   - 섹션 3: 개선 권장사항 (우선순위별 액션 플랜, 시뮬레이션)
   - 섹션 4: 동종업계 벤치마크 (비교 테이블, 레이더 차트)
   - 섹션 5: 재무제표 원본 데이터 (접을 수 있는 탭)
   - 섹션 6: PDF/Excel 리포트 다운로드

3. 새로운 기능 구현
   - 위험 요인 식별 알고리즘 (identify_critical_risks)
   - 개선 권장사항 생성 (generate_recommendations)
   - 업계 벤치마크 비교 (load_industry_benchmarks)
   - 시나리오 시뮬레이션 (simulate_improvement)

4. 시각화 컴포넌트 (src/visualization/charts.py)
   - SHAP waterfall 차트
   - 업계 비교 레이더 차트
   - 시뮬레이션 결과 차트

5. PDF 리포트 생성 (src/visualization/report.py)
   - reportlab 또는 weasyprint 사용
   - 7페이지 리포트 (표지, 종합평가, 위험분석, 권장사항, 벤치마크, 재무제표)

참고:
- 기존 app.py: streamlit_app/app.py
- 위 "Task 3.1: 메인 앱 리팩토링" 섹션의 코드 예시
- Plotly 차트 사용 (한글 폰트 설정 필수)

중요 요구사항:
- 한글 폰트 깨짐 방지 (CLAUDE.md 참고)
- 모바일 반응형 레이아웃
- 로딩 상태 표시 (st.spinner)
- 에러 핸들링 (try-except + st.error)

구현 완료 후:
1. streamlit run streamlit_app/app.py 실행
2. DART API 모드로 "삼성전자" 조회
3. 모든 섹션 정상 작동 확인
4. 스크린샷 저장
```

### 프롬프트 4: 배포 준비 및 Streamlit Cloud 배포

```
Streamlit Cloud 배포를 위한 모든 설정을 완료해주세요:

1. 의존성 관리
   - streamlit_app/requirements.txt 생성
   - 필수 패키지: streamlit, pandas, numpy, plotly, requests, scikit-learn, xgboost, lightgbm, shap, joblib, reportlab
   - 버전 고정 (pip freeze 기반)

2. 환경 설정
   - .env.example 생성 (DART_API_KEY 템플릿)
   - streamlit_app/.streamlit/config.toml 생성 (테마, 서버 설정)
   - streamlit_app/config.py 생성 (상수, 경로 관리)

3. Git LFS 설정
   - .gitattributes 생성
   - *.pkl, *.csv 파일을 LFS로 추적
   - git lfs install 및 track 명령 실행

4. README.md 업데이트
   - 프로젝트 소개
   - DART API Key 발급 방법
   - 로컬 실행 방법
   - Streamlit Cloud 배포 방법
   - 라이선스

5. 배포 가이드 문서 생성
   - docs/Deployment_Guide.md
   - Streamlit Cloud 배포 단계별 가이드
   - 환경 변수 설정 방법
   - 트러블슈팅

6. 최적화
   - 모델 로딩 캐싱 (@st.cache_resource)
   - API 응답 캐싱 (@st.cache_data)
   - 로깅 설정

필수 체크리스트:
- [ ] requirements.txt 생성
- [ ] .env.example 생성
- [ ] .gitattributes 생성 (Git LFS)
- [ ] config.toml 생성
- [ ] README.md 업데이트
- [ ] 모든 하드코딩 제거
- [ ] 상대 경로 사용 (절대 경로 금지)

구현 완료 후:
1. 로컬에서 streamlit run 테스트
2. Git 커밋 및 푸시 준비
3. Streamlit Cloud 배포 가이드 출력
```

---

## 🎯 성공 기준

### 기능적 요구사항
- [x] 상장기업 이름 입력 → 자동 재무제표 조회
- [x] 재무제표 → 65개 도메인 특성 자동 생성
- [x] 특성 → 부도 확률 예측 (모델 활용)
- [x] 위험 요인 상세 분석 및 설명
- [x] 실행 가능한 개선 권장사항 제공
- [x] 동종업계 벤치마크 비교
- [x] PDF 리포트 다운로드
- [x] Streamlit Cloud 배포 완료

### 비기능적 요구사항
- [x] 응답 시간 < 10초 (API 조회 + 예측)
- [x] 모바일 반응형 UI
- [x] 에러 발생 시 친절한 메시지
- [x] 한글 폰트 깨짐 없음
- [x] 코드 모듈화 (유지보수 용이)
- [x] 단위 테스트 커버리지 > 70%

### 사용자 경험
- [x] 비전문가도 5분 내 사용 가능
- [x] 결과 해석이 명확함
- [x] 다음 액션이 구체적으로 제시됨
- [x] 은행 대출 심사, 투자 검토에 활용 가능

---

## 📞 지원 및 참고 자료

### 공식 문서
- DART API: https://opendart.fss.or.kr/guide/main.do
- Streamlit Docs: https://docs.streamlit.io/
- Plotly: https://plotly.com/python/
- SHAP: https://shap.readthedocs.io/

### 내부 참고 자료
- CLAUDE.md (프로젝트 가이드)
- docs/notebook_summaries/ (노트북 요약)
- notebooks/ (원본 분석 노트북)

### 문의
- DART API 관련: opendart@fss.or.kr
- 프로젝트 관련: GitHub Issues

---

## 📝 변경 이력

| 버전 | 날짜 | 변경 내용 | 작성자 |
|------|------|-----------|--------|
| 1.0 | 2024-01-XX | 초안 작성 | Claude |

---

**이 문서는 Claude Code에게 프롬프트로 제공하여 자동 구현을 유도하기 위한 상세 개발 계획서입니다.**

**각 프롬프트를 순서대로 Claude Code에 제공하면서 단계적으로 구현하세요.**
