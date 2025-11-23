# Part 1: 프로젝트 개요 및 현황 분석

> **읽기 시간**: 5분 | **난이도**: ⭐ 기초

---

## 📋 프로젝트 개요

**프로젝트명**: 한국 기업 부도 예측 시스템
**목적**: Jupyter 노트북에서 개발된 분석 로직을 Streamlit 웹 대시보드로 전환

### 핵심 요구사항
- 노트북의 모든 분석, 특성 생성, 모델링 로직을 **정확히** 재현
- 비즈니스 가치 분석을 포함한 실용적인 대시보드 제공
- 프로덕션 품질: 에러 처리, 로깅, 사용자 경험 최적화

### 주요 기술 스택
- **Backend**: Python, pandas, numpy, scikit-learn
- **Model**: CatBoost (최종 선택), XGBoost, LightGBM
- **Frontend**: Streamlit
- **Visualization**: Plotly, SHAP
- **Data**: 한국 기업 50,105개, 159 변수, 부도율 1.5%

---

## 📚 참고 노트북 4개

### Part 1: 문제정의 및 핵심발견
- **파일**: `notebooks/발표_Part1_문제정의_및_핵심발견_executed.ipynb` (1.1MB)
- **내용**: 탐색적 데이터 분석, 클래스 불균형(1:66), 유동성이 핵심 변수
- **핵심 발견**:
  - 유동비율, 당좌비율, 현금비율이 정상 vs 부도 기업 구별력 높음
  - 업종별 부도율 2배 차이 (건설 2.8% vs 금융 0.9%)
  - 외감 여부가 신뢰성에 영향

### Part 2: 도메인 특성 공학
- **파일**: `notebooks/발표_Part2_도메인_특성_공학_완전판_executed.ipynb` (114KB)
- **내용**: 51개 도메인 기반 특성 생성
- **출력물**: `data/features/domain_based_features_완전판.csv` (51개 특성)
- **카테고리**:
  1. 유동성 위기 (9개): 즉각지급능력, 현금소진일수 등
  2. 지급불능 패턴 (11개): 자본잠식도, 이자보상배율 등
  3. 재무조작 탐지 (17개): M-Score, 발생액비율 등
  4. 이해관계자 행동 (10개): 연체, 신용등급 등
  5. 한국 시장 특화 (6개): 외감여부, 제조업리스크 등

### Part 3: 모델링 및 최적화
- **파일**: `notebooks/발표_Part3_모델링_및_최적화_완전판_executed.ipynb` (3.2MB)
- **내용**: 6개 모델 비교, CatBoost 최종 선택, SMOTE, 임계값 최적화
- **출력물**:
  - `data/processed/발표_Part3_v3_최종모델.pkl` (CatBoost)
  - `data/processed/발표_Part3_v3_임계값.pkl` (Red=0.0468, Yellow=0.0168)
- **성능**: PR-AUC 0.16, Recall 86.84%, F2-Score 0.20

### Part 4: 결과 및 비즈니스 가치
- **파일**: `notebooks/발표_Part4_결과_및_비즈니스_가치.ipynb` (852KB)
- **내용**: SHAP 분석, Traffic Light 시스템, 재무적 영향 분석
- **핵심 인사이트**:
  - TP 132개: 462M KRW 손실 방지
  - ROI 920%, Payback 1.3개월
  - 자동 승인 39%, 인력 70% 절감

---

## 📂 Streamlit 앱 현재 구조

```
streamlit_app/
├── app.py                    # 메인 앱 (320줄)
│   ├── 3가지 입력 모드: DART API, 직접 입력, 샘플 데이터
│   ├── run_analysis(): 분석 실행
│   └── display_*(): 결과 표시 함수들
├── config.py                 # 설정 파일 (54줄)
│   ├── 경로 설정 (MODEL_PATH, SCALER_PATH)
│   ├── RISK_THRESHOLDS (임계값) ❌ 노트북과 불일치!
│   └── 한글 폰트 설정
└── app_original.py           # 백업

src/
├── domain_features/          # 특성 생성 모듈
│   ├── feature_generator.py  # 메인 생성기 (276줄)
│   ├── liquidity_features.py
│   ├── insolvency_features.py
│   ├── manipulation_features.py
│   ├── korea_market_features.py
│   ├── stakeholder_features.py
│   └── composite_features.py  # ⚠️ 노트북에 없음?
├── models/
│   └── predictor.py          # 예측기 (202줄)
│       ├── load_model()
│       ├── predict()
│       └── _heuristic_prediction()
├── visualization/
│   └── charts.py             # Plotly 차트 (249줄)
│       ├── create_risk_gauge()
│       ├── create_shap_waterfall() ❌ 가짜 SHAP!
│       └── create_radar_chart()
├── dart_api/                 # DART API 연동
│   ├── client.py
│   └── parser.py
└── utils/
    └── helpers.py            # 헬퍼 함수 (276줄)
        ├── get_risk_level() ❌ 임계값 불일치!
        ├── identify_critical_risks()
        ├── identify_warnings()
        └── generate_recommendations()
```

---

## 🎯 작업 목표

### 최우선 목표 (Must-Have)
1. **노트북-앱 간 로직 일치**: 100% 동일하게 작동
   - 특성 생성 공식
   - 모델 파일 경로
   - Traffic Light 임계값
   - 전처리 파이프라인

2. **비즈니스 가치 시각화**: Part 4의 핵심 인사이트 통합
   - 실제 SHAP 분석
   - Traffic Light 성능
   - ROI, Confusion Matrix

3. **프로덕션 품질**: 안정성 및 UX
   - 에러 처리 (모델 없음, 잘못된 입력 등)
   - 로깅 및 디버깅
   - 명확한 사용자 메시지

### 부차적 목표 (Nice-to-Have)
- DART API 통합 개선
- 샘플 데이터 다양화
- 반응형 UI 최적화

---

## 🚨 발견된 주요 차이점 요약

총 **8개 Critical/High 이슈** 발견:

| # | 이슈 | 심각도 | 파일 | 상태 |
|---|------|--------|------|------|
| 1 | 모델 파일명 불일치 | 🔴 Critical | `config.py` | CatBoost ≠ XGBoost |
| 2 | Traffic Light 임계값 10배 차이 | 🔴 Critical | `config.py`, `helpers.py` | 0.0468 ≠ 0.3 |
| 3 | 특성 개수 불일치 | 🔴 Critical | `feature_generator.py` | 51개 ≠ 65개? |
| 4 | SHAP 값이 가짜 | 🟡 High | `charts.py` | 하드코딩 ≠ 실제 계산 |
| 5 | 비즈니스 가치 분석 누락 | 🟡 High | `app.py` | Part 4 미반영 |
| 6 | 데이터 유출 특성 미제거 | 🟡 High | `stakeholder_features.py` | 이해관계자_불신지수 |
| 7 | 전처리 파이프라인 불일치 | 🟡 High | `predictor.py` | LogTransformer 누락 |
| 8 | 특성 선택 로직 없음 | 🟡 High | `predictor.py` | IV/VIF 미적용 |

**상세 분석**: `02_critical_issues.md` 참조

---

## 📊 전체 작업 흐름

### Phase 1: Critical Fixes (1~2시간)
```
Priority 1 작업 (시스템 작동 불가 수준)
├── Task 1: 모델 파일명 수정 (10분)
├── Task 2: Traffic Light 임계값 수정 (15분)
├── Task 3: 특성 개수 일치 확인 (1시간)
└── 검증: 샘플 데이터로 전체 플로우 테스트
```

### Phase 2: Feature Enhancements (3~4시간)
```
Priority 2 작업 (기능 부정확성)
├── Task 4: 실제 SHAP 값 계산 (2시간)
├── Task 5: 비즈니스 가치 분석 추가 (2시간)
├── Task 6: 데이터 유출 특성 제거 (30분)
├── Task 7: 전처리 파이프라인 동기화 (1시간)
└── Task 8: 특성 선택 로직 구현 (1시간)
```

### Phase 3: Validation & Polish (30분)
```
전체 검증
├── 노트북 결과와 비교 (특성값, 예측 확률)
├── 에러 케이스 테스트
├── 성능 테스트
└── 문서화
```

---

## 🔗 다음 단계

**다음 파일을 읽으세요**: `02_critical_issues.md`

이 파일에서는 발견된 8개 주요 차이점을 상세히 분석합니다:
- 각 이슈의 원인 분석
- 노트북 vs 앱 코드 비교
- 예상 영향 및 리스크
- 수정 방향

---

**파일**: `01_overview.md`
**작성일**: 2025-11-23
**다음**: `02_critical_issues.md`
