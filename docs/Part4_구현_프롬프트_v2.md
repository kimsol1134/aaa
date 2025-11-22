# Part 4: 결과 및 비즈니스 가치 (v2 개선판 기반)

## 📙 Part 4: 결과 및 비즈니스 가치

당신은 시니어 데이터 사이언티스트이자 비즈니스 애널리스트입니다.

# 작업 목표
Part 3 v2 개선판에서 완성된 모델을 바탕으로 발표용 "Part 4: 결과 및 비즈니스 가치" 노트북을 생성하세요.

# Part 3 v2 개선판 핵심 개선사항 (반드시 반영할 것)

## P0 (Critical) 개선사항
1. **ImbPipeline 구조 도입**
   - 전처리 + 리샘플링 + 모델이 하나의 Pipeline으로 통합
   - 모델 로딩 시 `pipeline.named_steps['classifier']`로 접근 필요
   - SHAP 분석 시 Pipeline 구조 고려

2. **AutoML n_iter 200으로 증가**
   - LGBM/XGB: 200회, CatBoost: 150회
   - 더 나은 하이퍼파라미터 탐색 완료

3. **앙상블 다양성 체크**
   - Top 3 모델이 모두 GBM일 경우 LR/RF 강제 포함
   - 모델 선택 시 다양성 고려됨

## P1 (Recommended) 개선사항
4. **CV 기반 Threshold 최적화**
   - Validation 단독이 아닌 Validation + CV 평균 사용
   - 더 robust한 threshold 선정

5. **Winsorizer 실험**
   - 이상치 처리 유무 비교 실험 포함
   - 데이터 기반 의사결정

6. **Yellow Threshold 일관성**
   - Recall 95% 기반 일관된 로직
   - Conservative fallback (Red × 0.5)

## 데이터 구조
- **3-Way Split**: Train (60%), Validation (20%), Test (20%)
- **Data Leakage 방지**: 모든 결정이 Validation 기반, Test는 최종 평가만
- **Pipeline 메타데이터**: 모델, threshold, Traffic Light 정보 모두 저장됨

---

# 필수 요구사항

## 1. 기존 노트북 활용
- `notebooks/발표_Part3_모델링_및_최적화_v2_개선판.ipynb` (최종 모델 출처)
- `notebooks/05_모델_평가_및_해석.ipynb` (SHAP 분석 참고)
- **중요**: Part 3 v2 개선판의 메타데이터 파일 (`Part3_v2_메타데이터.pkl`) 활용

## 2. 노트북 구조

### Opening: Part 3 요약 (v2 개선판 반영)
```markdown
## 📌 최종 모델 (Part 3 v2 개선판)

### 모델 구조
- **Pipeline 기반**: 전처리 + 리샘플링 + 분류기가 통합된 ImbPipeline
- **최종 모델**: XGBoost (or 다른 선택된 모델)
- **샘플링**: SMOTE + Tomek Links (Pipeline 내부)
- **데이터 분할**: 3-Way Split (Train 60% / Validation 20% / Test 20%)

### 선택 과정
- **AutoML**: RandomizedSearchCV (n_iter=200, 5-Fold CV)
- **Threshold 최적화**: Validation + CV 평균 (더 robust)
- **앙상블 다양성**: GBM 편중 방지 체크 완료
- **Winsorizer 실험**: 데이터 기반 의사결정

### 최종 하이퍼파라미터
- [Part 3에서 선택된 최종 파라미터]
- PR-AUC: [Validation 성능]
- F2-Score: [Validation 성능]

### Data Leakage 방지
✅ Test 세트는 최종 평가에만 사용
✅ 모든 결정(모델 선택, threshold, Traffic Light)은 Validation 기반

이제 **Test 세트 성능 평가**와 **비즈니스 가치**를 분석합니다.
```

### 1. 환경 설정 및 모델 로딩

**Pipeline 구조 고려한 로딩 코드:**
```python
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    confusion_matrix,
    classification_report
)
import shap
import matplotlib.pyplot as plt

# 메타데이터 로딩
metadata = joblib.load('../data/processed/Part3_v2_메타데이터.pkl')

# Pipeline 로딩 (ImbPipeline 구조)
best_pipeline = metadata['best_pipeline']
selected_model_name = metadata['selected_model_name']
optimal_threshold = metadata['selected_threshold']

# Test 데이터 로딩
X_test = metadata['X_test']
y_test = metadata['y_test']

print(f"### 최종 선택 모델: {selected_model_name}")
print(f"### 최적 Threshold: {optimal_threshold:.4f}")
print(f"### Test 세트 크기: {len(X_test)} (부도: {y_test.sum()}건)")

# Pipeline 내부 확인
print("\n### Pipeline 구조:")
for name, step in best_pipeline.named_steps.items():
    print(f"  - {name}: {type(step).__name__}")
```

### 2. Test 세트 최종 성능 평가

**Bootstrap CI 추가:**
```python
from sklearn.utils import resample

# Test 세트 예측
y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= optimal_threshold).astype(int)

# PR-AUC Bootstrap CI
n_iterations = 1000
pr_aucs = []

np.random.seed(42)
for i in range(n_iterations):
    indices = resample(range(len(y_test)), random_state=i)
    y_test_boot = y_test.iloc[indices]
    y_pred_boot = y_test_prob[indices]

    pr_auc = average_precision_score(y_test_boot, y_pred_boot)
    pr_aucs.append(pr_auc)

print(f"### PR-AUC (Test)")
print(f"  Point Estimate: {np.mean(pr_aucs):.4f}")
print(f"  95% CI: [{np.percentile(pr_aucs, 2.5):.4f}, {np.percentile(pr_aucs, 97.5):.4f}]")

# F2-Score
f2_test = fbeta_score(y_test, y_test_pred, beta=2)
print(f"\n### F2-Score (Test): {f2_test:.4f}")

# Validation vs Test 비교
print(f"\n### Validation vs Test 성능 비교")
print(f"  Validation PR-AUC: {metadata.get('validation_pr_auc', 'N/A'):.4f}")
print(f"  Test PR-AUC: {np.mean(pr_aucs):.4f}")
print(f"  차이: {abs(metadata.get('validation_pr_auc', 0) - np.mean(pr_aucs)):.4f}")
```

### 3. Confusion Matrix (재무 관점 해석)

```python
# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

print("### Confusion Matrix (Test)")
print(f"  True Negative (정상→정상): {tn:,}건")
print(f"  False Positive (정상→부도): {fp:,}건")
print(f"  False Negative (부도→정상): {fn:,}건")
print(f"  True Positive (부도→부도): {tp:,}건")

# Type II Error
type2_error = fn / (fn + tp) if (fn + tp) > 0 else 0
print(f"\n### Type II Error: {type2_error:.2%}")
print(f"  → 부도 기업 중 {type2_error:.1%}를 놓침")
```

**재무 관점 해석 마크다운:**
```markdown
### 💡 재무 해석

#### True Positive (부도 기업을 부도로 예측): [tp]건
- **비즈니스 가치**: 부도 위험 사전 차단
- **예상 손실 회피**: [tp] × 67백만원 = [계산]억원
- **실무 액션**: 대출 거절 또는 담보 강화

#### False Negative (부도 기업을 정상으로 예측): [fn]건 ⚠️
- **비즈니스 리스크**: 부도 미탐지 → 실제 손실 발생
- **예상 손실**: [fn] × 67백만원 = [계산]억원
- **개선 필요**: FN 감소가 최우선 과제
- **원인 추정**:
  1. 데이터에 없는 정보 (소송, 경영진 이슈, 산업 환경 변화)
  2. 급격한 악화 (시계열 데이터 부재)
  3. 극도 불균형 (1:66) → 소수 클래스 학습 어려움

#### False Positive (정상 기업을 부도로 예측): [fp]건
- **비즈니스 기회 손실**: 정상 기업 대출 거절
- **예상 이자 수익 손실**: [fp] × 연 5% 이자 = [계산]억원
- **실무 영향**: 고객 불만, 시장 점유율 감소

#### Type II Error: [type2_error]%
- **의미**: 부도 기업의 절반 이상을 놓침
- **한계 인정**: 현재 데이터와 모델로는 달성 가능한 최선
- **개선 방향**: 외부 데이터(뉴스, 소송 이력) 통합 필요
```

### 4. SHAP 분석 (Pipeline 구조 반영)

**Pipeline에서 모델 추출:**
```python
# ImbPipeline에서 실제 classifier 추출
classifier = best_pipeline.named_steps['classifier']

# 전처리된 데이터 준비 (Pipeline의 앞단만 실행)
# Option 1: Pipeline의 transform 사용
preprocessed_steps = [
    step for name, step in best_pipeline.named_steps.items()
    if name not in ['resampler', 'classifier']
]

# X_test를 전처리
X_test_processed = X_test.copy()
for name, step in best_pipeline.named_steps.items():
    if name in ['inf_handler', 'imputer', 'scaler']:  # Preprocessing only
        X_test_processed = step.transform(X_test_processed)
    elif name == 'classifier':
        break

# SHAP TreeExplainer (XGBoost/LightGBM 등 tree 모델)
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_test_processed)

# Global Importance
shap.summary_plot(shap_values, X_test_processed,
                  feature_names=X_test.columns,
                  plot_type="bar",
                  max_display=10)

# Beeswarm plot
shap.summary_plot(shap_values, X_test_processed,
                  feature_names=X_test.columns,
                  max_display=10)
```

**Top 10 특성 재무 해석:**
```markdown
### Top 10 중요 특성 해석

1. **[Feature 1 name]** (SHAP: [value])
   - 재무 의미: [도메인 해석]
   - 위험 기준: [임계값]
   - 비즈니스 함의: [실무 의미]

2. **신용등급점수** (만약 상위권이면)
   - 재무 의미: 신용평가사의 종합 평가
   - ⚠️ **Data Leakage 가능성 검토 필요**
     - 신용등급이 부도 정보를 이미 반영했을 수 있음
     - 실무 사용 시 신용등급 제외 버전도 준비 권장

[나머지 특성도 동일하게 해석]
```

**Failure Case 분석:**
```python
# False Negative 케이스 분석 (모델이 놓친 부도 기업)
fn_indices = np.where((y_test == 1) & (y_test_pred == 0))[0]

print(f"### False Negative 분석 (총 {len(fn_indices)}건)")
print("\n샘플 케이스 (예측 확률이 낮았던 순):")

# 예측 확률 낮은 순으로 정렬
fn_sorted = sorted(fn_indices, key=lambda i: y_test_prob[i])[:5]

for rank, idx in enumerate(fn_sorted[:5], 1):
    print(f"\n--- Case {rank} ---")
    print(f"실제: 부도, 예측: 정상 (확률: {y_test_prob[idx]:.4f})")

    # 해당 케이스의 주요 특성 출력
    case_features = X_test.iloc[idx]
    print("주요 특성:")
    for col in X_test.columns[:5]:  # 상위 5개만
        print(f"  {col}: {case_features[col]:.3f}")

    # SHAP Force Plot (개별 예측 설명)
    shap.force_plot(explainer.expected_value,
                    shap_values[idx],
                    X_test_processed.iloc[idx],
                    feature_names=X_test.columns,
                    matplotlib=True)
```

### 5. Traffic Light 시스템

**메타데이터에서 Traffic Light Threshold 로딩:**
```python
# Part 3에서 저장된 Traffic Light threshold
red_threshold = metadata.get('red_threshold', optimal_threshold)
yellow_threshold = metadata.get('yellow_threshold', optimal_threshold * 0.5)

print(f"### Traffic Light 기준")
print(f"  🔴 Red (고위험): >= {red_threshold:.4f}")
print(f"  🟡 Yellow (중위험): [{yellow_threshold:.4f}, {red_threshold:.4f})")
print(f"  🟢 Green (저위험): < {yellow_threshold:.4f}")

# Test 세트에 적용
def classify_risk(prob):
    if prob >= red_threshold:
        return 'Red'
    elif prob >= yellow_threshold:
        return 'Yellow'
    else:
        return 'Green'

y_test_risk = pd.Series(y_test_prob).apply(classify_risk)

# 분포 출력
print("\n### Test 세트 위험도 분포")
print(y_test_risk.value_counts())

# 위험도별 실제 부도율
print("\n### 위험도별 실제 부도율")
for risk_level in ['Green', 'Yellow', 'Red']:
    mask = (y_test_risk == risk_level)
    if mask.sum() > 0:
        actual_default_rate = y_test[mask].mean()
        print(f"  {risk_level}: {actual_default_rate:.2%} (n={mask.sum()})")
```

**비즈니스 의사결정 규칙:**
```markdown
### 실무 적용 가이드

#### 🔴 Red Zone (고위험)
- **액션**: 즉시 대출 거절
- **예외**: 담보가 충분한 경우만 검토
- **실제 부도율**: [계산]%

#### 🟡 Yellow Zone (중위험)
- **액션**: 추가 실사 (Due Diligence)
  - 현장 방문
  - 경영진 인터뷰
  - 주요 거래처 확인
  - 최근 뉴스 검색
- **승인 조건**: 담보 또는 보증인 필수
- **실제 부도율**: [계산]%

#### 🟢 Green Zone (저위험)
- **액션**: 일반 프로세스 진행
- **주의**: Green이라도 정기 모니터링 필요
- **실제 부도율**: [계산]%
```

### 6. 비즈니스 가치 시뮬레이션

```python
# 시나리오: 연간 10,000건 심사 기준
n_applications = 10000
default_rate = 0.015  # 1.5%
avg_loan = 100_000_000  # 1억원
default_loss_rate = 0.67  # 회수율 33%
interest_rate = 0.05  # 연 5%

expected_defaults = int(n_applications * default_rate)

print("### 비즈니스 가치 시뮬레이션 (연간 10,000건 심사)")

# Baseline (모델 없이 모두 승인)
baseline_loss = expected_defaults * avg_loan * default_loss_rate
baseline_revenue = (n_applications - expected_defaults) * avg_loan * interest_rate
baseline_profit = baseline_revenue - baseline_loss

print(f"\n#### Baseline (모델 미사용)")
print(f"  예상 부도 건수: {expected_defaults}건")
print(f"  부도 손실: {baseline_loss/1e8:.1f}억원")
print(f"  이자 수익: {baseline_revenue/1e8:.1f}억원")
print(f"  순이익: {baseline_profit/1e8:.1f}억원")

# With Model
recall = tp / (tp + fn)  # Test 세트 Recall
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

detected_defaults = int(expected_defaults * recall)
missed_defaults = expected_defaults - detected_defaults
false_alarms = int(detected_defaults / precision - detected_defaults) if precision > 0 else 0

# 손실 및 수익 계산
model_loss = missed_defaults * avg_loan * default_loss_rate
rejected_good = false_alarms
opportunity_loss = rejected_good * avg_loan * interest_rate
approved = n_applications - detected_defaults - rejected_good
model_revenue = approved * avg_loan * interest_rate
model_profit = model_revenue - model_loss - opportunity_loss

print(f"\n#### With Model")
print(f"  탐지된 부도: {detected_defaults}건 (Recall {recall:.1%})")
print(f"  놓친 부도 (FN): {missed_defaults}건")
print(f"  오탐 (FP): {rejected_good}건")
print(f"  부도 손실: {model_loss/1e8:.1f}억원")
print(f"  기회 손실: {opportunity_loss/1e8:.1f}억원")
print(f"  이자 수익: {model_revenue/1e8:.1f}억원")
print(f"  순이익: {model_profit/1e8:.1f}억원")

print(f"\n#### 개선 효과")
improvement = model_profit - baseline_profit
print(f"  순이익 증가: {improvement/1e8:.1f}억원 ({improvement/baseline_profit*100:+.1f}%)")
```

### 7. 한계 및 개선 방향 (대폭 강화)

```markdown
## 한계 (Limitations) 및 개선 방향

### 1. 데이터 품질 이슈 ⚠️

**문제:**
- 63.7% 기업이 현금 = 0으로 기록
- 재고자산, 매출채권 등도 유사한 문제
- 원본 데이터 출처: [확인 필요]

**원인 추정:**
- 중소기업 회계 시스템 미비
- 실제 현금 부족 vs 기록 누락 구분 불가
- 간편 장부 기장 (복식부기 미적용)

**현재 대응:**
- Binary feature '현금보유여부' 추가
- Robust 통계량 사용 (median, IQR)
- Winsorizer로 극단값 처리 실험 완료

**향후 개선:**
- 원본 데이터 출처 확인 및 검증
- 외부 데이터 결합 (금융감독원, 국세청 사업자 데이터)
- 데이터 품질 스코어링 시스템 구축
- 데이터 신뢰도를 모델 입력으로 활용

---

### 2. 시계열 정보 부족 🕐

**한계:**
- 2021년 8월 단일 시점 스냅샷
- 재무 악화 속도, 추세 파악 불가
- "갑자기" 부도난 기업 탐지 어려움

**Impact:**
- 급격히 악화되는 기업 놓칠 수 있음
- FN [type2_error]%에 기여하는 주요 요인
- 계절성 효과 반영 불가

**향후 개선:**
- 분기별 데이터 확보 (최소 2년치)
- 변화율 특성 추가:
  - 매출 증가율 (YoY, QoQ)
  - 부채 증가율
  - 운전자본 변화율
  - 신용등급 변화 추이
- LSTM/Transformer 기반 시계열 모델 검토
- 조기 경보 시스템 (재무 악화 가속도 탐지)

---

### 3. 모델 성능 한계 📉

**현재 성능 (Test):**
- PR-AUC: [test_pr_auc] (95% CI: [ci_low]-[ci_high])
- F2-Score: [f2_test]
- **False Negative: [type2_error]%** ← 가장 큰 문제
- Recall: [recall]%

**원인 분석:**

#### 3.1 데이터에 없는 정보
- 소송 진행 상황 (법원 데이터)
- 경영진 교체/비리 (뉴스, 금융당국)
- 주요 거래처 부도 (연쇄 부도)
- 업계 환경 변화 (산업 트렌드)
- 노사 분규 (파업 등)

#### 3.2 극도 불균형 (1:66)
- 소수 클래스 학습 어려움
- SMOTE로 일부 완화했으나 근본 한계
- Class Weight 조정도 시도했으나 과적합 발생

#### 3.3 앙상블 효과 제한적 (Part 3에서 확인)
- 모든 Tree 기반 모델 예측 유사 (상관 > 0.95)
- 다양성 체크로 개선 시도했으나 한계
- Meta-learner 과적합 발생

**개선 방안:**

#### 단기 (3개월)
1. **외부 데이터 통합:**
   - 뉴스 감성 분석 (부정적 기사 탐지)
   - 소송 이력 데이터 (법원)
   - 경영진 이력 데이터

2. **앙상블 다양성 증대:**
   - Neural Network 추가 (TabNet, FT-Transformer)
   - Linear Model + Tree Model 조합
   - Stacking Meta-learner를 NN으로 변경

3. **Cost-sensitive Learning 강화:**
   - Focal Loss 적용 (어려운 샘플 집중)
   - FN 비용을 더 높게 설정 (현재 5배 → 10배)
   - Custom Loss Function (FN penalty 증가)

#### 중기 (6개월)
4. **시계열 모델 도입:**
   - 분기별 데이터 확보
   - LSTM/GRU 기반 시퀀스 모델
   - 변화율 특성 자동 생성

5. **Semi-supervised Learning:**
   - 부도 직전 기업 레이블 활용 (Label Spreading)
   - Pseudo-labeling

---

### 4. 일반화 가능성 🌍

**한계:**
- 2021년 데이터 → COVID-19 팬데믹 영향
- 정부 지원금으로 생존한 한계 기업 포함 가능
- 금리 상승기 vs 하락기 영향 미반영

**우려:**
- 2022-2023년에는 이들 기업이 부도날 가능성
- 모델이 이를 반영하지 못함
- 경제 환경 변화 시 성능 저하 가능

**검증 방안:**
- **시간적 검증 (Temporal Validation)**:
  - 2022-2023 데이터로 재평가
  - 성능 유지 여부 확인

- **공간적 검증 (Spatial Validation)**:
  - 다른 국가 데이터로 테스트
  - 한국 특화 특성만 제외하고 적용

- **경제 사이클 고려**:
  - 경기 지표 (GDP, 금리) 추가
  - 시기별 모델 재학습 전략

---

### 5. 해석 가능성 vs 성능 트레이드오프 ⚖️

**현재 선택:**
- 도메인 특성 기반 → 해석 가능성 우선
- Tree 기반 모델 (XGBoost) → SHAP으로 설명 용이
- Pipeline 구조 → 재현 가능성 높음

**트레이드오프:**
- Deep Learning 사용 시 성능 향상 가능 (예상 PR-AUC +0.02~0.03)
- 하지만 설명 어려움 → 실무 적용 제한
- Black-box 모델은 규제 이슈 발생 가능

**결정 근거:**
- **금융권 실무에서는 설명 가능성이 필수**
  - 규제 요구사항 (Basel III, AI 윤리 가이드라인)
  - 심사역이 "왜 거절했는가?" 고객에게 설명해야 함
  - 내부 감사 및 외부 검사 대비

- **사용자 신뢰 확보**
  - SHAP으로 모든 예측 근거 제시 가능
  - 도메인 전문가가 검증 가능한 로직
  - 이상 케이스 발견 시 원인 파악 용이

**향후 방향:**
- 해석 가능한 DL 모델 탐색 (TabNet 등)
- Hybrid 접근: DL로 특성 추출 + Tree로 분류
- 모델 독립적 해석 기법 (LIME, Anchors) 추가 적용

---

## 그럼에도 가치 있는 이유 ✅

### 1. 도메인 논리 명확 ✅
- 모든 특성이 재무 이론 기반 (Altman, Beaver, Beneish)
- 각 특성의 의미를 명확히 설명 가능
- 실무 심사역이 바로 이해하고 활용 가능
- 규제 당국 설명 용이

### 2. 재현 가능 및 확장 가능 ✅
- 다른 연도 데이터에 즉시 적용 가능 (Pipeline 재사용)
- 다른 국가 시장에도 확장 가능 (한국 특화 특성만 제외)
- 전체 파이프라인 자동화 완료
- Git으로 버전 관리 → 실험 재현 보장

### 3. 해석 가능한 AI ✅
- SHAP으로 모든 예측 근거 제시
- Global + Local Explanation 제공
- 규제 요구사항 충족 (AI 윤리)
- 사용자 신뢰 확보

### 4. 실용적 성능 향상 ✅
- **Baseline 대비 개선:**
  - PR-AUC: [baseline] → [final] ([improvement]% 향상)
  - F2-Score: [baseline] → [final] ([improvement]% 향상)
- 실제 손실 감소 효과 입증 (시뮬레이션)
- Traffic Light 시스템으로 의사결정 간소화

### 5. 확장 가능한 프레임워크 ✅
- 외부 데이터 추가 용이 (Pipeline 구조)
- 모델 업그레이드 가능 (Pipeline 교체만)
- 지속적 개선 가능 (A/B 테스트, Champion-Challenger)
- 모니터링 시스템 구축 용이

### 6. 견고한 방법론 (Part 3 v2 개선판) ✅
- **Data Leakage 완벽 방지**: 3-Way Split + Validation 기반 결정
- **Pipeline 구조**: 배포 시 전처리 오류 방지
- **CV 기반 Threshold**: 단일 split 편향 방지
- **앙상블 다양성 체크**: 모델 선택의 합리성 확보
- **통계적 엄격성**: Bootstrap CI, Wilcoxon test 등

---

## 향후 로드맵 🛣️

### Phase 1: 데이터 확장 (3개월)
- 2022-2023 데이터 확보
- 외부 데이터 통합 (뉴스, 소송)
- 시계열 변화율 특성 추가

### Phase 2: 모델 개선 (6개월)
- 시계열 모델 도입 (LSTM/Transformer)
- 앙상블 다양성 증대 (TabNet, NN 추가)
- Cost-sensitive Learning 강화

### Phase 3: 실무 배포 (9개월)
- 실시간 예측 API 개발
- 모니터링 시스템 구축
- A/B 테스트 (Champion-Challenger)
- 성능 추적 대시보드

### Phase 4: 지속 개선 (12개월~)
- 월별 재학습 파이프라인
- 피드백 루프 (실제 부도 vs 예측)
- 모델 드리프트 탐지 및 대응
```

### 8. 최종 요약

```markdown
## 🎯 프로젝트 성과

### 달성한 것 ✅
1. ✅ **도메인 지식 기반 특성 공학**
   - 75개 재무 이론 기반 특성 생성
   - 통계적 유의성 검증 완료
   - 27개 최종 선택 (VIF, IV, Correlation 기준)

2. ✅ **불균형 데이터 처리**
   - SMOTE + Tomek Links 적용
   - 1:66 불균형 비율 완화
   - Class Weight vs SMOTE 비교 완료

3. ✅ **견고한 모델링 프로세스 (Part 3 v2 개선판)**
   - 3-Way Split으로 Data Leakage 완벽 방지
   - ImbPipeline으로 배포 안정성 확보
   - AutoML (n_iter=200) 최적 하이퍼파라미터 탐색
   - CV 기반 robust threshold 선정
   - 앙상블 다양성 체크

4. ✅ **성능 달성 (Test)**
   - PR-AUC: [test_pr_auc] (Baseline 대비 [improvement]% 향상)
   - F2-Score: [f2_test]
   - 95% CI: [[ci_low], [ci_high]]

5. ✅ **해석 가능한 예측 시스템**
   - SHAP Global/Local Explanation
   - Traffic Light 의사결정 지원
   - 모든 특성의 재무적 의미 명확

---

### 주요 한계 ⚠️
1. ⚠️ **FN [type2_error]%** (부도 기업 절반 놓침)
   - 원인: 데이터 외 정보 부족, 극도 불균형
   - 개선: 외부 데이터, 시계열 모델 필요

2. ⚠️ **시계열 데이터 부재**
   - 스냅샷 데이터 → 악화 속도 파악 불가
   - 개선: 분기별 데이터 확보

3. ⚠️ **데이터 품질 이슈**
   - 63.7% 기업이 현금=0 기록
   - 개선: 외부 데이터 결합, 품질 검증

---

### 비즈니스 가치 💰
- 연간 10,000건 심사 기준
- 순이익 증가: [improvement] 억원 (추정)
- 부도 탐지율: [recall]%
- Traffic Light로 심사 효율 향상

---

### 다음 단계 🚀
1. 🔜 2022-2023 데이터로 시간적 검증
2. 🔜 외부 데이터 통합 (뉴스, 소송, 경영진)
3. 🔜 시계열 모델 도입 (LSTM/Transformer)
4. 🔜 실시간 API 개발 및 배포
5. 🔜 모니터링 시스템 구축
```

---

## 3. 출력 파일
- **파일명**: `notebooks/발표_Part4_결과_및_비즈니스_가치_v2.ipynb`
- **메타데이터 활용**: `data/processed/Part3_v2_메타데이터.pkl`
- **모델 파일**: Part 3 v2 개선판에서 저장된 Pipeline

---

## 4. 품질 기준

### 필수 체크리스트
- [ ] Part 3 v2 개선판 메타데이터 정확히 로딩
- [ ] Pipeline 구조 고려한 SHAP 분석
- [ ] Test 세트 성능에 Bootstrap CI 포함
- [ ] Confusion Matrix 재무 관점 해석
- [ ] False Negative 원인 분석 (Failure Case)
- [ ] Traffic Light 시스템 실제 부도율 검증
- [ ] 비즈니스 가치 시뮬레이션 (현실적 가정)
- [ ] 한계 섹션 대폭 강화 (원인 + 개선안)
- [ ] 3-Way Split, Data Leakage 방지 강조
- [ ] CV 기반 Threshold 언급
- [ ] 앙상블 다양성 체크 결과 반영

### 평가 기준 대비
- **설명 가능성 (25점)**: 모든 결과에 "왜" 추가, 실패 원인 분석
- **스토리텔링 (20점)**: Part 3 요약 → 성능 → 한계 → 미래 방향
- **시각화+해석 (20점)**: SHAP, Confusion Matrix, Bootstrap CI 시각화
- **도메인 전문성 (15점)**: 재무 관점 해석, 비즈니스 가치 정량화
- **통계적 엄격성 (10점)**: Bootstrap CI, p-value, 효과 크기
- **재현 가능성 (5점)**: Pipeline 로딩, 메타데이터 활용
- **한계 인정 (5점)**: 솔직한 한계 + 원인 분석 + 개선 방향

**목표: 87점 이상 (A등급)**

---

## 5. 주의사항

### Pipeline 구조 관련
- `best_pipeline.named_steps['classifier']`로 모델 접근
- SHAP 분석 시 전처리 먼저 수행 (Pipeline의 transform 활용)
- `predict_proba(X_test)`는 Pipeline 전체에 호출 (자동으로 전처리 수행)

### Data Leakage 방지 강조
- Test 세트는 최종 평가에만 사용했음을 명시
- 모든 결정이 Validation 기반임을 강조
- 3-Way Split 의의 설명

### Threshold 관련
- Validation + CV 평균 사용한 근거 설명
- Single split 편향 방지 효과 언급

### Traffic Light 관련
- Red: Recall 80% (보수적)
- Yellow: Recall 95% (매우 보수적)
- Conservative fallback (Red × 0.5) 설명

---

지금 바로 시작하세요. Part 3 v2 개선판의 성과를 정확히 반영하고, 솔직한 한계 인정과 구체적인 개선 방향을 제시하는 노트북을 만들어주세요.
