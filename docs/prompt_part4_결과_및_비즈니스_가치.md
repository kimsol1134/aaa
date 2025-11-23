# 📙 Part 4: 결과 및 비즈니스 가치 - 노트북 생성 프롬프트

---

## 🎭 역할 정의 (Role Assignment)

당신은 **시니어 데이터 사이언티스트이자 비즈니스 애널리스트**입니다.

**전문 영역:**
- 설명 가능한 AI (XAI) 및 SHAP 분석 전문가
- 기업 부도 예측 모델링 경험 10년+
- 재무 도메인 지식 보유 (유동성, 지급불능, 재무조작)
- 비즈니스 가치 측정 및 ROI 계산 전문가
- 모델 한계 분석 및 개선 방안 제시 능력

**커뮤니케이션 스타일:**
- 경영진에게 기술적 내용을 쉽게 설명
- 데이터 기반 의사결정 지원
- 객관적이고 균형 잡힌 평가 (성과와 한계 모두 투명하게)

---

## 🎯 작업 목표 (Primary Objective)

기존 3개의 발표용 노트북에서 **코드를 재사용**하여 "Part 4: 결과 및 비즈니스 가치" 노트북을 생성하세요.

**핵심 미션:**
1. **설명 가능성 (Explainability)**: SHAP 분석을 통해 모든 예측의 근거를 제시
2. **비즈니스 가치 (Business Value)**: 재무적 효과를 정량화 (손실 감소액, ROI)
3. **객관적 평가 (Honest Assessment)**: 성능뿐만 아니라 한계와 개선 방향도 명확히 제시
4. **의사결정 지원 (Decision Support)**: 실무에서 바로 활용 가능한 인사이트 제공

---

## 📚 필수 참조 자료

### 1. 기존 노트북 (코드 재사용 필수)

다음 **3개의 executed 노트북**에서 필요한 코드와 결과를 추출하세요:

```
notebooks/발표_Part1_문제정의_및_핵심발견_executed.ipynb
notebooks/발표_Part2_도메인_특성_공학_완전판_executed.ipynb
notebooks/발표_Part3_모델링_및_최적화_완전판_executed.ipynb
```

**중요:**
- Part2는 summary.md를 참조할 수 있음: `docs/notebook_summaries/발표_Part2_도메인_특성_공학_완전판_summary.md`
- Part1과 Part3는 executed 노트북에서 직접 코드를 추출
- 모든 코드는 **재사용 우선**, 새로 작성은 최소화

### 2. 프로젝트 규칙 (CLAUDE.md)

다음 규칙을 **반드시 준수**하세요:

#### ✅ 필수 준수 사항

1. **절대 하드 코딩 금지**
   - 파일 경로는 변수로: `model_path = '../data/processed/best_model_XGBoost.pkl'`
   - 상수는 별도 정의: `THRESHOLD = 0.15`

2. **한글 폰트 깨짐 방지**
   ```python
   import platform
   if platform.system() == 'Darwin':
       plt.rc('font', family='AppleGothic')
   elif platform.system() == 'Windows':
       plt.rc('font', family='Malgun Gothic')
   else:
       plt.rc('font', family='NanumGothic')
   plt.rc('axes', unicode_minus=False)
   ```

3. **데이터 특성 이해**
   - 시계열 데이터가 **아님** (2021년 8월 스냅샷)
   - 독립적인 횡단면 데이터로 처리
   - 시간 순서 의존적 로직 사용 금지

4. **평가 메트릭 우선순위**
   - 1순위: **PR-AUC** (불균형 데이터 핵심 지표)
   - 2순위: **F2-Score** (재현율 중시)
   - 3순위: **Type II Error** (부도 미탐지 최소화)
   - 참고: ROC-AUC (불균형 데이터에서 과대평가 위험)

5. **파일 경로 규칙**
   - 데이터 로딩: `pd.read_csv('../data/기업신용평가정보_210801.csv', encoding='utf-8')`
   - 모델 로딩: `joblib.load('../data/processed/best_model_XGBoost.pkl')`
   - 인코딩: 읽기 `utf-8`, 쓰기 `utf-8-sig`

---

## 📋 노트북 구조 (Structure Blueprint)

아래 구조를 **정확히** 따르세요. 각 섹션의 상세 요구사항도 함께 제시합니다.

---

### 🔹 Section 0: 환경 설정 및 데이터 로딩

**목적:** 필요한 라이브러리 임포트, 한글 폰트 설정, 모델/데이터 로딩

**코드 요구사항:**
```python
# 1. 필수 라이브러리
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 머신러닝
import joblib
from sklearn.metrics import (
    average_precision_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.utils import resample

# SHAP
import shap

# 한글 폰트 설정 (CLAUDE.md 규칙)
import platform
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)

# 2. 상수 정의 (하드코딩 금지)
MODEL_PATH = '../data/processed/best_model_XGBoost.pkl'  # Part3에서 확인
SCALER_PATH = '../data/processed/scaler.pkl'
FEATURES_PATH = '../data/features/domain_based_features_완전판.csv'
TARGET_COL = '모형개발용Performance(향후1년내부도여부)'
THRESHOLD = 0.15  # Part3에서 최적 threshold 확인

# 3. 데이터 및 모델 로딩
print("📂 데이터 및 모델 로딩 중...")

# 특성 데이터 로딩
df = pd.read_csv(FEATURES_PATH, encoding='utf-8')
print(f"✅ 데이터 로딩 완료: {df.shape}")

# 모델 로딩
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print(f"✅ 모델 로딩 완료: {type(model).__name__}")

# 4. Train/Test 분할 (Part3와 동일하게)
from sklearn.model_selection import train_test_split

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 예측 (테스트 세트)
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= THRESHOLD).astype(int)

print(f"\n✅ 테스트 세트 예측 완료")
print(f"   - 테스트 샘플: {len(y_test):,}")
print(f"   - 부도 기업: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")
print(f"   - 예측 부도: {y_pred.sum():,} ({y_pred.mean()*100:.2f}%)")
```

**출력 예시:**
```
📂 데이터 및 모델 로딩 중...
✅ 데이터 로딩 완료: (50105, 36)
✅ 모델 로딩 완료: XGBClassifier

✅ 테스트 세트 예측 완료
   - 테스트 샘플: 10,021
   - 부도 기업: 154 (1.54%)
   - 예측 부도: 345 (3.44%)
```

---

### 🔹 Section 1: Part 3 요약 (Opening)

**목적:** Part3에서 선택한 최종 모델을 간결하게 요약

**마크다운 요구사항:**
```markdown
## 📌 이전 Part 요약: 최종 모델 선택

Part 3에서 우리는 여러 모델과 샘플링 기법을 비교했습니다.

### ✅ 최종 선택 모델

| 항목 | 값 |
|------|-----|
| **모델** | XGBoost (또는 LightGBM, CatBoost) |
| **샘플링** | SMOTE + Tomek Links |
| **Threshold** | 0.15 |
| **선택 이유** | PR-AUC 최고, F2-Score 우수, Type II Error 최소 |

### 📊 Part 3 최종 성능 (Validation Set)

- **PR-AUC**: 0.145 (95% CI: 0.128-0.162)
- **F2-Score**: 0.21
- **Recall**: 0.46 (부도 기업의 46% 탐지)
- **Type II Error**: 53.95% (부도를 정상으로 오분류)

**→ 이제 테스트 세트에서 최종 성능을 평가하고, SHAP으로 모델을 해석합니다.**
```

**코드 요구사항:**
- Part3 노트북에서 최종 모델명, 샘플링 기법, 최적 threshold 추출
- Validation 성능 지표 가져오기 (PR-AUC, F2-Score, Recall, Type II Error)

---

### 🔹 Section 2: 테스트 세트 성능 평가 ⭐

**목적:** 테스트 세트에서 최종 성능을 측정하고 **Bootstrap CI 추가**

#### 2.1 성능 메트릭 계산 (Bootstrap CI 포함)

**코드 요구사항:**
```python
# 1. 기본 메트릭
pr_auc = average_precision_score(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
f2_score = fbeta_score(y_test, y_pred, beta=2)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# 2. Type II Error
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
type_ii_error = fn / (fn + tp)

# 3. Bootstrap으로 PR-AUC 신뢰구간 계산 ⭐
from sklearn.utils import resample

n_iterations = 1000
pr_aucs = []

np.random.seed(42)
for i in range(n_iterations):
    # 부트스트랩 샘플링
    indices = resample(range(len(y_test)), random_state=i)
    y_test_boot = y_test.iloc[indices]
    y_pred_boot = y_pred_proba[indices]

    # PR-AUC 계산
    pr_auc_boot = average_precision_score(y_test_boot, y_pred_boot)
    pr_aucs.append(pr_auc_boot)

pr_auc_mean = np.mean(pr_aucs)
pr_auc_ci_lower = np.percentile(pr_aucs, 2.5)
pr_auc_ci_upper = np.percentile(pr_aucs, 97.5)

# 4. 결과 출력
print("=" * 60)
print("📊 테스트 세트 최종 성능")
print("=" * 60)
print(f"PR-AUC:        {pr_auc_mean:.4f} (95% CI: [{pr_auc_ci_lower:.4f}, {pr_auc_ci_upper:.4f}])")
print(f"ROC-AUC:       {roc_auc:.4f}")
print(f"F2-Score:      {f2_score:.4f}")
print(f"Recall:        {recall:.4f} (부도 기업의 {recall*100:.1f}% 탐지)")
print(f"Precision:     {precision:.4f} (예측 부도 중 {precision*100:.1f}%가 실제 부도)")
print(f"Type II Error: {type_ii_error:.4f} (부도의 {type_ii_error*100:.1f}%를 정상으로 오분류)")
print("=" * 60)
```

**마크다운 해석:**
```markdown
### 💡 성능 해석

#### PR-AUC: 0.145 (95% CI: 0.128-0.162)
- **의미**: 불균형 데이터에서의 예측 정확도
- **해석**: Naive Baseline (1.54%) 대비 **9.4배 향상**
- **신뢰구간**: Bootstrap 1000회로 계산 → 안정적인 추정

#### F2-Score: 0.21
- **의미**: Recall을 2배 중시하는 조화평균
- **해석**: 부도 미탐지(FN)를 최소화하는 방향으로 최적화

#### Recall: 0.46
- **의미**: 실제 부도 기업 중 모델이 탐지한 비율
- **해석**: 부도 기업 10개 중 4.6개를 사전 차단 가능
- **한계**: 53.95%는 여전히 미탐지 → 개선 필요

#### Type II Error: 53.95%
- **의미**: 부도 기업을 정상으로 잘못 예측한 비율
- **리스크**: 이 기업들에게 대출 시 손실 발생
- **원인**: (Section 6 "한계" 섹션에서 상세 분석)
```

---

### 🔹 Section 3: Confusion Matrix 및 재무 해석 ⭐

**목적:** 혼동 행렬을 시각화하고, **각 셀을 재무 관점에서 해석**

#### 3.1 Confusion Matrix 시각화

**코드 요구사항:**
```python
# Confusion Matrix 계산
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# 시각화
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['정상', '부도'],
            yticklabels=['정상', '부도'],
            cbar_kws={'label': '기업 수'})
plt.xlabel('예측', fontsize=12, weight='bold')
plt.ylabel('실제', fontsize=12, weight='bold')
plt.title('혼동 행렬 (Confusion Matrix)', fontsize=14, weight='bold')

# 각 셀에 비율 추가
for i in range(2):
    for j in range(2):
        count = cm[i, j]
        total = cm.sum()
        pct = count / total * 100
        ax.text(j+0.5, i+0.7, f'({pct:.1f}%)',
                ha='center', va='center', fontsize=10, color='gray')

plt.tight_layout()
plt.show()

# 숫자 출력
print(f"True Negative (TN):  {tn:,}  (정상을 정상으로 예측)")
print(f"False Positive (FP): {fp:,}  (정상을 부도로 예측)")
print(f"False Negative (FN): {fn:,}  (부도를 정상으로 예측) ⚠️")
print(f"True Positive (TP):  {tp:,}  (부도를 부도로 예측)")
```

#### 3.2 재무 관점 해석 ⭐

**마크다운 요구사항:**
```markdown
### 💡 Confusion Matrix 재무 해석

#### ✅ True Positive (TP = 71개): 부도 기업을 부도로 예측
- **비즈니스 가치**: 부도 위험 사전 차단
- **예상 손실 회피**: 71개 × 평균 대출액 500만원 = **3.55억원 손실 방지**
- **실무 조치**: 대출 거절 또는 고금리 적용, 담보 요구

#### ⚠️ False Negative (FN = 83개): 부도 기업을 정상으로 예측
- **비즈니스 리스크**: 부도 미탐지 → 가장 큰 문제
- **예상 손실**: 83개 × 평균 대출액 500만원 = **4.15억원 잠재 손실**
- **원인 분석**:
  1. 재무제표에 나타나지 않는 리스크 (소송, 경영진 비리, 거래처 부도)
  2. 급격한 외부 환경 변화 (COVID-19, 원자재 가격 급등)
  3. 모델이 학습하지 못한 패턴 (극소수 케이스)
- **개선 필요**: FN 감소가 최우선 과제

#### ❌ False Positive (FP = 274개): 정상 기업을 부도로 예측
- **비즈니스 리스크**: 기회 비용 (정상 기업에게 대출 거절)
- **예상 기회 손실**: 274개 × 평균 이자수익 50만원 = **1.37억원**
- **완화 전략**:
  - Traffic Light 시스템 활용 (경고 구간 → 추가 심사)
  - 사람의 최종 판단 (모델은 1차 스크리닝)

#### ✅ True Negative (TN = 9,593개): 정상 기업을 정상으로 예측
- **비즈니스 가치**: 안전한 대출 집행
- **예상 수익**: 9,593개 × 평균 이자수익 50만원 = **47.97억원**
- **모델 기여**: 부도 리스크가 낮은 기업을 자동 승인 → 심사 인력 절감

---

### 📈 순 비즈니스 효과

| 항목 | 금액 |
|------|------|
| **손실 회피 (TP)** | +3.55억원 |
| **손실 발생 (FN)** | -4.15억원 |
| **기회 손실 (FP)** | -1.37억원 |
| **이자 수익 (TN)** | +47.97억원 |
| **순 효과** | **+46.00억원** |

**→ 모델이 없었다면?**
- FN + TP = 154개 모두 대출 → 154 × 500만원 = **7.7억원 손실**
- 모델 사용 시 → 4.15억원 손실
- **손실 감소**: 7.7억 - 4.15억 = **3.55억원 (46% 감소)** ✅
```

---

### 🔹 Section 4: SHAP 분석 ⭐⭐⭐ (가장 중요)

**목적:** SHAP으로 모델 예측의 근거를 제시하고, Top 특성을 재무적으로 해석

#### 4.1 SHAP Explainer 초기화

**코드 요구사항:**
```python
# SHAP Explainer 생성 (TreeExplainer - XGBoost/LightGBM/CatBoost 전용)
print("🔍 SHAP Explainer 초기화 중...")
explainer = shap.TreeExplainer(model)

# SHAP values 계산 (테스트 세트 샘플링 - 속도 향상)
sample_size = min(1000, len(X_test))
X_test_sample = X_test.sample(n=sample_size, random_state=42)

print(f"   - 샘플 크기: {sample_size}")
print(f"   - 계산 중... (시간이 걸릴 수 있습니다)")

shap_values = explainer.shap_values(X_test_sample)

print("✅ SHAP values 계산 완료")
```

#### 4.2 SHAP Summary Plot

**코드 요구사항:**
```python
# Summary Plot (Feature Importance + Distribution)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, plot_type="dot", show=False)
plt.title('SHAP Summary Plot: 특성 중요도 및 영향 방향', fontsize=14, weight='bold')
plt.tight_layout()
plt.show()
```

**마크다운 해석:**
```markdown
### 💡 Summary Plot 해석 가이드

#### 색상 의미:
- 🔴 **빨간색**: 특성 값이 높음 (예: 연체 건수 많음)
- 🔵 **파란색**: 특성 값이 낮음 (예: 현금 적음)

#### X축 (SHAP value):
- **양수 (+)**: 부도 확률 증가
- **음수 (-)**: 부도 확률 감소

#### 예시:
- **연체 건수**가 높으면 (빨간색) → SHAP value 양수 → 부도 위험 증가
- **즉각지급능력**이 높으면 (빨간색) → SHAP value 음수 → 부도 위험 감소
```

#### 4.3 SHAP Feature Importance (Bar Plot)

**코드 요구사항:**
```python
# Feature Importance (절댓값 평균)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
plt.title('SHAP Feature Importance: Top 20', fontsize=14, weight='bold')
plt.xlabel('Mean |SHAP value|', fontsize=12)
plt.tight_layout()
plt.show()

# Top 10 특성 추출
feature_importance = pd.DataFrame({
    'Feature': X_test_sample.columns,
    'SHAP_Importance': np.abs(shap_values).mean(axis=0)
}).sort_values('SHAP_Importance', ascending=False)

print("\n📊 Top 10 중요 특성:")
print(feature_importance.head(10).to_string(index=False))
```

#### 4.4 Top 10 특성 재무 해석 ⭐⭐

**마크다운 요구사항:**

Top 10 특성 각각에 대해 다음 형식으로 해석:

```markdown
### 🔍 Top 10 중요 특성 재무 해석

#### 1. **신용등급점수** (SHAP: 0.082)
- **재무 의미**: 신용평가사의 종합 평가 (1=AAA, 10=D)
- **부도 예측 메커니즘**: 등급이 낮을수록 (숫자 높을수록) 부도 확률 증가
- **위험 기준**: 등급 ≥ 5 (BB 이하) → 고위험
- **비즈니스 활용**: 1차 스크리닝 지표 (등급 5 이상 → 정밀 심사)
- ⚠️ **Data Leakage 가능성**: 신용등급 자체가 부도 예측 모델로 만들어짐 → 검토 필요

#### 2. **즉각지급능력** (SHAP: 0.065)
- **재무 의미**: (현금 + 현금성자산) / 유동부채
- **부도 예측 메커니즘**: 비율이 낮을수록 단기 부채 상환 불가 → 유동성 위기
- **위험 기준**:
  - < 0.1 → 매우 위험 (현금이 유동부채의 10%도 안 됨)
  - 0.1~0.3 → 위험
  - > 0.5 → 안전
- **비즈니스 활용**: "3개월 내 살아남을 수 있는가?" 판단
- **실제 사례**: 흑자도산 기업의 90%가 즉각지급능력 < 0.1

#### 3. **이해관계자_불신지수** (SHAP: 0.058)
- **재무 의미**: 연체 + 세금체납 + 법적리스크 + 신용등급 종합 점수
- **부도 예측 메커니즘**: 과거 행동이 미래를 예측 (연체 이력 → 재연 가능성 높음)
- **위험 기준**:
  - 0점 → 신뢰 (연체/체납 없음)
  - 5~10점 → 주의
  - > 10점 → 고위험
- **비즈니스 활용**: 재무제표보다 더 정직한 신호
- **핵심 인사이트**: "과거에 약속을 지키지 않은 기업은 미래에도 그럴 가능성이 높다"

#### 4. **현금소진일수** (SHAP: 0.052)
- **재무 의미**: 현금 / (영업비용 / 365)
- **부도 예측 메커니즘**: 현재 현금으로 며칠이나 버틸 수 있는가?
- **위험 기준**:
  - < 30일 → 매우 위험 (한 달도 못 버팀)
  - 30~90일 → 위험
  - > 180일 (6개월) → 안전
- **비즈니스 활용**: 긴급 자금 지원 필요 여부 판단
- **실무 중요성**: 부도 3개월 전에 급격히 감소 → 조기 경보

#### 5. **이자보상배율** (SHAP: 0.048)
- **재무 의미**: (영업이익 + 감가상각비) / 이자비용
- **부도 예측 메커니즘**: 영업으로 이자를 갚을 수 있는가?
- **위험 기준**:
  - < 1.0 → 영업이익 < 이자비용 → 버틸 수 없음
  - 1.0~1.5 → 위험
  - > 2.0 → 안전
- **비즈니스 활용**: 추가 차입 가능 여부 판단
- **경제적 의미**: 이자보상배율 < 1이면 영업할수록 손해

#### 6. **재무레버리지** (SHAP: 0.041)
- **재무 의미**: 자산총계 / 자본총계
- **부도 예측 메커니즘**: 레버리지가 높을수록 부채 의존도 증가 → 재무 위험
- **위험 기준**:
  - < 2 → 안전 (자산의 50% 이상이 자기자본)
  - 2~5 → 보통
  - > 5 → 고위험 (자본의 5배 자산 운영 → 부채 과다)
- **비즈니스 활용**: 추가 대출 시 담보 여력 판단

#### 7. **M_Score_한국형** (SHAP: 0.038)
- **재무 의미**: Beneish M-Score의 한국 시장 버전 (재무조작 탐지)
- **부도 예측 메커니즘**: 재무제표 조작 의심 → 실제 재무 상태 더 나쁠 가능성
- **위험 기준**:
  - M-Score > 0 → 조작 가능성 높음
  - M-Score ≤ 0 → 정상
- **비즈니스 활용**: 외부감사 미비 기업 대상 필수 검토
- **핵심 인사이트**: "부도 직전 기업은 실적을 부풀려 대출을 받으려 함"

#### 8. **발생액비율** (SHAP: 0.035)
- **재무 의미**: (당기순이익 - 영업현금흐름) / 자산총계
- **부도 예측 메커니즘**: 높을수록 "장부상 이익은 많은데 현금은 없음" → 위험
- **위험 기준**:
  - > 0.1 → 위험 (이익의 10% 이상이 현금 없는 이익)
  - 0.05~0.1 → 주의
  - < 0.05 → 안전
- **비즈니스 활용**: 현금흐름 품질 검증
- **실무 팁**: 발생액비율 높으면 → 재무제표 정밀 분석 필요

#### 9. **운전자본_대_자산** (SHAP: 0.032)
- **재무 의미**: (유동자산 - 유동부채) / 자산총계
- **부도 예측 메커니즘**: 음수면 → 단기 부채가 유동자산보다 많음 → 위험
- **위험 기준**:
  - < 0 → 매우 위험 (운전자본 부족)
  - 0~0.1 → 위험
  - > 0.2 → 안전
- **비즈니스 활용**: 단기 유동성 건전성 평가

#### 10. **연체심각도** (SHAP: 0.029)
- **재무 의미**: 총연체건수 × 부채비율 / 100
- **부도 예측 메커니즘**: 연체가 많고 + 부채가 많을수록 → 위험 극대화
- **위험 기준**:
  - 0 → 안전 (연체 없음)
  - > 1 → 위험
  - > 5 → 매우 위험
- **비즈니스 활용**: 과거 신용 행동 + 현재 재무 부담 종합 평가
- **핵심**: 연체 + 고부채 = 최악의 조합
```

**중요:**
- Part3 노트북에서 실제 SHAP Importance 값 확인
- 위 10개는 예시일 수 있으므로, 실제 Top 10을 추출하여 해석
- 각 특성의 재무 의미는 Part2 요약 문서 참조

---

### 🔹 Section 5: Traffic Light 시스템 (기존 코드 재사용)

**목적:** 예측 확률을 3단계 구간으로 나누어 실무 의사결정 지원

**코드 요구사항:**
```python
# Part3 노트북에서 Traffic Light 시스템 코드 그대로 가져오기

# Traffic Light 구간 정의
def traffic_light(prob):
    if prob < 0.05:
        return '안전 (Green)'
    elif prob < 0.15:
        return '경고 (Yellow)'
    else:
        return '위험 (Red)'

y_test_df = pd.DataFrame({
    'actual': y_test.values,
    'pred_proba': y_pred_proba,
    'pred_class': y_pred,
    'traffic_light': [traffic_light(p) for p in y_pred_proba]
})

# 구간별 통계
traffic_stats = y_test_df.groupby('traffic_light').agg({
    'actual': ['count', 'sum', 'mean']
}).round(4)

print("🚦 Traffic Light 시스템 성능:")
print(traffic_stats)

# 시각화 (Plotly)
# ... (Part3 코드 재사용)
```

**마크다운 해석:**
```markdown
### 💡 실무 활용 방안

#### 🟢 Green (안전): 확률 < 5%
- **조치**: 자동 승인
- **부도율**: 0.5% 이하
- **효과**: 심사 인력 80% 절감

#### 🟡 Yellow (경고): 5% ≤ 확률 < 15%
- **조치**: 추가 심사 (사람 개입)
- **부도율**: 3~8%
- **효과**: 정밀 분석으로 FN 감소

#### 🔴 Red (위험): 확률 ≥ 15%
- **조치**: 대출 거절 또는 고금리/담보 요구
- **부도율**: 15% 이상
- **효과**: 손실 사전 차단
```

---

### 🔹 Section 6: 한계 및 개선 방향 ⭐⭐ (대폭 강화)

**목적:** 모델의 한계를 **객관적으로** 분석하고, 구체적인 개선 방향 제시

**마크다운 요구사항:**

```markdown
## ⚠️ 한계 (Limitations) 및 개선 방향

### 1. 데이터 품질 이슈 🔴

#### 문제 상황:
- **현금 = 0인 기업이 63.7%** (Part1에서 발견)
- 재고자산, 매출채권 등도 유사한 문제
- "실제로 현금이 없는가?" vs "기록하지 않은 것인가?" 구분 불가

#### 원인 추정:
1. 중소기업 회계 시스템 미비
2. 세무 신고용 간편 장부 (정확도 낮음)
3. 외부감사 미대상 기업 (검증 안 됨)

#### 현재 대응 (Part2):
- Binary feature 추가: '현금보유여부'
- Robust 통계량 사용: median (평균 대신)
- 결측치 대체: median imputation

#### 향후 개선 방안:
1. **원본 데이터 출처 확인 및 품질 검증**
   - 나이스신용평가, KIS-Value 등 데이터 제공사에 문의
   - 현금 = 0의 진짜 의미 파악

2. **외부 데이터 결합**
   - 금융감독원 전자공시시스템 (DART) 데이터
   - 국세청 사업자 신고 데이터
   - 은행 거래 내역 (동의 시)

3. **데이터 품질 스코어링 시스템 구축**
   - 각 기업에 "데이터 신뢰도 점수" 부여
   - 신뢰도 낮은 기업 → 예측 신뢰구간 확대
   - 의사결정 시 참고

#### 비즈니스 임팩트:
- 데이터 품질 개선 시 예상 성능 향상: PR-AUC 0.145 → **0.18~0.20** (20~30% 향상)

---

### 2. 시계열 정보 부족 🔴

#### 문제 상황:
- **2021년 8월 단일 시점 스냅샷** (CLAUDE.md 명시)
- 재무 악화 "속도"를 알 수 없음
- 예: 부채비율 200% (작년에도 200%? 작년엔 100%?)

#### Impact:
1. **급격히 악화되는 기업 탐지 어려움**
   - 3개월 만에 부채비율 100% → 300% 급증 → 고위험
   - 하지만 모델은 "300%"만 보고 판단 → 맥락 누락

2. **False Negative (53.95%)에 기여하는 주요 요인**
   - 갑작스러운 환경 변화 (COVID-19, 원자재 가격 급등)
   - 주요 거래처 부도로 연쇄 부도
   - 모델은 이런 "변화"를 포착 못함

#### 현재 상황:
- 횡단면 데이터 (Cross-sectional)
- 시간 의존적 로직 없음 (CLAUDE.md 규칙 준수)

#### 향후 개선 방안:
1. **분기별/연도별 패널 데이터 확보**
   - 최소 3년치 시계열 (12분기)
   - 각 기업의 추세 파악 가능

2. **변화율 특성 추가**
   - 매출액 증가율 (YoY)
   - 부채비율 변화량 (QoQ)
   - 현금흐름 변동성 (표준편차)

3. **시계열 모델 검토**
   - LSTM/GRU: 시계열 패턴 학습
   - Transformer (Temporal Fusion Transformer): 시계열 + 정적 특성 결합
   - 단, 데이터 확보가 선행 조건

#### 비즈니스 임팩트:
- 시계열 데이터 추가 시 예상 Recall 향상: 0.46 → **0.60~0.65** (FN 30% 감소)

---

### 3. 모델 성능 한계 🔴

#### 현재 성능:
- **PR-AUC: 0.145** (95% CI: 0.128-0.162)
- **F2-Score: 0.21**
- **False Negative: 53.95%** ← **가장 큰 문제**

#### 원인 분석:

##### 3-1. 데이터에 없는 정보 (Non-financial Signals)

**재무제표로 포착 불가능한 부도 원인:**
1. **법적 리스크**
   - 대규모 소송 진행 중 (특허, 노동, 환경)
   - 경영진 비리/횡령
   - 인허가 취소 위험

2. **관계사 리스크**
   - 주요 거래처 부도 (연쇄 부도)
   - 대주주 건강 이상 (소기업)
   - 파트너사와 분쟁

3. **시장 환경 급변**
   - 업계 전체 불황 (예: COVID-19)
   - 경쟁사 신제품 출시
   - 규제 변화 (예: 환경 규제 강화)

4. **경영진 의사결정**
   - 무리한 M&A
   - 신사업 실패
   - 과도한 배당

**현재 대응**: 없음 (재무 데이터만 사용)

**향후 개선**:
- 뉴스 감성 분석 (부정적 기사 빈도)
- 소송 이력 데이터 (대법원 공개 데이터)
- 경영진 이력 데이터 (교체 빈도, 전문성)
- SNS/뉴스 크롤링 → NLP로 리스크 신호 탐지

##### 3-2. 극도 불균형 (1:66 비율)

**문제:**
- 부도 기업 1개 vs 정상 기업 66개
- SMOTE로 일부 완화했으나 한계 존재
- 소수 클래스 학습 어려움

**향후 개선**:
1. **Advanced Sampling**
   - ADASYN (Adaptive Synthetic Sampling)
   - Borderline-SMOTE (경계선만 오버샘플링)

2. **Cost-sensitive Learning 강화**
   - Focal Loss 적용 (hard example에 집중)
   - FN 비용을 더 높게 설정 (현재 5배 → 10배)

3. **Anomaly Detection 접근**
   - 부도 = 이상 탐지 문제로 재정의
   - Isolation Forest, One-Class SVM

##### 3-3. 모델 다양성 부족

**문제:**
- 모든 base model이 Tree 기반 (LightGBM, XGBoost, CatBoost)
- 예측 상관관계 > 0.95 (거의 같은 예측)
- Stacking Ensemble 효과 제한적

**향후 개선**:
1. **다른 학습 메커니즘 추가**
   - Neural Network (MLP, TabNet)
   - Tabular Transformer (FT-Transformer)
   - Support Vector Machine (비선형 커널)

2. **Stacking Meta-learner 변경**
   - 현재: Logistic Regression
   - 개선: XGBoost 또는 Neural Network

3. **Feature 서브셋 다양화**
   - 각 모델에 다른 특성 조합 제공
   - 앙상블 다양성 증대

#### 비즈니스 임팩트:
- 위 3가지 개선 시 예상 성능: PR-AUC 0.145 → **0.22~0.25** (50% 향상)

---

### 4. 해석 가능성 vs 성능 트레이드오프 ⚖️

#### 현재 선택: 해석 가능성 우선

**이유:**
1. **규제 요구사항**
   - 금융위원회 "AI 활용 가이드라인" (2021)
   - 대출 거절 시 사유 설명 의무
   - SHAP으로 근거 제시 가능

2. **실무 신뢰 확보**
   - "왜 이 기업을 거절했는가?" 설명 필요
   - Black Box 모델은 심사역이 신뢰 안 함
   - Tree 기반 모델 + SHAP = 완벽한 조합

#### 트레이드오프:

| 항목 | Tree 기반 (현재) | Deep Learning |
|------|------------------|---------------|
| **성능** | PR-AUC 0.145 | 0.18~0.20 (예상) |
| **해석력** | ✅ SHAP 완벽 지원 | ⚠️ 어려움 |
| **학습 속도** | ✅ 빠름 (분 단위) | ❌ 느림 (시간 단위) |
| **특성 중요도** | ✅ 명확 | ❌ 불명확 |
| **실무 신뢰** | ✅ 높음 | ❌ 낮음 |

#### 결정 근거:
- **성능 차이 < 0.05 (5%p)** → 해석력 손실을 정당화하기 어려움
- **실무 도입 가능성** → Tree 기반이 압도적 우위
- **유지보수** → Tree 기반이 단순하고 안정적

#### 향후 방향:
1. **Hybrid 접근**
   - Tree 기반 (1차 스크리닝) + Deep Learning (2차 정밀 심사)
   - 각각의 장점 활용

2. **Explainable Deep Learning**
   - TabNet: Attention 메커니즘으로 해석 가능
   - Layer-wise Relevance Propagation (LRP)

3. **Rule Extraction**
   - Deep Learning으로 학습 → Decision Tree로 근사
   - 성능 + 해석력 동시 확보

---

## ✅ 그럼에도 가치 있는 이유

### 1. 도메인 논리 명확 🎯

**강점:**
- 모든 특성이 재무 이론 기반 (Part2)
- 각 특성의 의미를 명확히 설명 가능
- "왜?"에 대한 답이 항상 존재

**실무 가치:**
- 대출 심사역이 바로 이해
- 규제 감사 시 근거 제시 용이
- 고객 불만 시 설명 가능

### 2. 재현 가능 및 확장 가능 ♻️

**강점:**
- 전체 파이프라인 자동화 (노트북 1~5)
- 다른 연도 데이터에 즉시 적용 가능
- 다른 국가 시장 확장 가능 (한국 특화 특성만 제외)

**실무 가치:**
- 매년/분기별 모델 업데이트 자동화
- 해외 지사 전개 가능

### 3. 해석 가능한 AI 🔍

**강점:**
- SHAP으로 모든 예측 근거 제시
- Top 10 특성 재무 해석 명확
- Traffic Light 시스템 → 의사결정 지원

**실무 가치:**
- 규제 요구사항 충족
- 사용자 신뢰 확보
- 감사 대응 용이

### 4. 실용적 성능 향상 📈

**Naive Baseline 대비:**
- PR-AUC: 0.0154 (1.54% 부도율) → 0.145 (**9.4배 향상**)
- 손실 감소: 7.7억 → 4.15억 (**46% 감소**)

**ROI:**
- 모델 개발 비용: 약 5,000만원 (인건비 + 인프라)
- 연간 손실 감소: 3.55억원
- **ROI: 710%** (1년 만에 7배 회수)

### 5. 확장 가능한 프레임워크 🚀

**현재 → 미래:**
- 외부 데이터 추가 용이 (뉴스, 소송, SNS)
- 모델 업그레이드 가능 (Tree → Hybrid → Deep Learning)
- 지속적 개선 프로세스 구축
```

**중요:**
- 한계를 숨기지 말고 **투명하게** 제시
- 각 한계에 대해 **구체적인 개선 방안** 제시
- 그럼에도 **현재 가치**를 명확히 설명

---

### 🔹 Section 7: 최종 요약

**마크다운 요구사항:**

```markdown
## 🎯 프로젝트 최종 성과

### ✅ 달성한 것

#### 1. 설명 가능한 부도 예측 모델 개발
- **SHAP 분석**: 모든 예측의 근거 제시
- **Top 10 특성**: 재무 도메인 지식 기반 해석
- **규제 요구사항 충족**: 대출 거절 사유 설명 가능

#### 2. 실용적 성능 달성
- **PR-AUC 0.145**: Naive Baseline 대비 9.4배 향상
- **Recall 0.46**: 부도 기업의 46% 사전 탐지
- **손실 46% 감소**: 7.7억 → 4.15억 (연간 3.55억 절감)

#### 3. 비즈니스 가치 정량화
- **ROI 710%**: 1년 만에 개발 비용 7배 회수
- **심사 효율 80% 향상**: Traffic Light Green 자동 승인
- **의사결정 지원**: Yellow 구간 정밀 심사 가이드

#### 4. 도메인 지식 기반 Feature Engineering
- **52개 특성 생성**: 7개 카테고리 (유동성, 지급불능, 재무조작 등)
- **Beneish M-Score 완전 구현**: 한국 시장 특화
- **Feature Validation**: 통계적 검증 완료 (Mann-Whitney U, Cliff's Delta, AUC)

#### 5. 재현 가능한 파이프라인 구축
- **전체 프로세스 자동화**: 노트북 Part 1~4
- **다른 연도 데이터 즉시 적용 가능**
- **해외 시장 확장 가능** (한국 특화 특성 제외)

---

### ⚠️ 주요 한계

#### 1. False Negative 53.95%
- 부도 기업의 절반 이상 미탐지
- 원인: 재무제표에 없는 정보 (소송, 경영진 비리, 시장 급변)
- 개선: 외부 데이터 통합 (뉴스, 소송 이력)

#### 2. 데이터 품질 이슈
- 현금 = 0인 기업 63.7%
- 원인: 중소기업 회계 시스템 미비
- 개선: 원본 데이터 검증, 외부 데이터 결합

#### 3. 시계열 정보 부족
- 단일 시점 스냅샷 → 추세 파악 불가
- 원인: 분기별/연도별 데이터 미확보
- 개선: 패널 데이터 확보, 변화율 특성 추가

#### 4. 극도 불균형 (1:66)
- 소수 클래스 학습 어려움
- 현재: SMOTE + Tomek Links
- 개선: Focal Loss, ADASYN, Anomaly Detection

---

### 🚀 향후 발전 방향

#### 단기 (3개월)
1. **외부 데이터 통합 파일럿**
   - 뉴스 감성 분석 (네이버 뉴스 크롤링)
   - 소송 이력 (대법원 공개 데이터)
   - 예상 효과: Recall 0.46 → 0.55

2. **Traffic Light 시스템 실무 적용**
   - Green 자동 승인 (80% 케이스)
   - Yellow 추가 심사 프로세스 정립
   - Red 거절 사유 템플릿 작성

#### 중기 (6개월)
3. **시계열 데이터 확보 및 분석**
   - 최소 3년치 분기별 데이터 수집
   - 변화율 특성 추가 (매출 증가율, 부채비율 변화)
   - 시계열 모델 검토 (LSTM, Transformer)
   - 예상 효과: Recall 0.55 → 0.65

4. **앙상블 다양성 증대**
   - Neural Network 추가 (TabNet)
   - Stacking Meta-learner 개선
   - 예상 효과: PR-AUC 0.145 → 0.18

#### 장기 (1년)
5. **Hybrid 시스템 구축**
   - Tree 기반 (1차 스크리닝) + Deep Learning (2차 정밀)
   - 성능과 해석력 동시 확보
   - 목표: PR-AUC 0.22, Recall 0.70, FN < 30%

6. **실시간 모니터링 대시보드**
   - Streamlit 앱 고도화
   - 매일 새 데이터 자동 예측
   - 위험 기업 조기 경보 시스템

---

## 📊 비즈니스 임팩트 요약표

| 지표 | Baseline | 현재 모델 | 향후 목표 (1년) |
|------|----------|-----------|-----------------|
| **PR-AUC** | 0.015 | 0.145 (9.4배↑) | 0.22 (14.7배↑) |
| **Recall** | - | 0.46 | 0.70 |
| **연간 손실** | 7.7억 | 4.15억 (46%↓) | 2.3억 (70%↓) |
| **ROI** | - | 710% | 1,000%+ |
| **심사 효율** | - | 80% 향상 | 90% 향상 |

---

## 🏁 결론

### 핵심 메시지:

> **"설명 가능하고, 실용적이며, 지속 개선 가능한 부도 예측 시스템을 구축했습니다."**

### 3가지 차별점:

1. **도메인 지식 기반**: 통계적 특성이 아닌 재무 이론 기반 특성 공학
2. **완전한 투명성**: SHAP으로 모든 예측 근거 제시 → 규제 충족
3. **실증된 비즈니스 가치**: 연간 3.55억 손실 감소, ROI 710%

### 다음 단계:

1. **즉시 실행**: Traffic Light 시스템 실무 적용
2. **3개월 내**: 외부 데이터 통합 파일럿
3. **1년 내**: Hybrid 시스템으로 FN < 30% 달성

**→ 이 모델은 "완성"이 아니라 "시작"입니다. 지속적 개선으로 더 나은 모델을 만들어갑니다.**
```

---

## 📂 출력 파일 정보

**파일명:**
```
notebooks/발표_Part4_결과_및_비즈니스_가치.ipynb
```

**저장 위치:**
- `notebooks/` 디렉토리
- 실행 후 `발표_Part4_결과_및_비즈니스_가치_executed.ipynb`도 생성 가능

**파일 구조:**
- Markdown 셀 + Code 셀 교대
- 모든 코드는 실행 가능해야 함 (경로 확인 필수)

---

## ✅ 품질 검증 체크리스트

노트북 생성 후 다음을 확인하세요:

### 1. 코드 품질
- [ ] 모든 코드 셀이 오류 없이 실행됨
- [ ] 하드코딩 없음 (파일 경로, 임계값 등 모두 변수화)
- [ ] 한글 폰트 설정 완료 (CLAUDE.md 규칙)
- [ ] 인코딩 올바름 (utf-8, utf-8-sig)

### 2. SHAP 분석
- [ ] SHAP values 계산 완료
- [ ] Summary Plot, Bar Plot 시각화
- [ ] Top 10 특성 재무 해석 (각 특성마다 4~6줄)
- [ ] 모든 해석이 Part2 도메인 지식과 일치

### 3. 비즈니스 가치
- [ ] Confusion Matrix 재무 해석 (TP, FP, FN, TN)
- [ ] 손실 감소액 계산 (구체적 금액)
- [ ] ROI 계산
- [ ] Traffic Light 시스템 성능 제시

### 4. 한계 분석
- [ ] 4가지 한계 모두 다룸 (데이터 품질, 시계열, 성능, 트레이드오프)
- [ ] 각 한계에 구체적 개선 방안 제시
- [ ] "그럼에도 가치 있는 이유" 섹션 포함

### 5. 문서 품질
- [ ] 마크다운 형식 올바름 (헤딩, 리스트, 테이블)
- [ ] 시각화 제목/라벨 한글 정상 표시
- [ ] 전체 스토리 흐름 자연스러움 (Part 3 → 성능 → SHAP → 가치 → 한계)

---

## 🎓 참고 자료

### Part 3에서 가져올 정보
- 최종 모델명 (XGBoost/LightGBM/CatBoost)
- 최적 Threshold
- Validation 성능 지표
- Traffic Light 시스템 코드

### Part 2에서 가져올 정보
- 각 특성의 재무 의미 (Summary.md 참조)
- Feature Validation 결과 (AUC, IV)

### Part 1에서 가져올 정보
- 데이터 품질 이슈 (현금 = 0 문제)
- 업종별 부도율

---

## 💬 최종 확인 질문

노트북 생성 전에 다음을 확인하세요:

1. **Part3 executed 노트북을 읽었는가?** (최종 모델 확인)
2. **SHAP 코드가 Part3에 있는가?** (있으면 재사용, 없으면 새로 작성)
3. **Bootstrap CI 코드를 이해했는가?** (PR-AUC 신뢰구간)
4. **각 특성의 재무 의미를 Part2에서 확인했는가?** (SHAP 해석 시 필요)
5. **한계 섹션이 충분히 솔직한가?** (객관성 유지)

---

## 🚀 실행 지침

**지금 바로 시작하세요!**

### Step 1: 노트북 읽기
```python
# Part1, Part3 executed 노트북을 읽고 코드 추출
# Part2는 summary.md로 대체
```

### Step 2: 노트북 생성
```python
# notebooks/발표_Part4_결과_및_비즈니스_가치.ipynb 생성
# 위 구조대로 Markdown + Code 셀 구성
```

### Step 3: 실행 및 검증
```python
# 모든 셀 실행 (Restart & Run All)
# 오류 없이 완료되는지 확인
# 시각화 한글 폰트 정상 표시 확인
```

### Step 4: 저장
```python
# 실행된 노트북을 _executed.ipynb로 저장
# git commit 및 push (CLAUDE.md 규칙)
```

---

**행운을 빕니다! 완벽한 Part 4 노트북을 만들어주세요. 🎯**
