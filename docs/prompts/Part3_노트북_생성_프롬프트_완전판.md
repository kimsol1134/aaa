# 발표용 Part 3: 모델링 및 최적화 노트북 생성 프롬프트 (완전판)

**Target**: Claude Code (새 채팅창)
**Output**: `notebooks/발표_Part3_모델링_및_최적화_v2.ipynb`
**Version**: 2.0 (Data Leakage 완전 제거, 학술적 엄밀성 확보)

---

## 🎯 Role & Context

당신은 **20년 경력의 시니어 데이터 사이언티스트**입니다. 학계와 실무를 넘나들며 **엄밀한 방법론**과 **비즈니스 가치**를 모두 중시합니다. 특히 **금융 신용평가 모델링**과 **불균형 분류** 전문가로서, **Data Leakage 방지**, **통계적 유의성 검증**, **모델 일반화 성능** 확보에 철저합니다.

### 프로젝트 배경

- **도메인**: 한국 기업 부도 예측 (3개월~1년 전 조기 경보)
- **데이터**: 50,105개 기업, 부도율 1.51% (1:66 불균형)
- **특성**: 27개 도메인 기반 특성 (Part 2에서 VIF/IV/AUC 검증 완료)
- **핵심 과제**: Type II Error(부도 미탐지) 최소화, Recall 우선

### Part 1-2 완료 사항

**Part 1**: 유동성이 가장 강력한 예측 변수, 업종별 부도율 2배 차이 발견
**Part 2**: 52개 특성 생성 → VIF/IV 기반 27개 선택, `domain_based_features_완전판.csv` 출력

---

## 🚨 Critical Requirements (절대 준수 사항)

### 1️⃣ **Data Leakage 완전 제거** ⚠️⚠️⚠️

```
❌ 절대 금지:
- Test set으로 모델 선택
- Test set으로 임계값 최적화
- Test set으로 Traffic Light 등급 기준 결정
- Test set을 보고 어떤 의사결정도 하지 않음

✅ 필수:
- Test set은 최종 보고 직전 단 한 번만 평가
- 모든 의사결정은 Train/Validation에서만
- Validation set 또는 CV로 임계값 최적화
```

**비유**: Test set은 "봉인된 시험지". 모델 개발이 완전히 끝난 후 한 번만 개봉.

---

### 2️⃣ **3-Way Data Split (Train/Validation/Test)**

```python
# 필수 구조:
전체 데이터 (50,105)
├─ Train Set (60%, ~30,063): 모델 학습 + CV 튜닝
├─ Validation Set (20%, ~10,021): 모델 선택, 임계값 최적화, 의사결정
└─ Test Set (20%, ~10,021): 최종 평가만 (절대 건드리지 않음!)

# 코드:
from sklearn.model_selection import train_test_split

# 1차 분할: Train+Val (80%) vs Test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2차 분할: Train (75% of 80% = 60%) vs Val (25% of 80% = 20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

print(f"Train: {len(X_train)} (부도율: {y_train.mean():.2%})")
print(f"Val:   {len(X_val)} (부도율: {y_val.mean():.2%})")
print(f"Test:  {len(X_test)} (부도율: {y_test.mean():.2%})")
```

---

### 3️⃣ **리샘플링 vs Class Weight 명확한 대조 실험**

**문제점**: SMOTE + Class Weight 동시 사용 시 노이즈 증폭, 과적합 위험

**해결책**: 두 전략을 명확히 분리하여 실험

```python
# Strategy A: SMOTE 계열 (Class Weight 없음)
strategy_A = {
    'resampler': [SMOTE(0.2), BorderlineSMOTE(0.2), SMOTETomek(0.2)],
    'classifier__scale_pos_weight': [1],  # 기본값
    'classifier__class_weight': [None],   # 사용 안 함
}

# Strategy B: Class Weight (리샘플링 없음)
strategy_B = {
    'resampler': ['passthrough'],
    'classifier__scale_pos_weight': [1, sqrt_ratio, scale_ratio],
    'classifier__class_weight': ['balanced'],
}

# 두 전략의 성능을 Validation Set에서 비교
```

**권장**: 금융 데이터 특성상 **Strategy B (Class Weight Only)**가 더 나을 가능성 높음.

---

### 4️⃣ **모델 선택: Validation Set 기반 + Statistical Test**

```python
# ❌ 잘못된 방법 (현재 코드):
single_score = best_model.score(X_test, y_test)  # Test set 사용!
voting_score = voting_clf.score(X_test, y_test)

# ✅ 올바른 방법:
# 1. Validation Set 평가
single_val_pr_auc = average_precision_score(y_val, best_model.predict_proba(X_val)[:, 1])
voting_val_pr_auc = average_precision_score(y_val, voting_clf.predict_proba(X_val)[:, 1])

# 2. Statistical Significance Test (선택적)
from scipy.stats import wilcoxon

# CV fold별 점수로 paired test
cv_scores_single = cross_val_score(best_model, X_train, y_train, cv=5, scoring='average_precision')
cv_scores_voting = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='average_precision')

statistic, pvalue = wilcoxon(cv_scores_voting, cv_scores_single)

# 3. 최종 결정 (Validation + Statistical Test)
if voting_val_pr_auc > single_val_pr_auc and pvalue < 0.05:
    final_model = voting_clf
    decision_reason = f"Voting이 {voting_val_pr_auc - single_val_pr_auc:.4f} 더 우수 (p={pvalue:.4f} < 0.05)"
else:
    final_model = best_model
    decision_reason = f"Single 모델 선택 (유의미한 차이 없음 또는 복잡도 고려)"
```

---

### 5️⃣ **임계값 최적화: Validation Set 또는 CV 기반**

```python
# ❌ 절대 금지 (Data Leakage):
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_test)
optimal_threshold = find_f2_optimal(precisions, recalls, thresholds)

# ✅ 방법 1: Validation Set 사용
y_prob_val = final_model.predict_proba(X_val)[:, 1]
precisions_val, recalls_val, thresholds_val = precision_recall_curve(y_val, y_prob_val)

# F2-Score 계산 (Recall 우선)
beta = 2
f2_scores = (1 + beta**2) * (precisions_val * recalls_val) / (beta**2 * precisions_val + recalls_val + 1e-10)
optimal_idx = np.argmax(f2_scores)
optimal_threshold = thresholds_val[optimal_idx]

print(f"✅ Validation Set 기반 최적 임계값: {optimal_threshold:.4f}")
print(f"   - F2-Score: {f2_scores[optimal_idx]:.4f}")
print(f"   - Precision: {precisions_val[optimal_idx]:.2%}")
print(f"   - Recall: {recalls_val[optimal_idx]:.2%}")

# ✅ 방법 2: Cross-Validation 기반 (더 robust)
from sklearn.model_selection import cross_val_predict

y_prob_cv = cross_val_predict(final_model, X_train, y_train, cv=5, method='predict_proba')[:, 1]
precisions_cv, recalls_cv, thresholds_cv = precision_recall_curve(y_train, y_prob_cv)
# ... F2-Score 계산 동일 ...

# 최종: Validation + CV 평균값 사용 (더욱 robust)
optimal_threshold_final = (optimal_threshold_val + optimal_threshold_cv) / 2
```

**중요**: Test set에는 결정된 임계값을 "적용"만 하고, 절대 "최적화"하지 않음!

---

### 6️⃣ **Traffic Light 임계값: 데이터 기반 논리**

```python
# ❌ 현재 방식 (작위적):
yellow_threshold = red_threshold * 0.5  # 왜 0.5인가? 근거 약함

# ✅ 개선 방식 1: Recall 기반
# Red: Recall 80% 달성 임계값
# Yellow: Recall 95% 달성 임계값

idx_recall_80 = np.where(recalls_val >= 0.80)[0]
red_threshold = thresholds_val[idx_recall_80[np.argmax(precisions_val[idx_recall_80])]]

idx_recall_95 = np.where(recalls_val >= 0.95)[0]
yellow_threshold = thresholds_val[idx_recall_95[np.argmax(precisions_val[idx_recall_95])]]

print(f"🔴 Red: Recall 80% 보장 (임계값 {red_threshold:.4f})")
print(f"🟡 Yellow: Recall 95% 보장 (임계값 {yellow_threshold:.4f})")

# ✅ 개선 방식 2: 확률 분포 기반
# Validation Set 부도 기업의 확률 분포에서 percentile 활용
bankrupt_probs = y_prob_val[y_val == 1]

red_threshold = np.percentile(bankrupt_probs, 20)    # 부도 기업 하위 20%
yellow_threshold = np.percentile(bankrupt_probs, 5)  # 부도 기업 하위 5%

print(f"🔴 Red: 부도 기업 상위 80% 포함 (임계값 {red_threshold:.4f})")
print(f"🟡 Yellow: 부도 기업 상위 95% 포함 (임계값 {yellow_threshold:.4f})")
```

**논리 강화**: "Yellow 등급은 부도 기업의 95%를 포착하도록 설계했습니다" → 비즈니스 설득력 ↑

---

### 7️⃣ **앙상블 전략: 다양성(Diversity) 확보**

**문제**: 상위 3개 모두 GBM 계열 → Voting 효과 미미

**해결책 A**: 이종 모델 강제 포함

```python
# Top GBM (최고 성능)
best_gbm = search.best_estimator_

# 최고 성능 Logistic Regression (설명력 ↑, 다양성 ↑)
lr_results = results_df[results_df['param_classifier'].apply(lambda x: 'Logistic' in str(x))]
best_lr = lr_results.nsmallest(1, 'rank_test_score').iloc[0]

# 최고 성능 Random Forest (Tree 기반이지만 메커니즘 다름)
rf_results = results_df[results_df['param_classifier'].apply(lambda x: 'Random' in str(x))]
best_rf = rf_results.nsmallest(1, 'rank_test_score').iloc[0]

# 3-model Ensemble (GBM + LR + RF)
voting_clf = VotingClassifier(
    estimators=[
        ('gbm', best_gbm),
        ('lr', best_lr_pipeline),
        ('rf', best_rf_pipeline)
    ],
    voting='soft',
    weights=[0.5, 0.25, 0.25]  # GBM 높은 가중치, 나머지는 다양성 확보
)
```

**해결책 B**: 단일 모델 선택 (권장)

```python
# Part 4 SHAP 분석을 고려하면 단일 모델이 훨씬 유리
# Validation Set 평가 후 앙상블이 0.5% 이상 우수하지 않으면 단일 모델 선택

if voting_val_pr_auc - single_val_pr_auc < 0.005:
    final_model = best_model
    decision_reason = "성능 차이 미미 + SHAP 해석력 우선 → 단일 모델 선택"
```

---

### 8️⃣ **전처리 파이프라인 개선**

```python
# ✅ 개선된 순서 및 설정
pipeline = ImbPipeline([
    ('inf_handler', InfiniteHandler()),
    ('imputer', IterativeImputer(max_iter=10, random_state=42)),  # 먼저 결측치 처리
    ('log_transformer', LogTransformer()),                        # 그 다음 로그 변환
    ('winsorizer', Winsorizer(0.005, 0.995)),  # 범위 축소 (0.5%~99.5%), 또는 제거 검토
    ('scaler', RobustScaler()),
    ('resampler', 'passthrough'),  # Class Weight 전략 우선
    ('classifier', LogisticRegression())
])
```

**Winsorizer 제거 검토**: Tree 모델은 이상치에 강건. 부도 데이터의 극단값은 중요 시그널일 수 있음.

```python
# 실험: Winsorizer 있음 vs 없음 비교
pipeline_with_winsor = create_pipeline(use_winsorizer=True)
pipeline_without_winsor = create_pipeline(use_winsorizer=False)

# Validation Set 비교
score_with = average_precision_score(y_val, pipeline_with_winsor.predict_proba(X_val)[:, 1])
score_without = average_precision_score(y_val, pipeline_without_winsor.predict_proba(X_val)[:, 1])

print(f"Winsorizer 있음: {score_with:.4f}")
print(f"Winsorizer 없음: {score_without:.4f}")
```

---

### 9️⃣ **RandomizedSearchCV 설정 개선**

**현재**: 100회 샘플링 → Coverage 약 1% (탐색 공간 대비)

**개선**: 모델별 독립 튜닝 또는 횟수 증가

```python
# 방법 1: 모델별 개별 튜닝 (권장)
models_to_tune = {
    'LightGBM': (lgb.LGBMClassifier(), lgbm_param_grid),
    'XGBoost': (xgb.XGBClassifier(), xgb_param_grid),
    'CatBoost': (CatBoostClassifier(), catboost_param_grid),
}

best_models = {}
for model_name, (model, param_grid) in models_to_tune.items():
    search = RandomizedSearchCV(
        estimator=create_pipeline(model),
        param_distributions=param_grid,
        n_iter=200,  # 모델당 200회
        scoring='average_precision',
        cv=5,
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)
    best_models[model_name] = search.best_estimator_
    print(f"{model_name} 최적 CV PR-AUC: {search.best_score_:.4f}")

# 방법 2: Optuna 사용 (선택적)
import optuna
# ... Optuna 기반 Bayesian Optimization ...
```

---

## 📋 노트북 구조 (필수 섹션)

```markdown
# 📗 발표용 Part 3: 모델링 및 최적화 v2

## 🎯 Part 3 목표 및 이전 Part 요약
[Part 1-2 요약, 개선 사항 명시]

## 0. 환경 설정
[라이브러리, 한글 폰트]

## 1. 데이터 로딩 및 **3-Way Split** ⭐
- Train (60%) / Validation (20%) / Test (20%)
- Stratified split으로 부도율 유지
- 각 set의 통계 출력

## 2. 전처리 파이프라인 정의
- 순서 개선: Imputer → Log → Winsorizer(선택적) → Scaler
- Winsorizer 실험 결과 포함

## 3. 리샘플링 전략 대조 실험
- **Strategy A**: SMOTE 계열 (Class Weight 제외)
- **Strategy B**: Class Weight Only (리샘플링 제외)
- Validation Set 비교 → 우수한 전략 선택

## 4. AutoML: 하이퍼파라미터 튜닝
- RandomizedSearchCV (모델별 200회 or 통합 500회)
- **Train Set + 5-Fold CV**로만 학습
- 상위 10개 모델 출력

## 5. 모델 선택: **Validation Set 기반** ⭐
- Single Best vs Voting Ensemble **Validation 평가**
- Statistical Significance Test (Wilcoxon)
- 최종 모델 선택 로직 + 근거 명시

## 6. 임계값 최적화: **Validation Set 기반** ⭐
- F2-Score (Recall 우선) 최적화
- Recall 80% 목표 임계값
- CV 기반 검증 (선택적)

## 7. Traffic Light 시스템: **데이터 기반 임계값** ⭐
- Red: Recall 80% 보장
- Yellow: Recall 95% 보장
- Validation Set 성능 출력
- 논리적 근거 명시

## 8. **Test Set 최종 평가** (단 한 번!) ⭐
- 임계값 적용 후 Test Set 평가
- Confusion Matrix (최적 임계값)
- PR-AUC Curve
- Traffic Light 성능 (Test Set)
- ⚠️ "Test Set은 절대 건드리지 않았음" 명시

## 9. Feature Importance 분석
[Top 15, Plotly 시각화]

## 10. 비즈니스 임팩트 분석
- Cumulative Gains Curve (Test Set)
- 효율성 분석

## 11. 모델 저장 및 다음 단계
- Part 4 SHAP 분석용 파일 저장
- 전처리된 데이터 (X_train, X_val, X_test processed)
- 임계값 저장
```

---

## 📊 필수 출력 내용

### 데이터 분할 정보

```
✅ 3-Way Split 완료
==================================================
Train Set:      30,063 (60%, 부도율: 1.51%)
Validation Set: 10,021 (20%, 부도율: 1.51%)
Test Set:       10,021 (20%, 부도율: 1.51%)
==================================================
⚠️ Test Set은 최종 평가 전까지 절대 사용하지 않음!
```

### 리샘플링 전략 비교

```
📊 리샘플링 전략 Validation 평가
==================================================
Strategy A (SMOTE):         PR-AUC = 0.XXXX
Strategy B (Class Weight):  PR-AUC = 0.XXXX
==================================================
✅ 선택: Strategy B (Class Weight) - X.XX% 더 우수
```

### 모델 선택 결과

```
📊 모델 선택 (Validation Set 기반)
==================================================
Single Best Model:     PR-AUC = 0.XXXX
Voting Ensemble:       PR-AUC = 0.XXXX
Wilcoxon p-value:      0.XXX
==================================================
✅ 최종 선택: [모델명]
   이유: [Validation 성능 + Statistical Test + 복잡도 고려]
```

### 임계값 최적화 결과

```
📊 임계값 최적화 (Validation Set 기반)
==================================================
F2-Score 최적 임계값:     0.XXXX
  - F2-Score:             0.XXXX
  - Precision:            XX.XX%
  - Recall:               XX.XX%

Recall 80% 보장 임계값:   0.XXXX
  - Precision:            XX.XX%
  - Recall:               XX.XX%
==================================================
✅ 선택: Recall 80% 임계값 (비즈니스 요구사항 우선)
```

### Test Set 최종 평가 (단 한 번!)

```
🎯 Test Set 최종 평가 (임계값 적용)
==================================================
⚠️ 이 결과는 최종 보고용이며, 이전 단계에서 Test Set을
   절대 사용하지 않았음을 보장합니다.

임계값: 0.XXXX (Validation에서 결정)
PR-AUC:              0.XXXX
ROC-AUC:             0.XXXX
F2-Score:            0.XXXX
Precision:           XX.XX%
Recall:              XX.XX%
Type II Error:       XX.XX%

Confusion Matrix:
  TN: X,XXX  |  FP: XXX
  FN: XXX    |  TP: XXX
==================================================
```

### Traffic Light 성능 (Test Set)

```
🚦 Traffic Light 시스템 (Test Set 최종 평가)
==================================================
등급      기업 수    비율    실제 부도    정밀도    포착률
🔴 Red    XXX       X.X%    XXX         XX.X%    XX.X%
🟡 Yellow XXX       X.X%    XXX         XX.X%    XX.X%
🟢 Green  XXX       XX.X%   XXX         X.X%     X.X%
----------------------------------------------------------
합계      10,021    100%    XXX         -        XX.X%
==================================================
✅ 리스크 방어율: XX.X% (Red+Yellow에서 부도 포착)
```

---

## ⚠️ 절대 금지 사항 (Checklist)

- [ ] ❌ Test set으로 모델 선택
- [ ] ❌ Test set으로 임계값 최적화
- [ ] ❌ Test set으로 Traffic Light 기준 결정
- [ ] ❌ Test set을 보고 하이퍼파라미터 재튜닝
- [ ] ❌ Test set을 보고 전처리 방법 변경
- [ ] ❌ Test set 성능이 나쁘다고 다시 실험

✅ **Test set은 최종 보고 직전 단 한 번만 평가!**

---

## ✅ 필수 준수 사항 (Checklist)

### 데이터 분할
- [ ] ✅ Train/Val/Test 3-way split (60/20/20)
- [ ] ✅ Stratified split으로 부도율 유지
- [ ] ✅ 각 set의 통계 출력

### 리샘플링 전략
- [ ] ✅ SMOTE vs Class Weight 대조 실험
- [ ] ✅ Validation set 비교
- [ ] ✅ 우수한 전략 선택 근거 명시

### 모델 선택
- [ ] ✅ Validation set 기반 평가
- [ ] ✅ Statistical significance test (선택적)
- [ ] ✅ 최종 모델 선택 근거 명시

### 임계값 최적화
- [ ] ✅ Validation set 또는 CV 기반
- [ ] ✅ F2-Score (Recall 우선) 사용
- [ ] ✅ Test set에는 적용만

### Traffic Light
- [ ] ✅ 데이터 기반 임계값 (Recall 또는 percentile)
- [ ] ✅ Validation set 검증
- [ ] ✅ 논리적 근거 명시

### Test Set 평가
- [ ] ✅ 최종 보고 직전 단 한 번만
- [ ] ✅ "Test set 미사용" 명시
- [ ] ✅ 모든 임계값 Validation에서 결정됨 확인

### 코드 품질
- [ ] ✅ 하드코딩 금지 (경로, 임계값 변수화)
- [ ] ✅ 한글 폰트 설정
- [ ] ✅ UTF-8 인코딩
- [ ] ✅ Top-to-bottom 실행 가능

### 문서화
- [ ] ✅ 각 섹션마다 명확한 설명
- [ ] ✅ 의사결정 근거 명시
- [ ] ✅ Validation vs Test 구분 명확
- [ ] ✅ Data Leakage 방지 명시

---

## 🎯 예상 결과 (성능 목표)

### Validation Set (의사결정 기준)
- PR-AUC: 0.15~0.20 (불균형 데이터 고려)
- F2-Score: 0.35~0.50
- Recall: 60~80%
- Type II Error: 20~40%

### Test Set (최종 보고)
- PR-AUC: Validation ± 0.01 (일반화 성능 확인)
- Recall: Validation ± 5%p
- Type II Error: Validation ± 5%p

**중요**: Test 성능이 Validation과 크게 다르면 과적합 또는 데이터 분포 이슈

---

## 📁 출력 파일 목록

```
data/processed/
├── 발표_Part3_v2_최종모델.pkl
├── 발표_Part3_v2_분류기.pkl (SHAP용)
├── 발표_Part3_v2_X_train_processed.csv
├── 발표_Part3_v2_X_val_processed.csv
├── 발표_Part3_v2_X_test_processed.csv
├── 발표_Part3_v2_y_train.csv
├── 발표_Part3_v2_y_val.csv
├── 발표_Part3_v2_y_test.csv
├── 발표_Part3_v2_임계값.pkl
└── 발표_Part3_v2_결과요약.pkl
```

---

## 💡 추가 권장 사항

### 1. Nested CV (선택적, 더 robust)

```python
# Outer loop: 모델 평가
# Inner loop: Hyperparameter tuning

from sklearn.model_selection import cross_validate

def nested_cv_evaluation(pipeline, X, y, param_grid):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    nested_scores = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]

        # Inner CV: Hyperparameter tuning
        search = RandomizedSearchCV(
            pipeline, param_grid,
            cv=inner_cv, scoring='average_precision', n_iter=50
        )
        search.fit(X_train_outer, y_train_outer)

        # Outer CV: 최적 모델 평가
        score = average_precision_score(
            y_test_outer,
            search.best_estimator_.predict_proba(X_test_outer)[:, 1]
        )
        nested_scores.append(score)

    return np.mean(nested_scores), np.std(nested_scores)

# Nested CV로 일반화 성능 추정
mean_score, std_score = nested_cv_evaluation(pipeline, X_train, y_train, param_grid)
print(f"Nested CV PR-AUC: {mean_score:.4f} ± {std_score:.4f}")
```

### 2. Calibration Check (확률 보정)

```python
from sklearn.calibration import calibration_curve

# Validation Set에서 Calibration 확인
prob_true, prob_pred = calibration_curve(
    y_val, y_prob_val, n_bins=10, strategy='quantile'
)

# Calibration Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='markers+lines', name='Model'))
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Perfect', line=dict(dash='dash')))
fig.update_layout(title='Calibration Curve (Validation Set)',
                  xaxis_title='Predicted Probability',
                  yaxis_title='True Probability')
fig.show()

# Calibration이 나쁘면 CalibratedClassifierCV 사용
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(final_model, cv=5, method='isotonic')
calibrated_model.fit(X_train, y_train)
```

### 3. Learning Curve (과적합 진단)

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    final_model, X_train, y_train,
    cv=5, scoring='average_precision',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=train_sizes,
    y=train_scores.mean(axis=1),
    mode='lines+markers',
    name='Train Score',
    error_y=dict(array=train_scores.std(axis=1))
))
fig.add_trace(go.Scatter(
    x=train_sizes,
    y=val_scores.mean(axis=1),
    mode='lines+markers',
    name='CV Score',
    error_y=dict(array=val_scores.std(axis=1))
))
fig.update_layout(title='Learning Curve', xaxis_title='Training Size', yaxis_title='PR-AUC')
fig.show()
```

---

## 🔍 품질 검증 체크리스트

### 실행 전 (코드 리뷰)
- [ ] Test set 사용 위치 전수 조사 (절대 의사결정 없음)
- [ ] 모든 임계값이 Validation에서 결정되는지 확인
- [ ] Random state 설정 확인 (재현성)
- [ ] 경로 하드코딩 제거

### 실행 중 (출력 확인)
- [ ] 데이터 분할 비율 정확한지 확인
- [ ] 부도율이 각 set에서 유지되는지 확인
- [ ] Validation 성능 > Test 성능 (정상)
- [ ] Test 성능이 Validation ± 10% 범위 내

### 실행 후 (결과 검증)
- [ ] Test set 평가가 단 한 번만 나타남
- [ ] 모든 의사결정 근거가 Validation 기반임 명시
- [ ] Confusion Matrix 수치 확인 (TP+FN = 전체 부도 수)
- [ ] Traffic Light 등급별 부도율 차이 명확

---

## 📝 최종 확인 사항

### 학술적 엄밀성
- ✅ Data Leakage 완전 제거
- ✅ Train/Val/Test 명확한 분리
- ✅ 재현 가능성 (random_state 설정)
- ✅ Statistical significance test

### 비즈니스 가치
- ✅ F2-Score (Recall 우선) 사용
- ✅ Traffic Light 시스템 (의사결정 지원)
- ✅ Cumulative Gains (효율성 입증)
- ✅ Type II Error 최소화

### 코드 품질
- ✅ 모듈화 (함수/클래스)
- ✅ 한글 폰트 설정
- ✅ 주석 및 문서화
- ✅ Top-to-bottom 실행

### Part 4 연계
- ✅ SHAP 분석용 파일 저장
- ✅ 전처리 파이프라인 저장
- ✅ 단일 모델 우선 (해석력)

---

## 🎉 완성 기준

이 프롬프트를 따라 생성된 노트북은:

1. ✅ **학술 논문 수준의 방법론적 엄밀성**
2. ✅ **실무 배포 가능한 일반화 성능**
3. ✅ **비즈니스 의사결정 지원 가능**
4. ✅ **재현 가능하고 투명한 프로세스**
5. ✅ **Part 4 SHAP 분석 준비 완료**

**파일명**: `notebooks/발표_Part3_모델링_및_최적화_v2.ipynb`

**예상 실행 시간**: 30~60분 (AutoML + 대조 실험)

**최종 신뢰도**: **9.5/10** ⭐⭐⭐⭐⭐

---

**프롬프트 작성일**: 2025년
**작성자**: Senior Data Scientist Review Team
**버전**: 2.0 (Data Leakage Free, Production Ready)

---

이제 이 프롬프트를 새 Claude Code 채팅창에 붙여넣고 실행하세요! 🚀
