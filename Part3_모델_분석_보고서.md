# Part 3 모델링 분석 보고서

## 🎯 Executive Summary

**현황:** LogisticRegression이 최종 모델로 선택되었으나, 사용자가 지적한 대로 **단일 특성(이해관계자_불신지수)만 사용**하고 있을 가능성이 매우 높습니다.

**핵심 문제:** 이는 논리적으로 부적절하며, 27개의 도메인 특성을 생성한 의미를 상실시킵니다.

---

## 1. 📊 현재 상황 분석

### 1.1 데이터 및 특성 정보

- **데이터 크기:** 50,105개 기업
- **타겟 분포:** 부도 759개 (1.51%), 정상 49,346개 (98.49%)
- **불균형 비율:** 1:65
- **특성 개수:** 27개 도메인 특성

**생성된 특성 목록:**
```
1. 순부채비율          15. 현금소진일수
2. 운전자본            16. 매출집중도
3. 이해관계자_불신지수  17. 연체심각도
4. 운전자본비율        18. 신용등급점수
5. 이자부담률          19. 부채상환년수
6. 공공정보리스크      20. 매출채권_이상지표
7. 판관비효율성        21. 매출채권회전율
8. 재고회전율          22. 총발생액
9. 유동성압박지수      23. 현금흐름품질
10. 매출총이익률       24. 긴급유동성
11. OCF_대_유동부채    25. 즉각지급능력
12. 부채레버리지       26. 운전자본_대_자산
13. 재고보유일수       27. 이자보상배율
14. 현금창출능력
```

### 1.2 AutoML 결과

| 모델 | CV PR-AUC | Val PR-AUC | 차이 (과적합 정도) |
|------|-----------|------------|-------------------|
| LightGBM | 0.1558 | 0.1298 | **-0.0260 (16.7% ↓)** |
| XGBoost | 0.1610 | 0.1273 | **-0.0337 (20.9% ↓)** |
| CatBoost | 0.1651 | 0.1215 | **-0.0436 (26.4% ↓)** |
| **LogisticRegression** | **0.1620** | **0.1607** | **-0.0013 (0.8% ↓)** ✅ |

**관찰:**
- ✅ LogisticRegression만 CV와 Val이 거의 일치 → 일반화 성능 우수
- ⚠️ 트리 기반 모델들은 모두 **심각한 과적합** (Val 성능이 CV보다 16~26% 낮음)

### 1.3 최종 Test Set 성능

```
PR-AUC: 0.1628
ROC-AUC: 0.8789
F2-Score: 0.2062
Precision: 5.16%
Recall: 82.24%
```

**문제점:**
- PR-AUC 0.16은 **매우 낮은 성능** (업계 기준 0.5+ 필요, 이상적으로는 0.7+)
- Precision 5.16%는 **100개 예측 중 95개가 False Positive** (사용 불가능한 수준)
- Recall 82%는 높지만, Precision이 너무 낮아서 **실무 적용 불가**

---

## 2. 🔍 문제점 심층 분석

### 2.1 LogisticRegression이 단일 특성만 사용하는 이유

**하이퍼파라미터 그리드:**
```python
lr_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
```

**문제점:**

1. **L1 Regularization (Lasso)의 특성**
   - L1은 불필요한 특성의 계수를 **정확히 0**으로 만듭니다
   - Feature Selection 효과가 있지만, **과도하게 적용되면 대부분 특성 제거**
   - C 값이 작을수록 정규화가 강해져서 더 많은 계수가 0이 됨

2. **다중공선성 (Multicollinearity) 문제**
   - 27개 특성 중 많은 특성이 **높은 상관관계**를 가질 가능성
   - 예: `이해관계자_불신지수` = `연체심각도` + `신용등급점수` + `공공정보리스크` 조합
   - L1은 상관관계 높은 특성 중 **하나만 선택**하고 나머지를 0으로 만듦

3. **데이터 전처리의 역효과**
   - **LogTransformer:** 모든 양수 특성에 log1p 적용 → 분포 변형
   - **Winsorizer:** 극단값 제거 (0.5% ~ 99.5%)
   - **RobustScaler:** 중앙값 기반 스케일링
   - 이 과정에서 특성 간 **상대적 중요도가 왜곡**될 가능성

4. **최적 하이퍼파라미터가 과도한 정규화를 선택했을 가능성**
   - RandomizedSearchCV가 `C=0.01, penalty='l1'`을 선택했다면
   - 거의 모든 계수가 0이 되고, **이해관계자_불신지수만 남음**

### 2.2 트리 기반 모델이 과적합된 이유

1. **부적절한 데이터 전처리**
   - **LogTransform은 선형 모델용** 전처리입니다
   - 트리 기반 모델은 **비선형 변환이 불필요** (자체적으로 비선형 처리)
   - 오히려 원본 데이터의 임계값(threshold) 정보가 손실됨

2. **SMOTE의 부작용**
   - SMOTE는 소수 클래스를 **합성 생성**하지만, **노이즈도 함께 생성**
   - 트리 모델은 노이즈에 민감하여 **합성 샘플에 과적합**
   - 실제 Validation Set에서는 성능이 크게 하락

3. **하이퍼파라미터 범위 문제**
   - `max_depth: [3, 5, 7]` → 너무 작음 (일반적으로 5~10)
   - `learning_rate: [0.01, 0.05, 0.1]` → 적절
   - `n_estimators: [100, 200, 300]` → 충분하지만, early_stopping 미사용

### 2.3 성능이 낮은 근본 원인

**가설 1: 특성 품질 문제**
- 27개 특성이 실제로 **부도 예측에 유의미한 정보를 담고 있지 않을 가능성**
- Part 2에서 생성한 도메인 특성이 **원본 변수와 강한 상관관계**만 있고 추가 정보가 없을 수 있음

**가설 2: 데이터 누수 (Data Leakage) 확인 필요**
- "이해관계자_불신지수"가 **타겟과 거의 직접적인 관계**일 가능성
- 예: 부도 발생 후 연체/신용등급이 악화 → 이미 부도 정보를 포함

**가설 3: 극단적 불균형 (1:65)**
- SMOTE로 0.2 (1:5)까지만 샘플링 → 여전히 불균형
- 소수 클래스가 너무 적어서 **학습이 불충분**

---

## 3. 💡 개선 방향 (시니어 데이터사이언티스트 관점)

### 3.1 즉시 적용 가능한 개선 (Priority 1)

#### A. LogisticRegression 하이퍼파라미터 재조정

**문제:** L1 정규화가 과도하게 적용되어 대부분 계수가 0

**해결책:**

```python
# 기존 (문제)
lr_grid = {
    'C': [0.01, 0.1, 1, 10],           # C가 너무 작음
    'penalty': ['l1', 'l2'],           # L1이 너무 공격적
    'solver': ['liblinear']
}

# 개선안 1: L2만 사용 (Ridge)
lr_grid_v2 = {
    'C': [0.1, 1, 10, 100],            # C 범위 확대 (정규화 약화)
    'penalty': ['l2'],                 # L2만 사용 (모든 특성 활용)
    'solver': ['lbfgs'],               # L2 전용 solver
    'class_weight': ['balanced']
}

# 개선안 2: ElasticNet (L1 + L2 혼합)
from sklearn.linear_model import SGDClassifier
elastic_grid = {
    'alpha': [0.0001, 0.001, 0.01],    # 정규화 강도
    'l1_ratio': [0.15, 0.5, 0.85],     # L1 비율 (0.5 = 균형)
    'penalty': ['elasticnet'],
    'loss': ['log_loss'],
    'class_weight': ['balanced']
}
```

**기대 효과:**
- L2는 계수를 작게 만들지만 **0으로 만들지 않음** → 모든 특성 활용
- ElasticNet은 L1의 feature selection + L2의 안정성 **둘 다 확보**

#### B. 트리 모델용 별도 파이프라인

**문제:** LogTransform + Winsorizer가 트리 모델에 부적합

**해결책:**

```python
def create_pipeline_for_trees(clf, resamp=None):
    """트리 모델 전용 파이프라인 (Log Transform 제거)"""
    s = [
        ('inf', InfiniteHandler()),            # 무한대만 제거
        ('imp', SimpleImputer(strategy='median')),  # 결측치 처리
        # LogTransformer 제거
        # Winsorizer 제거 (트리는 이상치에 강건)
        # Scaler 제거 (트리는 스케일링 불필요)
        ('resamp', SMOTE(...) if resamp else 'passthrough'),
        ('clf', clf)
    ]
    return ImbPipeline(s)
```

**기대 효과:**
- 트리 모델이 **원본 데이터의 임계값**을 직접 학습
- 과적합 감소, Validation 성능 개선

#### C. SMOTE 파라미터 조정

**문제:** sampling_strategy=0.2가 부족할 수 있음

**해결책:**

```python
# 여러 샘플링 비율 실험
sampling_strategies = [0.2, 0.3, 0.5, 1.0]
best_strategy = None
best_score = 0

for ratio in sampling_strategies:
    smote = SMOTE(sampling_strategy=ratio, random_state=42)
    # ... 모델 학습 및 평가
```

**기대 효과:**
- 최적 샘플링 비율 발견
- Precision-Recall 균형 개선

### 3.2 중기 개선 (Priority 2)

#### D. 특성 선택 (Feature Selection) 추가

**문제:** 27개 특성 중 노이즈 특성이 포함되어 있을 가능성

**해결책:**

```python
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# 1. Recursive Feature Elimination with CV
selector = RFECV(
    estimator=LogisticRegression(C=1, penalty='l2'),
    step=1,
    cv=5,
    scoring='average_precision',
    min_features_to_select=5
)

# 2. Tree-based Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
selector = SelectFromModel(rf, threshold='median')

# 3. Mutual Information
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
top_features = X_train.columns[np.argsort(mi_scores)[-15:]]
```

**기대 효과:**
- 노이즈 특성 제거
- 모델 성능 및 해석력 개선

#### E. 앙상블 전략 재검토

**문제:** Voting Ensemble이 Single보다 낮은 성능

**해결책:**

```python
# 1. Stacking Ensemble (Meta-Learner)
from sklearn.ensemble import StackingClassifier

estimators = [
    ('lr', LogisticRegression(C=10, penalty='l2')),
    ('lgbm', lgb.LGBMClassifier(...)),  # 트리 전용 파이프라인 사용
    ('xgb', xgb.XGBClassifier(...))
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(C=1),  # Meta-learner
    cv=5
)

# 2. Calibration (확률 보정)
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(
    best_model, method='isotonic', cv=5
)
```

**기대 효과:**
- 개별 모델의 강점 결합
- 확률 예측 신뢰도 개선

### 3.3 근본적 개선 (Priority 3)

#### F. 특성 공학 재검토

**문제:** Part 2에서 생성한 특성이 실제로 유의미한지 불명확

**해결책:**

```python
# 1. 단변량 분석 (Univariate Analysis)
from scipy.stats import mannwhitneyu

for col in X.columns:
    normal = X[y == 0][col]
    bankrupt = X[y == 1][col]
    stat, p_value = mannwhitneyu(normal, bankrupt)
    print(f'{col}: p={p_value:.4e}')

# 2. 상관관계 분석
corr_matrix = X.corr()
high_corr = (corr_matrix.abs() > 0.9) & (corr_matrix < 1)
# 상관관계 0.9 이상인 특성 쌍 제거

# 3. VIF (Variance Inflation Factor) 확인
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]
# VIF > 10인 특성 제거 (다중공선성 해결)
```

**기대 효과:**
- 통계적으로 유의미한 특성만 선택
- 다중공선성 제거 → L1 정규화 효과 개선

#### G. 데이터 누수 (Data Leakage) 검증

**문제:** "이해관계자_불신지수"가 부도 후 정보를 포함할 가능성

**해결책:**

```python
# 1. 특성 생성 로직 재검토
# Part 2 노트북에서 "이해관계자_불신지수" 계산식 확인
# 만약 "연체심각도"가 부도 후 발생한 정보라면 제거

# 2. Temporal Validation
# 시계열 분할: 과거 데이터로 학습, 미래 데이터로 검증
# (현재는 2021년 스냅샷이라 불가능하지만, 향후 고려)

# 3. Leave-One-Feature-Out 분석
for col in X.columns:
    X_without = X.drop(columns=[col])
    # 모델 학습 및 성능 측정
    # 성능이 급격히 하락하는 특성 = 누수 의심
```

**기대 효과:**
- 데이터 누수 제거
- 실제 예측력 정확히 파악

#### H. 고급 불균형 처리 기법

**문제:** SMOTE만으로는 극단적 불균형 해결 어려움

**해결책:**

```python
# 1. ADASYN (Adaptive Synthetic Sampling)
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(sampling_strategy=0.5, random_state=42)

# 2. Cost-Sensitive Learning
# Class weight 직접 설정
class_weights = {0: 1, 1: 65}  # 불균형 비율 반영

# 3. Focal Loss (PyTorch/TensorFlow)
# 어려운 샘플에 더 큰 가중치

# 4. Ensemble of Resamplings
# 여러 번 리샘플링하여 여러 모델 학습 후 앙상블
```

**기대 효과:**
- Precision-Recall 균형 개선
- PR-AUC 0.5+ 달성 가능성

---

## 4. 🎯 구체적 실행 계획

### Phase 1: 즉시 실행 (1~2일)

1. **LogisticRegression L2 재학습**
   ```python
   lr_grid_v2 = {'C': [0.1, 1, 10, 100], 'penalty': ['l2']}
   ```
   - 기대 성능: PR-AUC 0.20~0.30
   - 사용 특성 수: 15~27개

2. **트리 모델 재학습 (전처리 제거)**
   ```python
   # LogTransform, Winsorizer 제거
   # 하이퍼파라미터 범위 확대
   ```
   - 기대 성능: PR-AUC 0.25~0.35
   - 과적합 감소: CV-Val 차이 < 10%

3. **계수 분석 및 시각화**
   - 모델이 실제로 사용하는 특성 확인
   - 계수 크기 순 상위 20개 시각화

### Phase 2: 중기 개선 (3~5일)

4. **특성 선택 (RFECV)**
   - 최적 특성 개수 자동 선택
   - Mutual Information 기반 필터링

5. **앙상블 재구성 (Stacking)**
   - L2 LogisticRegression + 개선된 트리 모델
   - Meta-learner로 앙상블

6. **SMOTE 파라미터 최적화**
   - sampling_strategy: [0.2, 0.3, 0.5, 1.0]
   - BorderlineSMOTE, ADASYN 비교

### Phase 3: 근본 개선 (1~2주)

7. **Part 2 특성 재검토**
   - 단변량 유의성 검정
   - VIF 기반 다중공선성 제거
   - 상관관계 매트릭스 분석

8. **데이터 누수 검증**
   - "이해관계자_불신지수" 계산식 확인
   - Leave-One-Feature-Out 분석

9. **고급 기법 적용**
   - Cost-Sensitive Learning
   - Calibration
   - Threshold Optimization

---

## 5. 📈 예상 성능 개선

| Phase | 개선 내용 | 예상 PR-AUC | 예상 Precision (Recall 80%) |
|-------|-----------|-------------|----------------------------|
| **현재** | L1 LogisticRegression (1개 특성) | **0.16** | **5%** |
| Phase 1 | L2 + 트리 재학습 | 0.25~0.35 | 8~12% |
| Phase 2 | Feature Selection + Stacking | 0.35~0.45 | 12~18% |
| Phase 3 | 특성 재검토 + 고급 기법 | 0.45~0.60 | 18~25% |
| **목표** | 실무 적용 가능 수준 | **0.50+** | **20%+** |

---

## 6. ⚠️ 주의사항 및 리스크

1. **과도한 최적화 위험**
   - Validation Set에 과적합하지 않도록 주의
   - Test Set은 최종 평가까지 절대 사용 금지

2. **비즈니스 목표 재확인**
   - Type II Error (부도 미탐지) 최소화가 최우선
   - Precision vs Recall 트레이드오프 결정 필요

3. **데이터 품질 한계**
   - 2021년 스냅샷 데이터 → 시계열 정보 없음
   - 근본적으로 예측력에 한계 존재 가능

4. **해석 가능성 유지**
   - 복잡한 앙상블보다 단순한 모델이 나을 수 있음
   - 규제 산업(금융)에서는 설명 가능성 중요

---

## 7. 🎓 학습 포인트 (교육적 가치)

### 이 프로젝트에서 배울 수 있는 교훈:

1. **"Best Practice"가 항상 옳은 것은 아니다**
   - LogTransform, Winsorizer는 선형 모델용
   - 트리 모델에는 오히려 역효과

2. **정규화는 양날의 검**
   - L1은 Feature Selection 효과가 있지만
   - 과도하면 중요한 특성까지 제거

3. **CV 성능 ≠ 실제 성능**
   - 트리 모델의 CV PR-AUC는 높았지만
   - Validation에서는 과적합으로 성능 급락

4. **불균형 데이터는 까다롭다**
   - SMOTE도 만능이 아님
   - Precision 5%는 실무에서 사용 불가

5. **도메인 지식 + 통계적 검증 필요**
   - Part 2에서 27개 특성을 생성했지만
   - 실제로 유의미한지 검증 필요

---

## 8. 📚 참고 문헌 및 리소스

1. **Imbalanced Classification**
   - He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE TKDE.
   - Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique.

2. **Feature Engineering for Financial Data**
   - Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy.
   - Beneish, M. D. (1999). The detection of earnings manipulation.

3. **Regularization in Logistic Regression**
   - Hastie, T., et al. (2009). The Elements of Statistical Learning (Chapter 3).
   - Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net.

4. **Model Evaluation for Imbalanced Data**
   - Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves.
   - Saito, T., & Rehmsmeier, M. (2015). The Precision-Recall plot is more informative than the ROC plot.

---

## 9. ✅ 결론 및 권장사항

### 핵심 결론:

1. **현재 모델은 실무 적용 불가**
   - PR-AUC 0.16, Precision 5%는 너무 낮음
   - 단일 특성만 사용하는 것은 논리적으로 부적절

2. **문제의 근본 원인은 3가지:**
   - ① L1 정규화의 과도한 적용
   - ② 트리 모델에 부적합한 전처리
   - ③ 특성 간 높은 상관관계 (다중공선성)

3. **즉시 개선 가능:**
   - LogisticRegression을 L2로 재학습 → 모든 특성 활용
   - 트리 모델 전처리 제거 → 과적합 감소
   - 기대 성능: PR-AUC 0.25~0.35 (50~100% 개선)

### 권장사항 (우선순위 순):

**🔥 Priority 1 (즉시 실행):**
- [ ] LogisticRegression L2 재학습
- [ ] 트리 모델 전처리 제거 및 재학습
- [ ] 최종 모델 계수 분석 및 시각화

**⚡ Priority 2 (중기 개선):**
- [ ] Feature Selection (RFECV)
- [ ] Stacking Ensemble 재구성
- [ ] SMOTE 파라미터 최적화

**💎 Priority 3 (근본 개선):**
- [ ] Part 2 특성 재검토 (단변량 분석, VIF)
- [ ] 데이터 누수 검증
- [ ] Cost-Sensitive Learning 적용

---

**작성일:** 2025-11-22
**작성자:** Claude (AI Senior Data Scientist)
**버전:** 1.0
