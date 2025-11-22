# 발표_Part3_모델링_및_최적화_완전판.ipynb 생성 프롬프트

## 🎯 ROLE & EXPERTISE

당신은 **15년 경력의 시니어 머신러닝 엔지니어**이자 **금융 리스크 모델링 전문가**입니다. 다음 전문성을 갖추고 있습니다:

- ✅ **학술적 엄밀성**: Data Leakage, Selection Bias 등 ML 함정 완벽 회피
- ✅ **실무 배포 경험**: 금융권 신용평가 모델 상용화 경험 다수
- ✅ **불균형 데이터 전문가**: SMOTE, Class Weight, Threshold Optimization 마스터
- ✅ **모델 해석력**: SHAP, Feature Importance, Business Impact 분석 능력
- ✅ **코드 품질**: Production-ready, Reproducible, Well-documented

---

## 📋 CONTEXT & BACKGROUND

### 프로젝트 정보
- **주제**: 한국 기업 부도 예측 모델 (50,000+ 기업, 170+ 변수)
- **데이터 특성**:
  - 극심한 불균형 (부도율 ~1.5%, 비율 1:20)
  - 시계열이 아닌 횡단면 데이터 (2021년 8월 스냅샷)
  - 도메인 기반 Feature Engineering 완료 (65개 특성)
- **기존 작업**:
  - Part 1: 문제 정의 및 EDA 완료
  - Part 2: 도메인 기반 특성 공학 완료 (7개 카테고리 65개 특성)
  - Part 3 초안: 모델링 완료했으나 **치명적 Data Leakage 발견**

### 현재 Part 3 초안의 치명적 문제점

#### 🚨 Critical Issue 1: Test Set Leakage (가장 심각)
```python
# ❌ 잘못된 로직 (현재 코드)
# 1. Test set으로 모델 선택
single_best_metrics = evaluate_model(best_model, X_test, y_test)
voting_metrics = evaluate_model(voting_clf, X_test, y_test)
if pr_auc_diff > ENSEMBLE_THRESHOLD:
    final_model = voting_clf  # Test set 성능으로 선택!

# 2. Test set으로 임계값 최적화
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_test)
optimal_threshold = thresholds[np.argmax(f2_scores)]  # Test set에서 최적화!
```

**Impact**:
- 현재 보고된 성능은 신뢰할 수 없음
- 실제 배포 시 성능 5-10% 하락 예상
- 학술적으로 무효한 평가

#### 🚨 Critical Issue 2: Validation Set 부재
```
현재 구조:  Data → Train (80%) → Test (20%)
                    ↓ (5-Fold CV)
                  여기서만 검증

올바른 구조: Data → Train (60%) → Validation (20%) → Test (20%)
                     ↓ (CV)        ↓ (의사결정)      ↓ (최종평가만)
                   학습/튜닝      모델선택,임계값      보고용
```

#### ⚠️ Major Issue 3: Top 3 재학습 시 리샘플링 재적용 문제
- CV에서 찾은 최적 파라미터로 전체 Train set 재학습
- 하지만 SMOTE 등 리샘플링도 다시 랜덤하게 적용됨
- CV 성능 ≠ 재학습 후 성능 (재현성 문제)

#### ⚠️ Major Issue 4: Weighted Voting의 무의미한 가중치
```python
# 문제: CV 점수 차이가 미미
weights = [0.1601, 0.1598, 0.1595]  # 0.0003 차이
# 정규화 후: [0.333, 0.333, 0.333] ← 거의 균등 가중치
```

#### ⚠️ Design Flaw 5: RandomizedSearchCV 100회 부족
- 탐색 공간: 5개 모델 × 5개 리샘플링 × 수십 개 하이퍼파라미터 = 수만 조합
- 100회 샘플링 = 약 0.9% 커버리지 (턱없이 부족)

---

## 🎯 MISSION & OBJECTIVES

### Primary Mission
**학술적으로 완벽하고, 실무에서 즉시 배포 가능한 `발표_Part3_모델링_및_최적화_완전판.ipynb` 생성**

### Success Criteria
1. ✅ **Zero Data Leakage**: Train/Validation/Test 완벽 분리
2. ✅ **Reproducible**: random_state 통제, 실행 시마다 동일 결과
3. ✅ **Statistically Rigorous**: 모든 선택에 통계적 근거
4. ✅ **Business-Ready**: Traffic Light 시스템, ROI 분석, 배포 가이드
5. ✅ **Explainable**: 단일 최적 모델 선택 (SHAP 분석 대비)

---

## 📝 DETAILED TASK SPECIFICATION

### 노트북 구조 및 필수 구현 사항

#### Section 0: 설정 및 임포트
```python
# 필수 상수
RANDOM_STATE = 42
TEST_SIZE = 0.2      # 최종 Test set
VAL_SIZE = 0.25      # Train의 25% → 전체의 20%
CV_FOLDS = 5
N_ITER_PER_MODEL = 200  # 모델당 200회 (총 1000회)

# 평가 지표 우선순위
PRIMARY_METRIC = 'PR-AUC'  # 불균형 데이터 핵심 지표
SECONDARY_METRIC = 'F2-Score'  # Recall 중시
BUSINESS_CONSTRAINT = 'Type II Error < 20%'  # 부도 미탐지율
```

#### Section 1: 데이터 로딩 및 3-Way Split
```python
# ✅ 올바른 데이터 분할
# 1단계: Test set 분리 (20%) - 절대 건드리지 않음
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# 2단계: Train/Validation 분리 (60%/20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VAL_SIZE, stratify=y_temp, random_state=RANDOM_STATE
)

# 최종 비율: Train 60% / Validation 20% / Test 20%
print(f"Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"부도율 - Train: {y_train.mean():.3%}, Val: {y_val.mean():.3%}, Test: {y_test.mean():.3%}")
```

#### Section 2: 전처리 파이프라인 (개선)
```python
# ✅ 순서 최적화 및 Winsorizer 조정
def create_preprocessing_pipeline():
    return Pipeline([
        ('imputer', IterativeImputer(max_iter=20, random_state=RANDOM_STATE, verbose=0)),  # 먼저 결측치 처리
        ('winsorizer', Winsorizer(lower=0.005, upper=0.995)),  # 0.5%~99.5%로 완화 (극단값 보존)
        ('log_transformer', LogTransformer()),
        ('scaler', StandardScaler())
    ])
```

**주요 변경점**:
- `IterativeImputer` max_iter 10→20 (수렴 안정성)
- `Winsorizer` 1%~99% → 0.5%~99.5% (부도 예측에서 극단값은 중요 시그널)
- 순서: Imputer → Winsorizer → Log → Scaler

#### Section 3: 리샘플링 전략 비교 실험
```python
# ✅ 리샘플링 vs Class Weight 명확한 대조군 설정
resampling_strategies = {
    'baseline': None,  # 리샘플링 없음 (Class Weight만)
    'smote': SMOTE(random_state=RANDOM_STATE),
    'borderline_smote': BorderlineSMOTE(random_state=RANDOM_STATE),
    'smote_tomek': SMOTETomek(random_state=RANDOM_STATE),
    'adasyn': ADASYN(random_state=RANDOM_STATE)
}

# 전략별 성능 비교 (Train set + CV)
# → Validation set에서 최종 선택
```

#### Section 4: 하이퍼파라미터 튜닝 (개선)
```python
# ✅ 모델별 개별 튜닝 (각 200회)
models_to_tune = {
    'LightGBM': (LGBMClassifier, lgbm_param_dist),
    'XGBoost': (XGBClassifier, xgb_param_dist),
    'CatBoost': (CatBoostClassifier, cat_param_dist),
    'RandomForest': (RandomForestClassifier, rf_param_dist),  # 다양성 확보
    'LogisticRegression': (LogisticRegression, lr_param_dist)  # 이종 모델
}

all_results = []
for model_name, (model_class, param_dist) in models_to_tune.items():
    print(f"\n{'='*60}")
    print(f"🔧 {model_name} 튜닝 시작 (200회 탐색)")
    print(f"{'='*60}")

    search = RandomizedSearchCV(
        create_pipeline(model_class),
        param_distributions=param_dist,
        n_iter=N_ITER_PER_MODEL,  # 200회
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring='average_precision',  # PR-AUC
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )

    search.fit(X_train, y_train)
    all_results.append({
        'model': model_name,
        'best_params': search.best_params_,
        'cv_score': search.best_score_,
        'best_estimator': search.best_estimator_
    })
```

**주요 변경점**:
- 100회 → 모델당 200회 (총 1,000회)
- RandomForest, LogisticRegression 추가 (앙상블 다양성 확보)
- CV 결과를 DataFrame으로 체계적 저장

#### Section 5: 모델 선택 (Validation Set 활용) ⭐ 핵심!
```python
# ✅ Validation set으로 모델 선택 (Test set 절대 사용 금지!)
print("\n" + "="*80)
print("📊 Validation Set 기반 모델 평가 및 선택")
print("="*80)

# Top 5 모델을 Validation set에서 평가
top5_models = sorted(all_results, key=lambda x: x['cv_score'], reverse=True)[:5]

val_scores = []
for result in top5_models:
    model = result['best_estimator']
    y_prob_val = model.predict_proba(X_val)[:, 1]

    val_metrics = {
        'model': result['model'],
        'cv_score': result['cv_score'],
        'val_pr_auc': average_precision_score(y_val, y_prob_val),
        'val_roc_auc': roc_auc_score(y_val, y_prob_val)
    }
    val_scores.append(val_metrics)

val_df = pd.DataFrame(val_scores).sort_values('val_pr_auc', ascending=False)
print(val_df)

# 최고 성능 단일 모델 선택
best_single_model = top5_models[0]['best_estimator']
best_single_val_score = val_df.iloc[0]['val_pr_auc']
```

#### Section 6: 앙상블 구성 및 통계적 검증
```python
# ✅ Top 3 이종 모델로 앙상블 (다양성 확보)
# GBM 계열만 선택하지 않고, 상관계수 낮은 모델 조합
top3_diverse = select_diverse_models(top5_models, n_models=3)

# Exponential Weighting (점수 차이 강조)
cv_scores = np.array([m['cv_score'] for m in top3_diverse])
cv_scores_norm = (cv_scores - cv_scores.min()) / (cv_scores.max() - cv_scores.min() + 1e-10)
weights = np.exp(cv_scores_norm * 5)
weights = weights / weights.sum()

voting_clf = VotingClassifier(
    estimators=[(f"model_{i}", m['best_estimator']) for i, m in enumerate(top3_diverse)],
    voting='soft',
    weights=weights
)
voting_clf.fit(X_train, y_train)

# Validation set에서 앙상블 평가
y_prob_val_voting = voting_clf.predict_proba(X_val)[:, 1]
voting_val_score = average_precision_score(y_val, y_prob_val_voting)

# ✅ 통계적 유의성 검증 (McNemar's test 또는 Paired t-test)
from scipy.stats import wilcoxon

# CV fold별 점수로 paired test (간접 비교)
# 또는 Validation set에서 bootstrap CI 계산
print(f"\n단일 모델 Val PR-AUC: {best_single_val_score:.4f}")
print(f"앙상블 Val PR-AUC: {voting_val_score:.4f}")
print(f"차이: {voting_val_score - best_single_val_score:.4f}")

# 최종 선택 (실무적 판단: 복잡도 vs 성능)
if voting_val_score > best_single_val_score + 0.005:  # 0.5% 이상 개선
    print("\n✅ 앙상블 선택 (유의미한 성능 향상)")
    final_model = voting_clf
else:
    print("\n✅ 단일 모델 선택 (복잡도 대비 성능 향상 미미, SHAP 분석 용이)")
    final_model = best_single_model
```

#### Section 7: 임계값 최적화 (Validation Set) ⭐ 핵심!
```python
# ✅ Validation set에서 F2-optimal threshold 찾기
print("\n" + "="*80)
print("🎯 Validation Set 기반 임계값 최적화 (F2-Score)")
print("="*80)

y_prob_val_final = final_model.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob_val_final)

# F2-Score 계산 (Recall에 2배 가중치)
f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
f2_optimal_idx = np.argmax(f2_scores)
optimal_threshold_f2 = thresholds[f2_optimal_idx]

print(f"F2-optimal Threshold: {optimal_threshold_f2:.4f}")
print(f"해당 지점 - Precision: {precisions[f2_optimal_idx]:.3f}, Recall: {recalls[f2_optimal_idx]:.3f}")

# Type II Error 제약 확인 (부도 미탐지 < 20%)
type2_error = 1 - recalls[f2_optimal_idx]
print(f"Type II Error (부도 미탐지율): {type2_error:.1%}")

if type2_error > 0.20:
    print("⚠️ Type II Error가 20%를 초과합니다. Recall 80% 지점으로 조정합니다.")
    recall_80_idx = np.argmin(np.abs(recalls - 0.80))
    optimal_threshold = thresholds[recall_80_idx]
else:
    optimal_threshold = optimal_threshold_f2

print(f"\n✅ 최종 선택된 Threshold: {optimal_threshold:.4f}")
```

#### Section 8: Traffic Light 시스템 (개선)
```python
# ✅ 데이터 기반 Yellow 구간 설정
# Red: F2-optimal (Type II < 20%)
# Yellow: Recall 95% 커버 지점 (부도 기업의 95%를 최소한 경고)

red_threshold = optimal_threshold
recall_95_idx = np.argmin(np.abs(recalls - 0.95))
yellow_threshold = thresholds[recall_95_idx]

print(f"🚦 Traffic Light 시스템:")
print(f"  🔴 Red (고위험):   {red_threshold:.4f} 이상")
print(f"  🟡 Yellow (경계): {yellow_threshold:.4f} ~ {red_threshold:.4f}")
print(f"  🟢 Green (안전):  {yellow_threshold:.4f} 미만")
print(f"\n해석: Yellow 등급은 부도 기업의 95%를 커버하는 방어선입니다.")
```

#### Section 9: Test Set 최종 평가 (단 한 번!) ⭐ 핵심!
```python
# ✅ Test set은 최종 보고용으로만 사용 (의사결정 금지!)
print("\n" + "="*80)
print("📊 TEST SET 최종 평가 (Hold-out Performance)")
print("="*80)
print("⚠️ 주의: 이 결과는 unseen data 성능 추정치입니다.")
print("         모델 수정이나 재튜닝에 사용하지 마세요!\n")

y_prob_test = final_model.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= optimal_threshold).astype(int)

# 종합 평가
test_metrics = {
    'PR-AUC': average_precision_score(y_test, y_prob_test),
    'ROC-AUC': roc_auc_score(y_test, y_prob_test),
    'F2-Score': fbeta_score(y_test, y_pred_test, beta=2),
    'Recall': recall_score(y_test, y_pred_test),
    'Precision': precision_score(y_test, y_pred_test),
    'Type II Error': 1 - recall_score(y_test, y_pred_test)
}

print("최종 Test Set 성능:")
for metric, value in test_metrics.items():
    print(f"  {metric:20s}: {value:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print(f"\nConfusion Matrix:")
print(cm)
print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
```

#### Section 10: 모델 저장 및 재현성 보장
```python
# ✅ 모델, 전처리, 임계값 모두 저장
import joblib

model_artifacts = {
    'model': final_model,
    'optimal_threshold': optimal_threshold,
    'red_threshold': red_threshold,
    'yellow_threshold': yellow_threshold,
    'feature_names': X_train.columns.tolist(),
    'test_metrics': test_metrics,
    'random_state': RANDOM_STATE
}

joblib.dump(model_artifacts, '../data/processed/part3_final_model_artifacts.pkl')
print("✅ 모델 및 설정 저장 완료: part3_final_model_artifacts.pkl")
```

#### Section 11: 시각화 (한글 폰트 보장)
```python
# ✅ 한글 폰트 설정 (CLAUDE.md 규칙 준수)
import platform
import matplotlib.pyplot as plt

if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)

# Precision-Recall Curve (Validation + Test 비교)
# Confusion Matrix Heatmap
# Traffic Light 분포 히스토그램
# ... (기존 시각화 코드 활용)
```

---

## 🔒 CONSTRAINTS & REQUIREMENTS

### 필수 준수 사항 (절대적)
1. ✅ **Zero Data Leakage**: Test set은 Section 9에서 단 한 번만 사용
2. ✅ **Validation Set 필수**: 모든 의사결정은 Validation set 기반
3. ✅ **Reproducibility**: 모든 random_state = 42 통일
4. ✅ **한글 폰트**: 모든 시각화에서 한글 깨짐 방지 (CLAUDE.md 규칙)
5. ✅ **파일 경로**: `../data/` 형식 사용 (notebooks/ 기준)

### 코딩 규칙 (CLAUDE.md 준수)
```python
# ✅ 범주형 변수 처리
if '위험경보등급' in X.columns:
    X['위험경보등급'] = X['위험경보등급'].cat.codes

# ✅ 결측치 및 무한대 처리
X_filled = X.fillna(X.median())
X_filled = X_filled.replace([np.inf, -np.inf], 0)

# ✅ 데이터 인코딩
df = pd.read_csv('../data/기업신용평가정보_210801.csv', encoding='utf-8')
df.to_csv('../data/features/xxx.csv', encoding='utf-8-sig')
```

### 평가 지표 우선순위 (명시)
```python
# 모든 평가 결과는 다음 순서로 보고
metrics_priority = [
    'PR-AUC',        # Primary (불균형 데이터 핵심)
    'F2-Score',      # Secondary (Recall 중시)
    'Type II Error', # Business Constraint (< 20%)
    'Recall',        # 부도 탐지율
    'Precision',     # 오탐률
    'ROC-AUC'        # 참고용
]
```

---

## 📊 OUTPUT FORMAT & DELIVERABLES

### 노트북 구조 (최종)
```
발표_Part3_모델링_및_최적화_완전판.ipynb

├── Section 0: 환경 설정 및 임포트
├── Section 1: 데이터 로딩 및 3-Way Split ⭐
├── Section 2: 전처리 파이프라인 정의 (개선)
├── Section 3: 리샘플링 전략 비교 실험
├── Section 4: 하이퍼파라미터 튜닝 (모델별 200회)
├── Section 5: 모델 선택 (Validation Set) ⭐
├── Section 6: 앙상블 구성 및 통계적 검증
├── Section 7: 임계값 최적화 (Validation Set) ⭐
├── Section 8: Traffic Light 시스템 (데이터 기반)
├── Section 9: Test Set 최종 평가 (단 한 번!) ⭐
├── Section 10: 모델 저장 및 재현성 보장
├── Section 11: 종합 시각화 (한글 폰트 보장)
└── Section 12: 결론 및 배포 가이드
```

### 필수 산출물
1. **노트북 파일**: `발표_Part3_모델링_및_최적화_완전판.ipynb`
2. **모델 파일**: `../data/processed/part3_final_model_artifacts.pkl`
3. **실행 결과**: Test set 성능 메트릭 (신뢰할 수 있는)
4. **시각화**: PR Curve, CM, Traffic Light 분포 (한글 깨짐 없음)

### 문서화 요구사항
- 각 Section 시작 시 **마크다운 설명** (무엇을, 왜, 어떻게)
- 코드 주석: `# ✅ 올바른 방법`, `# ❌ 피해야 할 방법` 명시
- 주요 의사결정 지점에 **근거** 명시 (예: "Validation PR-AUC 기준 선택")

---

## 💡 EXAMPLES & ANTI-PATTERNS

### ✅ 올바른 패턴
```python
# 1. 3-way split
X_train, X_val, X_test = ...  # 60/20/20

# 2. Train set으로 CV 튜닝
search.fit(X_train, y_train)

# 3. Validation set으로 모델 선택
val_score = model.score(X_val, y_val)

# 4. Validation set으로 임계값 최적화
optimal_threshold = find_threshold(X_val, y_val)

# 5. Test set은 최종 보고만
test_score = model.score(X_test, y_test)  # 끝!
```

### ❌ 절대 금지 패턴
```python
# 1. Test set으로 모델 선택 (금지!)
if test_score_A > test_score_B:
    final_model = model_A  # ← Data Leakage!

# 2. Test set으로 임계값 최적화 (금지!)
threshold = optimize_f2(y_test, y_prob_test)  # ← Leakage!

# 3. Test set 보고 모델 재튜닝 (금지!)
# "Test에서 성능 낮네? 파라미터 수정하자" ← 절대 안 됨!
```

---

## 🚀 EXECUTION INSTRUCTIONS

### Step-by-Step 실행 가이드

1. **기존 Part 3 노트북 읽기**
   - `notebooks/발표_Part3_모델링_및_최적화.ipynb` 전체 코드 파악
   - 재사용 가능한 함수/클래스 식별 (Winsorizer, LogTransformer 등)

2. **새 노트북 생성**
   - 파일명: `notebooks/발표_Part3_모델링_및_최적화_완전판.ipynb`
   - 첫 셀에 명시: "⚠️ 이 노트북은 Data Leakage를 완전히 제거한 개선판입니다"

3. **Section별 구현**
   - 위 TASK SPECIFICATION의 구조를 정확히 따름
   - 각 Section마다 마크다운 설명 → 코드 → 결과 검증

4. **검증 체크리스트**
   ```python
   # 노트북 완성 후 다음 질문에 모두 "Yes"인지 확인
   checklist = {
       "Test set은 Section 9에서만 사용했는가?": "Yes/No",
       "모든 의사결정은 Validation set 기반인가?": "Yes/No",
       "random_state=42가 모든 곳에 설정되었는가?": "Yes/No",
       "한글 폰트가 모든 시각화에 적용되었는가?": "Yes/No",
       "Type II Error < 20% 제약을 확인했는가?": "Yes/No"
   }
   ```

5. **실행 및 저장**
   - 전체 셀 순차 실행 (Kernel Restart → Run All)
   - 에러 없이 완료 확인
   - `_executed.ipynb` 버전도 저장 (CLAUDE.md 규칙)

---

## 🎓 QUALITY ASSURANCE

### 학술적 엄밀성 체크
- [ ] Data Leakage 완전 제거
- [ ] Statistical significance test 수행
- [ ] Baseline 대조군 설정 (리샘플링 없는 버전)
- [ ] Cross-validation 올바른 사용

### 실무 배포 준비도 체크
- [ ] Reproducible (random_state 통제)
- [ ] 모델 artifacts 저장 (.pkl)
- [ ] Traffic Light 시스템 비즈니스 로직
- [ ] 한글 폰트 깨짐 방지

### 코드 품질 체크
- [ ] CLAUDE.md 규칙 100% 준수
- [ ] 주석 및 문서화 충분
- [ ] 매직 넘버 없음 (모두 상수화)
- [ ] 하드코딩 없음

---

## 🔥 FINAL REMINDER

이 노트북은 **포트폴리오 및 학술적 평가**에 사용될 것입니다.

다음 질문에 답할 수 있어야 합니다:
1. **"Test set을 어디에 사용했나요?"** → "Section 9에서 최종 보고만 했습니다."
2. **"모델과 임계값은 어떻게 선택했나요?"** → "Validation set 기반으로 선택했습니다."
3. **"재현 가능한가요?"** → "네, random_state=42로 통제했습니다."
4. **"실무 배포 가능한가요?"** → "네, Traffic Light와 artifacts 저장했습니다."

---

**이제 시작하세요. 당신은 할 수 있습니다. 🚀**
