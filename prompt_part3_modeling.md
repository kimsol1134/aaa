# 발표용 Part 3: 모델링 및 최적화 노트북 생성 프롬프트

당신은 한국 기업 부도 예측 프로젝트의 시니어 데이터 사이언티스트입니다. Part 2에서 생성한 도메인 기반 특성을 활용하여 **발표용 Part 3: 모델링 및 최적화** 노트북을 생성하는 것이 목표입니다.

---

## 📋 프로젝트 컨텍스트

### 프로젝트 개요
- **목표**: 한국 기업 부도 위험을 3개월~1년 전에 예측하는 AI 모델 개발
- **데이터**: 50,105개 기업, 부도율 1.51% (심각한 불균형)
- **핵심 과제**: 불균형 데이터 처리, 도메인 지식 반영, 해석 가능성

### Part 1-2 완료 사항

**Part 1: 문제 정의 및 탐색적 분석**
- 유동성이 가장 강력한 예측 변수 발견 (유동비율, 당좌비율, 현금비율)
- 업종별 부도율 2배 차이 (건설업 2.8% vs 금융업 0.9%)
- 외감 여부가 부도율에 영향

**Part 2: 도메인 특성 공학 완료 (실제 출력 결과)**
- **생성된 특성**: 52개 (유동성 10개, 지급불능 11개, 재무조작 15개, 이해관계자 10개, 한국시장 6개)
- **Feature Validation**: 49개 특성 검증 완료 (Mann-Whitney U test, Cliff's Delta, AUC)
- **VIF 다중공선성 분석**: 19개 고VIF 특성 발견 → 14개 제거, 5개 경고와 함께 유지
- **IV 기반 특성 선택**: IV > 0.02 기준으로 필터링
- **최종 선택된 특성**: 27개 특성 (다중공선성 제거 + 예측력 검증 완료)
- **출력 파일**: `data/features/domain_based_features_완전판.csv`

**Part 2 핵심 발견**:
- Beneish M-Score 완전 구현 (15개 재무조작 탐지 특성)
- 유동성 특성이 가장 높은 AUC (현금소진일수, 즉각지급능력)
- 이해관계자 행동 특성(연체, 신용등급)이 강한 예측력

---

## 🎯 Part 3 노트북 생성 요구사항

### 📂 입력 데이터
- **파일 경로**: `../data/features/domain_based_features_완전판.csv`
- **데이터 구조**:
  - 타겟: `모형개발용Performance(향후1년내부도여부)` (0: 정상, 1: 부도)
  - 특성: 27개 도메인 기반 특성 (Part 2에서 선택됨)
  - 크기: 50,105 rows

### 🛠️ 필수 구현 단계

#### 1️⃣ 환경 설정 및 데이터 로딩
```python
# 필수 라이브러리
- pandas, numpy, matplotlib, seaborn, plotly
- scikit-learn (train_test_split, RandomizedSearchCV, StratifiedKFold)
- imblearn (ImbPipeline, SMOTE, BorderlineSMOTE, RandomUnderSampler, SMOTETomek)
- lightgbm, xgboost, catboost
- BalancedRandomForestClassifier, VotingClassifier

# 한글 폰트 설정 (macOS: AppleGothic, Windows: Malgun Gothic, Linux: NanumGothic)
# UTF-8 인코딩 확인
```

**데이터 로딩 후 확인사항**:
- 데이터 shape 출력
- 부도율 확인 (약 1.51% 예상)
- 결측치 확인
- Train/Test split (80:20, stratify=y, random_state=42)

---

#### 2️⃣ 전처리 파이프라인 구축

**ImbPipeline 6단계 구조** (반드시 이 순서대로):

```python
from imblearn.pipeline import Pipeline as ImbPipeline

pipeline = ImbPipeline([
    ('inf_handler', InfiniteHandler()),         # 1. 무한대 값 처리
    ('winsorizer', Winsorizer(0.01, 0.99)),    # 2. 이상치 제어 (1%~99% 분위수)
    ('log_transformer', LogTransformer()),      # 3. 로그 변환 (양수만)
    ('imputer', IterativeImputer(max_iter=10)), # 4. 결측치 보간
    ('scaler', RobustScaler()),                 # 5. 스케일링
    ('resampler', 'passthrough'),               # 6. 리샘플링 (후보군에서 선택)
    ('classifier', LogisticRegression())        # 7. 분류기
])
```

**전처리 클래스 정의 필요**:
- `InfiniteHandler`: np.inf, -np.inf → np.nan
- `Winsorizer`: 1%~99% 분위수로 클리핑
- `LogTransformer`: 양수 컬럼에만 np.log1p 적용

---

#### 3️⃣ 리샘플링 전략 (5가지)

**반드시 포함해야 할 리샘플링 전략**:

```python
resampler_list = [
    'passthrough',                                      # 1. 리샘플링 없음 (베이스라인)
    SMOTE(sampling_strategy=0.2, random_state=42),     # 2. SMOTE (부도 기업 20%까지 증강)
    BorderlineSMOTE(sampling_strategy=0.2, random_state=42),  # 3. 경계선 샘플 중심
    RandomUnderSampler(sampling_strategy=0.3, random_state=42), # 4. 언더샘플링
    SMOTETomek(sampling_strategy=0.2, random_state=42) # 5. SMOTE + Tomek Links (추가 요청)
]
```

**리샘플링 비율 근거**:
- `sampling_strategy=0.2`: 부도율 1.5% → 20%로 증강 (과생성 방지)
- `sampling_strategy=0.3`: 언더샘플링은 30%까지만 (데이터 손실 최소화)

---

#### 4️⃣ AutoML: RandomizedSearchCV

**5개 모델 × 하이퍼파라미터 그리드**:

```python
param_grid = [
    # 1. LightGBM (자동 불균형 처리)
    {
        'resampler': resampler_list,
        'classifier': [lgb.LGBMClassifier(random_state=42, verbose=-1)],
        'classifier__n_estimators': [300, 500, 1000],
        'classifier__learning_rate': [0.01, 0.02, 0.05],
        'classifier__num_leaves': [31, 63, 127],
        'classifier__max_depth': [-1, 10, 20],
        'classifier__subsample': [0.7, 0.9],
        'classifier__reg_alpha': [0.1, 0.5],
        'classifier__reg_lambda': [0.1, 0.5],
        'classifier__is_unbalance': [True]
    },

    # 2. XGBoost
    {
        'resampler': resampler_list,
        'classifier': [xgb.XGBClassifier(random_state=42, eval_metric='logloss')],
        'classifier__n_estimators': [300, 500],
        'classifier__max_depth': [4, 6, 8],
        'classifier__learning_rate': [0.01, 0.05],
        'classifier__gamma': [0, 0.1, 0.5],
        'classifier__subsample': [0.7, 0.9],
        'classifier__reg_alpha': [0.1, 1.0],
        'classifier__scale_pos_weight': [1, 8, 66]  # sqrt_ratio, scale_ratio
    },

    # 3. CatBoost
    {
        'resampler': resampler_list,
        'classifier': [CatBoostClassifier(random_state=42, verbose=0)],
        'classifier__iterations': [500, 1000],
        'classifier__learning_rate': [0.01, 0.03, 0.1],
        'classifier__depth': [4, 6, 8],
        'classifier__l2_leaf_reg': [3, 5, 9],
        'classifier__auto_class_weights': ['Balanced', 'SqrtBalanced']
    },

    # 4. BalancedRandomForest
    {
        'resampler': resampler_list,
        'classifier': [BalancedRandomForestClassifier(random_state=42, n_jobs=-1)],
        'classifier__n_estimators': [300, 500],
        'classifier__max_depth': [10, 20, None],
        'classifier__max_features': ['sqrt', 'log2']
    },

    # 5. LogisticRegression (베이스라인)
    {
        'resampler': resampler_list,
        'classifier': [LogisticRegression(random_state=42, max_iter=1000)],
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__class_weight': ['balanced', None]
    }
]
```

**RandomizedSearchCV 설정**:

```python
search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=100,                          # 100회 랜덤 샘플링
    scoring='average_precision',         # PR-AUC 최적화
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1,
    random_state=42
)
```

**실행 후 출력**:
- 최적 모델 이름 (예: XGBClassifier, LGBMClassifier 등)
- 최적 리샘플링 전략
- 최적 하이퍼파라미터
- CV PR-AUC 점수
- 소요 시간

---

#### 5️⃣ Weighted Voting 앙상블

**Top 3 모델 기반 앙상블 구축**:

```python
# 1. AutoML 결과에서 상위 3개 추출
results_df = pd.DataFrame(search.cv_results_)
top3 = results_df.sort_values('mean_test_score', ascending=False).head(3)

# 2. VotingClassifier 구성
voting_clf = VotingClassifier(
    estimators=[
        ('Top1_ModelName', pipeline_1),
        ('Top2_ModelName', pipeline_2),
        ('Top3_ModelName', pipeline_3)
    ],
    voting='soft',                      # 확률 평균
    weights=[score1, score2, score3],   # CV PR-AUC 기반 가중치
    n_jobs=-1
)
```

**출력**:
- Top 3 모델 이름 및 CV 점수 (표 형식)
- 가중치 계산 방식 설명

---

#### 6️⃣ 최종 모델 선정 (Test Set 평가)

**성능 비교**:

```python
def evaluate_model(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        'PR-AUC': average_precision_score(y_test, y_prob),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'F1-Score': f1_score(y_test, (y_prob >= 0.5).astype(int))
    }

# Single Best vs Weighted Voting 비교
results = pd.DataFrame([
    evaluate_model(best_single_model, X_test, y_test),
    evaluate_model(voting_clf, X_test, y_test)
], index=['Single Best', 'Weighted Voting'])
```

**최종 모델 선정 로직**:
- PR-AUC가 높은 쪽 선택
- 단, 앙상블이 단일 모델보다 0.5% 이상 우수해야 선택 (복잡도 고려)
- 유지보수 용이성도 고려

---

#### 7️⃣ Traffic Light 시스템 (부도 위험 3등급 분류)

**임계값 결정 방법** (모델 결과 기반 최적화):

**Step 1: F1-Score 기반 최적 임계값 탐색**
```python
from sklearn.metrics import precision_recall_curve, f1_score

# PR Curve에서 모든 임계값 테스트
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# F1-Score 계산
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
optimal_threshold = thresholds[np.argmax(f1_scores)]

print(f"F1-Score 최적 임계값: {optimal_threshold:.4f}")
```

**Step 2: 비즈니스 요구사항 반영**
```python
# 목표: Recall >= 60% 보장 (부도 기업의 60% 이상 탐지)
target_recall = 0.6

# Recall >= 60%를 만족하는 최대 Precision 임계값 찾기
idx = np.where(recalls >= target_recall)[0]
if len(idx) > 0:
    business_threshold = thresholds[idx[np.argmax(precisions[idx])]]
    print(f"Recall 60% 보장 임계값: {business_threshold:.4f}")
```

**Step 3: Traffic Light 임계값 설정**
```python
# Red 임계값: 비즈니스 임계값 또는 상위 5% 백분위수
red_threshold = max(business_threshold, np.percentile(y_prob, 95))

# Yellow 임계값: Red의 40% 수준 또는 상위 15% 백분위수
yellow_threshold = max(red_threshold * 0.4, np.percentile(y_prob, 85))

print(f"\n최종 임계값:")
print(f"  Yellow (주의): >= {yellow_threshold:.4f}")
print(f"  Red (위험):    >= {red_threshold:.4f}")

# 분류 함수
def traffic_light_classification(y_prob, yellow_th, red_th):
    conditions = [
        (y_prob >= red_th),      # Red
        (y_prob >= yellow_th)    # Yellow
    ]
    choices = ['Red (위험)', 'Yellow (주의)']
    return np.select(conditions, choices, default='Green (안전)')

grades = traffic_light_classification(y_prob_test, yellow_threshold, red_threshold)
```

**참고 임계값 (04 노트북 결과)**:
- Yellow: 0.02 (2%), Red: 0.05 (5%)
- 하지만 모델 성능에 따라 조정 필요 (위 로직 사용)

**출력해야 할 통계**:

| 등급 | 기업 수 (비율) | 실제 부도 수 | 정밀도 (Precision) | 부도 포착률 (Recall 기여) |
|------|---------------|--------------|-------------------|------------------------|
| 🔴 Red (위험) | ??? (??%) | ??? | ??% | ??% |
| 🟡 Yellow (주의) | ??? (??%) | ??? | ??% | ??% |
| 🟢 Green (안전) | ??? (??%) | ??? | ??% | ??% |
| **합계** | 10,021 (100%) | ??? | - | **리스크 방어율: ??%** |

**리스크 방어율**: (Red + Yellow에서 포착한 부도 수) / 전체 부도 수 × 100

**임계값 결정 기준**:
- ✅ F1-Score 최적화
- ✅ Recall >= 60% 보장 (비즈니스 요구)
- ✅ 상위 5~15% 기업을 집중 관리 대상으로 선정

---

#### 8️⃣ 시각화 (필수 포함)

**1. Feature Importance (Top 15)**
- Plotly 가로 바 차트
- 중요도 점수와 특성명 표시

**2. PR-AUC Curve**
- Precision-Recall 곡선
- 현재 Red 임계값 위치 빨간 점 표시 (동적 임계값 사용)
- Yellow 임계값 위치 노란 점 표시
- AUC 값 제목에 표시

**3. Confusion Matrix (Red 임계값 기준)**
- Plotly Heatmap
- TN, FP, FN, TP 명확히 표기
- 제목에 사용된 임계값 표시

**4. 예측 확률 분포**
- 정상 기업(초록색) vs 부도 기업(빨간색) 히스토그램
- 로그 스케일 사용 (불균형 데이터)

**5. Traffic Light 시각화**
- 도넛 차트: 등급별 기업 비중
- 막대 차트: 등급별 실제 부도율

**6. Cumulative Gains Curve**
- 상위 N% 심사 시 부도 포착 비율
- 랜덤 기준선과 비교

---

#### 9️⃣ 모델 저장 (Part 4 SHAP 분석 대비)

**중요**: Part 4에서 SHAP 분석을 수행하므로, SHAP이 잘 작동하도록 모델을 저장해야 합니다.

```python
import joblib
import os

save_dir = '../data/processed/'
os.makedirs(save_dir, exist_ok=True)

# 1. 최종 모델 (전체 파이프라인 포함)
joblib.dump(final_model, os.path.join(save_dir, '발표_Part3_최종모델.pkl'))
print(f"✅ 전체 파이프라인 저장: 발표_Part3_최종모델.pkl")

# 2. 분류기만 (SHAP 분석용) ⭐ 중요!
if hasattr(final_model, 'named_steps'):
    classifier_only = final_model.named_steps['classifier']
else:
    classifier_only = final_model

joblib.dump(classifier_only, os.path.join(save_dir, '발표_Part3_분류기.pkl'))
print(f"✅ 분류기만 저장: 발표_Part3_분류기.pkl (SHAP용)")

# 3. 전처리된 데이터 저장 (SHAP 분석용) ⭐ 중요!
# SHAP은 전처리된 데이터가 필요하므로 Train/Test 데이터를 전처리 후 저장
from sklearn.pipeline import Pipeline

# 전처리만 수행하는 파이프라인 생성
preprocessing_only = Pipeline([
    ('inf_handler', final_model.named_steps['inf_handler']),
    ('winsorizer', final_model.named_steps['winsorizer']),
    ('log_transformer', final_model.named_steps['log_transformer']),
    ('imputer', final_model.named_steps['imputer']),
    ('scaler', final_model.named_steps['scaler'])
])

# 전처리 적용
X_train_processed = preprocessing_only.transform(X_train)
X_test_processed = preprocessing_only.transform(X_test)

# 데이터프레임으로 변환 (컬럼명 유지)
X_train_processed_df = pd.DataFrame(X_train_processed, columns=X_train.columns)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=X_test.columns)

# 저장
X_train_processed_df.to_csv(os.path.join(save_dir, '발표_Part3_X_train_processed.csv'),
                             index=False, encoding='utf-8-sig')
X_test_processed_df.to_csv(os.path.join(save_dir, '발표_Part3_X_test_processed.csv'),
                            index=False, encoding='utf-8-sig')
y_train.to_csv(os.path.join(save_dir, '발표_Part3_y_train.csv'),
               index=False, encoding='utf-8-sig', header=['target'])
y_test.to_csv(os.path.join(save_dir, '발표_Part3_y_test.csv'),
              index=False, encoding='utf-8-sig', header=['target'])

print(f"✅ 전처리된 Train/Test 데이터 저장 (SHAP 분석용)")

# 4. 임계값 저장 (Traffic Light 시스템용)
thresholds = {
    'yellow_threshold': yellow_threshold,
    'red_threshold': red_threshold,
    'optimal_f1_threshold': optimal_threshold
}
joblib.dump(thresholds, os.path.join(save_dir, '발표_Part3_임계값.pkl'))
print(f"✅ Traffic Light 임계값 저장")

print("\n" + "="*60)
print("Part 4 SHAP 분석을 위한 필수 파일:")
print("  1. 발표_Part3_분류기.pkl - SHAP explainer 생성용")
print("  2. 발표_Part3_X_train_processed.csv - Background 데이터")
print("  3. 발표_Part3_X_test_processed.csv - 설명 대상 데이터")
print("="*60)
```

**저장되는 파일 목록**:
1. `발표_Part3_최종모델.pkl`: 전체 파이프라인 (배포용)
2. `발표_Part3_분류기.pkl`: 분류기만 (SHAP 분석용) ⭐
3. `발표_Part3_X_train_processed.csv`: 전처리된 학습 데이터 (SHAP background) ⭐
4. `발표_Part3_X_test_processed.csv`: 전처리된 테스트 데이터 (SHAP 설명 대상) ⭐
5. `발표_Part3_y_train.csv`: 학습 타겟
6. `발표_Part3_y_test.csv`: 테스트 타겟
7. `발표_Part3_임계값.pkl`: Traffic Light 임계값

---

## 📊 노트북 구조 (마크다운 섹션)

```markdown
# 📗 발표용 Part 3: 모델링 및 최적화

## 🎯 Part 3 목표 및 이전 Part 요약

### Part 1-2 주요 발견
- [Part 1-2 요약 3-5줄]

### Part 3 목표
1. 불균형 데이터 처리 (5가지 리샘플링 전략)
2. AutoML로 최적 모델 탐색 (100회 랜덤 샘플링)
3. Weighted Voting 앙상블
4. 비즈니스 적용: Traffic Light 시스템

---

## 0. 환경 설정
[라이브러리 import, 한글 폰트 설정]

## 1. 데이터 로딩 및 분할
[Part 2 출력 파일 로딩, Train/Test split]

## 2. 전처리 클래스 정의
[InfiniteHandler, Winsorizer, LogTransformer]

## 3. 불균형 데이터 처리 전략
[ImbPipeline 구조, 5가지 리샘플링 전략 설명]

## 4. AutoML: 하이퍼파라미터 튜닝
[RandomizedSearchCV 실행, 결과 출력]

## 5. Weighted Voting 앙상블
[Top 3 모델 추출, VotingClassifier 구성]

## 6. 최종 모델 선정
[Single Best vs Voting 비교, 승자 선택]

## 7. 모델 성능 평가
[PR-AUC, ROC-AUC, Confusion Matrix]

## 8. Feature Importance 분석
[중요도 Top 15 시각화]

## 9. Traffic Light 시스템
[3등급 분류, 통계 출력, 시각화]

## 10. 비즈니스 임팩트 분석
[Cumulative Gains, 효율성 분석]

## 11. 모델 저장 및 다음 단계
[pkl 파일 저장, Part 4 예고]
```

---

## ✅ 품질 기준 (반드시 준수)

### 코드 품질
- [ ] 모든 셀이 순서대로 실행 가능 (top-to-bottom)
- [ ] 하드코딩 금지 (경로, 임계값 등 변수화)
- [ ] 한글 폰트 설정 완료 (깨짐 없음)
- [ ] UTF-8 인코딩 확인

### 데이터 처리
- [ ] 시계열 의존적 로직 사용 안 함 (횡단면 데이터)
- [ ] Train/Test split 전에 리샘플링 하지 않음 (Leakage 방지)
- [ ] Stratified split으로 부도율 유지

### 모델링
- [ ] 5가지 리샘플링 전략 모두 포함 (SMOTETomek 포함)
- [ ] 5개 모델 모두 테스트
- [ ] PR-AUC를 주요 메트릭으로 사용
- [ ] Weighted Voting 앙상블 구현

### 시각화
- [ ] Plotly 사용 (인터랙티브)
- [ ] 한글 깨짐 없음
- [ ] 6가지 핵심 시각화 모두 포함

### 출력
- [ ] Traffic Light 통계 표 완성
- [ ] 리스크 방어율 계산 및 출력
- [ ] 최종 모델 pkl 파일 저장
- [ ] SHAP 분석용 파일 저장 (분류기.pkl, X_train/test_processed.csv)

---

## 🚫 주의사항 (절대 하지 말 것)

1. **Optuna 사용 금지** → RandomizedSearchCV만 사용
2. **Stacking 앙상블 금지** → Weighted Voting만 사용
3. **시계열 로직 금지** → 시간 순서 의존 코드 작성 안 함
4. **Train/Test 전에 리샘플링 금지** → 파이프라인 내에서만 리샘플링
5. **Category dtype 수치 계산 금지** → 먼저 .cat.codes로 변환
6. **ROC-AUC 주요 메트릭 사용 금지** → PR-AUC가 핵심
7. **SHAP 분석용 파일 누락 금지** → 전처리된 데이터와 분류기 반드시 저장

---

## 📌 참고 자료

### 기존 노트북
- `notebooks/04_불균형_분류_모델링_final.ipynb`: 실제 구현 예시 (참고만, 복사 금지)
- `notebooks/발표_Part2_도메인_특성_공학_완전판.ipynb`: Part 2 출력 결과

### 문서
- `CLAUDE.md`: 프로젝트 전체 가이드
- `docs/notebook_summaries/`: 노트북 요약 문서

### 입력 파일
- `data/features/domain_based_features_완전판.csv`: 27개 특성 (Part 2 출력)
- `data/features/feature_metadata_완전판.csv`: 특성 메타데이터

---

## 🎯 최종 목표

**이 프롬프트를 따라 생성된 노트북은**:
1. ✅ Part 2 결과를 정확히 로딩하고 활용
2. ✅ 5가지 리샘플링 전략 모두 테스트 (SMOTETomek 포함)
3. ✅ 5개 모델 × 100회 랜덤 샘플링으로 최적 모델 탐색
4. ✅ Weighted Voting 앙상블 구축 및 비교
5. ✅ Traffic Light 시스템으로 비즈니스 가치 입증
6. ✅ 발표용으로 사용 가능한 완성도 (시각화, 설명 포함)
7. ✅ 실행 가능한 코드 (순서대로 실행 시 에러 없음)

**파일명**: `notebooks/발표_Part3_모델링_및_최적화.ipynb`

**예상 실행 시간**: 20~40분 (100회 랜덤 샘플링 기준)

---

이제 위 요구사항을 모두 만족하는 Jupyter Notebook을 생성해주세요. 마크다운 섹션은 이모지와 함께 명확하게 작성하고, 각 단계마다 충분한 설명을 포함하세요. 코드는 주석을 달아 이해하기 쉽게 작성하고, 출력 결과는 발표에 사용할 수 있도록 깔끔하게 포맷팅하세요.
