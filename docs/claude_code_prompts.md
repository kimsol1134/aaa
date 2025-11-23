# Claude Code 구현용 프롬프트

복사해서 Claude Code에 바로 붙여넣으세요.

---

## 📘 Part 1: 문제정의 및 핵심발견

```
당신은 시니어 데이터 사이언티스트이자 프롬프트 엔지니어링 전문가입니다.

# 작업 목표
기존 노트북에서 코드와 마크다운을 추출하여 발표용 "Part 1: 문제정의 및 핵심발견" 노트북을 생성하세요.

# 필수 요구사항

## 1. 기존 노트북 활용
다음 노트북에서 셀을 재사용하세요:
- `notebooks/01_도메인_기반_부도원인_분석.ipynb` (섹션 1-3)
- `notebooks/01_심화_재무_분석.ipynb` (섹션 2-3)

## 2. 노트북 구조
다음 순서로 구성하세요:

### Opening (문제 제기)
- 마크다운 셀: "50,000개 기업, 1.5% 부도율, 극도로 불균형한 데이터"
- 비즈니스 임팩트 강조

### 환경 설정 및 데이터 로딩
- 기존 노트북의 환경 설정 코드 재사용
- 데이터 로딩 코드 재사용

### 타겟 변수 및 불균형 분석
- 기존 시각화 코드 재사용
- **추가: 3-layer 해석 마크다운 셀**
  ```
  ### 📊 통계적 사실
  - 정상 기업: 98.49%
  - 부도 기업: 1.51%
  - 불균형 비율: 1:66

  ### 💡 재무 해석
  [이 비율이 의미하는 바를 재무 관점에서 설명]

  ### ➡️ 다음 액션
  [이 발견으로 인해 어떤 조치가 필요한지]
  ```

### 첫 번째 발견: 유동성이 핵심
- 기존 유동성 분석 코드 재사용
- **추가: 통계적 검정 코드**
  ```python
  from scipy.stats import mannwhitneyu

  # Mann-Whitney U test
  normal_liquidity = df[df[target_col] == 0]['현금비율']
  bankrupt_liquidity = df[df[target_col] == 1]['현금비율']
  u_stat, p_value = mannwhitneyu(normal_liquidity.dropna(), bankrupt_liquidity.dropna())

  # Cliff's delta (효과 크기)
  n1, n2 = len(normal_liquidity.dropna()), len(bankrupt_liquidity.dropna())
  cliff_delta = (u_stat - n1*n2/2) / (n1*n2)

  print(f"Mann-Whitney U test: p = {p_value:.2e}")
  print(f"Cliff's delta = {cliff_delta:.3f}")
  ```
- **추가: 3-layer 해석**

### 두 번째 발견: 업종별 차이
- 업종별 부도율 분석 코드 재사용
- **추가: Chi-square test**
- **추가: 3-layer 해석**

### 세 번째 발견: 한국 시장 특수성
- 외감 여부 분석 코드 재사용
- **추가: 통계적 검정**
- **추가: 3-layer 해석**

### Key Takeaways → Next Steps
- 마크다운 셀 추가:
  ```
  ## ✅ 핵심 발견
  1. 유동성(현금, 유동비율)이 가장 강력한 단변량 예측 변수
  2. 업종별 부도율 2배 차이 → 업종 더미 변수 필요
  3. 외감 여부가 중요한 신뢰도 지표

  ## ⚠️ 한계
  - 개별 변수로는 예측력 제한적 (단변량 AUC < 0.7)
  - 변수 간 상호작용 미고려

  ## ➡️ 다음 단계
  **Part 2: 도메인 지식 기반 복합 특성 생성**
  - 유동성 위기 지표 (현금소진일수 등)
  - 업종별 상대 비율
  - 재무조작 탐지 (Beneish M-Score)
  ```

## 3. 품질 기준
- 모든 주요 차트에 3-layer 해석 (📊 통계 → 💡 재무 → ➡️ 액션)
- 통계적 검정 (p-value, effect size) 추가
- 스토리가 자연스럽게 흐르도록

## 4. 출력 파일
- 파일명: `notebooks/발표_Part1_문제정의_및_핵심발견.ipynb`
- Jupyter notebook 형식 (.ipynb)

# 실행 단계
1. 기존 노트북 읽기
2. 필요한 셀 추출
3. 새 마크다운 셀 작성
4. 통계 검정 코드 추가
5. 노트북 생성 및 저장

지금 바로 시작하세요.
```

---

## 📗 Part 2: 도메인 특성 공학

```
당신은 시니어 데이터 사이언티스트이자 재무 도메인 전문가입니다.

# 작업 목표
기존 노트북에서 코드를 추출하여 발표용 "Part 2: 도메인 특성 공학" 노트북을 생성하세요.

# 필수 요구사항

## 1. 기존 노트북 활용
다음 노트북에서 셀을 재사용하세요:
- `notebooks/02_고급_도메인_특성공학.ipynb` (전체)
- `notebooks/03_상관관계_및_리스크_패턴_분석.ipynb` (VIF, IV, Feature Selection 부분)

## 2. 노트북 구조

### Opening: Part 1 요약
```markdown
## 📌 이전 Part 요약

Part 1에서 우리는 다음을 발견했습니다:
1. ✅ 유동성이 가장 강력한 예측 변수
2. ✅ 업종별 부도율 2배 차이
3. ✅ 외감 여부가 중요

하지만 **한계**가 있었습니다:
- ❌ 단변량 예측력 제한적 (AUC < 0.7)
- ❌ 변수 간 상호작용 미고려

**이제 도메인 지식을 활용한 복합 특성을 생성합니다.**
```

### Why 섹션: 왜 도메인 특성이 필요한가?
- 기존 노트북의 "Why" 섹션 재사용
- 부도의 3가지 경로 설명 (유동성/지급불능/신뢰상실)
- 이론적 배경 (Altman, Beneish 등)

### 특성 생성 (카테고리별)
각 카테고리마다:
1. 이론적 배경 마크다운
2. 특성 생성 코드
3. **특성 검증 코드 추가**:
   ```python
   # 예시: 즉각지급능력 검증
   feature_name = '즉각지급능력'

   normal_median = df[df[target_col] == 0][feature_name].median()
   bankrupt_median = df[df[target_col] == 1][feature_name].median()

   u_stat, p_value = mannwhitneyu(
       df[df[target_col] == 0][feature_name].dropna(),
       df[df[target_col] == 1][feature_name].dropna()
   )

   print(f"### {feature_name} 검증")
   print(f"- 정상 기업 median: {normal_median:.3f}")
   print(f"- 부도 기업 median: {bankrupt_median:.3f}")
   print(f"- Mann-Whitney U test: p = {p_value:.2e}")
   ```

### Feature Validation Matrix
- 새로운 코드 작성:
  ```python
  # 모든 생성된 특성 검증
  import pandas as pd
  from scipy.stats import mannwhitneyu, roc_auc_score

  validation_results = []

  for feature in all_features:
      try:
          normal = df[df[target_col] == 0][feature].dropna()
          bankrupt = df[df[target_col] == 1][feature].dropna()

          if len(normal) > 0 and len(bankrupt) > 0:
              # 통계 검정
              u_stat, p_value = mannwhitneyu(normal, bankrupt)

              # Cliff's delta
              n1, n2 = len(normal), len(bankrupt)
              cliff_delta = (u_stat - n1*n2/2) / (n1*n2)

              # AUC
              try:
                  auc = roc_auc_score(df[target_col], df[feature].fillna(df[feature].median()))
              except:
                  auc = None

              validation_results.append({
                  'Feature': feature,
                  'Normal_Median': normal.median(),
                  'Bankrupt_Median': bankrupt.median(),
                  'p_value': p_value,
                  'Cliff_Delta': cliff_delta,
                  'AUC': auc,
                  'Keep': '✅' if p_value < 0.01 and abs(cliff_delta) > 0.2 else '⚠️'
              })
      except:
          pass

  validation_df = pd.DataFrame(validation_results)
  print(validation_df.to_string(index=False))
  ```

### Feature Selection
- 기존 VIF, IV, Correlation 분석 코드 재사용
- **선택 기준 정당화 마크다운 추가**:
  ```markdown
  ### 왜 이 기준인가?

  #### VIF > 10 제거
  - **이유**: VIF 10 이상이면 분산이 10배 증가
  - **문제**: 계수 해석 불가능, 모델 불안정
  - **대안 검토**: VIF 5 (너무 엄격), VIF 15 (너무 관대)

  #### IV < 0.02 제거
  - **이유**: Information Value 0.02 미만은 예측력 없음
  - **기준**: IV 0.02-0.1 (약함), 0.1-0.3 (중간), 0.3+ (강함)

  #### Correlation > 0.9 제거
  - **이유**: 중복 정보, 한 개만 유지
  ```

### Key Takeaways → Next Steps
```markdown
## ✅ 생성된 특성
- 총 75개 → 선택 48개 (27개 제거)
- 모든 특성이 재무 이론 기반
- 통계적으로 유의미 (p < 0.01)

## ➡️ 다음 단계
**Part 3: 모델링 및 최적화**
- 불균형 데이터 처리 (SMOTE)
- 모델 비교 및 선택
- 하이퍼파라미터 튜닝
```

## 3. 출력 파일
- 파일명: `notebooks/발표_Part2_도메인_특성_공학.ipynb`

지금 바로 시작하세요.
```

---

## 📕 Part 3: 모델링 및 최적화

```
당신은 시니어 머신러닝 엔지니어이자 불균형 데이터 전문가입니다.

# 작업 목표
기존 노트북에서 코드를 추출하여 발표용 "Part 3: 모델링 및 최적화" 노트북을 생성하세요.

# 필수 요구사항

## 1. 기존 노트북 활용
- `notebooks/04_불균형_분류_모델링_final.ipynb` (전체)

## 2. 노트북 구조

### Opening: Part 2 요약
```markdown
## 📌 이전 Part 요약

Part 2에서 우리는:
- ✅ 75개 도메인 특성 생성
- ✅ 48개 최종 선택 (통계적 유의성 검증)

이제 **불균형 데이터 문제**를 해결하고 최적의 모델을 찾습니다.
```

### 불균형 데이터 문제
- 마크다운: 1:66 비율의 의미
- Naive baseline 성능 계산
- **왜 PR-AUC?** 설명:
  ```markdown
  ### 왜 PR-AUC를 사용하는가?

  #### ROC-AUC의 문제
  - 불균형 데이터에서 과대평가
  - 예: 모든 예측을 0으로 해도 높은 AUC

  #### PR-AUC의 장점
  - Precision-Recall 곡선 기반
  - 소수 클래스 성능에 집중
  - 불균형 데이터의 실질적 성능 측정

  #### 평가 지표 우선순위
  1. **PR-AUC** (핵심 지표)
  2. **F2-Score** (재현율 중시)
  3. **Type II Error** (부도 미탐지율)
  ```

###   
- 기존 코드 재사용
- **대안 비교표 추가**:
  ```python
  # 여러 샘플링 방법 비교
  from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
  from imblearn.combine import SMOTETomek

  methods = {
      'No Sampling': None,
      'SMOTE only': SMOTE(random_state=42),
      'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
      'SMOTE + Tomek': SMOTETomek(random_state=42),
      'ADASYN': ADASYN(random_state=42)
  }

  # 각 방법 비교 (간단한 모델로 테스트)
  comparison_results = []
  for name, sampler in methods.items():
      # [비교 코드]
      pass

  # 결과 테이블 출력
  ```

### 모델 비교 (5개 모델)
- 기존 모델 학습 코드 재사용
- **비교표 생성**:
  ```python
  model_comparison = pd.DataFrame({
      'Model': ['Logistic Regression', 'Random Forest', 'LightGBM', 'XGBoost', 'Stacking'],
      'PR-AUC': [0.089, 0.108, 0.142, 0.145, 0.135],
      'F2-Score': [0.14, 0.16, 0.20, 0.21, 0.19],
      'Training_Time': ['2s', '45s', '18s', '32s', '120s'],
      'Selected': ['', '', '', '✅', '']
  })
  print(model_comparison.to_string(index=False))
  ```

### 왜 Ensemble이 실패했나?
- **새로운 분석 코드 추가**:
  ```python
  # 모델 간 예측 상관관계
  predictions_df = pd.DataFrame({
      'LightGBM': lgb_pred,
      'XGBoost': xgb_pred,
      'CatBoost': cat_pred
  })

  print("### 모델 간 예측 상관관계")
  print(predictions_df.corr())

  # Meta-learner 과적합 분석
  print("\n### Meta-learner 성능")
  print(f"Train PR-AUC: {train_pr_auc:.3f}")
  print(f"Test PR-AUC: {test_pr_auc:.3f}")
  print(f"Overfitting: {(train_pr_auc - test_pr_auc) / train_pr_auc * 100:.1f}%")
  ```
- **마크다운 해석**

### Threshold 선정
- **비교 코드 추가**:
  ```python
  thresholds = [0.02, 0.05, 0.062, 0.08, 0.1, 0.15]

  for threshold in thresholds:
      y_pred = (y_pred_proba > threshold).astype(int)
      precision = precision_score(y_test, y_pred)
      recall = recall_score(y_test, y_pred)
      f2 = fbeta_score(y_test, y_pred, beta=2)

      print(f"Threshold {threshold:.3f}: P={precision:.3f}, R={recall:.3f}, F2={f2:.3f}")
  ```
- **왜 0.062?** 설명

### Key Takeaways → Next Steps
```markdown
## ✅ 최종 모델
- XGBoost + SMOTE+Tomek
- Threshold: 0.062
- PR-AUC: 0.145

## ➡️ 다음 단계
**Part 4: 성능 평가 및 비즈니스 가치**
```

## 3. 출력 파일
- 파일명: `notebooks/발표_Part3_모델링_및_최적화.ipynb`

지금 바로 시작하세요.
```

---

## 📙 Part 4: 결과 및 비즈니스 가치

```
당신은 시니어 데이터 사이언티스트이자 비즈니스 애널리스트입니다.

# 작업 목표
기존 노트북에서 코드를 추출하여 발표용 "Part 4: 결과 및 비즈니스 가치" 노트북을 생성하세요.

# 필수 요구사항

## 1. 기존 노트북 활용
  
- 발표_Part1_문제정의_및_핵심발견_executed
- 발표_Part2_도메인_특성_공학_완전판_executed
- 발표_Part3_모델링_및_최적화_완전판_executed

## 2. 노트북 구조

### Opening: Part 3 요약
```markdown
## 📌 최종 모델

- **모델**: 
- **샘플링**: 
- **Threshold**: 
- **선택 이유**: 

이제 **성능 평가**와 **비즈니스 가치**를 분석합니다.
```

### 성능 결과
- 기존 평가 코드 재사용
- **Bootstrap CI 추가**:
  ```python
  from sklearn.utils import resample

  # Bootstrap으로 PR-AUC 신뢰 구간 계산
  n_iterations = 1000
  pr_aucs = []

  np.random.seed(42)
  for i in range(n_iterations):
      # 부트스트랩 샘플링
      indices = resample(range(len(y_test)), random_state=i)
      y_test_boot = y_test.iloc[indices]
      y_pred_boot = y_pred_proba[indices]

      # PR-AUC 계산
      pr_auc = average_precision_score(y_test_boot, y_pred_boot)
      pr_aucs.append(pr_auc)

  print(f"PR-AUC: {np.mean(pr_aucs):.4f}")
  print(f"95% CI: [{np.percentile(pr_aucs, 2.5):.4f}, {np.percentile(pr_aucs, 97.5):.4f}]")
  ```

### Confusion Matrix
- 기존 코드 재사용
- **재무 관점 해석 추가**:
  ```markdown
  ### 💡 재무 해석

  #### True Positive (부도 기업을 부도로 예측): 
  - **비즈니스 가치**: 부도 위험 사전 차단
  - **예상 손실 회피**: 

  #### False Negative (부도 기업을 정상으로 예측): 
  - **비즈니스 리스크**: 부도 미탐지
  - **예상 손실**: 
  - **개선 필요**: FN 감소가 최우선 과제

  #### Type II Error: 
  - 원인: 
  ```

### SHAP 분석 ** 가장 중요
- 기존 SHAP 코드 재사용
- **Top 10 특성 재무 해석 추가**:
  ```markdown
  ### Top 10 중요 특성 해석

  1. **신용등급점수** (SHAP: 0.082)
     - 재무 의미: 신용평가사의 종합 평가
     - ⚠️ Data leakage 가능성 검토 필요

  2. **즉각지급능력** (SHAP: 0.065)
     - 재무 의미: 현금으로 단기 부채를 즉시 상환 가능한 비율
     - 위험 기준: < 0.1

  [나머지 특성도 동일하게 해석]
  ```

### Traffic Light 시스템
- 기존 코드 재사용

### 비즈니스 가치
- 기존 시뮬레이션 코드 재사용
- **결과 재검증**

### 한계 및 개선 방향
- **대폭 강화된 한계 섹션 작성**:
  ```markdown
  ## 한계 (Limitations) 및 개선 방향

  ### 1. 데이터 품질 이슈 ⚠️

  **문제:**
  - 63.7% 기업이 현금 = 0으로 기록
  - 재고자산, 매출채권 등도 유사한 문제

  **원인 추정:**
  - 중소기업 회계 시스템 미비
  - 실제 현금 부족 vs 기록 누락 구분 불가

  **현재 대응:**
  - Binary feature '현금보유여부' 추가
  - Robust 통계량 사용 (median)

  **향후 개선:**
  - 원본 데이터 출처 확인
  - 외부 데이터 결합 (금융감독원, 국세청)
  - 데이터 품질 스코어링 시스템 구축

  ### 2. 시계열 정보 부족 🕐

  **한계:**
  - 2021년 8월 단일 시점 스냅샷
  - 재무 악화 속도, 추세 파악 불가

  **Impact:**
  - 급격히 악화되는 기업 탐지 어려움
  - FN 53.95%에 기여하는 주요 요인

  **향후 개선:**
  - 분기별 데이터 확보
  - 변화율 특성 추가 (매출 증가율, 부채 증가율)
  - LSTM/Transformer 기반 시계열 모델 검토

  ### 3. 모델 성능 한계 📉

  **현재 성능:**
  - PR-AUC: 0.145 (95% CI: 0.128-0.162)
  - F2-Score: 0.21
  - **False Negative: 53.95%** ← 가장 큰 문제

  **원인 분석:**
  1. **데이터에 없는 정보:**
     - 소송 진행 상황
     - 경영진 교체/비리
     - 주요 거래처 부도
     - 업계 환경 변화

  2. **극도 불균형 (1:66):**
     - 소수 클래스 학습 어려움
     - SMOTE로 일부 완화했으나 한계

  3. **모델 다양성 부족:**
     - 모든 Tree 기반 모델 예측 유사 (상관 > 0.95)
     - Ensemble 효과 제한적

  **개선 방안:**
  1. **외부 데이터 통합:**
     - 뉴스 감성 분석 (부정적 기사 탐지)
     - 소송 이력 데이터
     - 경영진 이력 데이터

  2. **앙상블 다양성 증대:**
     - Neural Network 추가 (다른 학습 메커니즘)
     - Tabular Transformer (TabNet, FT-Transformer)
     - Stacking에서 Meta-learner를 XGBoost 대신 NN으로

  3. **Cost-sensitive Learning 강화:**
     - Focal Loss 적용
     - FN 비용을 더 높게 설정 (현재 5배 → 10배)

  ### 4. 해석 가능성 vs 성능 트레이드오프 ⚖️

  **현재 선택:**


  **트레이드오프:**


  **결정 근거:**


  ## 그럼에도 가치 있는 이유 ✅

  ### 1. 도메인 논리 명확
  - 모든 특성이 재무 이론 기반
  - 각 특성의 의미를 명확히 설명 가능
  - 실무에서 바로 활용 가능

  ### 2. 재현 가능 및 확장 가능
  - 다른 연도 데이터에 즉시 적용 가능
  - 다른 국가 시장에도 확장 가능 (한국 특화 특성만 제외)
  - 전체 파이프라인 자동화

  ### 3. 해석 가능한 AI
  - SHAP으로 모든 예측 근거 제시
  - 규제 요구사항 충족
  - 사용자 신뢰 확보

  ### 4. 실용적 성능 향상
  - Naive Baseline 대비:

  - 실제 손실 감소 효과 입증

  ### 5. 확장 가능한 프레임워크
  - 외부 데이터 추가 용이
  - 모델 업그레이드 가능
  - 지속적 개선 가능
  ```

### 최종 요약
```markdown
## 🎯 프로젝트 성과

### 달성한 것


### 주요 한계


### 향후 발전 방향
1. 🔜 외부 데이터 통합 (뉴스, 소송)
2. 🔜 시계열 모델 검토
3. 🔜 앙상블 다양성 증대
```

## 3. 출력 파일
- 파일명: `notebooks/발표_Part4_결과_및_비즈니스_가치.ipynb`

지금 바로 시작하세요.
```

---

## 📝 사용 방법

1. **Part 1부터 순차적으로 실행**
   - 각 프롬프트를 복사하여 Claude Code에 붙여넣기
   - 완료 후 다음 Part로 이동

2. **검증**
   - 각 노트북이 생성되면 실행해서 오류 확인
   - 스토리 흐름 검토

3. **최종 검토**
   - 4개 노트북 전체 순서대로 실행
   - 평가 기준 체크리스트 확인

---

## 체크리스트

각 노트북 완성 후 확인:

- [ ] 기존 코드가 제대로 재사용되었는가?
- [ ] 모든 주요 차트에 3-layer 해석이 있는가?
- [ ] 통계적 검정이 추가되었는가?
- [ ] Key Takeaways → Next Steps가 있는가?
- [ ] 노트북이 실행되는가?
- [ ] 스토리가 자연스럽게 흐르는가?
