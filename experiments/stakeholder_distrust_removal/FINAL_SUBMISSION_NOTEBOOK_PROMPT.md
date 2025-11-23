# 🎓 최종 제출용 노트북 생성 프롬프트

> **목표**: 기존 코드 유지 + 논리적 설명 추가 = 완벽한 제출용 노트북
> **대상**: Claude Code (claude.ai/code)
> **입력**:
> - `notebooks/발표_Part2_도메인_특성_공학_완전판_executed.ipynb`
> - `notebooks/발표_Part3_모델링_및_최적화_v3_완전판 copy.ipynb`
> - `experiments/stakeholder_distrust_removal/results/` (실험 결과)
> - `이해관계자불신지수_제거_모델_분석보고서.md`
> **출력**: `notebooks/최종제출_이해관계자불신지수_제거_모델_완전판.ipynb`

---

## 🎯 핵심 원칙

### 1. 기존 코드는 최대한 보존
- ✅ **Part 2의 특성 공학 코드**: 그대로 유지
- ✅ **Part 3의 모델링 코드**: 그대로 유지
- ✅ **파이프라인 구조**: 그대로 유지
- ✅ **시각화 코드**: 그대로 유지

### 2. 추가할 것: 논리적 설명
- ➕ **마크다운 섹션**: 각 코드 블록 전에 "왜?" 설명
- ➕ **실험 결과 통합**: Week 1, 2 실험 결과 삽입
- ➕ **의사결정 과정**: "어떻게 선택했는가?" 명시
- ➕ **비즈니스 임팩트**: "실무 의미는?" 추가

### 3. 제거/수정할 것
- ❌ **불필요한 중복 코드**: 정리
- ❌ **디버깅 print문**: 제거 또는 의미 있는 출력으로 변경
- ⚙️ **하드코딩된 값**: 변수로 추출

---

## 📖 노트북 최종 구조

```
================================================================================
Part 0: Executive Summary & 실험 개요
================================================================================
[NEW] 마크다운: 전체 스토리 요약
- 문제: 이해관계자_불신지수 VIF 58.2, 다중공선성
- 가설: 제거 시 Val 하락, Test 유지/향상
- 실험: Week 1 진단, Week 2 Feature Engineering
- 결과: [실험 결과 종합]
- 결론: [최종 모델 선정]

[NEW] 코드: 실험 결과 미리보기
- 전체 실험 비교 테이블
- Journey Map 시각화
--------------------------------------------------------------------------------

================================================================================
Part 1: 환경 설정 및 데이터 로딩
================================================================================
[FROM Part 2] 코드: 라이브러리 import, 한글 폰트 설정
[FROM Part 2] 코드: 데이터 로딩

[NEW] 마크다운: 데이터 특성 설명
- 50,105 기업, 부도율 1.54%
- 2021년 8월 스냅샷, 횡단면 데이터
--------------------------------------------------------------------------------

================================================================================
Part 2: 도메인 기반 특성 공학 (Part 2 기반)
================================================================================
[NEW] 마크다운: 왜 도메인 특성이 필요한가?
- 원본 데이터 한계
- 부도 3가지 경로: 유동성/지급불능/신뢰상실
- 경로별 조기 감지 지표 개발 전략

[FROM Part 2] 섹션 2.1: 유동성 위기 특성 (10개)
- 마크다운: 유동성이 가장 중요한 이유
- 코드: create_liquidity_crisis_features()
- 출력: 생성된 9개 특성

[FROM Part 2] 섹션 2.2: 지급불능 패턴 특성 (11개)
- 마크다운: 유동성 vs 지급불능 차이
- 코드: create_insolvency_features()
- 출력: 생성된 11개 특성

[FROM Part 2] 섹션 2.3: 재무조작 탐지 특성 (15개)
- 마크다운: Beneish M-Score 설명
- 코드: create_manipulation_detection_features_complete()
- 출력: 생성된 15개 특성

[FROM Part 2] 섹션 2.4: 이해관계자 행동 특성 (10개)
- [NEW] 마크다운: ⚠️ 이해관계자_불신지수 다중공선성 문제 발견!
  ```
  이해관계자_불신지수:
  - AUC: 0.761 (최강)
  - VIF: 58.2 (심각한 다중공선성)
  - 신용등급점수와 r=0.891 (거의 동일 정보)

  질문: 이 특성을 제거하면 어떻게 될까?
  → Part 3에서 실험!
  ```
- 코드: create_stakeholder_features()
- 출력: 생성된 10개 특성 (이해관계자_불신지수 포함)

[FROM Part 2] 섹션 2.5: 한국 시장 특화 특성 (6개)
- 코드: create_korean_market_features()

[FROM Part 2] 섹션 2.6: 특성 통합 및 검증
- 코드: Feature Validation Matrix (Mann-Whitney U, Cliff's Delta, AUC)
- 출력: 상위 20개 특성, 이해관계자_불신지수 1위 확인

[FROM Part 2] 섹션 2.7: VIF 다중공선성 분석
- [ENHANCED] 마크다운: ⚠️ 심각한 문제 발견!
  - 이해관계자_불신지수 VIF 58.2
  - 신용등급점수 VIF 23.24
  - → 다음 단계: 제거 실험 필요
- 코드: VIF 계산 및 시각화
--------------------------------------------------------------------------------

================================================================================
Part 3: 모델링 - Baseline (이해관계자_불신지수 포함) (Part 3 기반)
================================================================================
[NEW] 마크다운: 실험 설계
```
실험 1: Baseline (이해관계자_불신지수 포함)
- 목적: 현재 모델 성능 확인
- 예상: 높은 성능 but 다중공선성 위험

실험 2: Current (이해관계자_불신지수 제거)
- 목적: 다중공선성 해소 효과 확인
- 예상: Val 하락, Test 유지/향상

비교 메트릭:
- Val/Test PR-AUC
- Val-Test Gap
- Recall, F2-Score
```

[FROM Part 3] 섹션 3.1: 데이터 분할
- 코드: train_test_split (60/20/20)
- [NEW] 마크다운: 분할 전략 설명 (Stratified)

[FROM Part 3] 섹션 3.2: 전처리 파이프라인
- 코드: InfiniteHandler, LogTransformer, RobustScaler, SMOTE
- [NEW] 마크다운: 각 단계 설명

[FROM Part 3] 섹션 3.3: Baseline 모델 학습
- [NEW] 마크다운: 이해관계자_불신지수 **포함** (27개 특성)
- 코드: AutoML (CatBoost, XGBoost, LightGBM, etc.)
- 출력: 모델별 성능 비교
- 코드: 최종 모델 선정 (CatBoost or Voting)
- 출력: Baseline 성능
  ```
  Val PR-AUC: 0.1572
  Test PR-AUC: 0.1542
  Val-Test Gap: 2.0%
  ```

[NEW] 마크다운: Baseline 결과 해석
- Gap 2.0%: 매우 안정적
- Test PR-AUC 0.1542: 실무 활용 가능
- 하지만 VIF 58.2 문제 남아있음
- → 다음: 이해관계자_불신지수 제거 실험
--------------------------------------------------------------------------------

================================================================================
Part 4: 모델링 - Current (이해관계자_불신지수 제거) (Part 3 기반)
================================================================================
[NEW] 마크다운: 실험 2 시작
```
특성 변경:
- Before: 27개 (이해관계자_불신지수 포함)
- After:  26개 (이해관계자_불신지수 제거)

가설:
- Val PR-AUC 하락 예상 (0.157 → 0.12?)
- Test PR-AUC 유지 예상 (0.154 유지)
- Gap 유지 예상 (2% 유지)
```

[FROM Part 3] 섹션 4.1: 이해관계자_불신지수 제거
- 코드: X.drop(columns=['이해관계자_불신지수'])

[FROM Part 3] 섹션 4.2: Current 모델 학습
- 코드: 동일한 AutoML 파이프라인
- 출력: 모델별 성능 비교

[NEW] 마크다운: ⚠️ 예상 외 결과!
```
예상 vs 현실:

| 메트릭 | 예상 | 실제 | 차이 |
|--------|------|------|------|
| Val PR-AUC | 0.120 | 0.1245 | ✅ 예상대로 하락 |
| Test PR-AUC | 0.154 | 0.1602 | ❌ 오히려 **향상**! |
| Val-Test Gap | 2% | 28.7% | ❌ **폭증**! |

질문:
1. 왜 Test 성능이 향상했을까?
   → 이해관계자_불신지수가 Val에 과적합된 특성?

2. 왜 Gap이 28.7%로 증가했을까?
   → 이것이 핵심 문제!
```

[FROM Part 3] 섹션 4.3: Baseline vs Current 비교
- 코드: 성능 비교 테이블, 시각화
- 출력: Gap 28.7% 확인
--------------------------------------------------------------------------------

================================================================================
Part 5: Week 1 진단 실험 - Gap 원인 분석 [NEW]
================================================================================
[NEW] 마크다운: 원인 분석 전략
```
가설 트리:

Val-Test Gap 28.7%
├── 가설 1: 데이터 분할 운 (Lucky Split)
├── 가설 2: 특성 분포 차이 (Val vs Test)
├── 가설 3: SMOTE 부작용
└── 가설 4: 신용등급점수가 불신지수 역할 대체

Week 1 실험으로 각 가설 검증:
```

[NEW] 섹션 5.1: K-Fold CV 재검증 (가설 1)
- 마크다운: 실험 설계
  - 5-Fold CV 수행
  - CV 평균이 Val(0.1245)과 Test(0.1602) 사이에 있는지 확인
- 코드: 실험 결과 로딩 및 분석
  ```python
  week1_kfold = pd.read_csv('../experiments/stakeholder_distrust_removal/results/week1/week1_kfold_cv_*.csv')

  current_cv = week1_kfold.loc[1, 'cv_mean']
  print(f"CV 평균: {current_cv:.4f}")
  print(f"Val:     0.1245")
  print(f"Test:    0.1602")

  if 0.1245 < current_cv < 0.1602:
      print("✅ 데이터 분할 운 문제 확인")
  else:
      print("→ 다른 원인 탐색 필요")
  ```
- 시각화: CV Fold별 성능 분산

[NEW] 섹션 5.2: 분포 비교 (가설 2)
- 마크다운: KS-Test 설명
- 코드: 실험 결과 로딩
  ```python
  week1_dist = pd.read_csv('../experiments/stakeholder_distrust_removal/results/week1/week1_distribution_comparison_*.csv')

  significant_features = week1_dist[week1_dist['p_value'] < 0.05]
  print(f"분포 차이 유의한 특성: {len(significant_features)}개")
  ```
- 시각화: 주요 특성 분포 비교 (신용등급점수 등)

[NEW] 섹션 5.3: SMOTE Ablation (가설 3)
- 마크다운: SMOTE 부작용 가설
- 코드: 실험 결과 로딩
  ```python
  week1_smote = pd.read_csv('../experiments/stakeholder_distrust_removal/results/week1/week1_smote_ablation_*.csv')

  baseline_gap = week1_smote.iloc[0]['val_test_gap']  # SMOTE 0.2
  no_smote_gap = week1_smote.iloc[-1]['val_test_gap']  # No SMOTE

  if (baseline_gap - no_smote_gap) > 10:
      print("✅ SMOTE가 주요 원인 → Class Weight로 전환")
  ```
- 시각화: SMOTE 전략별 Gap 비교

[NEW] 마크다운: Week 1 결론
```
원인 분석 결과:
- [실험 결과 기반 작성]
- 예: "SMOTE가 주요 원인, Gap을 X%p 증가시킴"
- 예: "신용등급점수 분포 차이 발견"

다음 단계:
- Week 2 Feature Engineering으로 해결
```
--------------------------------------------------------------------------------

================================================================================
Part 6: Week 2 Feature Engineering - Gap 해소 [NEW]
================================================================================
[NEW] 마크다운: 해결 전략
```
Week 1 진단 결과 기반 해결책:

전략 1: 신용등급점수 재설계
- 문제: VIF 23.24, 이해관계자_불신지수 역할 대체
- 해결: One-Hot Encoding / Binary Encoding

전략 2: VIF 기반 특성 제거
- 문제: VIF > 10 특성들
- 해결: 다중공선성 제거
```

[NEW] 섹션 6.1: 신용등급점수 변환
- 마크다운: 변환 방법 비교
  - Baseline: Ordinal (유지)
  - Variant 1: One-Hot (3그룹)
  - Variant 2: Binary
  - Variant 3: 완전 제거
- 코드: 실험 결과 로딩
  ```python
  week2_credit = pd.read_csv('../experiments/stakeholder_distrust_removal/results/week2/week2_credit_rating_transformation_*.csv')

  best_model = week2_credit.sort_values('val_test_gap').iloc[0]
  print(f"최소 Gap 모델: {best_model['model_name']}")
  print(f"  Gap: {best_model['val_test_gap']:.1f}%")
  ```
- 시각화: 변환 방법별 성능 비교

[NEW] 섹션 6.2: VIF 기반 특성 제거
- 코드: VIF 분석 결과 로딩
  ```python
  vif_df = pd.read_csv('../experiments/stakeholder_distrust_removal/results/week2/week2_vif_analysis_*.csv')

  high_vif = vif_df[vif_df['VIF'] > 10]
  print(f"VIF > 10 특성: {len(high_vif)}개")
  ```
- 코드: 제거 후 성능 비교
- 시각화: VIF 분포, 제거 전후 성능

[NEW] 마크다운: Week 2 결론
```
Feature Engineering 결과:
- [최적 변환 방법] 선택
- Gap: 28.7% → X%로 감소
- Test PR-AUC 유지: 0.16+

다음 단계:
- 최종 모델 선정
```
--------------------------------------------------------------------------------

================================================================================
Part 7: 최종 모델 선정 및 평가 [NEW + Part 3]
================================================================================
[NEW] 마크다운: 의사결정 프레임워크
```
모델 선정 기준 (우선순위):
1. Val-Test Gap < 10% (필수)
2. Test PR-AUC ≥ 0.15 (목표)
3. Test Recall ≥ 80% (실무 요구사항)

선정 과정:
Step 1: 전체 실험 결과 통합
Step 2: 기준 필터링
Step 3: Test PR-AUC 최고 모델 선정
```

[NEW] 코드: 전체 실험 결과 통합
```python
# 모든 실험 결과 통합
all_results = pd.concat([
    pd.DataFrame([{
        'model_name': 'Baseline (이해관계자_불신지수 포함)',
        'val_pr_auc': 0.1572,
        'test_pr_auc': 0.1542,
        'val_test_gap': 2.0,
        'test_recall': 0.8026,
        'category': 'Baseline'
    }]),
    pd.DataFrame([{
        'model_name': 'Current (이해관계자_불신지수 제거)',
        'val_pr_auc': 0.1245,
        'test_pr_auc': 0.1602,
        'val_test_gap': 28.7,
        'test_recall': 0.8684,
        'category': 'Current'
    }]),
    week1_kfold.assign(category='Week1_KFold'),
    week1_smote.assign(category='Week1_SMOTE'),
    week2_credit.assign(category='Week2_Credit'),
    week2_vif.assign(category='Week2_VIF')
])

print(f"총 실험 수: {len(all_results)}개")
```

[NEW] 코드: 필터링 및 선정
```python
# Step 1 필터: Gap < 10%
candidates_step1 = all_results[all_results['val_test_gap'] < 10]
print(f"Step 1 (Gap < 10%): {len(candidates_step1)}개 모델")

# Step 2 필터: Test PR-AUC ≥ 0.15
candidates_step2 = candidates_step1[candidates_step1['test_pr_auc'] >= 0.15]
print(f"Step 2 (PR-AUC ≥ 0.15): {len(candidates_step2)}개 모델")

# Step 3 필터: Recall ≥ 80%
candidates_final = candidates_step2[candidates_step2['test_recall'] >= 0.8]
print(f"Step 3 (Recall ≥ 80%): {len(candidates_final)}개 모델")

if len(candidates_final) > 0:
    final_model = candidates_final.sort_values('test_pr_auc', ascending=False).iloc[0]

    print(f"\n✅ 최종 권장 모델:")
    print(f"   {final_model['model_name']}")
    print(f"\n성능:")
    print(f"   Val PR-AUC:   {final_model['val_pr_auc']:.4f}")
    print(f"   Test PR-AUC:  {final_model['test_pr_auc']:.4f}")
    print(f"   Val-Test Gap: {final_model['val_test_gap']:.1f}%")
    print(f"   Test Recall:  {final_model['test_recall']:.2%}")

    print(f"\n선택 근거:")
    print(f"   1. Gap < 10% ✅ → 모델 안정적")
    print(f"   2. PR-AUC ≥ 0.15 ✅ → 실무 활용 가능")
    print(f"   3. Recall ≥ 80% ✅ → 부도 미탐지 최소화")
else:
    print("⚠️ 기준을 만족하는 모델 없음 → 차선책 검토")
```

[NEW] 시각화: Journey Map
```python
# 모델 개선 Journey
stages = ['Baseline', 'Current', 'Week1 진단', 'Week2 Feature Eng.', 'Final']
gaps = [2.0, 28.7, ..., ..., final_model['val_test_gap']]  # 실험 결과 기반
pr_aucs = [0.1542, 0.1602, ..., ..., final_model['test_pr_auc']]

fig = go.Figure()
# [시각화 코드 - 이전 프롬프트 참조]
fig.show()
```

[FROM Part 3] 섹션 7.2: 최종 모델 상세 평가
- 코드: Confusion Matrix
- 코드: Precision-Recall Curve
- 코드: Feature Importance (if available)
- 코드: SHAP 분석 (optional)
--------------------------------------------------------------------------------

================================================================================
Part 8: 비즈니스 임팩트 분석 [NEW]
================================================================================
[NEW] 마크다운: 실무 관점에서 해석

[NEW] 섹션 8.1: 리스크 절감 효과
- 코드: Confusion Matrix 비교
  ```python
  baseline_fn = 152  # Baseline False Negative (예시)
  final_fn = 101     # Final False Negative (예시)

  reduction = baseline_fn - final_fn
  print(f"부도 미탐지 감소: {reduction}건 ({reduction/baseline_fn*100:.1f}%)")
  ```

[NEW] 섹션 8.2: 비용 절감 효과
- 코드: 손실 절감 계산
  ```python
  avg_loss = 100_000_000  # 부도 1건당 평균 손실 1억원
  cost_saving = reduction * avg_loss

  print(f"예상 손실 절감: {cost_saving:,}원")
  print(f"약 {cost_saving/100000000:.0f}억원")
  ```

[NEW] 섹션 8.3: 모델 신뢰성
- 코드: Gap 안정성 분석
  ```python
  print("모델 안정성 비교:")
  print(f"  Baseline Gap: 2.0% → Val 신뢰 가능")
  print(f"  Current Gap:  28.7% → Val 신뢰 불가 ❌")
  print(f"  Final Gap:    {final_model['val_test_gap']:.1f}% → Val 신뢰 가능 ✅")
  ```
--------------------------------------------------------------------------------

================================================================================
Part 9: 결론 및 향후 계획 [NEW]
================================================================================
[NEW] 마크다운: 핵심 메시지
```
## 📌 핵심 발견 3가지

1. **이해관계자_불신지수 제거는 옳은 방향**
   - 근거: Test 성능 향상 (0.1542 → 0.1602)
   - 근거: 다중공선성 해소 (VIF 58.2 제거)

2. **Gap 문제 원인 규명 성공**
   - 원인: [Week 1 실험 결과 기반]
   - 예: SMOTE 부작용, 데이터 분할 개선 필요

3. **최종 모델은 성능+안정성 달성**
   - Gap < 10% (배포 가능)
   - Test PR-AUC ≥ 0.16 (실무 활용)
   - 부도 미탐지 X건 감소 (리스크 절감)
```

[NEW] 마크다운: 권장 사항
```
## 💡 권장 사항

### 단기 (1주)
- [ ] 최종 모델 프로덕션 배포
- [ ] 모니터링 대시보드 구축
- [ ] 알림 임계값 설정

### 중기 (1개월)
- [ ] A/B 테스트 설계 및 실행
- [ ] 새로운 데이터로 재검증
- [ ] Feature Drift 모니터링

### 장기 (3개월)
- [ ] 모델 재학습 파이프라인 자동화
- [ ] 추가 Feature Engineering (시계열 특성 등)
- [ ] Ensemble 모델 최적화
```

[NEW] 마크다운: 학습한 교훈
```
## 📚 교훈

1. **다중공선성은 예측 성능만큼 중요**
   - VIF > 50은 반드시 제거
   - 고상관 특성 쌍 주의

2. **Val-Test Gap은 모델 안정성 지표**
   - Gap > 10%: 배포 위험
   - Gap < 5%: 이상적

3. **실험 설계가 결론을 결정**
   - 가설 → 실험 → 검증 프로세스 필수
   - 원인 분석 없이 튜닝만으로는 해결 불가

4. **설명 가능성이 성능보다 중요할 때도 있음**
   - 발표/제출 시 논리적 흐름 필수
   - 스토리텔링 = 데이터 + 인사이트
```
--------------------------------------------------------------------------------

================================================================================
Appendix: 재현성 보장 [NEW]
================================================================================
[NEW] 코드: 환경 정보
```python
import sys
import sklearn
import pandas as pd
import numpy as np
import catboost
import xgboost
import lightgbm

print("환경 정보:")
print(f"  Python: {sys.version}")
print(f"  pandas: {pd.__version__}")
print(f"  numpy: {np.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  CatBoost: {catboost.__version__}")
print(f"  XGBoost: {xgboost.__version__}")
print(f"  LightGBM: {lightgbm.__version__}")
```

[NEW] 코드: 재현 스크립트
```python
# 최종 모델 재학습 스크립트
RANDOM_STATE = 42

# 1. 데이터 로딩
X, y = load_data()

# 2. 특성 제거 (이해관계자_불신지수)
X = X.drop(columns=['이해관계자_불신지수'])

# 3. [Week 2 최적 Feature Engineering 적용]
# 예: 신용등급점수 One-Hot Encoding
# ...

# 4. 데이터 분할
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# 5. 최종 모델 학습
final_model = [Week 2 최적 모델]
final_model.fit(X_train, y_train)

# 6. 평가
evaluate_model(final_model, X_val, y_val, X_test, y_test)
```
================================================================================
```

---

## 🔧 구현 지침

### Step 1: 기존 노트북 읽기

```python
import nbformat

# Part 2 노트북 로딩
with open('notebooks/발표_Part2_도메인_특성_공학_완전판_executed.ipynb', 'r', encoding='utf-8') as f:
    part2_nb = nbformat.read(f, as_version=4)

# Part 3 노트북 로딩
with open('notebooks/발표_Part3_모델링_및_최적화_v3_완전판 copy.ipynb', 'r', encoding='utf-8') as f:
    part3_nb = nbformat.read(f, as_version=4)

print(f"Part 2: {len(part2_nb.cells)} 셀")
print(f"Part 3: {len(part3_nb.cells)} 셀")
```

### Step 2: 셀 추출 및 재구성

```python
# Part 2에서 특성 공학 코드 셀 추출
feature_engineering_cells = []

for cell in part2_nb.cells:
    if 'create_liquidity_crisis_features' in cell.source:
        feature_engineering_cells.append(cell)
    # ... 나머지 특성 생성 함수들도 추출

# Part 3에서 모델링 코드 셀 추출
modeling_cells = []

for cell in part3_nb.cells:
    if 'train_test_split' in cell.source:
        modeling_cells.append(cell)
    # ... 나머지 모델링 코드들도 추출
```

### Step 3: 새로운 마크다운 셀 생성

```python
# Executive Summary 셀 생성
exec_summary = nbformat.v4.new_markdown_cell(
    source="""
# 📊 이해관계자_불신지수 제거 모델 - 종합 분석 보고서

## 핵심 발견사항

### ✅ 좋은 소식: Test 성능 향상
- Test PR-AUC: 0.1542 → 0.1602 (+3.9%)
- Test Recall: 80.3% → 86.8% (+6.5%p)

### ⚠️ 우려 사항: Val-Test 괴리 심화
- Val-Test Gap: 2.0% → 28.7% (14배 증가)
- 원인: [Week 1 실험으로 규명]

### 💡 해결책: [최종 모델]
- Gap: 28.7% → [최종 Gap]%
- 방법: [Week 2 Feature Engineering]
"""
)
```

### Step 4: 실험 결과 로딩 셀 생성

```python
# 실험 결과 로딩 셀
experiment_loading_cell = nbformat.v4.new_code_cell(
    source="""
import pandas as pd
import glob
from pathlib import Path

results_dir = Path('../experiments/stakeholder_distrust_removal/results')

# Week 1 결과 로딩
week1_files = {
    'kfold': glob.glob(str(results_dir / 'week1' / 'week1_kfold_cv_*.csv')),
    'dist': glob.glob(str(results_dir / 'week1' / 'week1_distribution_comparison_*.csv')),
    'smote': glob.glob(str(results_dir / 'week1' / 'week1_smote_ablation_*.csv'))
}

# 최신 파일 로딩
week1_kfold = pd.read_csv(max(week1_files['kfold'], key=os.path.getctime)) if week1_files['kfold'] else pd.DataFrame()
week1_dist = pd.read_csv(max(week1_files['dist'], key=os.path.getctime)) if week1_files['dist'] else pd.DataFrame()
week1_smote = pd.read_csv(max(week1_files['smote'], key=os.path.getctime)) if week1_files['smote'] else pd.DataFrame()

print("✅ Week 1 실험 결과 로딩 완료")
"""
)
```

### Step 5: 셀 조립

```python
# 최종 노트북 생성
final_notebook = nbformat.v4.new_notebook()

# 셀 순서대로 추가
final_notebook.cells = [
    exec_summary,
    experiment_loading_cell,
    # ... Part 2 셀들 (순서대로)
    # ... Part 3 셀들 (순서대로)
    # ... Week 1 분석 셀들
    # ... Week 2 분석 셀들
    # ... 최종 결론 셀들
]

# 메타데이터 설정
final_notebook.metadata = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'codemirror_mode': {'name': 'ipython', 'version': 3},
        'file_extension': '.py',
        'mimetype': 'text/x-python',
        'name': 'python',
        'nbconvert_exporter': 'python',
        'pygments_lexer': 'ipython3',
        'version': '3.8.0'
    }
}
```

### Step 6: 저장

```python
# 노트북 저장
output_path = 'notebooks/최종제출_이해관계자불신지수_제거_모델_완전판.ipynb'

with open(output_path, 'w', encoding='utf-8') as f:
    nbformat.write(final_notebook, f)

print(f"✅ 최종 노트북 저장: {output_path}")
```

---

## ✅ 검증 체크리스트

생성된 노트북이 다음 조건을 만족하는지 확인하세요:

### 구조 검증
- [ ] Part 0: Executive Summary 존재
- [ ] Part 1: 환경 설정 (Part 2 코드 유지)
- [ ] Part 2: 특성 공학 (Part 2 코드 유지)
- [ ] Part 3-4: Baseline/Current 모델링 (Part 3 코드 유지)
- [ ] Part 5: Week 1 진단 (NEW)
- [ ] Part 6: Week 2 Feature Engineering (NEW)
- [ ] Part 7: 최종 모델 선정 (NEW)
- [ ] Part 8: 비즈니스 임팩트 (NEW)
- [ ] Part 9: 결론 (NEW)

### 코드 검증
- [ ] Part 2 코드 90% 이상 유지
- [ ] Part 3 코드 90% 이상 유지
- [ ] 실험 결과 로딩 코드 정상 작동
- [ ] 한글 폰트 설정 포함
- [ ] UTF-8 인코딩 적용

### 논리성 검증
- [ ] 문제 → 가설 → 실험 → 결과 → 해결 흐름 명확
- [ ] 각 섹션에 "왜?" 설명 존재
- [ ] 의사결정 기준 사전 정의
- [ ] 최종 모델 선택 근거 명시
- [ ] 비즈니스 임팩트 정량화

### 실행 가능성
- [ ] 모든 셀 순차 실행 시 에러 없음
- [ ] 실험 결과 파일 경로 정확
- [ ] 시각화 정상 작동
- [ ] 한글 깨짐 없음

---

## 🎬 최종 실행

```bash
# 노트북 생성 (Python 스크립트 또는 Claude Code 직접 실행)
python create_final_notebook.py

# Jupyter로 실행 및 검증
jupyter notebook notebooks/최종제출_이해관계자불신지수_제거_모델_완전판.ipynb
```

---

**Ready to create the perfect submission notebook! 🎓**
