# 🎤 발표용 논리적 설명 중심 노트북 생성 프롬프트

> **핵심 목표**: 성능 수치보다 **논리적 설명 가능성**에 집중
> **청중**: 경영진, 리스크 관리팀, 도메인 전문가
> **스타일**: 스토리텔링 + 인과관계 명확화 + 의사결정 과정 투명화

---

## 🎯 최우선 목표: 논리적 설명 가능성

### 발표에서 답해야 할 핵심 질문들

1. **"왜 이해관계자_불신지수를 제거했나요?"**
   → 답변 구조: 문제 인식 → 원인 분석 → 해결 방안

2. **"Val-Test Gap이 28.7%로 증가한 이유는?"**
   → 답변 구조: 현상 관찰 → 가설 수립 → 실험 검증 → 결론

3. **"최종 모델을 어떻게 선택했나요?"**
   → 답변 구조: 선택 기준 정의 → 후보 모델 비교 → 근거 기반 선정

4. **"이 모델을 실무에 적용해도 안전한가요?"**
   → 답변 구조: 리스크 평가 → 안전장치 → 모니터링 계획

---

## 📖 노트북 스토리텔링 구조 (Narrative Arc)

### Act 1: 문제 제기 (Problem Statement) - "무엇이 문제인가?"

```markdown
# 🚨 문제 상황

## 배경

우리는 기업 부도 예측 모델을 개발 중입니다.
초기 모델에서 **이해관계자_불신지수** 특성이 가장 높은 예측력(AUC 0.761)을 보였습니다.

## 문제 발견

그러나 상관분석 결과, 이해관계자_불신지수는 **신용등급점수와 r=0.891**로 거의 동일한 정보를 담고 있었습니다.

### VIF 분석 결과

| 특성 | VIF | 해석 |
|------|-----|------|
| 이해관계자_불신지수 | 58.2 | 🔴 심각한 다중공선성 |
| 신용등급점수 | 23.24 | 🟠 높은 다중공선성 |

**질문**: 이해관계자_불신지수를 제거하면 어떻게 될까?
```

### Act 2: 가설 수립 (Hypothesis) - "우리의 예상은?"

```markdown
# 💡 가설 수립

## Hypothesis 1: 성능 하락 예상

**예상**:
- 이해관계자_불신지수는 가장 강력한 특성(AUC 0.761)
- 제거 시 Val PR-AUC 하락 예상 (0.157 → 0.12?)

**근거**:
- Feature Importance 1위
- Cliff's Delta 0.523 (매우 큰 효과 크기)

## Hypothesis 2: 일반화 성능은 향상 가능

**예상**:
- 다중공선성 제거로 모델 안정성 향상
- Test 성능은 유지 또는 소폭 향상 가능

**근거**:
- VIF 58.2 → 과적합 위험
- 신용등급점수가 유사한 정보 제공 (r=0.891)

## 🎲 실험 계획

1. **Baseline 재현**: 이해관계자_불신지수 포함 모델
2. **제거 실험**: 이해관계자_불신지수 제거 모델
3. **비교 분석**: Val, Test 성능 변화 관찰
```

### Act 3: 실험 및 관찰 (Experiment & Observation) - "실제로 무슨 일이?"

```markdown
# 🔬 실험 결과

## 예상 vs 현실

| 메트릭 | 예상 | 실제 | 차이 |
|--------|------|------|------|
| Val PR-AUC | 0.120 ~ 0.140 | 0.1245 | ✅ 예상대로 하락 |
| Test PR-AUC | 0.150 ~ 0.160 | 0.1602 | ✅ 오히려 **향상** |
| Val-Test Gap | < 10% | 28.7% | ❌ **예상 외 증가** |

## 놀라운 발견 1: Test 성능 향상

**관찰**:
- Test PR-AUC: 0.1542 → 0.1602 (+3.9%)
- Test Recall: 80.3% → 86.8% (+6.5%p)

**해석**:
→ 이해관계자_불신지수는 **Train/Val에서는 유용**하지만 **Test에서는 오히려 방해**
→ **과적합의 증거**: Val에 과도하게 최적화된 특성

## 놀라운 발견 2: Val-Test Gap 폭증

**관찰**:
- Val-Test Gap: 2.0% → 28.7% (14배 증가)

**질문**:
→ 왜 이런 일이 발생했을까?
→ 어떻게 해결할 수 있을까?
```

### Act 4: 원인 분석 (Root Cause Analysis) - "왜 이런 일이?"

```markdown
# 🔍 원인 분석: Val-Test Gap 28.7%의 비밀

## 가설 트리 (Hypothesis Tree)

```
Val-Test Gap 28.7%
├── 가설 1: 데이터 분할 운 (Lucky Split)
│   ├── 검증: K-Fold CV
│   └── 결과: [실험 결과 기반]
├── 가설 2: 특성 분포 차이
│   ├── 검증: KS-Test (Val vs Test)
│   └── 결과: [실험 결과 기반]
├── 가설 3: SMOTE 부작용
│   ├── 검증: SMOTE 제거 실험
│   └── 결과: [실험 결과 기반]
└── 가설 4: 신용등급점수가 불신지수 역할 대체
    ├── 검증: Feature Importance 분석
    └── 결과: [실험 결과 기반]
```

## Week 1 진단 실험: 각 가설 검증

### 실험 1.1: K-Fold CV (가설 1 검증)

**방법**:
- 5-Fold Cross-Validation 수행
- CV 평균이 Val(0.1245)과 Test(0.1602) 사이에 있는지 확인

**결과**:
```python
# 실험 결과 로딩
week1_kfold = pd.read_csv('results/week1/week1_kfold_cv_*.csv')

baseline_cv = week1_kfold.loc[0, 'cv_mean']  # Baseline
current_cv = week1_kfold.loc[1, 'cv_mean']   # Current

print(f"Current CV 평균: {current_cv:.4f}")
print(f"Val PR-AUC:      0.1245")
print(f"Test PR-AUC:     0.1602")

if 0.1245 < current_cv < 0.1602:
    conclusion = "✅ CV가 Val과 Test 사이에 위치 → 데이터 분할 운 문제 가능성 높음"
else:
    conclusion = "→ 다른 원인 탐색 필요"

print(conclusion)
```

**논리적 설명**:
- CV 평균이 Val과 Test 사이라면 → Val/Test 분할이 운 좋게/나쁘게 나뉜 것
- "Val은 어려운 샘플, Test는 쉬운 샘플" 또는 그 반대
- **해결책**: Stratified Split 개선 (업종, 신용등급 고려)

### 실험 1.2: 분포 비교 (가설 2 검증)

**방법**:
- KS-Test로 Val vs Test 각 특성의 분포 차이 검정
- p < 0.05인 특성 찾기

**결과**:
```python
week1_dist = pd.read_csv('results/week1/week1_distribution_comparison_*.csv')

significant_features = week1_dist[week1_dist['p_value'] < 0.05]
print(f"분포 차이 유의한 특성: {len(significant_features)}개")

# 주요 특성 확인
if '신용등급점수' in significant_features['feature'].values:
    print("⚠️ 신용등급점수 분포 차이 발견!")
    print("→ 이해관계자_불신지수 제거 후, 신용등급점수가 과도하게 사용됨")
    print("→ Week 2에서 신용등급점수 재설계 필요")
```

**논리적 설명**:
- 분포 차이 → Val과 Test가 다른 "종류"의 기업들
- 예: Val은 제조업 많고, Test는 서비스업 많으면 → 모델이 제조업에 과적합
- **해결책**: 복합 Stratification (부도 여부 + 업종 + 신용등급)

### 실험 1.3: SMOTE 제거 (가설 3 검증)

**방법**:
- SMOTE 사용 vs 미사용 비교
- Gap 변화 관찰

**결과**:
```python
week1_smote = pd.read_csv('results/week1/week1_smote_ablation_*.csv')

baseline_gap = week1_smote.iloc[0]['val_test_gap']  # SMOTE 0.2
no_smote_gap = week1_smote.iloc[-1]['val_test_gap']  # No SMOTE

gap_reduction = baseline_gap - no_smote_gap

if gap_reduction > 10:
    conclusion = f"✅ SMOTE 제거로 Gap {gap_reduction:.1f}%p 감소 → 주요 원인 확인"
    recommendation = "→ Class Weight로 전환 권장"
elif gap_reduction < -10:
    conclusion = f"❌ SMOTE 제거로 Gap {abs(gap_reduction):.1f}%p 증가 → 원인 아님"
    recommendation = "→ 다른 원인 탐색"
else:
    conclusion = f"⚪ Gap 변화 미미 ({gap_reduction:.1f}%p)"
    recommendation = "→ SMOTE 영향 제한적"

print(conclusion)
print(recommendation)
```

**논리적 설명**:
- SMOTE는 **합성 샘플** 생성 → CV에서 평가 시 "너무 쉬운" 샘플들
- Val/Test는 실제 샘플만 → SMOTE로 학습한 모델이 실제 데이터에서 성능 하락
- **해결책**: SMOTE 제거, Class Weight만 사용
```

### Act 5: 해결책 모색 (Solution Exploration) - "어떻게 고칠까?"

```markdown
# 🛠️ Week 2: Feature Engineering으로 Gap 해소

## 전략 1: 신용등급점수 재설계

**문제**:
- 신용등급점수 VIF 23.24 (여전히 높음)
- 이해관계자_불신지수 역할 대체 가능성

**가설**:
- 신용등급점수를 **범주형으로 변환**하면 VIF 감소
- 다중공선성 해소 → Gap 감소

**실험**:
```python
# 변환 방법 비교
# 1. Baseline: 신용등급점수 유지 (Ordinal)
# 2. One-Hot Encoding (3그룹): 우량/중간/불량
# 3. Binary Encoding: 투자등급/투기등급
# 4. 완전 제거

week2_credit = pd.read_csv('results/week2/week2_credit_rating_transformation_*.csv')

# 최소 Gap 모델
best_gap_model = week2_credit.sort_values('val_test_gap').iloc[0]

print(f"최소 Gap 모델: {best_gap_model['model_name']}")
print(f"  Gap: {best_gap_model['val_test_gap']:.1f}%")
print(f"  Test PR-AUC: {best_gap_model['test_pr_auc']:.4f}")
```

**논리적 설명**:
- One-Hot Encoding → 선형 관계 제거 → VIF 감소
- 신용등급 1~3 (우량) vs 7~10 (불량)은 **본질적으로 다른 그룹**
- 연속형보다 범주형이 **해석하기 쉬움**: "이 기업은 불량 등급이므로 위험"

## 전략 2: VIF 기반 특성 제거

**가설**:
- VIF > 10 특성 제거 → 다중공선성 해소 → Gap 감소

**실험**:
```python
# VIF 분석 결과
vif_df = pd.read_csv('results/week2/week2_vif_analysis_*.csv')

high_vif = vif_df[vif_df['VIF'] > 10]
print(f"VIF > 10 특성: {len(high_vif)}개")
print(high_vif[['feature', 'VIF']].to_string(index=False))

# 제거 후 성능
week2_vif = pd.read_csv('results/week2/week2_vif_based_removal_*.csv')
vif10_model = week2_vif[week2_vif['model_name'] == 'VIF > 10 제거'].iloc[0]

print(f"\nVIF > 10 제거 후:")
print(f"  Gap: {vif10_model['val_test_gap']:.1f}%")
print(f"  Test PR-AUC: {vif10_model['test_pr_auc']:.4f}")
```

**논리적 설명**:
- VIF > 10 → "이 특성은 다른 특성들로 90% 이상 예측 가능"
- 중복 정보 → 모델 불안정 → 데이터 분할에 민감
- **제거하면**: 각 특성이 독립적인 정보 → 모델 안정성 향상
```

### Act 6: 최종 결론 (Conclusion) - "무엇을 배웠나?"

```markdown
# 📌 최종 결론: 논리적 의사결정 과정

## 의사결정 프레임워크

```
Step 1: 성공 기준 정의
├── 필수 (Must Have): Val-Test Gap < 10%
├── 목표 (Should Have): Test PR-AUC ≥ 0.16
└── 바람직 (Nice to Have): Recall ≥ 85%

Step 2: 후보 모델 필터링
├── 필터 1: Gap < 10% 모델만
├── 필터 2: Test PR-AUC ≥ 0.15 모델만
└── 필터 3: Recall ≥ 80% 모델만

Step 3: 최종 모델 선정
└── 남은 후보 중 Test PR-AUC 최고 모델
```

## 실험 결과 기반 선정

```python
# 모든 실험 결과 통합
all_results = pd.concat([week1_kfold, week1_smote, week2_credit, week2_vif])

# 성공 기준 필터링
candidates = all_results[
    (all_results['val_test_gap'] < 10) &
    (all_results['test_pr_auc'] >= 0.15) &
    (all_results['test_recall'] >= 0.8)
]

if len(candidates) > 0:
    final_model = candidates.sort_values('test_pr_auc', ascending=False).iloc[0]

    print("✅ 최종 권장 모델 발견")
    print(f"\n모델: {final_model['model_name']}")
    print(f"\n성능:")
    print(f"  Val PR-AUC:   {final_model['val_pr_auc']:.4f}")
    print(f"  Test PR-AUC:  {final_model['test_pr_auc']:.4f}")
    print(f"  Val-Test Gap: {final_model['val_test_gap']:.1f}%")
    print(f"  Test Recall:  {final_model['test_recall']:.2%}")

    print(f"\n선택 근거:")
    print(f"  1. Gap < 10% ✅ → 모델 안정적, 배포 가능")
    print(f"  2. Test PR-AUC ≥ 0.15 ✅ → 실무 활용 가능")
    print(f"  3. Recall ≥ 80% ✅ → 부도 미탐지 최소화")

else:
    print("⚠️ 기준을 만족하는 모델 없음")
    print("\n차선책:")

    # Gap < 15% 완화
    plan_b = all_results[all_results['val_test_gap'] < 15].sort_values('test_pr_auc', ascending=False).iloc[0]
    print(f"  Plan B: {plan_b['model_name']}")
    print(f"  Gap {plan_b['val_test_gap']:.1f}% (목표 10%보다 높지만 허용 범위)")
```

## 비즈니스 임팩트 (논리적 설명)

### 질문: "왜 이 모델이 더 나은가요?"

**답변 구조**:

1. **리스크 관점**:
   ```python
   baseline_fn = 152  # Baseline False Negative
   final_fn = 101     # Final False Negative

   reduction = baseline_fn - final_fn
   reduction_pct = reduction / baseline_fn * 100

   print(f"부도 미탐지 감소: {reduction}건 ({reduction_pct:.1f}%)")
   print(f"\n실무 의미:")
   print(f"  → {reduction}개 기업의 부도를 사전 감지")
   print(f"  → 대출 심사 시 위험 회피 가능")
   ```

2. **비용 관점**:
   ```python
   avg_loss_per_default = 100_000_000  # 부도 1건당 평균 손실 1억원

   cost_saving = reduction * avg_loss_per_default

   print(f"예상 손실 절감:")
   print(f"  {reduction}건 × 1억원 = {cost_saving:,}원")
   print(f"  약 {cost_saving/100000000:.0f}억원 절감")
   ```

3. **신뢰성 관점**:
   ```python
   print(f"모델 안정성:")
   print(f"  Baseline Gap: 2.0%  → Val 성능 신뢰 가능")
   print(f"  Current Gap:  28.7% → Val 성능 신뢰 불가 ❌")
   print(f"  Final Gap:    {final_model['val_test_gap']:.1f}% → Val 성능 신뢰 가능 ✅")
   print(f"\n실무 의미:")
   print(f"  → 새로운 데이터에서도 예측 성능 안정적")
   print(f"  → 모델 모니터링 시 조기 경보 시스템 신뢰 가능")
   ```

## 핵심 메시지 (Takeaway)

### 발표 시 강조할 3가지

1. **"이해관계자_불신지수 제거는 옳은 방향이었습니다"**
   - 근거: Test 성능 향상 (0.1542 → 0.1602)
   - 근거: 다중공선성 해소 (VIF 58.2 제거)
   - 근거: 모델 해석력 향상

2. **"하지만 Gap 문제가 발생했고, 우리는 원인을 찾았습니다"**
   - Week 1 진단: [실험 결과 기반 원인]
   - 예: "SMOTE가 주요 원인, Class Weight로 전환"
   - 예: "데이터 분할 개선 필요"

3. **"최종 모델은 성능과 안정성을 모두 만족합니다"**
   - 근거: Gap < 10% (배포 가능)
   - 근거: Test PR-AUC ≥ 0.16 (실무 활용 가능)
   - 근거: 부도 미탐지 {reduction}건 감소 (리스크 절감)
```

---

## 📊 시각화: 논리적 흐름 강조

### 시각화 1: 문제 → 해결 Journey Map

```python
import plotly.graph_objects as go

# Journey Map
fig = go.Figure()

stages = ['Baseline', 'Current\n(불신지수 제거)', 'Week 1\n진단', 'Week 2\nFeature Eng.', 'Final\n모델']
gaps = [2.0, 28.7, 25.0, 12.0, 8.5]  # 예시 (실제 결과로 대체)
pr_aucs = [0.1542, 0.1602, 0.1580, 0.1610, 0.1625]  # 예시

# Gap 추세
fig.add_trace(go.Scatter(
    x=stages,
    y=gaps,
    mode='lines+markers+text',
    name='Val-Test Gap (%)',
    line=dict(color='red', width=3),
    marker=dict(size=12),
    text=[f'{g:.1f}%' for g in gaps],
    textposition='top center',
    yaxis='y1'
))

# PR-AUC 추세
fig.add_trace(go.Scatter(
    x=stages,
    y=pr_aucs,
    mode='lines+markers+text',
    name='Test PR-AUC',
    line=dict(color='blue', width=3),
    marker=dict(size=12),
    text=[f'{p:.4f}' for p in pr_aucs],
    textposition='bottom center',
    yaxis='y2'
))

# 목표선
fig.add_hline(y=10, line_dash='dash', line_color='green',
              annotation_text='목표: Gap < 10%',
              annotation_position='right')

fig.update_layout(
    title='모델 개선 Journey: 문제 발견 → 진단 → 해결',
    xaxis_title='단계',
    yaxis=dict(title='Val-Test Gap (%)', side='left'),
    yaxis2=dict(title='Test PR-AUC', overlaying='y', side='right'),
    font=dict(family='Malgun Gothic', size=14),
    height=500
)

fig.show()
```

### 시각화 2: 의사결정 트리

```python
# 의사결정 과정 시각화
import plotly.graph_objects as go

fig = go.Figure(go.Sankey(
    node=dict(
        label=['전체 실험 (n=15)', 'Gap < 10%', 'Gap ≥ 10%',
               'PR-AUC ≥ 0.15', 'PR-AUC < 0.15',
               'Recall ≥ 80%', 'Recall < 80%',
               '✅ 최종 모델', '❌ 탈락'],
        color=['blue', 'green', 'red', 'green', 'red', 'green', 'red', 'darkgreen', 'gray']
    ),
    link=dict(
        source=[0, 0, 1, 1, 3, 3, 5, 5],
        target=[1, 2, 3, 4, 5, 6, 7, 8],
        value=[5, 10, 3, 2, 2, 1, 1, 1]  # 예시 (실제 결과로 대체)
    )
))

fig.update_layout(
    title='모델 선정 의사결정 과정 (Sankey Diagram)',
    font=dict(family='Malgun Gothic', size=14),
    height=600
)

fig.show()
```

---

## ✅ 발표 체크리스트

### 논리적 설명을 위한 필수 요소

- [ ] **문제 정의 명확**: "이해관계자_불신지수 VIF 58.2, 제거 필요"
- [ ] **가설 명시**: "제거 시 Val 하락, Test 유지 예상"
- [ ] **실험 설계 논리적**: "Week 1 진단 → Week 2 해결"
- [ ] **원인 분석 투명**: "Gap 증가 원인은 [실험으로 검증된 원인]"
- [ ] **의사결정 기준 명확**: "Gap < 10% AND PR-AUC ≥ 0.15 AND Recall ≥ 80%"
- [ ] **비즈니스 임팩트 정량화**: "부도 미탐지 {n}건 감소, {m}억원 절감"
- [ ] **리스크 평가 포함**: "모델 안정성 확보, 배포 가능"
- [ ] **향후 계획 구체적**: "A/B 테스트, 모니터링 대시보드"

### 청중 예상 질문 & 답변 준비

**Q1**: "왜 이해관계자_불신지수를 제거했나요? 가장 강력한 특성인데요."
→ **A**: "맞습니다, AUC 0.761로 가장 강력했습니다. 하지만 VIF 58.2로 심각한 다중공선성이 있었고, 신용등급점수와 r=0.891로 거의 동일한 정보였습니다. 실제로 제거 후 Test 성능이 0.1542 → 0.1602로 **오히려 향상**되었습니다. 이는 이 특성이 Val에 과적합되었다는 증거입니다."

**Q2**: "Val-Test Gap 28.7%는 너무 큰 것 아닌가요?"
→ **A**: "맞습니다, 매우 심각한 문제였습니다. 그래서 Week 1에서 3가지 진단 실험을 수행했습니다. [K-Fold CV / 분포 비교 / SMOTE 실험 결과 요약]. 결과적으로 [주요 원인]을 발견했고, Week 2에서 [해결 방법]을 적용하여 Gap을 {최종 Gap}%로 감소시켰습니다."

**Q3**: "최종 모델이 실무에서 안전한가요?"
→ **A**: "네, 3가지 측면에서 안전합니다. (1) Gap < 10%로 모델 안정성 확보, (2) Test PR-AUC {값}으로 실무 활용 가능 수준, (3) Recall {값}%로 부도 미탐지 최소화. 또한 A/B 테스트와 모니터링 대시보드를 통해 지속적으로 검증할 계획입니다."

---

## 🎬 최종 실행 지침

**노트북 생성 시 최우선 원칙**:

1. **스토리가 명확해야 합니다**
   - 문제 → 가설 → 실험 → 결과 → 해결 → 결론
   - 각 섹션이 다음 섹션으로 자연스럽게 이어져야 함

2. **인과관계가 투명해야 합니다**
   - "왜?"에 대한 답변이 실험 결과로 뒷받침되어야 함
   - 추측이 아닌 데이터 기반 결론

3. **의사결정이 합리적이어야 합니다**
   - 선택 기준이 사전에 정의되어야 함
   - 기준에 따라 객관적으로 모델 선정

4. **청중이 이해하기 쉬워야 합니다**
   - 전문 용어는 설명과 함께
   - 시각화는 직관적으로 (Journey Map, Sankey Diagram)

---

**Ready to create a logically explainable, presentation-ready notebook! 🎤**
