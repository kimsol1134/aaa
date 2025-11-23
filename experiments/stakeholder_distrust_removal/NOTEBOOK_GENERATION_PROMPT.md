# 📘 Jupyter Notebook 생성 프롬프트 - 이해관계자_불신지수 제거 모델 분석

> **대상**: Claude Code (claude.ai/code)
> **목적**: 실험 결과를 종합한 분석 노트북 자동 생성
> **기법**: Role-based + Chain-of-Thought + Structured Output + Few-shot Examples
> **날짜**: 2025-11-23

---

## 🎯 프롬프트 엔지니어링 구조

이 프롬프트는 다음 고급 기법을 사용합니다:

1. **Role-based Prompting** → 시니어 데이터 사이언티스트 역할 부여
2. **Chain-of-Thought (CoT)** → 단계별 추론 과정 명시
3. **Few-shot Examples** → 노트북 셀 예시 제공
4. **Structured Output** → 명확한 출력 형식 (Jupyter Notebook JSON)
5. **Constraints** → 한국어, UTF-8, 폰트 설정 등 제약 조건
6. **Self-Consistency** → 실험 결과 검증 로직 포함

---

## 🧑‍💼 Role Definition (역할 정의)

당신은 **시니어 데이터 사이언티스트**이자 **ML 엔지니어**입니다.

**전문 분야**:
- 한국 기업 부도 예측 모델링
- 불균형 데이터 처리 (SMOTE, Class Weight)
- Feature Engineering & Selection
- 모델 진단 (Val-Test 괴리, 과적합 분석)
- 실험 설계 및 결과 해석

**당신의 임무**:
`/home/user/aaa/experiments/stakeholder_distrust_removal/results/` 디렉토리에 저장된 **모든 실험 결과**를 종합하여, **발표용 Jupyter Notebook**을 생성하는 것입니다.

---

## 📋 Task Description (작업 설명)

### 배경 컨텍스트

**이전 모델 (Baseline)**:
- 특성: 27개 (이해관계자_불신지수 **포함**)
- Val PR-AUC: 0.1572
- Test PR-AUC: 0.1542
- Val-Test Gap: 2.0%

**현재 모델 (Current)**:
- 특성: 26개 (이해관계자_불신지수 **제거**)
- Val PR-AUC: 0.1245
- Test PR-AUC: 0.1602
- Val-Test Gap: **28.7%** ← 심각한 괴리!

**핵심 문제**:
이해관계자_불신지수 제거 후 Test 성능은 향상(0.1542 → 0.1602)되었으나, Val-Test 괴리가 14배 증가(2.0% → 28.7%)했습니다.

**목표**:
Week 1, 2, 3 실험을 통해 **Val-Test Gap < 10%**를 달성하면서 **Test PR-AUC ≥ 0.16**을 유지하는 모델을 찾는 것입니다.

---

### 생성할 노트북 구조

다음 섹션으로 구성된 Jupyter Notebook (`.ipynb`)을 생성하세요:

#### **Part 1: Executive Summary (경영진 요약)**

```markdown
# 📊 이해관계자_불신지수 제거 모델 - 종합 분석 보고서

## 핵심 발견사항

### ✅ 좋은 소식: Test 성능 향상
- ...

### ⚠️ 우려 사항: Val-Test 괴리 심화
- ...

### 💡 해결책: [실험 결과 기반 최적 모델]
- ...
```

#### **Part 2: Week 1 진단 실험 결과**

**섹션 2.1: K-Fold Cross-Validation 재검증**

```python
# 실험 결과 로딩
import pandas as pd
week1_kfold = pd.read_csv('results/week1/week1_kfold_cv_*.csv')

# 결과 시각화
import plotly.graph_objects as go
fig = go.Figure(...)
fig.show()

# 분석
if week1_kfold['cv_mean'].iloc[1] between 0.1245 and 0.1602:
    print("✅ 데이터 분할 운(Lucky Split) 문제 확인")
    print("→ Stratified Split 개선 필요")
else:
    print("→ 다른 원인 탐색 필요")
```

**섹션 2.2: Val vs Test 분포 비교**

```python
# 실험 결과 로딩
week1_dist = pd.read_csv('results/week1/week1_distribution_comparison_*.csv')

# 분포 차이 유의한 특성
significant_features = week1_dist[week1_dist['p_value'] < 0.05]
print(f"분포 차이 유의한 특성: {len(significant_features)}개")

# 시각화 (Plotly 사용)
...
```

**섹션 2.3: SMOTE Ablation Study**

```python
# 실험 결과 로딩
week1_smote = pd.read_csv('results/week1/week1_smote_ablation_*.csv')

# 비교 테이블
display(week1_smote[['model_name', 'val_pr_auc', 'test_pr_auc', 'val_test_gap']])

# Gap 변화 분석
baseline_gap = week1_smote.iloc[0]['val_test_gap']
no_smote_gap = week1_smote.iloc[-1]['val_test_gap']

if (baseline_gap - no_smote_gap) > 10:
    print("✅ SMOTE 제거로 Gap 감소 → Class Weight 전환 권장")
else:
    print("→ SMOTE는 주요 원인 아님")
```

#### **Part 3: Week 2 Feature Engineering 결과**

**섹션 3.1: 신용등급점수 변환**

```python
# 실험 결과 로딩
week2_credit = pd.read_csv('results/week2/week2_credit_rating_transformation_*.csv')

# 최적 변환 방법 찾기
best_model = week2_credit.sort_values('test_pr_auc', ascending=False).iloc[0]

print(f"🏆 최적 변환 방법: {best_model['model_name']}")
print(f"   Test PR-AUC: {best_model['test_pr_auc']:.4f}")
print(f"   Val-Test Gap: {best_model['val_test_gap']:.1f}%")
```

**섹션 3.2: VIF 기반 특성 제거**

```python
# VIF 분석 결과 로딩
vif_df = pd.read_csv('results/week2/week2_vif_analysis_*.csv')

# VIF > 10 특성
high_vif = vif_df[vif_df['VIF'] > 10]
print(f"VIF > 10 특성: {len(high_vif)}개")

# 제거 후 성능 변화
week2_vif = pd.read_csv('results/week2/week2_vif_based_removal_*.csv')
display(week2_vif)
```

#### **Part 4: 최종 권장 모델**

```python
# 모든 실험 결과 종합
all_results = pd.concat([
    week1_kfold.assign(category='Week1_KFold'),
    week1_smote.assign(category='Week1_SMOTE'),
    week2_credit.assign(category='Week2_Credit'),
    week2_vif.assign(category='Week2_VIF')
])

# 최적 모델 선택 기준
# 1. Val-Test Gap < 10% (필수)
# 2. Test PR-AUC ≥ 0.15 (목표)
# 3. Test Recall ≥ 80% (실무 요구사항)

candidates = all_results[
    (all_results['val_test_gap'] < 10) &
    (all_results['test_pr_auc'] >= 0.15) &
    (all_results['test_recall'] >= 0.8)
]

if len(candidates) > 0:
    final_model = candidates.sort_values('test_pr_auc', ascending=False).iloc[0]
    print(f"✅ 최종 권장 모델 발견:")
    print(f"   {final_model['model_name']}")
else:
    print("⚠️ 기준을 만족하는 모델 없음")
    # 차선책 제시
    ...
```

#### **Part 5: 비즈니스 임팩트 분석**

```python
# 혼동 행렬 비교
baseline_cm = [[baseline_tn, baseline_fp], [baseline_fn, baseline_tp]]
final_cm = [[final_tn, final_fp], [final_fn, final_tp]]

# 실무 임팩트 계산
print("부도 미탐지 감소:")
print(f"  Baseline: {baseline_fn}건")
print(f"  Final:    {final_fn}건")
print(f"  개선:     {baseline_fn - final_fn}건 ({(baseline_fn - final_fn)/baseline_fn*100:.1f}%)")

# 비용 절감 효과
# 가정: 부도 1건당 평균 손실 1억원
cost_saving = (baseline_fn - final_fn) * 100000000
print(f"\n💰 예상 비용 절감: {cost_saving:,}원")
```

#### **Part 6: 결론 및 향후 계획**

```markdown
## 📌 결론

### 핵심 발견

1. **[실험 결과 기반 작성]**
   - ...

2. **Val-Test 괴리 원인**
   - ...

3. **최적 모델 구성**
   - ...

### 권장 사항

1. **단기 (1주)**:
   - ...

2. **중기 (1개월)**:
   - ...

3. **장기 (3개월)**:
   - ...

### 다음 단계

- [ ] 최종 모델 프로덕션 배포
- [ ] A/B 테스트 설계
- [ ] 모니터링 대시보드 구축
```

---

## 🔧 Chain-of-Thought 추론 과정

노트북 생성 시 다음 단계를 따르세요:

### Step 1: 실험 결과 파일 탐색

```python
# 1. results/ 디렉토리 탐색
import os
from pathlib import Path

results_dir = Path('results')
all_results = {}

for week in ['week1', 'week2']:
    week_dir = results_dir / week
    if week_dir.exists():
        for file in week_dir.glob('*.csv'):
            # 파일 로딩 및 분석
            df = pd.read_csv(file)
            all_results[file.stem] = df
```

### Step 2: 실험 결과 검증

```python
# 2. 각 실험 결과의 유효성 검증
for exp_name, df in all_results.items():
    # 필수 컬럼 확인
    required_cols = ['model_name', 'val_pr_auc', 'test_pr_auc', 'val_test_gap']
    if not all(col in df.columns for col in required_cols):
        print(f"⚠️ {exp_name}: 필수 컬럼 누락")
        continue

    # 값 범위 검증
    if not (0 <= df['val_pr_auc'].max() <= 1):
        print(f"⚠️ {exp_name}: PR-AUC 값 이상")
```

### Step 3: 최적 모델 선정 로직

```python
# 3. 다단계 필터링으로 최적 모델 선정

# 필터 1: Gap < 10%
candidates_step1 = all_results[all_results['val_test_gap'] < 10]

# 필터 2: Test PR-AUC ≥ 0.15
candidates_step2 = candidates_step1[candidates_step1['test_pr_auc'] >= 0.15]

# 필터 3: Recall ≥ 80%
candidates_step3 = candidates_step2[candidates_step2['test_recall'] >= 0.8]

# 정렬: Test PR-AUC 내림차순
final_model = candidates_step3.sort_values('test_pr_auc', ascending=False).iloc[0]
```

### Step 4: 시각화 생성

```python
# 4. Plotly로 인터랙티브 차트 생성

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 예시: Val vs Test PR-AUC 비교
fig = go.Figure()

fig.add_trace(go.Bar(
    name='Val PR-AUC',
    x=all_results['model_name'],
    y=all_results['val_pr_auc'],
    marker_color='lightblue'
))

fig.add_trace(go.Bar(
    name='Test PR-AUC',
    x=all_results['model_name'],
    y=all_results['test_pr_auc'],
    marker_color='salmon'
))

fig.update_layout(
    title='실험별 Val vs Test PR-AUC 비교',
    xaxis_title='모델',
    yaxis_title='PR-AUC',
    barmode='group',
    font=dict(family='Malgun Gothic', size=12)  # 한글 폰트
)

fig.show()
```

### Step 5: 결과 해석 및 권장사항

```python
# 5. 실험 결과 기반 자동 인사이트 생성

insights = []

# Gap 분석
if final_model['val_test_gap'] < 5:
    insights.append("✅ Val-Test 괴리 매우 작음 (< 5%) → 모델 안정적")
elif final_model['val_test_gap'] < 10:
    insights.append("✅ Val-Test 괴리 허용 범위 (5~10%) → 배포 가능")
else:
    insights.append("⚠️ Val-Test 괴리 여전히 큼 (> 10%) → 추가 개선 필요")

# SMOTE 분석
if 'No SMOTE' in final_model['model_name']:
    insights.append("→ SMOTE 제거가 효과적, Class Weight 사용 권장")

# Feature 분석
if 'VIF' in final_model['model_name']:
    insights.append("→ 다중공선성 제거가 중요, VIF > 10 특성 제거")

print("\n".join(insights))
```

---

## 📐 Few-shot Examples (예시)

### 예시 1: 마크다운 셀 (Executive Summary)

```json
{
  "cell_type": "markdown",
  "metadata": {},
  "source": [
    "# 📊 이해관계자_불신지수 제거 모델 - 종합 분석 보고서\n",
    "\n",
    "**분석 기간**: 2025-11-23 ~ 2025-11-30  \n",
    "**분석자**: 시니어 데이터 사이언티스트  \n",
    "**목적**: Val-Test 괴리 해소 및 최적 모델 선정\n",
    "\n",
    "---\n",
    "\n",
    "## 핵심 발견사항\n",
    "\n",
    "### ✅ 좋은 소식: Test 성능 향상\n",
    "\n",
    "- **Test PR-AUC**: 0.1542 → 0.1602 (+3.9%)\n",
    "- **Test Recall**: 80.3% → 86.8% (+6.5%p)\n",
    "- **부도 미탐지 감소**: 실무 임팩트 향상\n",
    "\n",
    "### ⚠️ 우려 사항: Val-Test 괴리 심화\n",
    "\n",
    "- **Val-Test Gap**: 2.0% → 28.7% (14배 증가)\n",
    "- **원인**: [실험 결과 기반 작성]\n",
    "- **영향**: 모델 선택 신뢰도 저하, 배포 시 예측 불확실성\n"
  ]
}
```

### 예시 2: 코드 셀 (데이터 로딩 및 분석)

```json
{
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import plotly.graph_objects as go\n",
    "from pathlib import Path\n",
    "\n",
    "# 한글 폰트 설정\n",
    "import platform\n",
    "if platform.system() == 'Windows':\n",
    "    font_family = 'Malgun Gothic'\n",
    "elif platform.system() == 'Darwin':\n",
    "    font_family = 'AppleGothic'\n",
    "else:\n",
    "    font_family = 'NanumGothic'\n",
    "\n",
    "# 실험 결과 로딩\n",
    "results_dir = Path('../results')\n",
    "\n",
    "# Week 1 실험 결과\n",
    "week1_kfold = pd.read_csv(results_dir / 'week1' / 'week1_kfold_cv_20251123_*.csv')\n",
    "week1_smote = pd.read_csv(results_dir / 'week1' / 'week1_smote_ablation_20251123_*.csv')\n",
    "\n",
    "print('✅ 실험 결과 로딩 완료')\n",
    "print(f'   Week 1 K-Fold CV: {len(week1_kfold)} 실험')\n",
    "print(f'   Week 1 SMOTE: {len(week1_smote)} 실험')"
  ]
}
```

### 예시 3: 코드 셀 (시각화)

```json
{
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
    "# Val-Test Gap 비교 시각화\n",
    "fig = go.Figure()\n",
    "\n",
    "# Baseline\n",
    "fig.add_trace(go.Bar(\n",
    "    name='Baseline (이해관계자_불신지수 포함)',\n",
    "    x=['Val PR-AUC', 'Test PR-AUC', 'Gap (%)'],\n",
    "    y=[0.1572, 0.1542, 2.0],\n",
    "    marker_color='lightblue',\n",
    "    text=[0.1572, 0.1542, 2.0],\n",
    "    textposition='outside'\n",
    "))\n",
    "\n",
    "# Current\n",
    "fig.add_trace(go.Bar(\n",
    "    name='Current (이해관계자_불신지수 제거)',\n",
    "    x=['Val PR-AUC', 'Test PR-AUC', 'Gap (%)'],\n",
    "    y=[0.1245, 0.1602, 28.7],\n",
    "    marker_color='salmon',\n",
    "    text=[0.1245, 0.1602, 28.7],\n",
    "    textposition='outside'\n",
    "))\n",
    "\n",
    "fig.update_layout(\n",
    "    title='Baseline vs Current 모델 비교',\n",
    "    xaxis_title='메트릭',\n",
    "    yaxis_title='값',\n",
    "    barmode='group',\n",
    "    font=dict(family=font_family, size=12),\n",
    "    height=500\n",
    ")\n",
    "\n",
    "fig.show()"
  ]
}
```

---

## ✅ Constraints & Requirements (제약 조건)

### 필수 준수 사항

1. **언어**: 모든 텍스트는 한국어로 작성
2. **인코딩**: UTF-8 (파일 읽기/쓰기 시 `encoding='utf-8'` 또는 `encoding='utf-8-sig'`)
3. **폰트**: 한글 폰트 설정 필수 (Matplotlib, Plotly)
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

4. **경로**: 상대 경로 사용 (`../results/`, `../data/`)
5. **파일 로딩**: glob 패턴으로 최신 결과 파일 찾기
   ```python
   from pathlib import Path
   import glob

   # 예시: week1 kfold 결과 중 가장 최신 파일
   pattern = str(results_dir / 'week1' / 'week1_kfold_cv_*.csv')
   files = glob.glob(pattern)
   if files:
       latest_file = max(files, key=os.path.getctime)
       df = pd.read_csv(latest_file)
   ```

6. **시각화**: Plotly 우선 사용 (인터랙티브), Matplotlib/Seaborn은 보조
7. **결측치/무한대 처리**: 항상 안전하게 처리
   ```python
   df = df.fillna(0).replace([np.inf, -np.inf], 0)
   ```

8. **에러 핸들링**: try-except로 안전하게 처리
   ```python
   try:
       # 실험 결과 로딩
       df = pd.read_csv(file_path)
   except Exception as e:
       print(f"⚠️ 파일 로딩 실패: {e}")
       df = pd.DataFrame()  # 빈 데이터프레임
   ```

9. **출력 형식**: 깔끔한 포맷팅
   ```python
   print(f"{'='*80}")
   print(f"실험 결과 요약")
   print(f"{'='*80}")
   ```

10. **저장**: 최종 결과는 CSV로 저장
    ```python
    final_summary.to_csv('../results/final_summary.csv', index=False, encoding='utf-8-sig')
    ```

---

## 🎨 Structured Output Format (출력 형식)

생성할 노트북은 다음 JSON 구조를 따릅니다:

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["# 타이틀"]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": ["import pandas as pd"]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}
```

---

## 🚀 Execution Instructions (실행 지침)

### Step-by-Step 노트북 생성 프로세스

1. **실험 결과 수집**
   - `/home/user/aaa/experiments/stakeholder_distrust_removal/results/` 탐색
   - 모든 `.csv` 파일 로딩
   - 결과 유효성 검증

2. **데이터 전처리**
   - 결측치 처리
   - 컬럼명 표준화
   - 타임스탬프 파싱

3. **분석 및 인사이트 생성**
   - Week 1: 진단 (K-Fold, 분포, SMOTE)
   - Week 2: Feature Engineering (신용등급, VIF)
   - 최적 모델 선정

4. **시각화 생성**
   - Plotly 차트 (Bar, Line, Heatmap)
   - 한글 폰트 적용
   - 레이아웃 최적화

5. **노트북 조립**
   - 마크다운 셀 + 코드 셀 순차 배치
   - 실행 순서 최적화
   - 메타데이터 설정

6. **저장 및 검증**
   - `.ipynb` 파일로 저장
   - JSON 형식 검증
   - Jupyter에서 실행 가능 여부 확인

---

## 💡 Self-Consistency Verification (자가 검증)

노트북 생성 후 다음 사항을 검증하세요:

### 검증 체크리스트

- [ ] **실험 결과 파일 존재 확인**
  ```python
  assert Path('results/week1/week1_kfold_cv_*.csv').exists()
  ```

- [ ] **필수 컬럼 존재 확인**
  ```python
  required_cols = ['model_name', 'val_pr_auc', 'test_pr_auc', 'val_test_gap']
  assert all(col in df.columns for col in required_cols)
  ```

- [ ] **값 범위 검증**
  ```python
  assert 0 <= df['val_pr_auc'].max() <= 1
  assert 0 <= df['test_pr_auc'].max() <= 1
  ```

- [ ] **한글 폰트 적용 확인**
  ```python
  assert 'family' in plt.rcParams['font']
  ```

- [ ] **상대 경로 사용 확인**
  ```python
  assert all('../' in path for path in file_paths)
  ```

- [ ] **실행 가능성 확인**
  ```python
  # 노트북 셀 순차 실행 시 에러 없음
  ```

---

## 📝 Example Notebook Outline (노트북 개요 예시)

```
노트북 제목: 이해관계자_불신지수 제거 모델 - 종합 분석 및 최적화

1. Executive Summary (1 markdown cell)
   - 핵심 발견사항
   - 최종 권장 모델

2. 환경 설정 및 데이터 로딩 (2 code cells)
   - 라이브러리 import
   - 실험 결과 파일 로딩

3. Week 1: 진단 실험 (3 섹션)
   3.1. K-Fold CV (1 markdown + 2 code cells)
   3.2. 분포 비교 (1 markdown + 2 code cells)
   3.3. SMOTE Ablation (1 markdown + 2 code cells)

4. Week 2: Feature Engineering (2 섹션)
   4.1. 신용등급점수 변환 (1 markdown + 2 code cells)
   4.2. VIF 기반 제거 (1 markdown + 2 code cells)

5. 최적 모델 선정 (1 markdown + 3 code cells)
   - 전체 실험 결과 종합
   - 필터링 및 선정
   - 최종 모델 평가

6. 비즈니스 임팩트 (1 markdown + 2 code cells)
   - 혼동 행렬 비교
   - 비용 절감 효과

7. 결론 및 향후 계획 (1 markdown cell)
   - 핵심 인사이트
   - 권장 사항
   - 다음 단계

총 셀 수: ~25개 (markdown 10개 + code 15개)
```

---

## 🎯 Final Output Specification (최종 출력 사양)

### 파일 정보

- **파일명**: `이해관계자_불신지수_제거_모델_종합분석_완전판.ipynb`
- **경로**: `/home/user/aaa/notebooks/`
- **인코딩**: UTF-8
- **Jupyter 버전**: nbformat 4, nbformat_minor 4

### 품질 기준

1. **정확성**: 실험 결과를 정확히 반영
2. **가독성**: 깔끔한 포맷팅, 명확한 섹션 구분
3. **재현성**: 다른 사용자가 실행 시 동일한 결과 재현 가능
4. **실행 가능성**: 모든 셀이 순차 실행 시 에러 없음
5. **시각화**: 고품질 Plotly 차트, 한글 폰트 적용
6. **인사이트**: 데이터 기반 명확한 결론 및 권장사항

---

## 🔍 Troubleshooting (문제 해결)

### 자주 발생하는 문제

**문제 1**: 실험 결과 파일이 없음

```python
# 해결: glob 패턴으로 유연하게 찾기
import glob
pattern = 'results/week1/week1_kfold_cv_*.csv'
files = glob.glob(pattern)
if not files:
    print(f"⚠️ 파일 없음: {pattern}")
    print(f"→ 먼저 실험을 실행하세요: python run_all_experiments.py")
```

**문제 2**: 한글 깨짐

```python
# 해결: 폰트 설정 확인
import matplotlib.pyplot as plt
print(plt.rcParams['font.family'])  # 폰트 확인

# 재설정
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
```

**문제 3**: 결측치/무한대 에러

```python
# 해결: 전처리 함수
def clean_data(df):
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    return df

df = clean_data(df)
```

---

## ✨ Bonus: Advanced Features (추가 기능)

### 인터랙티브 대시보드

```python
# Plotly Dash로 인터랙티브 대시보드 생성 (선택 사항)
import plotly.express as px

# 실험 결과 비교 슬라이더
fig = px.bar(
    all_results,
    x='model_name',
    y='test_pr_auc',
    color='category',
    animation_frame='week',
    range_y=[0, 0.2],
    title='실험별 Test PR-AUC 변화'
)

fig.show()
```

### 통계적 유의성 검정

```python
# Wilcoxon Signed-Rank Test로 모델 비교
from scipy.stats import wilcoxon

baseline_scores = [...]  # Baseline 모델 CV scores
final_scores = [...]     # Final 모델 CV scores

stat, p_value = wilcoxon(baseline_scores, final_scores)

if p_value < 0.05:
    print(f"✅ 통계적 유의 (p={p_value:.4f})")
else:
    print(f"⚠️ 유의하지 않음 (p={p_value:.4f})")
```

---

## 📚 References (참고 자료)

- **분석 보고서**: `/home/user/aaa/이해관계자불신지수_제거_모델_분석보고서.md`
- **프로젝트 가이드**: `/home/user/aaa/CLAUDE.md`
- **실험 스크립트**: `/home/user/aaa/experiments/stakeholder_distrust_removal/`

---

## 🎬 Action Items (실행 항목)

**당신이 지금 해야 할 일**:

1. ✅ 이 프롬프트를 **완전히 이해**하세요
2. ✅ `/home/user/aaa/experiments/stakeholder_distrust_removal/results/` 디렉토리를 **탐색**하세요
3. ✅ **모든 실험 결과 파일**을 로딩하고 검증하세요
4. ✅ 위에서 제시한 **구조와 예시**를 따라 노트북을 생성하세요
5. ✅ **자가 검증 체크리스트**를 통과하는지 확인하세요
6. ✅ 최종 노트북을 `/home/user/aaa/notebooks/이해관계자_불신지수_제거_모델_종합분석_완전판.ipynb`에 **저장**하세요

---

**Ready? Let's generate an amazing analysis notebook! 🚀**
