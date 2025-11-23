# 🔬 이해관계자_불신지수 제거 모델 - 실험 프로젝트

> **목적**: 이해관계자_불신지수 제거 후 발생한 Val-Test Gap 28.7% 문제 해결
> **기간**: 2025-11-23 ~ 2025-11-30 (예상)
> **담당**: Claude Code + 사용자

---

## 📊 프로젝트 개요

### 문제 상황

**Baseline 모델 (이해관계자_불신지수 포함)**:
- 특성: 27개
- Val PR-AUC: 0.1572
- Test PR-AUC: 0.1542
- Val-Test Gap: 2.0% ✅

**Current 모델 (이해관계자_불신지수 제거)**:
- 특성: 26개
- Val PR-AUC: 0.1245 📉
- Test PR-AUC: 0.1602 📈
- Val-Test Gap: 28.7% ❌ (14배 증가!)

### 핵심 질문

1. 왜 Test 성능은 향상했는데 Val 성능은 하락했는가?
2. Val-Test Gap 28.7%의 원인은 무엇인가?
3. Gap을 10% 이하로 낮추면서 Test 성능을 유지할 수 있는가?

---

## 🗂️ 프로젝트 구조

```
experiments/stakeholder_distrust_removal/
├── README.md                          # 이 파일
├── run_all_experiments.py             # 전체 실험 실행 스크립트
├── NOTEBOOK_GENERATION_PROMPT.md      # 노트북 생성 프롬프트 (기술 중심)
├── FINAL_PROMPT_발표용_논리적_설명.md   # 논리적 설명 프롬프트 (발표 중심)
├── FINAL_SUBMISSION_NOTEBOOK_PROMPT.md # 최종 제출용 프롬프트 (코드 보존 + 논리)
│
├── scripts/                           # 공통 유틸리티
│   └── common_utils.py                # 데이터 로딩, 파이프라인, 평가 함수
│
├── week1_diagnosis/                   # Week 1: 진단 실험
│   ├── exp1_kfold_cv.py              # K-Fold CV 재검증
│   ├── exp2_distribution_comparison.py # Val vs Test 분포 비교
│   └── exp3_smote_ablation.py        # SMOTE 제거 실험
│
├── week2_feature_engineering/         # Week 2: Feature Engineering
│   ├── exp1_credit_rating_transformation.py # 신용등급점수 변환
│   └── exp2_vif_based_removal.py     # VIF 기반 특성 제거
│
├── week3_model_improvement/           # Week 3: 모델 개선 (선택 사항)
│   ├── exp1_stacking_ensemble.py     # Stacking Ensemble
│   ├── exp2_early_stopping.py        # Early Stopping
│   └── exp3_dart_mode.py             # LightGBM DART
│
└── results/                          # 실험 결과 (자동 생성)
    ├── week1/
    │   ├── week1_kfold_cv_20251123_*.csv
    │   ├── week1_distribution_comparison_20251123_*.csv
    │   └── week1_smote_ablation_20251123_*.csv
    ├── week2/
    │   ├── week2_credit_rating_transformation_20251123_*.csv
    │   ├── week2_vif_analysis_20251123_*.csv
    │   └── week2_vif_based_removal_20251123_*.csv
    └── experiment_summary_20251123_*.json
```

---

## 🚀 빠른 시작

### 1. 환경 확인

```bash
# Python 3.8+
python --version

# 필요한 패키지 확인
pip list | grep -E "pandas|numpy|sklearn|catboost|xgboost|lightgbm|imblearn"
```

### 2. 전체 실험 실행 (권장)

```bash
cd /home/user/aaa/experiments/stakeholder_distrust_removal

# 전체 실험 실행 (Week 1 + Week 2)
python run_all_experiments.py
```

**예상 소요 시간**: 30분 ~ 1시간

### 3. 개별 실험 실행

```bash
# Week 1: K-Fold CV
python week1_diagnosis/exp1_kfold_cv.py

# Week 1: 분포 비교
python week1_diagnosis/exp2_distribution_comparison.py

# Week 1: SMOTE Ablation
python week1_diagnosis/exp3_smote_ablation.py

# Week 2: 신용등급점수 변환
python week2_feature_engineering/exp1_credit_rating_transformation.py

# Week 2: VIF 기반 제거
python week2_feature_engineering/exp2_vif_based_removal.py
```

### 4. 결과 확인

```bash
# 결과 파일 확인
ls -lh results/week1/
ls -lh results/week2/

# CSV 파일 미리보기
head -10 results/week1/week1_kfold_cv_*.csv
```

---

## 📋 실험 설계

### Week 1: 진단 실험 (Root Cause Analysis)

**목표**: Val-Test Gap 28.7%의 원인 규명

| 실험 | 가설 | 방법 | 성공 기준 |
|------|------|------|-----------|
| **1.1 K-Fold CV** | 데이터 분할 운 (Lucky Split) | 5-Fold CV 수행, CV 평균 확인 | CV 평균이 Val~Test 사이 |
| **1.2 분포 비교** | Val vs Test 특성 분포 차이 | KS-Test (p < 0.05) | 유의한 차이 5개 이상 발견 |
| **1.3 SMOTE Ablation** | SMOTE 부작용 | SMOTE 제거 후 Gap 변화 | Gap 10%p 이상 감소 |

### Week 2: Feature Engineering

**목표**: Gap을 10% 이하로 낮추면서 Test 성능 유지

| 실험 | 전략 | 방법 | 성공 기준 |
|------|------|------|-----------|
| **2.1 신용등급 변환** | VIF 감소, 해석력 향상 | One-Hot / Binary Encoding | Gap < 10% AND Test PR-AUC ≥ 0.16 |
| **2.2 VIF 제거** | 다중공선성 해소 | VIF > 10 특성 제거 | Gap < 10% AND Test PR-AUC ≥ 0.15 |

### Week 3: 모델 개선 (Optional)

| 실험 | 전략 | 성공 기준 |
|------|------|-----------|
| **3.1 Stacking** | 모델 다양성 | Test PR-AUC ≥ 0.17 |
| **3.2 Early Stopping** | 과적합 방지 | Gap < 5% |
| **3.3 DART** | Tree Dropout | CV-Val 안정성 향상 |

---

## 📊 결과 해석 가이드

### 실험 결과 CSV 파일 구조

**week1_kfold_cv_*.csv**:
```csv
experiment,n_features,cv_mean,cv_std,cv_min,cv_max,cv_fold1,cv_fold2,cv_fold3,cv_fold4,cv_fold5
Baseline (이해관계자_불신지수 포함),27,0.1560,0.0120,0.1420,0.1680,0.1550,0.1570,0.1520,0.1600,0.1560
Current (이해관계자_불신지수 제거),26,0.1420,0.0180,0.1200,0.1620,0.1400,0.1450,0.1380,0.1500,0.1370
```

**해석**:
- `cv_mean`: 5-Fold CV 평균 PR-AUC
- `cv_std`: 표준편차 (높을수록 불안정)
- `cv_fold1~5`: 각 Fold별 성능

**분석 질문**:
1. CV 평균이 Val(0.1245)과 Test(0.1602) 사이에 있는가?
   - Yes → 데이터 분할 운 문제 가능성
   - No → 다른 원인 탐색

2. CV 표준편차가 큰가? (> 0.03)
   - Yes → 모델 불안정
   - No → 모델 안정적

---

**week1_smote_ablation_*.csv**:
```csv
model_name,val_pr_auc,test_pr_auc,val_test_gap,test_recall,test_f2
Baseline: SMOTE (0.2),0.1245,0.1602,28.7,0.8684,0.2046
SMOTE (0.5),0.1280,0.1590,24.2,0.8550,0.2010
No SMOTE (Class Weight Only),0.1310,0.1580,20.6,0.8420,0.1980
```

**분석 질문**:
1. Baseline Gap (28.7%) vs No SMOTE Gap (20.6%)
   - 차이 > 10%p → SMOTE가 주요 원인
   - 차이 < 10%p → SMOTE 영향 제한적

2. Test PR-AUC 변화
   - 증가 → SMOTE 제거 후 성능 향상
   - 감소 → SMOTE 필요

---

**week2_credit_rating_transformation_*.csv**:
```csv
model_name,n_features,val_pr_auc,test_pr_auc,val_test_gap,test_recall
Baseline: 신용등급점수 유지,26,0.1245,0.1602,28.7,0.8684
One-Hot Encoding (3그룹),28,0.1320,0.1610,22.0,0.8550
Binary Encoding,27,0.1280,0.1595,24.6,0.8600
신용등급점수 완전 제거,25,0.1100,0.1520,38.2,0.8200
```

**분석**:
- One-Hot Encoding: Gap 감소 (28.7% → 22.0%)
- 완전 제거: Gap 오히려 증가 (28.7% → 38.2%) ❌

**결론**: One-Hot Encoding 선택

---

## 🎯 성공 기준

### 최종 모델 선정 기준 (우선순위)

1. **Val-Test Gap < 10%** (필수)
   - 모델 안정성 확보
   - 배포 시 예측 성능 신뢰 가능

2. **Test PR-AUC ≥ 0.15** (목표)
   - 실무 활용 가능 수준
   - Baseline (0.1542) 이상 유지

3. **Test Recall ≥ 80%** (실무 요구사항)
   - 부도 기업의 80% 이상 탐지
   - False Negative 최소화

### 모델 비교 예시

| 모델 | Val PR-AUC | Test PR-AUC | Gap | Recall | 선정 |
|------|-----------|------------|-----|--------|------|
| Baseline | 0.1572 | 0.1542 | 2.0% | 80.3% | ⚪ (Gap 우수, but 다중공선성) |
| Current | 0.1245 | 0.1602 | 28.7% | 86.8% | ❌ (Gap 과도) |
| Week2 One-Hot | 0.1320 | 0.1610 | 22.0% | 85.5% | ⚠️ (Gap 여전히 높음) |
| Week2 VIF 제거 | 0.1450 | 0.1590 | 9.7% | 84.2% | ✅ (모든 기준 만족!) |

---

## 📈 시각화 가이드

### Journey Map (모델 개선 과정)

```python
import plotly.graph_objects as go

stages = ['Baseline', 'Current', 'Week1 진단', 'Week2 One-Hot', 'Week2 VIF', 'Final']
gaps = [2.0, 28.7, 25.0, 22.0, 9.7, 9.7]
pr_aucs = [0.1542, 0.1602, 0.1590, 0.1610, 0.1590, 0.1590]

fig = go.Figure()

# Gap 추세
fig.add_trace(go.Scatter(
    x=stages, y=gaps,
    mode='lines+markers+text',
    name='Val-Test Gap (%)',
    line=dict(color='red', width=3),
    marker=dict(size=12),
    text=[f'{g:.1f}%' for g in gaps],
    textposition='top center'
))

# 목표선
fig.add_hline(y=10, line_dash='dash', line_color='green',
              annotation_text='목표: Gap < 10%')

fig.update_layout(
    title='모델 개선 Journey: Gap 감소 과정',
    xaxis_title='단계',
    yaxis_title='Val-Test Gap (%)',
    font=dict(family='Malgun Gothic', size=14),
    height=500
)

fig.show()
```

---

## 🛠️ 트러블슈팅

### 문제 1: 실험 결과 파일이 없음

```bash
# 증상
FileNotFoundError: [Errno 2] No such file or directory: '.../week1_kfold_cv_*.csv'

# 원인
실험을 아직 실행하지 않음

# 해결
python run_all_experiments.py  # 전체 실험 실행
```

### 문제 2: 한글 깨짐

```python
# 해결
import matplotlib.pyplot as plt
import platform

if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='NanumGothic')

plt.rc('axes', unicode_minus=False)
```

### 문제 3: 메모리 부족

```python
# 증상
MemoryError: Unable to allocate ...

# 해결
# common_utils.py에서 샘플링 비율 조정
def calculate_vif_all_features(X, sample_ratio=0.1):  # 0.2 → 0.1로 감소
    ...
```

---

## 📚 참고 문서

- **분석 보고서**: `/home/user/aaa/이해관계자불신지수_제거_모델_분석보고서.md`
- **프로젝트 가이드**: `/home/user/aaa/CLAUDE.md`
- **Part 2 요약**: `/home/user/aaa/docs/notebook_summaries/발표_Part2_도메인_특성_공학_완전판_summary.md`

---

## 🎓 노트북 생성

실험 완료 후 최종 제출용 노트북 생성:

### 방법 1: 기술 중심 노트북

```bash
# NOTEBOOK_GENERATION_PROMPT.md 사용
# - 실험 결과 종합
# - 시각화 중심
# - 기술적 분석
```

### 방법 2: 발표용 논리적 설명 노트북

```bash
# FINAL_PROMPT_발표용_논리적_설명.md 사용
# - 스토리텔링 구조
# - 인과관계 명확화
# - 의사결정 과정 투명화
```

### 방법 3: 최종 제출용 (기존 코드 보존 + 논리)

```bash
# FINAL_SUBMISSION_NOTEBOOK_PROMPT.md 사용 (권장)
# - Part 2, 3 코드 90% 유지
# - 논리적 설명 추가
# - 실험 결과 통합
```

**Claude Code에게 프롬프트 전달**:
```
FINAL_SUBMISSION_NOTEBOOK_PROMPT.md 파일을 읽고,
최종 제출용 노트북을 생성해줘.

입력:
- notebooks/발표_Part2_도메인_특성_공학_완전판_executed.ipynb
- notebooks/발표_Part3_모델링_및_최적화_v3_완전판 copy.ipynb
- experiments/stakeholder_distrust_removal/results/

출력:
- notebooks/최종제출_이해관계자불신지수_제거_모델_완전판.ipynb
```

---

## ✅ 체크리스트

### 실험 수행 전
- [ ] 데이터 파일 존재 확인 (`data/features/domain_based_features_완전판.csv`)
- [ ] 필요한 패키지 설치 확인
- [ ] RANDOM_STATE = 42 설정 확인

### 실험 수행 중
- [ ] Week 1 실험 3개 완료
- [ ] Week 2 실험 2개 완료
- [ ] 결과 파일 생성 확인 (`results/` 디렉토리)

### 실험 수행 후
- [ ] 모든 실험 결과 CSV 파일 존재
- [ ] Val-Test Gap < 10% 모델 발견
- [ ] Test PR-AUC ≥ 0.15 유지
- [ ] 최종 모델 선정 완료

### 노트북 생성 후
- [ ] 기존 코드 90% 이상 유지
- [ ] 논리적 흐름 명확 (문제 → 가설 → 실험 → 결과 → 해결)
- [ ] 모든 셀 순차 실행 시 에러 없음
- [ ] 한글 깨짐 없음
- [ ] 시각화 정상 작동

---

**Good luck with your experiments! 🚀**
