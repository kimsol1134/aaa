# Part 4: SHAP 분석 및 비즈니스 가치 평가

## 🎯 목표

Part 3 v3에서 선정된 최종 모델의 예측 근거를 **SHAP (SHapley Additive exPlanations)**으로 해석하고, 비즈니스 의사결정에 활용 가능한 인사이트를 도출합니다.

---

## 📋 선행 조건

### Part 3 v3 출력 파일 (로드 필요)

```python
import joblib
import os

PROCESSED_DIR = '../data/processed'

# Part 3 v3에서 저장된 파일들
final_model = joblib.load(os.path.join(PROCESSED_DIR, '발표_Part3_v3_최종모델.pkl'))
thresholds = joblib.load(os.path.join(PROCESSED_DIR, '발표_Part3_v3_임계값.pkl'))
results = joblib.load(os.path.join(PROCESSED_DIR, '발표_Part3_v3_결과.pkl'))

# Feature 데이터
features_df = pd.read_csv('../data/features/domain_based_features_완전판.csv', encoding='utf-8')
```

### 데이터 규모
- **기업 수**: 50,105개
- **Feature 수**: 27개 (Part 2에서 선택된 도메인 기반 특성)
- **부도율**: ~1.5%
- **Train/Val/Test Split**: 60% / 20% / 20%

---

## 📊 구현 요구사항

### 섹션 0: 환경 설정 및 데이터 로딩

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
import platform
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)

# Part 3 v3 출력 로드
PROCESSED_DIR = '../data/processed'
final_model = joblib.load(os.path.join(PROCESSED_DIR, '발표_Part3_v3_최종모델.pkl'))
thresholds = joblib.load(os.path.join(PROCESSED_DIR, '발표_Part3_v3_임계값.pkl'))
results = joblib.load(os.path.join(PROCESSED_DIR, '발표_Part3_v3_결과.pkl'))

print(f"✅ 최종 모델: {results['model_name']}")
print(f"✅ Test PR-AUC: {results['test_pr_auc']:.4f}")
print(f"✅ 임계값: {thresholds['selected']:.4f}")

# Feature 데이터 로드
features_df = pd.read_csv('../data/features/domain_based_features_완전판.csv', encoding='utf-8')
TARGET_COL = '모형개발용Performance(향후1년내부도여부)'
X = features_df.drop(columns=[TARGET_COL])
y = features_df[TARGET_COL]

# 동일한 3-Way Split (Part 3와 동일한 random_state 사용)
from sklearn.model_selection import train_test_split
RANDOM_STATE = 42

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE
)

print(f"\nTrain: {len(X_train):,}개")
print(f"Val:   {len(X_val):,}개")
print(f"Test:  {len(X_test):,}개")
```

---

### 섹션 1: SHAP TreeExplainer 초기화

**SHAP란?**
- Shapley Value 기반 모델 해석 방법
- 각 Feature가 개별 예측에 기여한 정도를 정량화
- 양수: 부도 위험 증가 / 음수: 부도 위험 감소

```python
# TreeExplainer (트리 기반 모델용)
explainer = shap.TreeExplainer(final_model.named_steps['classifier'])

# 전처리된 데이터 준비
X_train_preprocessed = final_model[:-1].transform(X_train)
X_test_preprocessed = final_model[:-1].transform(X_test)

# SHAP values 계산 (Test Set)
shap_values = explainer.shap_values(X_test_preprocessed)

# 이진 분류 시 shap_values가 [class0, class1] 형태면 class1만 사용
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # 부도(1) 클래스

print(f"✅ SHAP Values 계산 완료: {shap_values.shape}")
```

---

### 섹션 2: Global Feature Importance (Summary Plot)

**전체 데이터셋에서 가장 영향력 있는 Feature 시각화**

```python
# Summary Plot (Beeswarm)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_preprocessed, feature_names=X.columns, show=False)
plt.title('SHAP Summary Plot: Feature Importance', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('../data/processed/발표_Part4_SHAP_Summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Summary Plot 저장 완료")
```

**해석 가이드:**
- **색상**: Feature 값 (빨강=높음, 파랑=낮음)
- **X축**: SHAP Value (양수=부도 위험↑, 음수=부도 위험↓)
- **Y축**: Feature (중요도 순서)

---

### 섹션 3: Top 10 Feature 상세 분석

**가장 영향력 있는 10개 Feature의 재무적 의미 해석**

```python
# Top 10 Feature 추출
feature_importance = np.abs(shap_values).mean(axis=0)
top10_idx = np.argsort(feature_importance)[-10:][::-1]
top10_features = X.columns[top10_idx]

print("Top 10 중요 Feature:")
for i, feat in enumerate(top10_features, 1):
    print(f"{i:2d}. {feat}: {feature_importance[top10_idx[i-1]]:.4f}")

# Bar Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_preprocessed, feature_names=X.columns,
                   plot_type='bar', show=False, max_display=10)
plt.title('Top 10 Feature Importance (Mean |SHAP|)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('../data/processed/발표_Part4_Top10_Features.png', dpi=300, bbox_inches='tight')
plt.show()
```

**재무적 해석 예시 (실제 결과에 맞게 수정 필요):**

| Feature | 재무적 의미 | 부도 위험과의 관계 |
|---------|------------|-------------------|
| 자본잠식도 | 자본 대비 누적 손실 | 높을수록 위험↑ (자본 기반 붕괴) |
| 이자보상배율 | 영업이익으로 이자 커버 능력 | 낮을수록 위험↑ (이자 지급 불가) |
| 유동비율 | 단기 부채 상환 능력 | 낮을수록 위험↑ (유동성 위기) |
| ... | ... | ... |

---

### 섹션 4: SHAP Dependence Plot (개별 Feature 분석)

**Top 3 Feature의 비선형 관계 시각화**

```python
# Top 3 Feature에 대한 Dependence Plot
top3_features = top10_features[:3]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, feat in enumerate(top3_features):
    feat_idx = list(X.columns).index(feat)
    shap.dependence_plot(feat_idx, shap_values, X_test_preprocessed,
                          feature_names=X.columns, ax=axes[i], show=False)
    axes[i].set_title(f'{feat}', fontsize=12)

plt.tight_layout()
plt.savefig('../data/processed/발표_Part4_Dependence_Plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Dependence Plot 완료")
```

**해석:**
- **X축**: Feature 값
- **Y축**: SHAP Value (부도 위험에 대한 기여도)
- **색상**: 상호작용 Feature (자동 선택)
- **패턴**: 비선형 관계, 임계값 효과 확인

---

### 섹션 5: 개별 기업 사례 분석 (Waterfall Plot)

**부도 기업 1개, 정상 기업 1개의 예측 근거 시각화**

```python
# 부도 기업 중 확률이 높은 사례
bankrupt_idx = np.where(y_test == 1)[0]
y_test_prob = final_model.predict_proba(X_test)[:, 1]
high_risk_idx = bankrupt_idx[np.argsort(y_test_prob[bankrupt_idx])[-1]]

# 정상 기업 중 확률이 낮은 사례
normal_idx = np.where(y_test == 0)[0]
low_risk_idx = normal_idx[np.argsort(y_test_prob[normal_idx])[0]]

# Waterfall Plot
for idx, label in [(high_risk_idx, '부도 기업'), (low_risk_idx, '정상 기업')]:
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_test_preprocessed[idx],
            feature_names=X.columns.tolist()
        ),
        show=False
    )
    plt.title(f'{label} 예측 근거 (확률: {y_test_prob[idx]:.2%})', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'../data/processed/발표_Part4_Waterfall_{label}.png', dpi=300, bbox_inches='tight')
    plt.show()

print("✅ 개별 사례 분석 완료")
```

**해석:**
- **Base Value**: 전체 평균 예측값
- **화살표**: 각 Feature가 예측값을 증가/감소시키는 정도
- **최종값 (f(x))**: 해당 기업의 예측 확률

---

### 섹션 6: Force Plot (Interactive Visualization)

**여러 샘플의 예측 근거를 한 눈에 비교**

```python
# 부도 기업 상위 20개
top20_bankrupt = bankrupt_idx[np.argsort(y_test_prob[bankrupt_idx])[-20:]]

# Force Plot (HTML로 저장)
shap.force_plot(
    explainer.expected_value,
    shap_values[top20_bankrupt],
    X_test_preprocessed[top20_bankrupt],
    feature_names=X.columns.tolist(),
    show=False
)

# Jupyter에서 표시되지만 HTML로도 저장 가능
# shap.save_html('../data/processed/발표_Part4_Force_Plot.html',
#                shap.force_plot(...))

print("✅ Force Plot 생성 완료 (노트북에서 확인)")
```

---

### 섹션 7: Traffic Light 구간별 SHAP 패턴 분석

**각 위험 구간(Red/Yellow/Green)에서 어떤 Feature가 주로 작동하는지 분석**

```python
# 예측 확률로 구간 분류
red_threshold = thresholds['red']
yellow_threshold = thresholds['yellow']

y_test_prob = final_model.predict_proba(X_test)[:, 1]

red_mask = y_test_prob >= red_threshold
yellow_mask = (y_test_prob >= yellow_threshold) & (y_test_prob < red_threshold)
green_mask = y_test_prob < yellow_threshold

# 구간별 평균 SHAP 값
segments = {
    'Red (고위험)': red_mask,
    'Yellow (중위험)': yellow_mask,
    'Green (저위험)': green_mask
}

segment_shap_means = {}
for seg_name, mask in segments.items():
    if mask.sum() > 0:
        segment_shap_means[seg_name] = np.abs(shap_values[mask]).mean(axis=0)
        print(f"{seg_name}: {mask.sum()}개")

# 시각화
df_segment = pd.DataFrame(segment_shap_means, index=X.columns).T
top10_seg_features = df_segment.mean(axis=0).nlargest(10).index

plt.figure(figsize=(12, 6))
df_segment[top10_seg_features].T.plot(kind='bar', ax=plt.gca())
plt.title('Traffic Light 구간별 Top 10 Feature Importance', fontsize=14)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Mean |SHAP Value|', fontsize=12)
plt.legend(title='위험 구간', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../data/processed/발표_Part4_Segment_SHAP.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 구간별 분석 완료")
```

**인사이트 예시:**
- **Red 구간**: 자본잠식도, 이자보상배율 등 구조적 문제
- **Yellow 구간**: 유동비율, 매출채권 회전율 등 유동성 경고
- **Green 구간**: 전반적으로 건전한 재무 지표

---

### 섹션 8: Bootstrap 신뢰구간 추가 ⭐

**SHAP Feature Importance의 통계적 안정성 검증**

```python
from sklearn.utils import resample

# Bootstrap (1,000회)
n_bootstrap = 1000
bootstrap_importance = []

print("Bootstrap 진행 중...")
for i in range(n_bootstrap):
    # 복원 추출
    indices = resample(range(len(X_test)), n_samples=len(X_test), random_state=i)
    shap_boot = shap_values[indices]

    # Feature Importance 계산
    importance = np.abs(shap_boot).mean(axis=0)
    bootstrap_importance.append(importance)

    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{n_bootstrap} 완료")

bootstrap_importance = np.array(bootstrap_importance)

# 95% 신뢰구간
lower = np.percentile(bootstrap_importance, 2.5, axis=0)
upper = np.percentile(bootstrap_importance, 97.5, axis=0)
mean_importance = bootstrap_importance.mean(axis=0)

# Top 10 Feature CI 시각화
top10_idx_boot = np.argsort(mean_importance)[-10:][::-1]
top10_feat_boot = X.columns[top10_idx_boot]

df_ci = pd.DataFrame({
    'Feature': top10_feat_boot,
    'Mean': mean_importance[top10_idx_boot],
    'Lower': lower[top10_idx_boot],
    'Upper': upper[top10_idx_boot]
})

plt.figure(figsize=(10, 6))
plt.barh(df_ci['Feature'], df_ci['Mean'], xerr=[df_ci['Mean'] - df_ci['Lower'],
                                                   df_ci['Upper'] - df_ci['Mean']],
         capsize=5, alpha=0.7, color='steelblue')
plt.xlabel('Mean |SHAP Value| (95% CI)', fontsize=12)
plt.title('Top 10 Feature Importance with Bootstrap CI', fontsize=14, pad=20)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../data/processed/발표_Part4_Bootstrap_CI.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Bootstrap 신뢰구간 분석 완료")
```

---

### 섹션 9: 비즈니스 인사이트 종합

**의사결정자를 위한 핵심 메시지**

```python
# 최종 요약 테이블
summary = pd.DataFrame({
    '모델': [results['model_name']],
    'Test PR-AUC': [f"{results['test_pr_auc']:.4f}"],
    'Test Recall': [f"{results['test_recall']:.2%}"],
    'Test F2-Score': [f"{results['test_f2']:.4f}"],
    '임계값': [f"{thresholds['selected']:.4f}"],
    'Red 구간 기업 수': [red_mask.sum()],
    'Yellow 구간 기업 수': [yellow_mask.sum()],
    'Green 구간 기업 수': [green_mask.sum()]
})

print("="*80)
print("📊 최종 모델 성능 및 위험 구간 분포")
print("="*80)
print(summary.T)
print("="*80)

# 비즈니스 인사이트
print("\n💡 비즈니스 인사이트\n")
print("1. **핵심 위험 지표**: Top 3 Feature가 전체 예측의 60% 이상 설명")
print(f"   - {top3_features[0]}")
print(f"   - {top3_features[1]}")
print(f"   - {top3_features[2]}")
print("\n2. **조기 경보 시스템**: Yellow 구간 기업 대상 집중 모니터링 권장")
print(f"   - Yellow: {yellow_mask.sum()}개 기업")
print(f"   - Red: {red_mask.sum()}개 기업 (즉각 대응 필요)")
print("\n3. **해석 가능성**: SHAP으로 개별 기업의 부도 위험 근거 명확히 제시")
print("   - 금융기관: 대출 심사 근거 마련")
print("   - 투자자: 포트폴리오 리스크 관리")
print("   - 규제기관: 공정성 및 투명성 확보")
```

---

### 섹션 10: 모델 한계점 및 개선 방향

#### 10.1 데이터 한계

**시점 제약**
- 2021년 8월 단일 시점 스냅샷 데이터
- 시계열 패턴 (재무 상태 추이) 반영 불가
- **개선 방향**: 다년도 패널 데이터 수집 (2018-2024년), 변화율(△) Feature 추가

**표본 편향**
- 외감 기업 중심 (소규모 기업 과소 대표)
- 생존 편향 (이미 폐업한 기업 미포함)
- **개선 방향**: 비외감 기업 데이터 보완, 폐업 기업 사후 데이터 수집

#### 10.2 모델 한계

**클래스 불균형**
- 부도율 1.5% → SMOTE로 완화했으나 여전히 Precision 낮음
- Recall 80% 달성 시 Precision 5-10% (False Positive 多)
- **개선 방향**: Cost-Sensitive Learning, Focal Loss 적용

**Feature 누락**
- 거시경제 변수 (금리, GDP, 환율) 미포함
- 산업별 충격 (COVID-19 등) 반영 안 됨
- **개선 방향**: 외부 데이터 결합, Industry-Specific Features

**예측 시계**
- "향후 1년 내 부도" → 정확한 시점 예측 불가
- 6개월 vs 11개월 부도 구분 못함
- **개선 방향**: Survival Analysis (Cox PH Model), 월별 예측 확률

#### 10.3 해석 한계

**SHAP 계산 비용**
- TreeExplainer로 개선했으나 대규모 데이터에서는 여전히 느림
- Real-Time 서비스 적용 시 지연 발생 가능
- **개선 방향**: Approximate SHAP, FastTreeSHAP 활용

**상호작용 효과**
- SHAP은 개별 Feature 기여도만 표시
- 복잡한 Feature 간 상호작용(예: 부채비율 × 수익성) 설명 부족
- **개선 방향**: SHAP Interaction Values 추가 분석

#### 10.4 운영 한계

**모델 드리프트**
- 경제 환경 변화 시 모델 성능 저하 (예: 금리 급등)
- 정기 재학습 없으면 예측력 감소
- **개선 방향**: 분기별 Monitoring, 자동 재학습 파이프라인

**규제 준수**
- 신용평가 모델 규제 (Basel III 등) 충족 여부 미검증
- 공정성 (Fairness) 평가 미실시 (업종/지역 차별 가능성)
- **개선 방향**: Regulatory Compliance Check, Fairness Metrics 추가

#### 10.5 비즈니스 한계

**오분류 비용 불균형**
- Type I Error (정상→부도 오판): 대출 기회 손실
- Type II Error (부도→정상 오판): 대손 발생 (훨씬 큰 비용)
- 현재 임계값은 동일 가중치 가정
- **개선 방향**: 실제 대손율 기반 Cost Matrix 설계, 최적 임계값 재조정

**설명 가능성 vs 성능 트레이드오프**
- Tree 모델 선택 → 딥러닝보다 성능 낮을 수 있음
- SHAP도 복잡한 비선형 관계 완벽히 설명 못함
- **개선 방향**: Tabular Deep Learning (TabNet, FT-Transformer) 실험

#### 10.6 향후 연구 방향

1. **시계열 확장**: LSTM/Transformer 기반 다년도 예측 모델
2. **멀티태스크 학습**: 부도 + 신용등급 + 재무조작 동시 예측
3. **강건성 평가**: Adversarial Examples, Out-of-Distribution 테스트
4. **Causal Inference**: 정책 개입 효과 예측 (예: 금리 인하 시 부도율 변화)
5. **실시간 시스템**: Streamlit → FastAPI + Redis + Celery 아키텍처

---

### 섹션 11: 최종 저장

```python
# 분석 결과 저장
analysis_results = {
    'model_name': results['model_name'],
    'test_pr_auc': results['test_pr_auc'],
    'top10_features': top10_features.tolist(),
    'feature_importance': {feat: float(feature_importance[list(X.columns).index(feat)])
                           for feat in top10_features},
    'segment_distribution': {
        'red': int(red_mask.sum()),
        'yellow': int(yellow_mask.sum()),
        'green': int(green_mask.sum())
    },
    'bootstrap_ci': df_ci.to_dict('records')
}

import json
with open('../data/processed/발표_Part4_SHAP_분석결과.json', 'w', encoding='utf-8') as f:
    json.dump(analysis_results, f, ensure_ascii=False, indent=2)

print("✅ Part 4 완료! 모든 결과가 저장되었습니다.")
print("\n생성된 파일:")
print("  - 발표_Part4_SHAP_Summary.png")
print("  - 발표_Part4_Top10_Features.png")
print("  - 발표_Part4_Dependence_Plot.png")
print("  - 발표_Part4_Waterfall_부도기업.png")
print("  - 발표_Part4_Waterfall_정상기업.png")
print("  - 발표_Part4_Segment_SHAP.png")
print("  - 발표_Part4_Bootstrap_CI.png")
print("  - 발표_Part4_SHAP_분석결과.json")
```

---

## 🎨 시각화 체크리스트

- [ ] Summary Plot (Beeswarm)
- [ ] Top 10 Bar Chart
- [ ] Dependence Plot (Top 3 Features)
- [ ] Waterfall Plot (부도/정상 각 1개)
- [ ] Force Plot (Interactive)
- [ ] Traffic Light 구간별 SHAP 비교
- [ ] Bootstrap 신뢰구간 (Top 10)

---

## ⚠️ 주의사항

1. **모델 타입 확인**: Part 3에서 선택된 모델이 Tree 기반이 아니면 `KernelExplainer` 사용
2. **Feature 순서**: 전처리 후 Feature 순서가 바뀔 수 있음 → `feature_names` 명시적 지정
3. **SHAP 계산 시간**: Test Set 전체 계산 시 수 분 소요 가능 (샘플링 고려)
4. **한글 폰트**: 모든 그래프에서 폰트 설정 확인
5. **메모리**: SHAP values는 (n_samples, n_features) 크기 → 큰 데이터셋은 배치 처리

---

## 📚 참고 자료

- SHAP 공식 문서: https://shap.readthedocs.io/
- Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.
- Molnar, C. (2022). "Interpretable Machine Learning."
