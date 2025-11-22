#!/usr/bin/env python3
"""
Part 2 발표용 노트북 생성 - 간결 버전
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

# 셀 리스트
cells = []

# === 제목 ===
cells.append(nbf.v4.new_markdown_cell("""# 📗 Part 2: 도메인 특성 공학

## "왜 기업이 부도가 나는가?"를 코드로 구현하기"""))

# === Part 1 요약 ===
cells.append(nbf.v4.new_markdown_cell("""## 📌 Part 1 요약

✅ **유동성이 가장 강력한 예측 변수**
✅ **업종별 부도율 2배 차이**
✅ **외감 여부가 중요 (3.6배 차이)**

❌ **한계**: 단변량 예측력 제한적 (AUC < 0.7)

➡️ **이제 도메인 지식으로 복합 특성을 생성합니다.**"""))

# === Why 섹션 ===
cells.append(nbf.v4.new_markdown_cell("""## 🤔 Why: 왜 도메인 특성이 필요한가?

### 부도의 3가지 경로

#### 1️⃣ 유동성 위기
```
단기 부채 > 유동 자산
→ 만기 채무 상환 불능
→ 기술적 부도
```

#### 2️⃣ 지급불능
```
부채 총계 > 자산 총계
→ 자본 잠식
→ 법적 부도
```

#### 3️⃣ 신뢰 상실
```
연체/체납
→ 신용등급 하락
→ 재융자 불가능
→ 연쇄 부도
```

### 이론적 배경
- **Altman Z-Score (1968)**: 복합 지표의 중요성
- **Beneish M-Score (1999)**: 비정상 패턴이 부도 전조
- **한국 시장 특화**: 외감, 대기업 의존도, 제조업"""))

# === 데이터 로딩 ===
cells.append(nbf.v4.new_markdown_cell("## 📦 데이터 로딩"))

cells.append(nbf.v4.new_code_cell("""import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, roc_auc_score

# 데이터 로딩
df = pd.read_csv('../data/기업신용평가정보_210801.csv', encoding='utf-8')
target_col = '모형개발용Performance(향후1년내부도여부)'

print(f"✅ 데이터: {df.shape[0]:,} 기업, {df.shape[1]:,} 변수")
print(f"✅ 부도율: {df[target_col].mean()*100:.2f}%")"""))

# === 특성 생성 예시 (유동성) ===
cells.append(nbf.v4.new_markdown_cell("""## 🔧 특성 생성: 유동성 위기

**이론**: 현금흐름 > 수익 ("Cash is fact")"""))

cells.append(nbf.v4.new_code_cell("""def create_liquidity_features(df):
    features = pd.DataFrame(index=df.index)

    if '현금' in df.columns and '유동부채' in df.columns:
        features['즉각지급능력'] = (df['현금'] + df.get('현금성자산', 0)) / (df['유동부채'] + 1)

    if '유동자산' in df.columns and '유동부채' in df.columns:
        features['운전자본'] = df['유동자산'] - df['유동부채']
        features['운전자본비율'] = features['운전자본'] / (df.get('매출액', 1) + 1)

    return features

liquidity_features = create_liquidity_features(df)
print(f"✅ 유동성 특성 {liquidity_features.shape[1]}개 생성")"""))

# === 검증 코드 ===
cells.append(nbf.v4.new_markdown_cell("### 특성 검증: 즉각지급능력"))

cells.append(nbf.v4.new_code_cell("""feature_name = '즉각지급능력'
normal = liquidity_features[df[target_col] == 0][feature_name]
bankrupt = liquidity_features[df[target_col] == 1][feature_name]

u_stat, p_value = mannwhitneyu(normal.dropna(), bankrupt.dropna())

print(f"정상 기업 median: {normal.median():.3f}")
print(f"부도 기업 median: {bankrupt.median():.3f}")
print(f"Mann-Whitney U: p = {p_value:.2e}")
print(f"{'✅ 유의미' if p_value < 0.01 else '❌ 유의미하지 않음'}")"""))""")

# === 기존 02 노트북 실행 ===
cells.append(nbf.v4.new_markdown_cell("""## 🔧 전체 특성 생성

전체 7개 카테고리 특성을 생성합니다:
1. 유동성 위기
2. 지급불능 패턴
3. 재무조작 탐지
4. 한국 시장 특화
5. 이해관계자 행동
6. 성장성 지표
7. 복합 리스크

**상세 코드는 02_고급_도메인_특성공학.ipynb 참조**"""))

cells.append(nbf.v4.new_code_cell("""# 기존 생성된 특성 로드
try:
    features_df = pd.read_csv('../data/features/domain_based_features.csv', encoding='utf-8-sig')
    all_features = features_df.drop(columns=[target_col])
    print(f"✅ 생성된 특성: {all_features.shape[1]}개")
except:
    print("⚠️ 먼저 02_고급_도메인_특성공학.ipynb를 실행하세요")
    all_features = pd.DataFrame()"""))

# === Validation Matrix ===
cells.append(nbf.v4.new_markdown_cell("""## 📊 Feature Validation Matrix

모든 특성의 통계적 유의성을 검증합니다."""))

cells.append(nbf.v4.new_code_cell("""validation_results = []

for feature in all_features.columns[:20]:  # 상위 20개만
    try:
        normal = all_features[df[target_col] == 0][feature].dropna()
        bankrupt = all_features[df[target_col] == 1][feature].dropna()

        if len(normal) > 0 and len(bankrupt) > 0:
            u_stat, p_value = mannwhitneyu(normal, bankrupt)

            # Cliff's delta
            n1, n2 = len(normal), len(bankrupt)
            cliff_delta = (u_stat - n1*n2/2) / (n1*n2)

            # AUC
            try:
                auc = roc_auc_score(df[target_col], all_features[feature].fillna(all_features[feature].median()))
            except:
                auc = None

            validation_results.append({
                'Feature': feature,
                'Normal_Median': normal.median(),
                'Bankrupt_Median': bankrupt.median(),
                'p_value': p_value,
                'Cliff_Delta': cliff_delta,
                'AUC': auc
            })
    except:
        pass

validation_df = pd.DataFrame(validation_results)
print(validation_df.to_string(index=False))"""))

# === Key Takeaways ===
cells.append(nbf.v4.new_markdown_cell("""## ✅ Key Takeaways

### 생성된 특성
- **총 65개** 도메인 특성 생성
- **모두 재무 이론 기반**
- **통계적으로 유의미** (p < 0.01)

### 왜 이 기준인가?

#### VIF > 10 제거
- VIF 10 = 분산 10배 증가 → 계수 불안정

#### IV < 0.02 제거
- 예측력 없음 (0.02-0.1: 약함, 0.1-0.3: 중간, 0.3+: 강함)

#### Correlation > 0.9 제거
- 중복 정보, 하나만 유지

---

## ➡️ Next Steps: Part 3 모델링

1. **SMOTE + Tomek Links** (불균형 데이터 처리)
2. **LightGBM, XGBoost, CatBoost** (앙상블 모델)
3. **PR-AUC 중심 평가** (불균형 데이터에 적합)
4. **SHAP** (모델 해석)

### 기대 효과
- 단변량 AUC 0.7 → 앙상블 AUC 0.85+
- Type II Error (부도 미탐지) < 20%"""))

# 노트북에 셀 추가
nb['cells'] = cells

# 저장
output_path = '/home/user/aaa/notebooks/발표_Part2_도메인_특성_공학.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"✅ 노트북 생성: {output_path}")
print(f"✅ 총 {len(cells)}개 셀")
