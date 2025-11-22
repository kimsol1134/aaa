#!/usr/bin/env python3
"""
Part 2 발표용 노트북 생성 스크립트
기존 노트북에서 코드를 추출하여 발표용 구조로 재구성
"""

import json

# 노트북 구조 정의
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
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

def add_markdown(text):
    """마크다운 셀 추가"""
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.split("\n")
    })

def add_code(code):
    """코드 셀 추가"""
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split("\n")
    })

# === 제목 ===
add_markdown("""# 📗 Part 2: 도메인 특성 공학 (Domain Feature Engineering)

## 한국 기업 부도 예측: 재무 이론을 코드로 구현하기""")

# === Part 1 요약 ===
add_markdown("""## 📌 이전 Part 요약

Part 1에서 우리는 다음을 발견했습니다:

### ✅ 핵심 발견
1. **유동성이 가장 강력한 예측 변수**
   - 현금/당좌비율이 부도와 가장 높은 상관관계
   - 유동성 위기 → 부도의 직접 경로 확인

2. **업종별 부도율 2배 차이**
   - 제조업(1.8%) vs 금융업(0.9%)
   - 업종 특성이 중요한 리스크 요인

3. **외감 여부가 중요**
   - 외부감사 기업: 부도율 0.7%
   - 비외감 기업: 부도율 2.5% (3.6배 차이)

### ❌ 한계
- 단변량 예측력 제한적 (AUC < 0.7)
- 변수 간 상호작용 미고려
- 원본 변수만으로는 복잡한 패턴 포착 어려움

---

## 🎯 Part 2의 목표

**도메인 지식을 활용하여 예측력 높은 복합 특성을 생성합니다.**

- 재무 이론 기반 특성 생성 (Altman, Beneish, etc.)
- 변수 간 상호작용 효과 포착
- 한국 시장 특화 리스크 반영
- 통계적 검증을 통한 특성 선별""")

# === Why 섹션 ===
add_markdown("""## 🤔 Why: 왜 도메인 특성이 필요한가?

### 부도의 3가지 경로

재무 이론에 따르면 기업 부도는 다음 3가지 경로로 발생합니다:

#### 1️⃣ 유동성 위기 (Liquidity Crisis)
```
단기 부채 > 유동 자산
→ 만기 채무 상환 불능
→ 기술적 부도 (Technical Default)
```

**예시**: 흑자 기업도 현금흐름 문제로 부도 가능

#### 2️⃣ 지급불능 (Insolvency)
```
부채 총계 > 자산 총계
→ 자본 잠식 (자본총계 < 0)
→ 법적 부도 (Legal Insolvency)
```

**예시**: 누적 적자로 자본이 완전히 소진

#### 3️⃣ 신뢰 상실 (Loss of Confidence)
```
연체/체납 발생
→ 신용등급 하락
→ 재융자 불가능
→ 연쇄 부도
```

**예시**: 세금 체납 → 은행 여신 회수 → 부도

---

### 이론적 배경

#### Altman Z-Score (1968)
- 재무 비율 5개를 결합하여 부도 예측
- 제조업 중심 모델 (한국 시장과 유사)
- **핵심**: 단일 변수가 아닌 **복합 지표**의 중요성

#### Beneish M-Score (1999)
- 재무제표 조작 탐지 모델
- 매출채권/재고 이상 증가, 발생액 품질 등
- **핵심**: **비정상적 패턴**이 부도 전조

#### Ohlson O-Score (1980)
- 로지스틱 회귀 기반 부도 예측
- 기업 규모, 레버리지, 수익성 조합
- **핵심**: **변수 간 상호작용**이 중요

---

### 도메인 특성 공학의 원칙

1. **이론 기반 (Theory-Driven)**
   - "왜 이 특성이 부도를 예측하는가?"에 답할 수 있어야 함
   - 재무 이론, 실무 경험, 선행 연구 근거

2. **한국 시장 특화 (Market-Specific)**
   - 외감 여부, 대기업 의존도, 제조업 중심
   - 글로벌 모델을 그대로 적용하지 않음

3. **통계적 검증 (Statistically Validated)**
   - 생성한 특성이 실제로 부도와 유의미한 관계가 있는지 검증
   - Mann-Whitney U test, Information Value 등

4. **해석 가능성 (Interpretable)**
   - 금융 실무자가 이해하고 신뢰할 수 있어야 함
   - 블랙박스가 아닌 설명 가능한 특성""")

# === 데이터 로딩 ===
add_markdown("## 📦 데이터 로딩 및 환경 설정")

add_code("""import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, roc_auc_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 한글 폰트 설정
import platform
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)

# 데이터 로딩
df = pd.read_csv('../data/기업신용평가정보_210801.csv', encoding='utf-8')
target_col = '모형개발용Performance(향후1년내부도여부)'

print(f"✅ 데이터 로딩 완료: {df.shape[0]:,} 기업, {df.shape[1]:,} 변수")
print(f"✅ 부도율: {df[target_col].mean()*100:.2f}%")
print(f"✅ 부도 기업: {df[target_col].sum():,}개")
print(f"✅ 정상 기업: {(~df[target_col].astype(bool)).sum():,}개")""")

# === 특성 생성 섹션들 ===
# 간결함을 위해 핵심 카테고리만 포함
add_markdown("""## 🔧 카테고리 1: 유동성 위기 특성 (Liquidity Crisis Features)

### 이론적 배경
- **현금흐름 > 수익**: "Profits are opinion, cash is fact"
- **운전자본 건전성**: 유동자산 - 유동부채 > 0이어야 안전
- **현금소진일수**: 현재 현금으로 며칠 버틸 수 있는가?""")

add_code("""def create_liquidity_crisis_features(df):
    features = pd.DataFrame(index=df.index)

    if '현금' in df.columns and '유동부채' in df.columns:
        features['즉각지급능력'] = (df['현금'] + df.get('현금성자산', 0)) / (df['유동부채'] + 1)
        features['현금소진일수'] = (df['현금'] + df.get('현금성자산', 0)) / (df.get('영업비용', df['매출원가']) / 365 + 1)

    if '유동자산' in df.columns and '유동부채' in df.columns:
        features['운전자본'] = df['유동자산'] - df['유동부채']
        features['운전자본비율'] = features['운전자본'] / (df.get('매출액', 1) + 1)

    if '영업활동현금흐름' in df.columns:
        features['OCF_대_유동부채'] = df['영업활동현금흐름'] / (df.get('유동부채', 1) + 1)

    print(f"✅ 유동성 위기 특성 {features.shape[1]}개 생성 완료")
    return features

liquidity_features = create_liquidity_crisis_features(df)
liquidity_features.head()""")

# 검증 코드 추가
add_markdown("### 특성 검증: 즉각지급능력")

add_code("""# 즉각지급능력 검증
feature_name = '즉각지급능력'

normal_median = liquidity_features[df[target_col] == 0][feature_name].median()
bankrupt_median = liquidity_features[df[target_col] == 1][feature_name].median()

u_stat, p_value = mannwhitneyu(
    liquidity_features[df[target_col] == 0][feature_name].dropna(),
    liquidity_features[df[target_col] == 1][feature_name].dropna()
)

print(f"### {feature_name} 검증")
print(f"- 정상 기업 median: {normal_median:.3f}")
print(f"- 부도 기업 median: {bankrupt_median:.3f}")
print(f"- 차이: {normal_median - bankrupt_median:.3f} ({(normal_median/bankrupt_median):.1f}배)")
print(f"- Mann-Whitney U test: p = {p_value:.2e}")
print(f"- 결론: {'✅ 통계적으로 유의미' if p_value < 0.01 else '❌ 유의미하지 않음'}")""")

# 나머지 카테고리들을 계속 추가...
# (간결함을 위해 핵심 부분만 작성)

# 노트북 저장
output_path = "/home/user/aaa/notebooks/발표_Part2_도메인_특성_공학.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"✅ 노트북 생성 완료: {output_path}")
print(f"✅ 총 {len(notebook['cells'])}개 셀")
