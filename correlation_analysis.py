#!/usr/bin/env python3
"""
특성 간 상관관계 분석 스크립트

목표:
1. 모든 특성 간 상관관계 매트릭스 생성
2. 이해관계자_불신지수와 다른 특성들의 상관관계 분석
3. 높은 상관관계 (> 0.8) 특성 쌍 찾기
4. VIF (Variance Inflation Factor) 계산
5. 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
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

print('='*80)
print('📊 특성 상관관계 분석')
print('='*80)

# 데이터 로딩
df = pd.read_csv('data/features/domain_based_features_완전판.csv', encoding='utf-8')
TARGET_COL = '모형개발용Performance(향후1년내부도여부)'
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print(f'데이터 shape: {df.shape}')
print(f'특성 개수: {X.shape[1]}')

# ============================================================================
# 1. 전체 상관관계 매트릭스
# ============================================================================

print('\n' + '='*80)
print('1️⃣ 전체 상관관계 매트릭스 (Pearson)')
print('='*80)

corr_matrix = X.corr(method='pearson')

print(f'\n상관관계 매트릭스 shape: {corr_matrix.shape}')
print(f'\n상관관계 통계:')
print(f'  평균 (절대값): {corr_matrix.abs().mean().mean():.4f}')
print(f'  중앙값: {corr_matrix.median().median():.4f}')
print(f'  최대값: {corr_matrix.abs().max().max():.4f}')

# ============================================================================
# 2. 이해관계자_불신지수와의 상관관계
# ============================================================================

print('\n' + '='*80)
print('2️⃣ 이해관계자_불신지수와 다른 특성들의 상관관계')
print('='*80)

DISTRUST_COL = '이해관계자_불신지수'
if DISTRUST_COL in corr_matrix.columns:
    distrust_corr = corr_matrix[DISTRUST_COL].drop(DISTRUST_COL).sort_values(ascending=False, key=abs)

    print(f'\n상위 10개 (절대값 기준):')
    print('-'*80)
    print(f'{"특성":<30s} {"상관계수":>12s} {"절대값":>12s}')
    print('-'*80)
    for feat, corr in distrust_corr.head(10).items():
        print(f'{feat:<30s} {corr:12.6f} {abs(corr):12.6f}')

    print(f'\n하위 10개 (절대값 기준):')
    print('-'*80)
    for feat, corr in distrust_corr.tail(10).items():
        print(f'{feat:<30s} {corr:12.6f} {abs(corr):12.6f}')

    # 매우 높은 상관관계 (> 0.9)
    very_high = distrust_corr[distrust_corr.abs() > 0.9]
    if len(very_high) > 0:
        print(f'\n⚠️ 매우 높은 상관관계 (> 0.9): {len(very_high)}개')
        for feat, corr in very_high.items():
            print(f'  {feat}: {corr:.6f}')
else:
    print(f'⚠️ {DISTRUST_COL} 컬럼을 찾을 수 없습니다')

# ============================================================================
# 3. 높은 상관관계 쌍 찾기
# ============================================================================

print('\n' + '='*80)
print('3️⃣ 높은 상관관계 특성 쌍 (|r| > 0.8)')
print('='*80)

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.8:
            high_corr_pairs.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_val,
                'Abs_Corr': abs(corr_val)
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Abs_Corr', ascending=False)
    print(f'\n찾은 쌍: {len(high_corr_df)}개')
    print('\n' + high_corr_df.to_string(index=False))

    # 파일 저장
    high_corr_df.to_csv('data/processed/high_correlation_pairs.csv', index=False, encoding='utf-8-sig')
    print(f'\n✅ 저장: data/processed/high_correlation_pairs.csv')
else:
    print('\n상관관계 > 0.8인 쌍이 없습니다')

# ============================================================================
# 4. VIF (Variance Inflation Factor) 계산
# ============================================================================

print('\n' + '='*80)
print('4️⃣ VIF (다중공선성) 분석')
print('='*80)

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # 무한대/결측치 제거
    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    print('\nVIF 계산 중... (시간이 걸릴 수 있습니다)')

    vif_data = []
    for i, col in enumerate(X_clean.columns):
        try:
            vif = variance_inflation_factor(X_clean.values, i)
            vif_data.append({'Feature': col, 'VIF': vif})
            if (i + 1) % 5 == 0:
                print(f'  진행: {i+1}/{len(X_clean.columns)}')
        except Exception as e:
            print(f'  {col} 계산 실패: {e}')
            vif_data.append({'Feature': col, 'VIF': np.nan})

    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

    print(f'\nVIF 결과 (상위 20개):')
    print('-'*80)
    print(vif_df.head(20).to_string(index=False))

    # VIF > 10인 특성
    high_vif = vif_df[vif_df['VIF'] > 10]
    print(f'\n⚠️ VIF > 10 (다중공선성 문제): {len(high_vif)}개')
    for _, row in high_vif.head(10).iterrows():
        print(f'  {row["Feature"]}: {row["VIF"]:.2e}')

    # 저장
    vif_df.to_csv('data/processed/vif_analysis.csv', index=False, encoding='utf-8-sig')
    print(f'\n✅ 저장: data/processed/vif_analysis.csv')

except ImportError:
    print('\n⚠️ statsmodels 패키지가 필요합니다')
    print('   설치: pip install statsmodels')

# ============================================================================
# 5. 시각화
# ============================================================================

print('\n' + '='*80)
print('5️⃣ 시각화')
print('='*80)

# 5.1 상관관계 히트맵 (전체)
fig, axes = plt.subplots(2, 2, figsize=(20, 20))

# 전체 히트맵
sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            cbar_kws={'shrink': 0.8}, ax=axes[0, 0], square=True)
axes[0, 0].set_title('전체 상관관계 매트릭스', fontsize=16, pad=20)

# 이해관계자_불신지수 관련 (있는 경우)
if DISTRUST_COL in X.columns:
    distrust_idx = X.columns.get_loc(DISTRUST_COL)
    distrust_row = corr_matrix.iloc[distrust_idx, :].values.reshape(1, -1)

    sns.heatmap(distrust_row, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                xticklabels=X.columns, yticklabels=[DISTRUST_COL],
                cbar_kws={'shrink': 0.8}, ax=axes[0, 1])
    axes[0, 1].set_title(f'{DISTRUST_COL}와의 상관관계', fontsize=16, pad=20)
    axes[0, 1].tick_params(axis='x', rotation=90)

# 상위 10개 특성 간 히트맵
if DISTRUST_COL in X.columns:
    top10_features = [DISTRUST_COL] + distrust_corr.head(9).index.tolist()
    top10_corr = corr_matrix.loc[top10_features, top10_features]

    sns.heatmap(top10_corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, cbar_kws={'shrink': 0.8}, ax=axes[1, 0], square=True)
    axes[1, 0].set_title('상위 10개 특성 간 상관관계 (세부)', fontsize=16, pad=20)
    axes[1, 0].tick_params(axis='x', labelrotation=45)
    axes[1, 0].tick_params(axis='y', labelrotation=0)

# 상관관계 분포 히스토그램
corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
axes[1, 1].hist(corr_values, bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].axvline(0.8, color='orange', linestyle='--', linewidth=2, label='|r| = 0.8')
axes[1, 1].axvline(-0.8, color='orange', linestyle='--', linewidth=2)
axes[1, 1].set_title('상관계수 분포', fontsize=16, pad=20)
axes[1, 1].set_xlabel('상관계수', fontsize=12)
axes[1, 1].set_ylabel('빈도', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/processed/correlation_analysis.png', dpi=300, bbox_inches='tight')
print('\n✅ 저장: data/processed/correlation_analysis.png')

# 5.2 이해관계자_불신지수 상관관계 바 차트
if DISTRUST_COL in X.columns:
    fig, ax = plt.subplots(figsize=(12, 8))

    top15 = distrust_corr.head(15)
    colors = ['red' if x > 0 else 'blue' for x in top15.values]

    ax.barh(range(len(top15)), top15.values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15.index)
    ax.set_xlabel('상관계수', fontsize=12)
    ax.set_title(f'{DISTRUST_COL}와의 상관관계 (상위 15개)', fontsize=16, pad=20)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

    # 값 표시
    for i, v in enumerate(top15.values):
        ax.text(v, i, f' {v:.3f}', va='center', ha='left' if v > 0 else 'right', fontsize=10)

    plt.tight_layout()
    plt.savefig('data/processed/distrust_correlation_bar.png', dpi=300, bbox_inches='tight')
    print('✅ 저장: data/processed/distrust_correlation_bar.png')

# ============================================================================
# 6. 요약 보고서 생성
# ============================================================================

print('\n' + '='*80)
print('6️⃣ 요약 보고서 생성')
print('='*80)

summary = []
summary.append('# 특성 상관관계 분석 보고서\n')
summary.append(f'생성 일시: {pd.Timestamp.now()}\n\n')

summary.append('## 1. 데이터 정보\n')
summary.append(f'- 특성 개수: {X.shape[1]}\n')
summary.append(f'- 샘플 개수: {X.shape[0]}\n\n')

summary.append('## 2. 전체 상관관계 통계\n')
summary.append(f'- 평균 (절대값): {corr_matrix.abs().mean().mean():.4f}\n')
summary.append(f'- 중앙값: {corr_matrix.median().median():.4f}\n')
summary.append(f'- 최대값: {corr_matrix.abs().max().max():.4f}\n\n')

if DISTRUST_COL in X.columns:
    summary.append(f'## 3. {DISTRUST_COL}와의 상관관계\n\n')
    summary.append('### 상위 10개 (절대값 기준)\n\n')
    for feat, corr in distrust_corr.head(10).items():
        summary.append(f'- {feat}: {corr:.6f}\n')

    very_high = distrust_corr[distrust_corr.abs() > 0.9]
    if len(very_high) > 0:
        summary.append(f'\n### ⚠️ 매우 높은 상관관계 (> 0.9): {len(very_high)}개\n\n')
        for feat, corr in very_high.items():
            summary.append(f'- {feat}: {corr:.6f}\n')

if high_corr_pairs:
    summary.append(f'\n## 4. 높은 상관관계 쌍 (|r| > 0.8): {len(high_corr_df)}개\n\n')
    for _, row in high_corr_df.head(20).iterrows():
        summary.append(f'- {row["Feature 1"]} ↔ {row["Feature 2"]}: {row["Correlation"]:.6f}\n')

summary.append('\n## 5. 결론\n\n')
if DISTRUST_COL in X.columns and len(very_high) > 0:
    summary.append(f'⚠️ **이해관계자_불신지수는 {len(very_high)}개 특성과 매우 높은 상관관계(> 0.9)를 가짐**\n\n')
    summary.append('이는 다중공선성 문제를 야기하며, L1 정규화가 다른 특성을 억제한 주요 원인입니다.\n\n')
    summary.append('**권장사항:**\n')
    summary.append('1. 이해관계자_불신지수 제외 후 모델 재학습\n')
    summary.append('2. 또는 L2 정규화 + StandardScaler 사용\n')
    summary.append('3. 또는 VIF > 10인 특성들을 순차적으로 제거\n')

with open('data/processed/correlation_summary.md', 'w', encoding='utf-8') as f:
    f.writelines(summary)

print('\n✅ 저장: data/processed/correlation_summary.md')

print('\n' + '='*80)
print('✅ 상관관계 분석 완료!')
print('='*80)
