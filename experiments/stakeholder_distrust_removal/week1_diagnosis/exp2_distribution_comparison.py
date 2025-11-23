"""
Week 1 실험 2: Val vs Test 분포 비교
목적: Validation과 Test 세트의 특성 분포 차이 확인
"""

import sys
sys.path.append('/home/user/aaa/experiments/stakeholder_distrust_removal/scripts')

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

from common_utils import (
    load_data, split_data, remove_feature_from_data,
    save_results
)


def run_distribution_comparison():
    """
    Val vs Test 분포 비교 실험

    KS-Test로 각 특성의 분포 차이 검정
    p < 0.05: 분포 차이 유의
    """
    print("=" * 80)
    print("Week 1 실험 2: Val vs Test 분포 비교")
    print("=" * 80)

    # 데이터 로딩 및 분할
    X, y = load_data()
    X_removed = remove_feature_from_data(X, '이해관계자_불신지수')

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_removed, y)

    print(f"\n검사할 특성 수: {X_removed.shape[1]}개")

    # KS-Test 수행
    ks_results = []

    for feature in X_removed.columns:
        val_data = X_val[feature].dropna()
        test_data = X_test[feature].dropna()

        if len(val_data) > 0 and len(test_data) > 0:
            ks_stat, p_value = ks_2samp(val_data, test_data)

            # 중앙값, 평균, 표준편차 비교
            val_median = val_data.median()
            test_median = test_data.median()
            val_mean = val_data.mean()
            test_mean = test_data.mean()
            val_std = val_data.std()
            test_std = test_data.std()

            ks_results.append({
                'feature': feature,
                'ks_stat': ks_stat,
                'p_value': p_value,
                'significant': '✅' if p_value < 0.05 else '⚪',
                'val_median': val_median,
                'test_median': test_median,
                'val_mean': val_mean,
                'test_mean': test_mean,
                'val_std': val_std,
                'test_std': test_std,
                'median_diff_%': abs(test_median - val_median) / (abs(val_median) + 1e-8) * 100,
                'mean_diff_%': abs(test_mean - val_mean) / (abs(val_mean) + 1e-8) * 100
            })

    results_df = pd.DataFrame(ks_results).sort_values('p_value')

    # 유의한 차이가 있는 특성
    significant_features = results_df[results_df['p_value'] < 0.05]

    print(f"\n📊 분포 차이 유의한 특성: {len(significant_features)}개 / {len(results_df)}개")

    print("\n상위 20개 특성 (p-value 기준):")
    print(results_df.head(20)[['feature', 'ks_stat', 'p_value', 'significant',
                                 'median_diff_%', 'mean_diff_%']].to_string(index=False))

    # 분석
    print("\n" + "=" * 80)
    print("🔍 분석")
    print("=" * 80)

    if len(significant_features) >= 5:
        print(f"\n⚠️ {len(significant_features)}개 특성에서 분포 차이 유의 (p < 0.05)")
        print(f"→ Stratified Split 개선 필요")
        print(f"→ 업종, 기업 규모, 신용등급 등 복합 Stratification 고려")
    else:
        print(f"\n✅ 분포 차이 유의한 특성이 적음 ({len(significant_features)}개 < 5개)")
        print(f"→ 데이터 분할은 적절함")

    # 주요 특성 확인
    important_features = ['신용등급점수', '연체심각도', 'OCF_대_유동부채', '현금창출능력']

    print(f"\n주요 특성 분포 확인:")
    for feature in important_features:
        if feature in results_df['feature'].values:
            row = results_df[results_df['feature'] == feature].iloc[0]
            print(f"\n  {feature}:")
            print(f"    Val 중앙값:  {row['val_median']:.4f}")
            print(f"    Test 중앙값: {row['test_median']:.4f}")
            print(f"    차이:        {row['median_diff_%']:.1f}%")
            print(f"    p-value:     {row['p_value']:.4f} {'(유의)' if row['p_value'] < 0.05 else ''}")

    # 신용등급점수 특별 체크
    if '신용등급점수' in results_df['feature'].values:
        credit_row = results_df[results_df['feature'] == '신용등급점수'].iloc[0]
        if credit_row['p_value'] < 0.05:
            print(f"\n⚠️ 신용등급점수 분포 차이 발견!")
            print(f"   → 이해관계자_불신지수와 고상관 특성")
            print(f"   → 신용등급점수가 불신지수 역할 대체 가능성")
            print(f"   → Week 2에서 신용등급점수 재설계 필요")

    # 결과 저장
    save_results(results_df, 'week1', 'distribution_comparison')

    return results_df


if __name__ == '__main__':
    results = run_distribution_comparison()
    print("\n✅ Week 1 실험 2 완료")
