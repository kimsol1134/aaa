"""
Week 2 실험 2: VIF 기반 특성 제거
목적: 다중공선성이 높은 특성 제거하여 모델 안정성 향상
"""

import sys
sys.path.append('/home/user/aaa/experiments/stakeholder_distrust_removal/scripts')

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

from common_utils import (
    load_data, split_data, remove_feature_from_data,
    create_pipeline, evaluate_model, save_results, RANDOM_STATE
)


def calculate_vif_all_features(X, sample_ratio=0.2):
    """
    모든 특성의 VIF 계산

    Args:
        X: 특성 데이터프레임
        sample_ratio: 샘플링 비율 (계산 속도 향상)

    Returns:
        VIF 데이터프레임
    """
    print(f"VIF 계산 중 (샘플링 {sample_ratio*100:.0f}%)...")

    # 샘플링
    sample_size = int(len(X) * sample_ratio)
    X_sample = X.sample(n=sample_size, random_state=RANDOM_STATE)

    # 전처리
    X_clean = X_sample.fillna(0).replace([np.inf, -np.inf], 0)

    # VIF 계산
    vif_data = []
    for i, feature in enumerate(X_clean.columns):
        try:
            vif = variance_inflation_factor(X_clean.values, i)
            if np.isinf(vif) or np.isnan(vif):
                vif = 999
        except:
            vif = 999

        vif_data.append({
            'feature': feature,
            'VIF': vif,
            'status': '🔴 제거 고려' if vif > 10 else ('🟡 주의' if vif > 5 else '✅ 양호')
        })

        if (i + 1) % 10 == 0:
            print(f"  진행: {i+1}/{len(X_clean.columns)} 특성")

    return pd.DataFrame(vif_data).sort_values('VIF', ascending=False)


def remove_high_vif_features(X, vif_threshold=10):
    """
    VIF가 높은 특성 제거

    Args:
        X: 특성 데이터프레임
        vif_threshold: VIF 임계값

    Returns:
        X_reduced: VIF 기반 제거된 데이터프레임
        removed_features: 제거된 특성 리스트
    """
    # VIF 계산
    vif_df = calculate_vif_all_features(X)

    # VIF > threshold인 특성
    high_vif_features = vif_df[vif_df['VIF'] > vif_threshold]['feature'].tolist()

    print(f"\n📊 VIF > {vif_threshold} 특성: {len(high_vif_features)}개")
    if high_vif_features:
        print(vif_df[vif_df['VIF'] > vif_threshold][['feature', 'VIF', 'status']].to_string(index=False))

    # 제거
    X_reduced = X.drop(columns=high_vif_features)

    print(f"\n✅ 특성 제거: {X.shape[1]} → {X_reduced.shape[1]} ({len(high_vif_features)}개 제거)")

    return X_reduced, high_vif_features, vif_df


def run_vif_based_removal_experiment():
    """
    VIF 기반 특성 제거 실험

    비교 대상:
    1. Baseline: VIF 제거 안 함
    2. VIF > 10 제거
    3. VIF > 5 제거 (더 보수적)
    4. 고상관 쌍 제거 (|r| > 0.9)
    """
    print("=" * 80)
    print("Week 2 실험 2: VIF 기반 특성 제거")
    print("=" * 80)

    # 데이터 로딩
    X, y = load_data()
    X_removed = remove_feature_from_data(X, '이해관계자_불신지수')

    # 데이터 분할
    X_train_base, X_val_base, X_test_base, y_train, y_val, y_test = split_data(X_removed, y)

    # 실험 1: Baseline (VIF 제거 안 함)
    print(f"\n{'='*80}")
    print(f"실험 1: Baseline (VIF 제거 안 함)")
    print(f"{'='*80}")

    model = CatBoostClassifier(
        iterations=100, depth=5, learning_rate=0.05,
        random_state=RANDOM_STATE, verbose=0, eval_metric='Precision'
    )
    pipe = create_pipeline(model, wins=False, resamp='smote', resamp_ratio=0.2)

    print(f"학습 중...")
    pipe.fit(X_train_base, y_train)

    baseline_result = evaluate_model(
        pipe, X_val_base, y_val, X_test_base, y_test,
        model_name='Baseline'
    )
    baseline_result['n_features'] = X_train_base.shape[1]

    # 실험 2: VIF > 10 제거
    print(f"\n{'='*80}")
    print(f"실험 2: VIF > 10 제거")
    print(f"{'='*80}")

    X_train_vif10, removed_vif10, vif_df = remove_high_vif_features(X_train_base, vif_threshold=10)
    X_val_vif10 = X_val_base.drop(columns=removed_vif10)
    X_test_vif10 = X_test_base.drop(columns=removed_vif10)

    model = CatBoostClassifier(
        iterations=100, depth=5, learning_rate=0.05,
        random_state=RANDOM_STATE, verbose=0, eval_metric='Precision'
    )
    pipe = create_pipeline(model, wins=False, resamp='smote', resamp_ratio=0.2)

    print(f"\n학습 중...")
    pipe.fit(X_train_vif10, y_train)

    vif10_result = evaluate_model(
        pipe, X_val_vif10, y_val, X_test_vif10, y_test,
        model_name='VIF > 10 제거'
    )
    vif10_result['n_features'] = X_train_vif10.shape[1]
    vif10_result['removed_features'] = ', '.join(removed_vif10)

    # 실험 3: VIF > 5 제거
    print(f"\n{'='*80}")
    print(f"실험 3: VIF > 5 제거")
    print(f"{'='*80}")

    X_train_vif5, removed_vif5, _ = remove_high_vif_features(X_train_base, vif_threshold=5)
    X_val_vif5 = X_val_base.drop(columns=removed_vif5)
    X_test_vif5 = X_test_base.drop(columns=removed_vif5)

    model = CatBoostClassifier(
        iterations=100, depth=5, learning_rate=0.05,
        random_state=RANDOM_STATE, verbose=0, eval_metric='Precision'
    )
    pipe = create_pipeline(model, wins=False, resamp='smote', resamp_ratio=0.2)

    print(f"\n학습 중...")
    pipe.fit(X_train_vif5, y_train)

    vif5_result = evaluate_model(
        pipe, X_val_vif5, y_val, X_test_vif5, y_test,
        model_name='VIF > 5 제거'
    )
    vif5_result['n_features'] = X_train_vif5.shape[1]

    # 실험 4: 고상관 쌍 제거
    print(f"\n{'='*80}")
    print(f"실험 4: 고상관 쌍 제거 (|r| > 0.9)")
    print(f"{'='*80}")

    # 상관계수 계산
    corr_matrix = X_train_base.fillna(0).replace([np.inf, -np.inf], 0).corr()

    # 고상관 쌍 찾기
    high_corr_pairs = []
    removed_by_corr = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                feat1 = corr_matrix.columns[i]
                feat2 = corr_matrix.columns[j]
                high_corr_pairs.append((feat1, feat2, corr_matrix.iloc[i, j]))

                # VIF가 더 높은 것 제거
                if feat1 in vif_df['feature'].values and feat2 in vif_df['feature'].values:
                    vif1 = vif_df[vif_df['feature'] == feat1]['VIF'].values[0]
                    vif2 = vif_df[vif_df['feature'] == feat2]['VIF'].values[0]

                    if vif1 > vif2:
                        removed_by_corr.add(feat1)
                    else:
                        removed_by_corr.add(feat2)

    removed_by_corr = list(removed_by_corr)

    print(f"고상관 쌍: {len(high_corr_pairs)}개")
    print(f"제거할 특성: {len(removed_by_corr)}개")
    if removed_by_corr:
        print(f"  {', '.join(removed_by_corr)}")

    X_train_corr = X_train_base.drop(columns=removed_by_corr)
    X_val_corr = X_val_base.drop(columns=removed_by_corr)
    X_test_corr = X_test_base.drop(columns=removed_by_corr)

    model = CatBoostClassifier(
        iterations=100, depth=5, learning_rate=0.05,
        random_state=RANDOM_STATE, verbose=0, eval_metric='Precision'
    )
    pipe = create_pipeline(model, wins=False, resamp='smote', resamp_ratio=0.2)

    print(f"\n학습 중...")
    pipe.fit(X_train_corr, y_train)

    corr_result = evaluate_model(
        pipe, X_val_corr, y_val, X_test_corr, y_test,
        model_name='고상관 쌍 제거'
    )
    corr_result['n_features'] = X_train_corr.shape[1]

    # 결과 비교
    results_df = pd.DataFrame([baseline_result, vif10_result, vif5_result, corr_result])

    print("\n" + "=" * 80)
    print("📊 최종 비교 결과")
    print("=" * 80)
    print(results_df[['model_name', 'n_features', 'val_pr_auc', 'test_pr_auc',
                       'val_test_gap', 'test_recall']].to_string(index=False))

    # 분석
    print("\n" + "=" * 80)
    print("🔍 분석")
    print("=" * 80)

    best_model = results_df.sort_values('test_pr_auc', ascending=False).iloc[0]
    best_gap = results_df.sort_values('val_test_gap').iloc[0]

    print(f"\n최고 Test PR-AUC:")
    print(f"  {best_model['model_name']}")
    print(f"  Test PR-AUC:  {best_model['test_pr_auc']:.4f}")
    print(f"  특성 수:      {best_model['n_features']}개")

    print(f"\n최소 Val-Test Gap:")
    print(f"  {best_gap['model_name']}")
    print(f"  Val-Test Gap: {best_gap['val_test_gap']:.1f}%")
    print(f"  Test PR-AUC:  {best_gap['test_pr_auc']:.4f}")

    # VIF 데이터프레임 저장
    save_results(vif_df, 'week2', 'vif_analysis')
    save_results(results_df, 'week2', 'vif_based_removal')

    return results_df, vif_df


if __name__ == '__main__':
    results, vif_df = run_vif_based_removal_experiment()
    print("\n✅ Week 2 실험 2 완료")
