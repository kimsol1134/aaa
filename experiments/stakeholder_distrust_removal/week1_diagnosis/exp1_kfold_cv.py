"""
Week 1 실험 1: K-Fold Cross-Validation 재검증
목적: Val-Test 괴리가 데이터 분할 운(Lucky Split) 때문인지 확인
"""

import sys
sys.path.append('/home/user/aaa/experiments/stakeholder_distrust_removal/scripts')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

from common_utils import (
    load_data, remove_feature_from_data, create_pipeline,
    save_results, RANDOM_STATE
)


def run_kfold_cv_experiment():
    """
    K-Fold CV 실험 실행

    비교 대상:
    1. Baseline (이해관계자_불신지수 포함)
    2. Current (이해관계자_불신지수 제거)

    각각에 대해 5-Fold CV 수행하여 성능 분산 확인
    """
    print("=" * 80)
    print("Week 1 실험 1: K-Fold Cross-Validation 재검증")
    print("=" * 80)

    # 데이터 로딩
    X, y = load_data()

    experiments = {
        'Baseline (이해관계자_불신지수 포함)': X.copy(),
        'Current (이해관계자_불신지수 제거)': remove_feature_from_data(X, '이해관계자_불신지수')
    }

    results = []

    for exp_name, X_exp in experiments.items():
        print(f"\n{'='*80}")
        print(f"{exp_name}")
        print(f"{'='*80}")

        # CatBoost 모델 생성 (간단한 파라미터)
        model = CatBoostClassifier(
            iterations=100,
            depth=5,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbose=0,
            eval_metric='Precision'
        )

        # 파이프라인 생성
        pipe = create_pipeline(model, wins=False, resamp='smote', resamp_ratio=0.2)

        # 5-Fold CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        print("\n5-Fold Cross-Validation 수행 중...")

        cv_scores = cross_val_score(
            pipe, X_exp, y,
            cv=cv,
            scoring='average_precision',
            n_jobs=-1,
            verbose=0
        )

        # 통계 정보
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        min_score = cv_scores.min()
        max_score = cv_scores.max()

        print(f"\n📊 CV 결과:")
        print(f"   평균 PR-AUC: {mean_score:.4f} ± {std_score:.4f}")
        print(f"   최소값:      {min_score:.4f}")
        print(f"   최대값:      {max_score:.4f}")
        print(f"   개별 Fold:   {', '.join([f'{s:.4f}' for s in cv_scores])}")

        # 결과 저장
        results.append({
            'experiment': exp_name,
            'n_features': X_exp.shape[1],
            'cv_mean': mean_score,
            'cv_std': std_score,
            'cv_min': min_score,
            'cv_max': max_score,
            'cv_fold1': cv_scores[0],
            'cv_fold2': cv_scores[1],
            'cv_fold3': cv_scores[2],
            'cv_fold4': cv_scores[3],
            'cv_fold5': cv_scores[4]
        })

    # 결과 데이터프레임
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("📊 최종 비교 결과")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # 분석
    print("\n" + "=" * 80)
    print("🔍 분석")
    print("=" * 80)

    baseline_cv = results_df.loc[0, 'cv_mean']
    current_cv = results_df.loc[1, 'cv_mean']

    print(f"\nBaseline CV:  {baseline_cv:.4f}")
    print(f"Current CV:   {current_cv:.4f}")
    print(f"차이:         {(current_cv - baseline_cv):.4f}")

    # 노트북 결과와 비교
    print(f"\n노트북 결과 비교:")
    print(f"  Baseline: Val 0.1572, Test 0.1542 (Gap 2.0%)")
    print(f"  Current:  Val 0.1245, Test 0.1602 (Gap 28.7%)")
    print(f"\n  CV 평균이 Val과 Test 사이에 있는가?")
    if 0.1245 < current_cv < 0.1602:
        print(f"  ✅ 예 ({current_cv:.4f}는 0.1245~0.1602 사이)")
        print(f"  → 데이터 분할 운(Lucky Split) 문제일 가능성 높음")
    else:
        print(f"  ❌ 아니오 ({current_cv:.4f}는 0.1245~0.1602 범위 밖)")
        print(f"  → 다른 원인 탐색 필요")

    # CV 분산 분석
    baseline_std = results_df.loc[0, 'cv_std']
    current_std = results_df.loc[1, 'cv_std']

    print(f"\nCV 분산 분석:")
    print(f"  Baseline 분산: ±{baseline_std:.4f}")
    print(f"  Current 분산:  ±{current_std:.4f}")

    if current_std > 0.03:
        print(f"  ⚠️ Current 모델의 분산이 큼 (±{current_std:.4f} > ±0.03)")
        print(f"  → 모델 불안정, 하이퍼파라미터 재튜닝 필요")
    else:
        print(f"  ✅ 분산이 적당함 (±{current_std:.4f} ≤ ±0.03)")

    # 결과 저장
    save_results(results_df, 'week1', 'kfold_cv')

    return results_df


if __name__ == '__main__':
    results = run_kfold_cv_experiment()
    print("\n✅ Week 1 실험 1 완료")
