"""
Week 1 실험 3: SMOTE 제거 실험 (Ablation Study)
목적: SMOTE가 Val-Test 괴리의 원인인지 확인
"""

import sys
sys.path.append('/home/user/aaa/experiments/stakeholder_distrust_removal/scripts')

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

from common_utils import (
    load_data, split_data, remove_feature_from_data,
    create_pipeline, evaluate_model, save_results, RANDOM_STATE
)


def run_smote_ablation_study():
    """
    SMOTE Ablation Study

    비교 대상:
    1. SMOTE + sampling_strategy=0.2 (현재)
    2. SMOTE + sampling_strategy=0.5
    3. BorderlineSMOTE
    4. ADASYN
    5. SMOTE + ENN
    6. Class Weight만 사용 (SMOTE 제거)
    """
    print("=" * 80)
    print("Week 1 실험 3: SMOTE Ablation Study")
    print("=" * 80)

    # 데이터 로딩 및 분할
    X, y = load_data()
    X_removed = remove_feature_from_data(X, '이해관계자_불신지수')

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_removed, y)

    # 실험 설정
    experiments = [
        {'name': 'Baseline: SMOTE (0.2)', 'resamp': 'smote', 'ratio': 0.2, 'scale_pos_weight': 1},
        {'name': 'SMOTE (0.5)', 'resamp': 'smote', 'ratio': 0.5, 'scale_pos_weight': 1},
        {'name': 'BorderlineSMOTE', 'resamp': 'borderline', 'ratio': 0.2, 'scale_pos_weight': 1},
        {'name': 'ADASYN', 'resamp': 'adasyn', 'ratio': 0.2, 'scale_pos_weight': 1},
        {'name': 'SMOTE + ENN', 'resamp': 'smote_enn', 'ratio': None, 'scale_pos_weight': 1},
        {'name': 'No SMOTE (Class Weight Only)', 'resamp': None, 'ratio': None, 'scale_pos_weight': 66.5}
    ]

    results = []

    for exp in experiments:
        print(f"\n{'='*80}")
        print(f"{exp['name']}")
        print(f"{'='*80}")

        # 부도율 계산
        bankruptcy_rate = y_train.mean()
        normal_rate = 1 - bankruptcy_rate

        # scale_pos_weight 계산 (클래스 불균형 비율)
        if exp['scale_pos_weight'] == 66.5:
            scale_pos_weight = normal_rate / bankruptcy_rate
        else:
            scale_pos_weight = exp['scale_pos_weight']

        # CatBoost 모델
        model = CatBoostClassifier(
            iterations=100,
            depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            verbose=0,
            eval_metric='Precision'
        )

        # 파이프라인
        pipe = create_pipeline(
            model,
            wins=False,
            resamp=exp['resamp'],
            resamp_ratio=exp['ratio'] if exp['ratio'] is not None else 0.2
        )

        # 학습
        print(f"학습 중...")
        pipe.fit(X_train, y_train)

        # 평가
        result = evaluate_model(
            pipe, X_val, y_val, X_test, y_test,
            model_name=exp['name']
        )

        results.append(result)

    # 결과 데이터프레임
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("📊 최종 비교 결과")
    print("=" * 80)
    print(results_df[['model_name', 'val_pr_auc', 'test_pr_auc',
                       'val_test_gap', 'test_recall', 'test_f2']].to_string(index=False))

    # 분석
    print("\n" + "=" * 80)
    print("🔍 분석")
    print("=" * 80)

    baseline = results_df.iloc[0]
    no_smote = results_df.iloc[-1]

    print(f"\nBaseline (SMOTE 0.2):")
    print(f"  Val-Test Gap: {baseline['val_test_gap']:.1f}%")
    print(f"  Test PR-AUC:  {baseline['test_pr_auc']:.4f}")

    print(f"\nNo SMOTE (Class Weight만):")
    print(f"  Val-Test Gap: {no_smote['val_test_gap']:.1f}%")
    print(f"  Test PR-AUC:  {no_smote['test_pr_auc']:.4f}")

    gap_reduction = baseline['val_test_gap'] - no_smote['val_test_gap']

    if gap_reduction > 10:
        print(f"\n✅ SMOTE 제거로 Gap {gap_reduction:.1f}%p 감소")
        print(f"   → SMOTE가 주요 원인, Class Weight로 전환 권장")
    elif gap_reduction < -10:
        print(f"\n⚠️ SMOTE 제거로 Gap {abs(gap_reduction):.1f}%p 증가")
        print(f"   → SMOTE는 원인 아님, 다른 원인 탐색 필요")
    else:
        print(f"\n⚪ Gap 변화 미미 ({gap_reduction:.1f}%p)")
        print(f"   → SMOTE 영향 제한적")

    # 최고 성능 모델
    best_model = results_df.sort_values('test_pr_auc', ascending=False).iloc[0]

    print(f"\n🏆 최고 성능 모델:")
    print(f"   {best_model['model_name']}")
    print(f"   Test PR-AUC:  {best_model['test_pr_auc']:.4f}")
    print(f"   Val-Test Gap: {best_model['val_test_gap']:.1f}%")

    # 권장 사항
    print(f"\n💡 권장 사항:")
    if best_model['val_test_gap'] < 10:
        print(f"   ✅ '{best_model['model_name']}' 사용 권장")
        print(f"   → Gap < 10% 달성")
    else:
        print(f"   ⚠️ 모든 설정에서 Gap > 10%")
        print(f"   → Week 2에서 Feature Engineering으로 개선 필요")

    # 결과 저장
    save_results(results_df, 'week1', 'smote_ablation')

    return results_df


if __name__ == '__main__':
    results = run_smote_ablation_study()
    print("\n✅ Week 1 실험 3 완료")
