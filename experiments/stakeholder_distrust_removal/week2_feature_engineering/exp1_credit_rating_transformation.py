"""
Week 2 실험 1: 신용등급점수 변환
목적: 신용등급점수의 VIF를 낮추고 이해관계자_불신지수 대체 방지
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


def calculate_vif_for_features(X, features):
    """특정 특성들의 VIF 계산"""
    X_subset = X[features].fillna(0).replace([np.inf, -np.inf], 0)

    vif_data = []
    for i, feature in enumerate(features):
        try:
            vif = variance_inflation_factor(X_subset.values, i)
            if np.isinf(vif) or np.isnan(vif):
                vif = 999
        except:
            vif = 999

        vif_data.append({'feature': feature, 'VIF': vif})

    return pd.DataFrame(vif_data).sort_values('VIF', ascending=False)


def transform_credit_rating(X, method='onehot'):
    """
    신용등급점수 변환

    Args:
        X: 특성 데이터프레임
        method: 변환 방법
            - 'onehot': One-Hot Encoding (3그룹)
            - 'binary': Binary Encoding (High/Low)
            - 'ordinal': Ordinal Encoding (유지, 스케일링만)

    Returns:
        변환된 데이터프레임
    """
    X_transformed = X.copy()

    if '신용등급점수' not in X.columns:
        print("⚠️ 신용등급점수 특성이 없습니다.")
        return X_transformed

    if method == 'onehot':
        # 3그룹으로 분할
        # 1~3: 우량 (AAA, AA, A)
        # 4~6: 중간 (BBB, BB, B)
        # 7~10: 불량 (CCC 이하)
        X_transformed['신용등급_우량'] = (X['신용등급점수'] <= 3).astype(int)
        X_transformed['신용등급_중간'] = ((X['신용등급점수'] > 3) & (X['신용등급점수'] <= 6)).astype(int)
        X_transformed['신용등급_불량'] = (X['신용등급점수'] > 6).astype(int)

        # 원본 제거
        X_transformed = X_transformed.drop(columns=['신용등급점수'])

        print(f"✅ 신용등급점수 → One-Hot Encoding (3그룹)")
        print(f"   우량 (1~3): {X_transformed['신용등급_우량'].sum():,}개")
        print(f"   중간 (4~6): {X_transformed['신용등급_중간'].sum():,}개")
        print(f"   불량 (7~10): {X_transformed['신용등급_불량'].sum():,}개")

    elif method == 'binary':
        # 2그룹으로 분할
        # 1~5: 투자등급 (Investment Grade)
        # 6~10: 투기등급 (Speculative Grade)
        X_transformed['신용등급_투자등급'] = (X['신용등급점수'] <= 5).astype(int)

        # 원본 제거
        X_transformed = X_transformed.drop(columns=['신용등급점수'])

        print(f"✅ 신용등급점수 → Binary Encoding")
        print(f"   투자등급 (1~5):  {X_transformed['신용등급_투자등급'].sum():,}개")
        print(f"   투기등급 (6~10): {(1 - X_transformed['신용등급_투자등급']).sum():,}개")

    elif method == 'ordinal':
        # 그대로 유지
        print(f"✅ 신용등급점수 유지 (Ordinal)")

    return X_transformed


def run_credit_rating_transformation_experiment():
    """
    신용등급점수 변환 실험

    비교 대상:
    1. Baseline: 신용등급점수 유지
    2. One-Hot Encoding (3그룹)
    3. Binary Encoding
    4. 신용등급점수 완전 제거
    """
    print("=" * 80)
    print("Week 2 실험 1: 신용등급점수 변환")
    print("=" * 80)

    # 데이터 로딩
    X, y = load_data()
    X_removed = remove_feature_from_data(X, '이해관계자_불신지수')

    # 데이터 분할
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_removed, y)

    # 실험 설정
    experiments = [
        {'name': 'Baseline: 신용등급점수 유지', 'method': 'ordinal'},
        {'name': 'One-Hot Encoding (3그룹)', 'method': 'onehot'},
        {'name': 'Binary Encoding', 'method': 'binary'},
        {'name': '신용등급점수 완전 제거', 'method': 'remove'}
    ]

    results = []

    for exp in experiments:
        print(f"\n{'='*80}")
        print(f"{exp['name']}")
        print(f"{'='*80}")

        # 데이터 변환
        if exp['method'] == 'remove':
            X_train_exp = X_train.drop(columns=['신용등급점수']) if '신용등급점수' in X_train.columns else X_train
            X_val_exp = X_val.drop(columns=['신용등급점수']) if '신용등급점수' in X_val.columns else X_val
            X_test_exp = X_test.drop(columns=['신용등급점수']) if '신용등급점수' in X_test.columns else X_test
            print(f"✅ 신용등급점수 제거")
        else:
            X_train_exp = transform_credit_rating(X_train, method=exp['method'])
            X_val_exp = transform_credit_rating(X_val, method=exp['method'])
            X_test_exp = transform_credit_rating(X_test, method=exp['method'])

        print(f"\n특성 수: {X_train_exp.shape[1]}개")

        # VIF 계산 (샘플링)
        if '신용등급' in ' '.join(X_train_exp.columns):
            credit_features = [col for col in X_train_exp.columns if '신용등급' in col]
            if credit_features:
                print(f"\nVIF 계산 (신용등급 관련 특성):")
                sample_size = min(1000, len(X_train_exp))
                X_sample = X_train_exp.sample(n=sample_size, random_state=RANDOM_STATE)
                vif_df = calculate_vif_for_features(X_sample, credit_features)
                print(vif_df.to_string(index=False))

        # CatBoost 모델
        model = CatBoostClassifier(
            iterations=100,
            depth=5,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbose=0,
            eval_metric='Precision'
        )

        # 파이프라인
        pipe = create_pipeline(model, wins=False, resamp='smote', resamp_ratio=0.2)

        # 학습
        print(f"\n학습 중...")
        pipe.fit(X_train_exp, y_train)

        # 평가
        result = evaluate_model(
            pipe, X_val_exp, y_val, X_test_exp, y_test,
            model_name=exp['name']
        )

        result['n_features'] = X_train_exp.shape[1]
        results.append(result)

    # 결과 데이터프레임
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("📊 최종 비교 결과")
    print("=" * 80)
    print(results_df[['model_name', 'n_features', 'val_pr_auc', 'test_pr_auc',
                       'val_test_gap', 'test_recall']].to_string(index=False))

    # 분석
    print("\n" + "=" * 80)
    print("🔍 분석")
    print("=" * 80)

    baseline = results_df.iloc[0]
    best_model = results_df.sort_values('test_pr_auc', ascending=False).iloc[0]
    best_gap = results_df.sort_values('val_test_gap').iloc[0]

    print(f"\nBaseline (신용등급점수 유지):")
    print(f"  Val-Test Gap: {baseline['val_test_gap']:.1f}%")
    print(f"  Test PR-AUC:  {baseline['test_pr_auc']:.4f}")

    print(f"\n최고 Test PR-AUC:")
    print(f"  {best_model['model_name']}")
    print(f"  Test PR-AUC:  {best_model['test_pr_auc']:.4f}")
    print(f"  Val-Test Gap: {best_model['val_test_gap']:.1f}%")

    print(f"\n최소 Val-Test Gap:")
    print(f"  {best_gap['model_name']}")
    print(f"  Val-Test Gap: {best_gap['val_test_gap']:.1f}%")
    print(f"  Test PR-AUC:  {best_gap['test_pr_auc']:.4f}")

    # 권장 사항
    print(f"\n💡 권장 사항:")
    if best_gap['val_test_gap'] < 10:
        print(f"   ✅ '{best_gap['model_name']}' 사용 권장")
        print(f"   → Gap < 10% 달성")
    else:
        print(f"   ⚠️ 모든 변환 방법에서 Gap > 10%")
        print(f"   → 추가 Feature Engineering 필요")

    # 결과 저장
    save_results(results_df, 'week2', 'credit_rating_transformation')

    return results_df


if __name__ == '__main__':
    results = run_credit_rating_transformation_experiment()
    print("\n✅ Week 2 실험 1 완료")
