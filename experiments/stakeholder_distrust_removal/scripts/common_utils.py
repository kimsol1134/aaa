"""
공통 유틸리티 함수 모음
실험 전반에서 사용되는 공통 함수들
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import average_precision_score, recall_score, f1_score, fbeta_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import scipy.stats as stats
from scipy.stats import ks_2samp

import joblib
import json
from pathlib import Path
from datetime import datetime

# 경로 설정
BASE_DIR = Path('/home/user/aaa')
DATA_DIR = BASE_DIR / 'data'
FEATURES_DIR = DATA_DIR / 'features'
PROCESSED_DIR = DATA_DIR / 'processed'
EXPERIMENT_DIR = BASE_DIR / 'experiments' / 'stakeholder_distrust_removal'
RESULTS_DIR = EXPERIMENT_DIR / 'results'

RANDOM_STATE = 42


def load_data():
    """
    도메인 기반 특성 데이터 로딩

    Returns:
        X: 특성 데이터프레임
        y: 타겟 변수 (부도 여부)
    """
    # 완전판 데이터 로딩
    data_path = FEATURES_DIR / 'domain_based_features_완전판.csv'
    df = pd.read_csv(data_path, encoding='utf-8')

    target_col = '모형개발용Performance(향후1년내부도여부)'

    y = df[target_col]
    X = df.drop(columns=[target_col])

    print(f"✅ 데이터 로딩 완료: {X.shape[0]:,} 기업, {X.shape[1]:,} 특성")
    print(f"   부도율: {y.mean()*100:.2f}% ({y.sum():,} / {len(y):,})")

    return X, y


def split_data(X, y, test_size=0.2, val_size=0.25, random_state=RANDOM_STATE):
    """
    데이터를 Train/Val/Test로 분할

    Args:
        X: 특성 데이터
        y: 타겟 변수
        test_size: Test set 비율 (전체의 20%)
        val_size: Validation set 비율 (temp의 25% = 전체의 20%)
        random_state: 랜덤 시드

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Test set 분할
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Validation set 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state
    )

    print(f"\n📊 데이터 분할 완료:")
    print(f"   Train: {len(y_train):,} ({y_train.mean()*100:.2f}% 부도)")
    print(f"   Val:   {len(y_val):,} ({y_val.mean()*100:.2f}% 부도)")
    print(f"   Test:  {len(y_test):,} ({y_test.mean()*100:.2f}% 부도)")

    return X_train, X_val, X_test, y_train, y_val, y_test


class InfiniteHandler:
    """무한대 값 처리"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.replace([np.inf, -np.inf], np.nan)
        return X_copy

    def fit_transform(self, X, y=None):
        return self.transform(X)


class LogTransformer:
    """로그 변환 (음수 처리 포함)"""
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in X_copy.columns:
            if X_copy[col].min() < 0:
                # 음수가 있으면 변환 스킵
                continue
            else:
                # 양수만 있으면 로그 변환
                X_copy[col] = np.log1p(X_copy[col] + self.epsilon)
        return X_copy

    def fit_transform(self, X, y=None):
        return self.transform(X)


def create_pipeline(clf, wins=False, resamp='smote', resamp_ratio=0.2):
    """
    전처리 + 리샘플링 + 분류기 파이프라인 생성

    Args:
        clf: 분류기
        wins: Winsorization 적용 여부
        resamp: 리샘플링 방법 ('smote', 'borderline', 'adasyn', 'smote_enn', None)
        resamp_ratio: 리샘플링 비율

    Returns:
        ImbPipeline 객체
    """
    steps = [
        ('inf', InfiniteHandler()),
        ('imp', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ]

    # Resampling
    if resamp == 'smote':
        resampler = SMOTE(sampling_strategy=resamp_ratio, random_state=RANDOM_STATE)
    elif resamp == 'borderline':
        resampler = BorderlineSMOTE(sampling_strategy=resamp_ratio, random_state=RANDOM_STATE)
    elif resamp == 'adasyn':
        resampler = ADASYN(sampling_strategy=resamp_ratio, random_state=RANDOM_STATE)
    elif resamp == 'smote_enn':
        resampler = SMOTEENN(random_state=RANDOM_STATE)
    elif resamp is None:
        resampler = 'passthrough'
    else:
        resampler = 'passthrough'

    steps.append(('resamp', resampler))
    steps.append(('clf', clf))

    return ImbPipeline(steps)


def evaluate_model(model, X_val, y_val, X_test, y_test, model_name='Model'):
    """
    모델 평가 및 메트릭 계산

    Args:
        model: 학습된 모델
        X_val: Validation 특성
        y_val: Validation 타겟
        X_test: Test 특성
        y_test: Test 타겟
        model_name: 모델 이름

    Returns:
        결과 딕셔너리
    """
    # Validation 성능
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_pr_auc = average_precision_score(y_val, y_val_pred_proba)

    # Test 성능
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    test_pr_auc = average_precision_score(y_test, y_test_pred_proba)

    # Recall@80% threshold (validation 기준)
    threshold = np.percentile(y_val_pred_proba, 80)
    y_test_pred = (y_test_pred_proba >= threshold).astype(int)

    test_recall = recall_score(y_test, y_test_pred)
    test_f2 = fbeta_score(y_test, y_test_pred, beta=2)

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

    # Val-Test Gap
    gap = abs(test_pr_auc - val_pr_auc) / val_pr_auc * 100

    results = {
        'model_name': model_name,
        'val_pr_auc': val_pr_auc,
        'test_pr_auc': test_pr_auc,
        'val_test_gap': gap,
        'test_recall': test_recall,
        'test_f2': test_f2,
        'threshold': threshold,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }

    print(f"\n📊 {model_name} 평가 결과:")
    print(f"   Val PR-AUC:    {val_pr_auc:.4f}")
    print(f"   Test PR-AUC:   {test_pr_auc:.4f}")
    print(f"   Val-Test Gap:  {gap:.1f}%")
    print(f"   Test Recall:   {test_recall:.2%}")
    print(f"   Test F2-Score: {test_f2:.4f}")

    return results


def compare_distributions(X_val, X_test, features):
    """
    Validation과 Test 세트의 특성 분포 비교 (KS-Test)

    Args:
        X_val: Validation 특성
        X_test: Test 특성
        features: 비교할 특성 리스트

    Returns:
        분포 차이 유의한 특성 리스트
    """
    significant_diffs = []

    for feature in features:
        ks_stat, p_value = ks_2samp(X_val[feature], X_test[feature])

        if p_value < 0.05:
            significant_diffs.append({
                'feature': feature,
                'ks_stat': ks_stat,
                'p_value': p_value
            })

    return pd.DataFrame(significant_diffs).sort_values('p_value')


def save_results(results, experiment_name, suffix=''):
    """
    실험 결과 저장

    Args:
        results: 결과 딕셔너리 또는 데이터프레임
        experiment_name: 실험 이름 (week1, week2, week3)
        suffix: 파일명 접미사
    """
    output_dir = RESULTS_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{experiment_name}_{suffix}_{timestamp}.json" if suffix else f"{experiment_name}_{timestamp}.json"

    output_path = output_dir / filename

    # 데이터프레임인 경우 CSV로 저장
    if isinstance(results, pd.DataFrame):
        csv_path = output_path.with_suffix('.csv')
        results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 결과 저장: {csv_path}")
    else:
        # 딕셔너리인 경우 JSON으로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ 결과 저장: {output_path}")

    return output_path


def remove_feature_from_data(X, feature_name):
    """
    특성 제거 (이해관계자_불신지수 등)

    Args:
        X: 특성 데이터프레임
        feature_name: 제거할 특성명

    Returns:
        특성 제거된 데이터프레임
    """
    if feature_name in X.columns:
        X_removed = X.drop(columns=[feature_name])
        print(f"✅ '{feature_name}' 제거: {X.shape[1]} → {X_removed.shape[1]} 특성")
        return X_removed
    else:
        print(f"⚠️ '{feature_name}' 특성이 데이터에 없습니다.")
        return X.copy()


if __name__ == '__main__':
    # 테스트
    print("=" * 70)
    print("공통 유틸리티 테스트")
    print("=" * 70)

    # 데이터 로딩 테스트
    X, y = load_data()

    # 데이터 분할 테스트
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 이해관계자_불신지수 제거 테스트
    X_removed = remove_feature_from_data(X, '이해관계자_불신지수')

    print("\n✅ 공통 유틸리티 정상 작동 확인")
