#!/usr/bin/env python3
"""
개선된 모델 학습 스크립트

실험 목표:
1. L2 정규화 사용 (L1 대신)
2. StandardScaler 사용 (RobustScaler 대신)
3. 이해관계자_불신지수 제외 실험
4. 모든 조합 비교
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, fbeta_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb
import joblib
import os
from datetime import datetime

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# Transformer Classes
# ============================================================================

class InfiniteHandler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X.replace([np.inf, -np.inf], np.nan)

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, eps=1e-10): self.eps = eps
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_c = X.copy()
        for c in X_c.columns:
            if (X_c[c] >= 0).all(): X_c[c] = np.log1p(X_c[c] + self.eps)
        return X_c

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, l=0.005, u=0.995): self.l, self.u, self.b = l, u, {}
    def fit(self, X, y=None):
        for c in X.columns: self.b[c] = (X[c].quantile(self.l), X[c].quantile(self.u))
        return self
    def transform(self, X):
        X_c = X.copy()
        for c in X_c.columns: X_c[c] = X_c[c].clip(*self.b[c])
        return X_c

# ============================================================================
# Pipeline Factory
# ============================================================================

def create_pipeline(clf, use_log=True, use_wins=True, scaler='robust', resamp='smote'):
    """파이프라인 생성

    Args:
        clf: 분류기
        use_log: LogTransform 사용 여부
        use_wins: Winsorizer 사용 여부
        scaler: 'robust', 'standard', None
        resamp: 'smote', None
    """
    steps = [
        ('inf', InfiniteHandler()),
        ('imp', SimpleImputer(strategy='median').set_output(transform='pandas'))
    ]

    if use_log:
        steps.append(('log', LogTransformer()))

    if use_wins:
        steps.append(('wins', Winsorizer()))

    if scaler == 'robust':
        steps.append(('scaler', RobustScaler()))
    elif scaler == 'standard':
        steps.append(('scaler', StandardScaler()))

    if resamp == 'smote':
        steps.append(('resamp', SMOTE(sampling_strategy=0.2, random_state=RANDOM_STATE)))
    else:
        steps.append(('resamp', 'passthrough'))

    steps.append(('clf', clf))

    return ImbPipeline(steps)

# ============================================================================
# Data Loading
# ============================================================================

print('='*80)
print('📊 데이터 로딩')
print('='*80)

df = pd.read_csv('data/features/domain_based_features_완전판.csv', encoding='utf-8')
TARGET_COL = '모형개발용Performance(향후1년내부도여부)'
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print(f'데이터 shape: {df.shape}')
print(f'특성 개수: {X.shape[1]}')
print(f'부도율: {y.mean()*100:.2f}%')
print(f'\n특성 목록:')
for i, col in enumerate(X.columns, 1):
    print(f'  {i:2d}. {col}')

# Train/Val/Test split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE)

print(f'\nTrain: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}')

# ⚠️ 이해관계자_불신지수 제외 버전 준비
print('\n' + '='*80)
print('⚠️ 이해관계자_불신지수 제외 데이터 준비')
print('='*80)

DISTRUST_COL = '이해관계자_불신지수'
if DISTRUST_COL in X.columns:
    X_train_no_distrust = X_train.drop(columns=[DISTRUST_COL])
    X_val_no_distrust = X_val.drop(columns=[DISTRUST_COL])
    X_test_no_distrust = X_test.drop(columns=[DISTRUST_COL])
    print(f'✅ {DISTRUST_COL} 제외됨')
    print(f'   특성 개수: {X.shape[1]} → {X_train_no_distrust.shape[1]}')
else:
    X_train_no_distrust = X_train
    X_val_no_distrust = X_val
    X_test_no_distrust = X_test
    print(f'⚠️ {DISTRUST_COL} 컬럼을 찾을 수 없습니다')

# ============================================================================
# Experiment Setup
# ============================================================================

experiments = {
    # 기존 설정 (비교 기준)
    'Baseline_L1': {
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'grid': {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__penalty': ['l1'],
            'clf__solver': ['liblinear'],
            'clf__class_weight': [None]
        },
        'pipeline_args': {'use_log': True, 'use_wins': True, 'scaler': 'robust', 'resamp': 'smote'}
    },

    # ✅ Experiment 1: L2 정규화 (가장 중요)
    'Improved_L2': {
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'grid': {
            'clf__C': [0.1, 1, 10, 100],  # C 범위 확대
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs'],
            'clf__class_weight': ['balanced']
        },
        'pipeline_args': {'use_log': True, 'use_wins': True, 'scaler': 'robust', 'resamp': 'smote'}
    },

    # ✅ Experiment 2: StandardScaler
    'Improved_L2_StdScaler': {
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'grid': {
            'clf__C': [0.1, 1, 10, 100],
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs'],
            'clf__class_weight': ['balanced']
        },
        'pipeline_args': {'use_log': True, 'use_wins': True, 'scaler': 'standard', 'resamp': 'smote'}
    },

    # ✅ Experiment 3: ElasticNet
    'Improved_ElasticNet': {
        'model': SGDClassifier(loss='log_loss', random_state=RANDOM_STATE, max_iter=1000),
        'grid': {
            'clf__alpha': [0.0001, 0.001, 0.01],
            'clf__l1_ratio': [0.15, 0.5, 0.85],
            'clf__penalty': ['elasticnet'],
            'clf__class_weight': ['balanced']
        },
        'pipeline_args': {'use_log': True, 'use_wins': True, 'scaler': 'standard', 'resamp': 'smote'}
    },

    # ✅ Experiment 4: 정규화 없음 (L2 + C=매우 큼)
    'Improved_NoRegularization': {
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'grid': {
            'clf__C': [100, 1000, 10000],
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs'],
            'clf__class_weight': ['balanced']
        },
        'pipeline_args': {'use_log': True, 'use_wins': True, 'scaler': 'standard', 'resamp': 'smote'}
    },

    # ✅ Experiment 5: LogTransform 제거 (트리 모델용)
    'Improved_NoLogTransform': {
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'grid': {
            'clf__C': [1, 10, 100],
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs'],
            'clf__class_weight': ['balanced']
        },
        'pipeline_args': {'use_log': False, 'use_wins': False, 'scaler': 'standard', 'resamp': 'smote'}
    },

    # ✅ Experiment 6: 이해관계자_불신지수 제외 + L2
    'Improved_L2_NoDistrust': {
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'grid': {
            'clf__C': [0.1, 1, 10, 100],
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs'],
            'clf__class_weight': ['balanced']
        },
        'pipeline_args': {'use_log': True, 'use_wins': True, 'scaler': 'standard', 'resamp': 'smote'},
        'use_no_distrust': True  # 특별 플래그
    },

    # ✅ Experiment 7: 이해관계자_불신지수 제외 + 정규화 거의 없음
    'Improved_NoReg_NoDistrust': {
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'grid': {
            'clf__C': [100, 1000, 10000],
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs'],
            'clf__class_weight': ['balanced']
        },
        'pipeline_args': {'use_log': False, 'use_wins': False, 'scaler': 'standard', 'resamp': 'smote'},
        'use_no_distrust': True
    }
}

# ============================================================================
# Training Loop
# ============================================================================

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print('\n' + '='*80)
print('🔬 실험 시작')
print('='*80)

for exp_name, config in experiments.items():
    print(f'\n📌 {exp_name}')
    print('-'*80)

    # 이해관계자_불신지수 제외 여부 확인
    use_no_distrust = config.get('use_no_distrust', False)

    if use_no_distrust and DISTRUST_COL in X.columns:
        X_train_exp = X_train_no_distrust
        X_val_exp = X_val_no_distrust
        X_test_exp = X_test_no_distrust
        print(f'⚠️ {DISTRUST_COL} 제외 모드')
    else:
        X_train_exp = X_train
        X_val_exp = X_val
        X_test_exp = X_test

    # Create pipeline
    pipe = create_pipeline(config['model'], **config['pipeline_args'])

    # Create grid
    pipe_grid = config['grid']

    # RandomizedSearchCV
    search = RandomizedSearchCV(
        pipe, pipe_grid,
        n_iter=min(50, len(list(pipe_grid.values())[0]) * 4),  # 적절한 iter 수
        scoring='average_precision',
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0
    )

    print(f'학습 중...')
    search.fit(X_train_exp, y_train)

    # Best model
    best_model = search.best_estimator_

    # Validation performance
    y_val_prob = best_model.predict_proba(X_val_exp)[:, 1]
    y_val_pred = (y_val_prob >= 0.15).astype(int)  # 임계값 0.15

    val_pr_auc = average_precision_score(y_val, y_val_prob)
    val_f2 = fbeta_score(y_val, y_val_pred, beta=2)

    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
    val_recall = tp / (tp + fn)
    val_precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Coefficient analysis
    clf = best_model.named_steps['clf']
    if hasattr(clf, 'coef_'):
        coef = clf.coef_[0]
        nonzero_count = (coef != 0).sum()
        max_coef = np.max(np.abs(coef))

        # Top 5 features
        top5_idx = np.argsort(np.abs(coef))[-5:][::-1]
        feature_cols = X_train_exp.columns
        top5_features = [(feature_cols[i], coef[i]) for i in top5_idx]
    else:
        nonzero_count = 'N/A'
        max_coef = 'N/A'
        top5_features = []

    # Store results
    results[exp_name] = {
        'cv_pr_auc': search.best_score_,
        'val_pr_auc': val_pr_auc,
        'val_f2': val_f2,
        'val_recall': val_recall,
        'val_precision': val_precision,
        'nonzero_coef': nonzero_count,
        'max_coef': max_coef,
        'top5_features': top5_features,
        'best_params': search.best_params_,
        'best_model': best_model
    }

    print(f'  CV PR-AUC:     {search.best_score_:.4f}')
    print(f'  Val PR-AUC:    {val_pr_auc:.4f}')
    print(f'  Val F2-Score:  {val_f2:.4f}')
    print(f'  Val Recall:    {val_recall:.2%}')
    print(f'  Val Precision: {val_precision:.2%}')
    print(f'  0이 아닌 계수: {nonzero_count}')
    print(f'  최대 계수:     {max_coef}')
    print(f'\n  최적 하이퍼파라미터:')
    for k, v in search.best_params_.items():
        print(f'    {k}: {v}')

    if top5_features:
        print(f'\n  상위 5개 특성:')
        for feat, c in top5_features:
            print(f'    {feat:30s}: {c:10.6f}')

# ============================================================================
# 결과 비교 및 저장
# ============================================================================

print('\n' + '='*80)
print('📊 실험 결과 요약')
print('='*80)

# Create comparison DataFrame
comparison = []
for exp_name, res in results.items():
    comparison.append({
        'Experiment': exp_name,
        'CV_PR_AUC': res['cv_pr_auc'],
        'Val_PR_AUC': res['val_pr_auc'],
        'Gap': res['cv_pr_auc'] - res['val_pr_auc'],
        'F2_Score': res['val_f2'],
        'Recall': res['val_recall'],
        'Precision': res['val_precision'],
        'NonZero_Coef': res['nonzero_coef']
    })

comp_df = pd.DataFrame(comparison)
comp_df = comp_df.sort_values('Val_PR_AUC', ascending=False)

print('\n성능 비교:')
print(comp_df.to_string(index=False))

# Best experiment
best_exp = comp_df.iloc[0]['Experiment']
best_model = results[best_exp]['best_model']

print(f'\n✅ 최고 성능 실험: {best_exp}')
print(f'   Val PR-AUC: {results[best_exp]["val_pr_auc"]:.4f}')
print(f'   CV-Val Gap: {results[best_exp]["cv_pr_auc"] - results[best_exp]["val_pr_auc"]:.4f}')

# Test Set 평가
print('\n' + '='*80)
print('🎯 Test Set 최종 평가')
print('='*80)

y_test_prob = best_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= 0.15).astype(int)

test_pr_auc = average_precision_score(y_test, y_test_prob)
test_f2 = fbeta_score(y_test, y_test_pred, beta=2)

tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
test_recall = tp / (tp + fn)
test_precision = tp / (tp + fp) if (tp + fp) > 0 else 0

print(f'\nPR-AUC:    {test_pr_auc:.4f}')
print(f'F2-Score:  {test_f2:.4f}')
print(f'Recall:    {test_recall:.2%}')
print(f'Precision: {test_precision:.2%}')
print(f'\nConfusion Matrix:')
print(f'  TN: {tn:,}  |  FP: {fp:,}')
print(f'  FN: {fn:,}  |  TP: {tp:,}')

# 계수 상세 분석
print('\n' + '='*80)
print('📈 계수 상세 분석 (최고 성능 모델)')
print('='*80)

clf = best_model.named_steps['clf']
if hasattr(clf, 'coef_'):
    coef = clf.coef_[0]
    feature_names = X.columns.tolist()

    # 모든 계수 정렬
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coef,
        'Abs_Coef': np.abs(coef)
    })
    coef_df = coef_df.sort_values('Abs_Coef', ascending=False)

    print('\n모든 특성 계수 (절대값 순):')
    print(coef_df.to_string(index=False))

    # 계수 분포 통계
    print(f'\n계수 통계:')
    print(f'  평균 절대값: {coef_df["Abs_Coef"].mean():.6f}')
    print(f'  중앙값:      {coef_df["Abs_Coef"].median():.6f}')
    print(f'  최대값:      {coef_df["Abs_Coef"].max():.6f}')
    print(f'  최소값:      {coef_df["Abs_Coef"].min():.6f}')
    print(f'  표준편차:    {coef_df["Abs_Coef"].std():.6f}')

    # 계수 균형 지수 (최대 / 평균)
    balance_idx = coef_df["Abs_Coef"].max() / (coef_df["Abs_Coef"].mean() + 1e-10)
    print(f'  균형 지수:   {balance_idx:.2f} (작을수록 균형적)')

# 저장
OUTPUT_DIR = 'data/processed/improved_models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 최고 성능 모델 저장
joblib.dump(best_model, f'{OUTPUT_DIR}/best_model_{best_exp}_{timestamp}.pkl')
print(f'\n✅ 최고 성능 모델 저장: {OUTPUT_DIR}/best_model_{best_exp}_{timestamp}.pkl')

# 결과 저장
comp_df.to_csv(f'{OUTPUT_DIR}/experiment_results_{timestamp}.csv', index=False, encoding='utf-8-sig')
print(f'✅ 실험 결과 저장: {OUTPUT_DIR}/experiment_results_{timestamp}.csv')

# 계수 저장
if hasattr(clf, 'coef_'):
    coef_df.to_csv(f'{OUTPUT_DIR}/coefficients_{best_exp}_{timestamp}.csv', index=False, encoding='utf-8-sig')
    print(f'✅ 계수 저장: {OUTPUT_DIR}/coefficients_{best_exp}_{timestamp}.csv')

print('\n✅ 모든 실험 완료!')
