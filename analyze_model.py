#!/usr/bin/env python3
"""
Part 3 모델 분석 스크립트
LogisticRegression이 어떤 특성을 사용하는지 확인
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Transformer classes
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

def create_pipeline(clf, wins=False, resamp=None):
    s = [
        ('inf', InfiniteHandler()),
        ('imp', SimpleImputer(strategy='median').set_output(transform='pandas')),
        ('log', LogTransformer())
    ]
    if wins: s.append(('wins', Winsorizer()))
    s.append(('scaler', RobustScaler()))
    s.append(('resamp', SMOTE(sampling_strategy=0.2, random_state=RANDOM_STATE) if resamp else 'passthrough'))
    s.append(('clf', clf))
    return ImbPipeline(s)

# 데이터 로드
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

# Train/Val split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE)

print(f'\nTrain: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}')

# LogisticRegression AutoML
print('\n' + '='*80)
print('🔬 LogisticRegression 하이퍼파라미터 튜닝')
print('='*80)

lr_grid = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l1', 'l2'],
    'clf__solver': ['liblinear'],
    'clf__class_weight': [None]  # SMOTE 사용하므로 None
}

lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
pipe = create_pipeline(lr, wins=True, resamp='smote')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
search = RandomizedSearchCV(pipe, lr_grid, n_iter=200, scoring='average_precision',
                             cv=cv, n_jobs=-1, random_state=RANDOM_STATE, verbose=1)

print('\n튜닝 시작 (n_iter=200)...')
search.fit(X_train, y_train)

print(f'\n최적 하이퍼파라미터:')
for k, v in search.best_params_.items():
    print(f'  {k}: {v}')

print(f'\nCV PR-AUC: {search.best_score_:.4f}')

# Validation 성능
best_model = search.best_estimator_
y_val_prob = best_model.predict_proba(X_val)[:, 1]
val_pr = average_precision_score(y_val, y_val_prob)
print(f'Val PR-AUC: {val_pr:.4f}')

# 계수 분석
print('\n' + '='*80)
print('📈 계수 분석')
print('='*80)

clf = best_model.named_steps['clf']
coef = clf.coef_[0]
feature_names = X.columns.tolist()

print(f'\n계수 shape: {coef.shape}')
print(f'0이 아닌 계수: {(coef != 0).sum()} / {len(coef)}')

# 0이 아닌 계수만 정렬
nonzero_mask = coef != 0
if nonzero_mask.any():
    nonzero_features = [(feature_names[i], coef[i]) for i in range(len(feature_names)) if nonzero_mask[i]]
    nonzero_features_sorted = sorted(nonzero_features, key=lambda x: abs(x[1]), reverse=True)

    print(f'\n✅ 사용된 특성 ({len(nonzero_features_sorted)}개):')
    print('-'*80)
    for feat, c in nonzero_features_sorted:
        print(f'  {feat:50s}: {c:12.6f}')
else:
    print('\n⚠️ 모든 계수가 0 (모델이 특성을 사용하지 않음)')

if hasattr(clf, 'intercept_'):
    print(f'\n절편 (Intercept): {clf.intercept_[0]:.6f}')

# 데이터 전처리 영향 분석
print('\n' + '='*80)
print('🔍 데이터 전처리 영향 분석')
print('='*80)

# 원본 데이터 통계
print('\n원본 데이터:')
print(X_train.describe().loc[['mean', 'std', 'min', 'max']].T.head(10))

# 전처리 후 데이터 확인
from sklearn.pipeline import Pipeline
prep_pipe = Pipeline([
    ('inf', InfiniteHandler()),
    ('imp', SimpleImputer(strategy='median').set_output(transform='pandas')),
    ('log', LogTransformer()),
    ('wins', Winsorizer()),
    ('scaler', RobustScaler())
])

X_train_prep = prep_pipe.fit_transform(X_train, y_train)
print('\n전처리 후 데이터:')
if isinstance(X_train_prep, pd.DataFrame):
    print(X_train_prep.describe().loc[['mean', 'std', 'min', 'max']].T.head(10))
else:
    X_train_prep_df = pd.DataFrame(X_train_prep, columns=X.columns)
    print(X_train_prep_df.describe().loc[['mean', 'std', 'min', 'max']].T.head(10))

print('\n✅ 분석 완료')
