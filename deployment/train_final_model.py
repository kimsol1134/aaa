"""
Part3 노트북의 핵심 부분 추출 - 최종 모델 학습 및 저장

이 스크립트는 Part3 노트북과 동일한 파이프라인으로 모델을 학습합니다.
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score, fbeta_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

# 설정
RANDOM_STATE = 42
DATA_DIR = 'data'
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

INPUT_FILE = os.path.join(DATA_DIR, 'features', 'domain_based_features_완전판.csv')
OUTPUT_PREFIX = '발표_Part3_v3'

print("=" * 80)
print("🚀 Part3 최종 모델 학습 시작")
print("=" * 80)

# 1. 데이터 로드
print("\n📂 1. 데이터 로드 중...")
# 특성 데이터
features_df = pd.read_csv(INPUT_FILE, index_col=0, encoding='utf-8-sig')
print(f"   특성 데이터 shape: {features_df.shape}")

# 원본 데이터에서 타겟 변수 로드
print("   원본 데이터에서 타겟 변수 로드...")
original_file = os.path.join(DATA_DIR, '기업신용평가정보_210801.csv')
original_df = pd.read_csv(original_file, index_col=0, encoding='cp949')

target_col = '모형개발용Performance(향후1년내부도여부)'
if target_col not in original_df.columns:
    # 컬럼명 확인
    print("   사용 가능한 타겟 변수:")
    target_candidates = [col for col in original_df.columns if 'Performance' in col or '부도' in col]
    print(f"   {target_candidates}")
    if target_candidates:
        target_col = target_candidates[0]
        print(f"   → '{target_col}' 사용")
    else:
        raise ValueError("타겟 변수를 찾을 수 없습니다.")

# 타겟 변수 추출 (인덱스 기준 매칭)
y = original_df.loc[features_df.index, target_col]
print(f"   타겟 변수 shape: {y.shape}")

# 2. 특성과 타겟 분리
print("\n🎯 2. 특성과 타겟 확인...")
X = features_df
print(f"   X shape: {X.shape}")
print(f"   y 분포: 정상={np.sum(y==0)}, 부도={np.sum(y==1)}, 비율={np.mean(y):.4f}")

# 3. Train/Val/Test 분할
print("\n✂️  3. 데이터 분할 (Train/Val/Test)...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# 4. 커스텀 Transformer 정의
print("\n🔧 4. 전처리 파이프라인 구성...")

from sklearn.base import BaseEstimator, TransformerMixin

class InfiniteHandler(BaseEstimator, TransformerMixin):
    """무한대 값을 0으로 변환"""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X = X.replace([np.inf, -np.inf], 0)
        return X

class LogTransformer(BaseEstimator, TransformerMixin):
    """로그 변환 (양수만)"""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            if (X[col] > 0).all():
                X[col] = np.log1p(X[col])
        return X

def create_pipeline(clf, use_smote=True):
    """
    Part3 노트북과 동일한 파이프라인 생성

    Args:
        clf: 분류기
        use_smote: SMOTE 사용 여부

    Returns:
        Pipeline
    """
    steps = [
        ('inf_handler', InfiniteHandler()),
        ('imputer', SimpleImputer(strategy='median')),
        ('log_transform', LogTransformer()),
        ('scaler', RobustScaler()),
    ]

    if use_smote:
        steps.append(('smote', SMOTE(sampling_strategy=0.2, random_state=RANDOM_STATE)))

    steps.append(('classifier', clf))

    return Pipeline(steps)

# 5. 간단한 모델로 빠른 학습 (Logistic Regression)
print("\n🎓 5. Logistic Regression (L1) 모델 학습...")
print("   (빠른 학습을 위해 간단한 모델 사용)")

lr_clf = LogisticRegression(
    penalty='l1',
    C=1.0,
    solver='liblinear',
    class_weight='balanced',
    random_state=RANDOM_STATE,
    max_iter=1000
)

pipeline = create_pipeline(lr_clf, use_smote=True)
print("   학습 중...")
pipeline.fit(X_train, y_train)
print("   ✅ 학습 완료!")

# 6. 검증
print("\n📊 6. 모델 검증...")
y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]
val_pr_auc = average_precision_score(y_val, y_val_pred_proba)
print(f"   Validation PR-AUC: {val_pr_auc:.4f}")

# 7. 테스트
print("\n🧪 7. 최종 테스트...")
y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
test_pr_auc = average_precision_score(y_test, y_test_pred_proba)

# 임계값 설정 (간단히 중앙값 사용)
threshold = 0.05
y_test_pred = (y_test_pred_proba >= threshold).astype(int)
test_recall = recall_score(y_test, y_test_pred)
test_f2 = fbeta_score(y_test, y_test_pred, beta=2)

print(f"   Test PR-AUC: {test_pr_auc:.4f}")
print(f"   Test Recall: {test_recall:.4f}")
print(f"   Test F2-Score: {test_f2:.4f}")

# 8. 모델 저장
print("\n💾 8. 모델 및 파이프라인 저장...")

# 파이프라인 전체 저장 (전처리 + 모델)
model_path = os.path.join(PROCESSED_DIR, f'{OUTPUT_PREFIX}_최종모델.pkl')
joblib.dump(pipeline, model_path)
print(f"   ✅ {OUTPUT_PREFIX}_최종모델.pkl")

# 임계값 저장
thresholds = {
    'selected': threshold,
    'red': 0.05,
    'yellow': 0.02,
    'green': 0.01
}
threshold_path = os.path.join(PROCESSED_DIR, f'{OUTPUT_PREFIX}_임계값.pkl')
joblib.dump(thresholds, threshold_path)
print(f"   ✅ {OUTPUT_PREFIX}_임계값.pkl")

# 결과 저장
results = {
    'model_name': 'LogisticRegression_L1_SMOTE',
    'test_pr_auc': test_pr_auc,
    'test_recall': test_recall,
    'test_f2': test_f2,
    'val_pr_auc': val_pr_auc
}
results_path = os.path.join(PROCESSED_DIR, f'{OUTPUT_PREFIX}_결과.pkl')
joblib.dump(results, results_path)
print(f"   ✅ {OUTPUT_PREFIX}_결과.pkl")

# 전처리만 분리해서 저장 (Streamlit용)
preprocessing_steps = pipeline.steps[:-1]  # 마지막 classifier 제외
preprocessing_pipeline = Pipeline(preprocessing_steps)
preprocess_path = os.path.join(PROCESSED_DIR, 'preprocessing_pipeline.pkl')
joblib.dump(preprocessing_pipeline, preprocess_path)
print(f"   ✅ preprocessing_pipeline.pkl")

print(f"\n저장 위치: {PROCESSED_DIR}")
print("=" * 80)
print("✅ 모델 학습 및 저장 완료!")
print("=" * 80)

# deployment로 복사
print("\n📦 deployment 폴더로 복사 중...")
import shutil

deploy_model_dir = 'deployment/data/processed'
os.makedirs(deploy_model_dir, exist_ok=True)

files_to_copy = [
    f'{OUTPUT_PREFIX}_최종모델.pkl',
    f'{OUTPUT_PREFIX}_임계값.pkl',
    'preprocessing_pipeline.pkl'
]

for fname in files_to_copy:
    src = os.path.join(PROCESSED_DIR, fname)
    dst = os.path.join(deploy_model_dir, fname)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"   ✅ {fname} → deployment/")

print("\n🎉 완료! deployment 폴더에 최종 모델이 준비되었습니다.")
