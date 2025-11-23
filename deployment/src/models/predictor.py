"""
부도 예측 모델 로딩 및 예측

Part3 노트북과 동일한 파이프라인으로 예측 수행
학습된 모델을 로드하고 새로운 데이터에 대해 예측 수행
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging
import sys

# 전처리 모듈 import
try:
    from src.preprocessing.transformers import create_preprocessing_pipeline
except ImportError:
    # deployment 폴더에서 실행될 경우
    try:
        from preprocessing.transformers import create_preprocessing_pipeline
    except ImportError:
        create_preprocessing_pipeline = None
        logging.warning("전처리 모듈을 import할 수 없습니다. 기본 전처리 사용")

logger = logging.getLogger(__name__)


class BankruptcyPredictor:
    """
    부도 예측 모델

    Part3 노트북과 동일한 파이프라인 지원:
    - 전처리 파이프라인 (InfiniteHandler, LogTransformer, Scaler 등)
    - 전체 파이프라인 (전처리 + 모델)
    - 휴리스틱 방식 (모델 없을 때)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        pipeline_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        use_pipeline: bool = True
    ):
        """
        Args:
            model_path: 모델 파일 경로 (단독 모델)
            pipeline_path: 파이프라인 파일 경로 (전처리 + 모델)
            scaler_path: 스케일러 파일 경로 (단독 스케일러)
            use_pipeline: 파이프라인 사용 여부 (Part3 방식)
        """
        self.model = None
        self.pipeline = None
        self.scaler = None
        self.preprocessing_pipeline = None

        self.model_path = model_path
        self.pipeline_path = pipeline_path
        self.scaler_path = scaler_path
        self.use_pipeline = use_pipeline
        self.expected_features = None

    def load_model(self):
        """
        모델 로드 (우선순위):
        1. 전체 파이프라인 (전처리 + 모델) - Part3 방식
        2. 모델 + 스케일러 분리
        3. 휴리스틱 방식
        """
        try:
            # 1. 전체 파이프라인 로드 시도 (Part3 방식)
            if self.use_pipeline and self.pipeline_path and self.pipeline_path.exists():
                logger.info(f"📦 전체 파이프라인 로딩 중: {self.pipeline_path}")
                self.pipeline = joblib.load(self.pipeline_path)
                logger.info("✓ Part3 파이프라인 로드 성공!")
                logger.info(f"   파이프라인 단계: {len(self.pipeline.steps)}개")
                for step_name, _ in self.pipeline.steps:
                    logger.info(f"   - {step_name}")
                return

            # 2. 모델 단독 로드
            if self.model_path and self.model_path.exists():
                logger.info(f"🎯 모델 로딩 중: {self.model_path}")
                self.model = joblib.load(self.model_path)
                logger.info("✓ 모델 로드 성공")
            else:
                logger.warning("모델 파일을 찾을 수 없습니다.")
                self.model = None

            # 3. 스케일러 로드
            if self.scaler_path and self.scaler_path.exists():
                logger.info(f"📏 스케일러 로딩 중: {self.scaler_path}")
                self.scaler = joblib.load(self.scaler_path)
                logger.info("✓ 스케일러 로드 성공")
            else:
                logger.warning("스케일러 파일을 찾을 수 없습니다.")

                # 스케일러 없으면 전처리 파이프라인 생성
                if create_preprocessing_pipeline:
                    logger.info("기본 전처리 파이프라인 생성 중...")
                    self.preprocessing_pipeline = create_preprocessing_pipeline(
                        use_log_transform=True,
                        use_winsorizer=False,
                        scaler_type='robust'
                    )
                    logger.info("✓ Part3 전처리 파이프라인 생성 완료")

        except Exception as e:
            logger.error(f"모델 로딩 실패: {str(e)}")
            logger.warning("휴리스틱 방식으로 전환합니다.")
            self.model = None
            self.pipeline = None
            self.scaler = None

    def predict(self, features_df: pd.DataFrame) -> Dict:
        """
        부도 확률 예측

        Args:
            features_df: 특성 DataFrame (1행)

        Returns:
            {
                'bankruptcy_probability': 0.15,
                'risk_level': '주의',
                'confidence': 0.85,
                'features_used': [...],
                'model_info': {...}
            }
        """
        try:
            # 1. 전체 파이프라인 사용 (Part3 방식)
            if self.pipeline is not None:
                logger.info("Part3 파이프라인으로 예측 중...")
                X = self._prepare_features(features_df)

                # 파이프라인으로 직접 예측
                if hasattr(self.pipeline, 'predict_proba'):
                    proba = self.pipeline.predict_proba(X)[0]
                    bankruptcy_prob = proba[1]
                    confidence = max(proba)
                else:
                    prediction = self.pipeline.predict(X)[0]
                    bankruptcy_prob = 0.8 if prediction == 1 else 0.2
                    confidence = 0.7

                # 파이프라인 내부 모델 추출 (SHAP용)
                model_for_shap = self.pipeline.named_steps.get('classifier', None)
                X_for_shap = X

            # 2. 전처리 파이프라인 + 모델 분리 사용
            elif self.preprocessing_pipeline is not None and self.model is not None:
                logger.info("전처리 파이프라인 + 모델로 예측 중...")
                X = self._prepare_features(features_df)
                X_preprocessed = self.preprocessing_pipeline.transform(X)

                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(X_preprocessed)[0]
                    bankruptcy_prob = proba[1]
                    confidence = max(proba)
                else:
                    prediction = self.model.predict(X_preprocessed)[0]
                    bankruptcy_prob = 0.8 if prediction == 1 else 0.2
                    confidence = 0.7

                model_for_shap = self.model
                X_for_shap = X_preprocessed

            # 3. 모델만 사용 (스케일러 포함)
            elif self.model is not None:
                logger.info("모델 단독 예측 중...")
                X = self._prepare_features(features_df)

                # 스케일링
                if self.scaler is not None:
                    X_scaled = self.scaler.transform(X)
                else:
                    X_scaled = X

                # 예측
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(X_scaled)[0]
                    bankruptcy_prob = proba[1]
                    confidence = max(proba)
                else:
                    prediction = self.model.predict(X_scaled)[0]
                    bankruptcy_prob = 0.8 if prediction == 1 else 0.2
                    confidence = 0.7

                model_for_shap = self.model
                X_for_shap = X_scaled

            # 4. 모델이 없으면 휴리스틱
            else:
                logger.warning("모델 없음. 휴리스틱 기반 예측 사용")
                return self._heuristic_prediction(features_df)

            # SHAP 값 계산
            shap_values = None
            shap_base_value = None
            try:
                import shap
                if model_for_shap is not None:
                    explainer = shap.TreeExplainer(model_for_shap)
                    shap_values_result = explainer.shap_values(X_for_shap)
                else:
                    raise ValueError("SHAP을 위한 모델을 찾을 수 없습니다.")

                # CatBoost는 리스트 반환 → 부도(1) 클래스만 사용
                if isinstance(shap_values_result, list):
                    shap_values = shap_values_result[1][0]
                    shap_base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                else:
                    shap_values = shap_values_result[0]
                    shap_base_value = float(explainer.expected_value)

                logger.info("✓ SHAP 값 계산 완료")
            except Exception as e:
                logger.warning(f"SHAP 계산 실패: {e}")
                shap_values = None
                shap_base_value = None

            # 결과 생성
            from src.utils.helpers import get_risk_level
            risk_level, icon, msg = get_risk_level(bankruptcy_prob)

            # 모델 타입 결정
            if self.pipeline is not None:
                model_type = f"Pipeline({type(model_for_shap).__name__})"
            elif self.model is not None:
                model_type = type(self.model).__name__
            else:
                model_type = "Heuristic"

            result = {
                'bankruptcy_probability': float(bankruptcy_prob),
                'risk_level': risk_level,
                'risk_icon': icon,
                'risk_message': msg,
                'confidence': float(confidence),
                'features_used': list(X_for_shap.columns) if hasattr(X_for_shap, 'columns') else [],
                'model_info': {
                    'model_type': model_type,
                    'n_features': X_for_shap.shape[1] if hasattr(X_for_shap, 'shape') else 0
                }
            }

            # SHAP 정보 추가
            if shap_values is not None:
                result['shap_values'] = shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values
                result['shap_base_value'] = float(shap_base_value)
                result['feature_names'] = list(X_for_shap.columns) if hasattr(X_for_shap, 'columns') else []

            logger.info(f"예측 완료: 부도 확률 {bankruptcy_prob:.1%}, 등급 {risk_level}")

            return result

        except Exception as e:
            logger.error(f"예측 실패: {str(e)}")
            # 에러 시 휴리스틱 예측
            return self._heuristic_prediction(features_df)

    def _prepare_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        모델에 맞게 특성 준비

        Args:
            features_df: 생성된 특성 DataFrame

        Returns:
            모델 입력용 DataFrame
        """
        # 모델이 기대하는 특성 목록 로드 (선택된 특성)
        # 실제로는 학습 시 사용한 특성 목록을 저장해두고 로드해야 함
        # 여기서는 모든 특성 사용

        X = features_df.copy()

        # 범주형 변수 제거 (숫자형만)
        X = X.select_dtypes(include=[np.number])

        # NaN/Inf 제거
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)

        return X

    def _heuristic_prediction(self, features_df: pd.DataFrame) -> Dict:
        """
        휴리스틱 기반 부도 확률 예측 (모델 없을 때)

        주요 지표들을 조합하여 경험적으로 부도 확률 추정

        Args:
            features_df: 특성 DataFrame

        Returns:
            예측 결과
        """
        logger.info("휴리스틱 기반 예측 실행")

        # 주요 위험 지표 추출
        유동성위기 = features_df.get('유동성위기지수', pd.Series([0.5])).iloc[0]
        지급불능위험 = features_df.get('지급불능위험지수', pd.Series([0.5])).iloc[0]
        재무조작위험 = features_df.get('재무조작위험지수', pd.Series([0.3])).iloc[0]

        # 조기경보신호
        경보신호수 = features_df.get('조기경보신호수', pd.Series([0])).iloc[0]

        # 종합 부도 위험 스코어 (가중평균)
        bankruptcy_prob = (
            0.35 * 유동성위기 +
            0.35 * 지급불능위험 +
            0.20 * 재무조작위험 +
            0.10 * min(1.0, 경보신호수 / 5)
        )

        # 0~1 범위로 클리핑
        bankruptcy_prob = max(0.0, min(1.0, bankruptcy_prob))

        from src.utils.helpers import get_risk_level
        risk_level, icon, msg = get_risk_level(bankruptcy_prob)

        result = {
            'bankruptcy_probability': float(bankruptcy_prob),
            'risk_level': risk_level,
            'risk_icon': icon,
            'risk_message': msg,
            'confidence': 0.7,  # 휴리스틱이므로 신뢰도 낮음
            'features_used': ['유동성위기지수', '지급불능위험지수', '재무조작위험지수', '조기경보신호수'],
            'model_info': {
                'model_type': 'Heuristic',
                'n_features': 4,
                'note': '학습된 모델이 없어 경험적 규칙 기반으로 예측했습니다.'
            }
        }

        logger.info(f"휴리스틱 예측 완료: 부도 확률 {bankruptcy_prob:.1%}")

        return result
