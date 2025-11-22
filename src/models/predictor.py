"""
부도 예측 모델 로딩 및 예측

학습된 모델을 로드하고 새로운 데이터에 대해 예측 수행
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BankruptcyPredictor:
    """부도 예측 모델"""

    def __init__(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        """
        Args:
            model_path: 모델 파일 경로
            scaler_path: 스케일러 파일 경로
        """
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.expected_features = None

    def load_model(self):
        """모델 및 스케일러 로드"""
        try:
            if self.model_path and self.model_path.exists():
                logger.info(f"모델 로딩 중: {self.model_path}")
                self.model = joblib.load(self.model_path)
                logger.info("✓ 모델 로드 성공")
            else:
                logger.warning("모델 파일을 찾을 수 없습니다. 더미 모델을 사용합니다.")
                self.model = None

            if self.scaler_path and self.scaler_path.exists():
                logger.info(f"스케일러 로딩 중: {self.scaler_path}")
                self.scaler = joblib.load(self.scaler_path)
                logger.info("✓ 스케일러 로드 성공")
            else:
                logger.warning("스케일러 파일을 찾을 수 없습니다.")
                self.scaler = None

        except Exception as e:
            logger.error(f"모델 로딩 실패: {str(e)}")
            self.model = None
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
            # 모델이 없으면 휴리스틱 기반 예측
            if self.model is None:
                logger.warning("모델 없음. 휴리스틱 기반 예측 사용")
                return self._heuristic_prediction(features_df)

            # 특성 준비
            X = self._prepare_features(features_df)

            # 스케일링
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            # 예측
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_scaled)[0]
                bankruptcy_prob = proba[1]  # 부도(1) 클래스의 확률
                confidence = max(proba)
            else:
                # predict만 있는 경우
                prediction = self.model.predict(X_scaled)[0]
                bankruptcy_prob = 0.8 if prediction == 1 else 0.2
                confidence = 0.7

            # 결과 생성
            from src.utils.helpers import get_risk_level
            risk_level, icon, msg = get_risk_level(bankruptcy_prob)

            result = {
                'bankruptcy_probability': float(bankruptcy_prob),
                'risk_level': risk_level,
                'risk_icon': icon,
                'risk_message': msg,
                'confidence': float(confidence),
                'features_used': list(X.columns),
                'model_info': {
                    'model_type': type(self.model).__name__,
                    'n_features': X.shape[1]
                }
            }

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
