"""
Integration Tests

각 모듈이 올바르게 통합되는지 테스트
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# 프로젝트 루트 추가
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.domain_features import DomainFeatureGenerator
from src.models import BankruptcyPredictor
from src.utils.business_value import BusinessValueCalculator
from src.utils.helpers import get_risk_level
from src.visualization.charts import create_shap_waterfall_real


class TestDomainFeatureGeneration:
    """도메인 특성 생성 테스트"""

    @pytest.fixture
    def sample_financial_data(self):
        """샘플 재무 데이터"""
        return {
            '자산총계': 1_000_000,
            '부채총계': 400_000,
            '자본총계': 600_000,
            '유동자산': 600_000,
            '비유동자산': 400_000,
            '유동부채': 200_000,
            '비유동부채': 200_000,
            '현금및현금성자산': 200_000,
            '단기금융상품': 100_000,
            '매출채권': 150_000,
            '재고자산': 80_000,
            '유형자산': 250_000,
            '무형자산': 50_000,
            '단기차입금': 50_000,
            '장기차입금': 100_000,
            '매출액': 2_000_000,
            '매출원가': 1_200_000,
            '매출총이익': 800_000,
            '판매비와관리비': 400_000,
            '영업이익': 400_000,
            '이자비용': 10_000,
            '당기순이익': 300_000,
            '영업활동현금흐름': 350_000,
            '매입채무': 100_000,
        }

    def test_feature_generation_success(self, sample_financial_data):
        """특성 생성 성공 테스트"""
        generator = DomainFeatureGenerator()
        features_df = generator.generate_all_features(sample_financial_data)

        # 기본 검증
        assert features_df is not None
        assert len(features_df) == 1  # 1개 행
        assert len(features_df.columns) > 0  # 특성 존재

        print(f"✓ 생성된 특성 개수: {len(features_df.columns)}개")

    def test_feature_categories(self, sample_financial_data):
        """특성 카테고리별 검증"""
        generator = DomainFeatureGenerator()
        features_df = generator.generate_all_features(sample_financial_data)

        # 주요 특성 존재 확인
        expected_features = [
            '유동비율',
            '부채비율',
            '이자보상배율',
            '영업이익률'
        ]

        for feature in expected_features:
            assert feature in features_df.columns, f"{feature} 특성이 없습니다"

        print("✓ 주요 특성 존재 확인 완료")

    def test_feature_values_valid(self, sample_financial_data):
        """특성 값 유효성 검증"""
        generator = DomainFeatureGenerator()
        features_df = generator.generate_all_features(sample_financial_data)

        # NaN이나 Inf 값이 없어야 함
        assert not features_df.isnull().any().any(), "NaN 값이 존재합니다"
        assert not np.isinf(features_df.values).any(), "Inf 값이 존재합니다"

        print("✓ 특성 값 유효성 검증 완료 (NaN/Inf 없음)")


class TestPrediction:
    """예측 모델 테스트"""

    @pytest.fixture
    def sample_features(self):
        """샘플 특성 데이터"""
        generator = DomainFeatureGenerator()
        financial_data = {
            '자산총계': 1_000_000, '부채총계': 400_000, '자본총계': 600_000,
            '유동자산': 600_000, '비유동자산': 400_000,
            '유동부채': 200_000, '비유동부채': 200_000,
            '현금및현금성자산': 200_000, '단기금융상품': 100_000,
            '매출채권': 150_000, '재고자산': 80_000,
            '유형자산': 250_000, '무형자산': 50_000,
            '단기차입금': 50_000, '장기차입금': 100_000,
            '매출액': 2_000_000, '매출원가': 1_200_000, '매출총이익': 800_000,
            '판매비와관리비': 400_000, '영업이익': 400_000,
            '이자비용': 10_000, '당기순이익': 300_000,
            '영업활동현금흐름': 350_000, '매입채무': 100_000,
        }
        return generator.generate_all_features(financial_data)

    def test_prediction_output_structure(self, sample_features):
        """예측 결과 구조 검증"""
        predictor = BankruptcyPredictor()
        predictor.load_model()

        result = predictor.predict(sample_features)

        # 필수 키 존재 확인
        required_keys = [
            'bankruptcy_probability',
            'risk_level',
            'risk_icon',
            'risk_message',
            'confidence',
            'features_used',
            'model_info'
        ]

        for key in required_keys:
            assert key in result, f"{key} 키가 결과에 없습니다"

        print("✓ 예측 결과 구조 검증 완료")

    def test_prediction_probability_range(self, sample_features):
        """예측 확률 범위 검증"""
        predictor = BankruptcyPredictor()
        predictor.load_model()

        result = predictor.predict(sample_features)

        prob = result['bankruptcy_probability']
        assert 0 <= prob <= 1, f"부도 확률이 범위를 벗어났습니다: {prob}"

        print(f"✓ 예측 확률 범위 검증 완료: {prob:.2%}")

    def test_risk_level_consistency(self, sample_features):
        """위험 등급 일관성 검증"""
        predictor = BankruptcyPredictor()
        predictor.load_model()

        result = predictor.predict(sample_features)

        prob = result['bankruptcy_probability']
        level, icon, msg = get_risk_level(prob)

        # 예측 결과와 helpers 함수 결과가 일치해야 함
        assert result['risk_level'] == level
        assert result['risk_icon'] == icon

        print(f"✓ 위험 등급 일관성 검증 완료: {level} {icon}")


class TestBusinessValue:
    """비즈니스 가치 계산 테스트"""

    def test_single_company_calculation(self):
        """단일 기업 비즈니스 가치 계산"""
        calc = BusinessValueCalculator()

        # 낮은 부도 확률
        result_low = calc.calculate_single_company(0.01)
        assert result_low['net'] > 0, "낮은 부도 확률에서 순 기대값이 양수여야 함"

        # 높은 부도 확률
        result_high = calc.calculate_single_company(0.8)
        assert result_high['net'] < 0, "높은 부도 확률에서 순 기대값이 음수여야 함"

        print("✓ 비즈니스 가치 계산 검증 완료")

    def test_portfolio_calculation(self):
        """포트폴리오 계산 테스트"""
        calc = BusinessValueCalculator()

        predictions = [0.01, 0.02, 0.05, 0.10]
        result = calc.calculate_portfolio(predictions)

        assert result['num_companies'] == 4
        assert 'total_net' in result
        assert 'avg_loss_per_company' in result

        print("✓ 포트폴리오 계산 검증 완료")


class TestSHAPIntegration:
    """SHAP 통합 테스트"""

    @pytest.fixture
    def sample_shap_data(self):
        """샘플 SHAP 데이터"""
        np.random.seed(42)
        n_features = 10
        return {
            'shap_values': np.random.randn(n_features) * 0.1,
            'feature_names': [f'특성_{i}' for i in range(n_features)],
            'feature_values': pd.Series({f'특성_{i}': np.random.rand() for i in range(n_features)}),
            'base_value': 0.015
        }

    def test_shap_waterfall_creation(self, sample_shap_data):
        """SHAP Waterfall 차트 생성 테스트"""
        fig = create_shap_waterfall_real(
            shap_values=sample_shap_data['shap_values'],
            feature_values=sample_shap_data['feature_values'],
            feature_names=sample_shap_data['feature_names'],
            base_value=sample_shap_data['base_value']
        )

        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0

        print("✓ SHAP Waterfall 차트 생성 검증 완료")


def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 60)
    print("Integration Tests 실행 중...")
    print("=" * 60)

    # pytest 실행
    exit_code = pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-p', 'no:warnings'
    ])

    return exit_code


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
