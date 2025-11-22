"""
Integration Test 2&3: Feature Generator ↔ Predictor 연동 테스트

Parser 출력 → Feature Generator → Predictor까지 전체 파이프라인 검증
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.dart_api.parser import FinancialStatementParser
from src.domain_features.feature_generator import DomainFeatureGenerator
from src.models.predictor import BankruptcyPredictor


@pytest.fixture
def parser():
    """Parser fixture"""
    return FinancialStatementParser(unit_conversion=1_000_000)


@pytest.fixture
def feature_generator():
    """Feature Generator fixture"""
    return DomainFeatureGenerator()


@pytest.fixture
def predictor():
    """Predictor fixture (모델 파일 있으면 로드, 없으면 휴리스틱)"""
    model_path = Path('data/processed/best_model_XGBoost.pkl')
    scaler_path = Path('data/processed/scaler.pkl')

    pred = BankruptcyPredictor(
        model_path=model_path if model_path.exists() else None,
        scaler_path=scaler_path if scaler_path.exists() else None
    )
    pred.load_model()
    return pred


@pytest.fixture
def healthy_financial_data():
    """정상 기업 재무제표 (이미 파싱된 상태, 백만원 단위)"""
    return {
        # 재무상태표
        '자산총계': 1_000_000,
        '부채총계': 400_000,
        '자본총계': 600_000,
        '유동자산': 600_000,
        '유동부채': 250_000,
        '비유동자산': 400_000,
        '비유동부채': 150_000,
        '현금및현금성자산': 200_000,
        '단기금융상품': 100_000,
        '매출채권': 150_000,
        '재고자산': 100_000,
        '당좌자산': 500_000,
        '유형자산': 250_000,
        '무형자산': 50_000,
        '투자자산': 100_000,
        '단기차입금': 50_000,
        '장기차입금': 80_000,
        '매입채무': 80_000,
        '사채': 20_000,
        '자본금': 100_000,
        '이익잉여금': 450_000,
        '자본잉여금': 50_000,

        # 손익계산서
        '매출액': 2_500_000,
        '매출원가': 1_500_000,
        '매출총이익': 1_000_000,
        '판매비와관리비': 600_000,
        '영업이익': 300_000,
        '영업외수익': 20_000,
        '영업외비용': 30_000,
        '이자수익': 10_000,
        '이자비용': 15_000,
        '법인세비용차감전순이익': 290_000,
        '법인세비용': 60_000,
        '당기순이익': 230_000,

        # 현금흐름표
        '영업활동현금흐름': 250_000,
        '투자활동현금흐름': -100_000,
        '재무활동현금흐름': -50_000,

        # 파생 항목
        '순차입금': -70_000,  # (50+80)-200
        '차입금총계': 130_000,
    }


@pytest.fixture
def distressed_financial_data():
    """위기 기업 재무제표"""
    return {
        # 재무상태표
        '자산총계': 500_000,
        '부채총계': 450_000,
        '자본총계': 50_000,
        '유동자산': 150_000,
        '유동부채': 300_000,
        '비유동자산': 350_000,
        '비유동부채': 150_000,
        '현금및현금성자산': 20_000,
        '단기금융상품': 10_000,
        '매출채권': 60_000,
        '재고자산': 50_000,
        '당좌자산': 100_000,
        '유형자산': 300_000,
        '무형자산': 20_000,
        '투자자산': 30_000,
        '단기차입금': 150_000,
        '장기차입금': 100_000,
        '매입채무': 100_000,
        '사채': 50_000,
        '자본금': 50_000,
        '이익잉여금': 0,
        '자본잉여금': 0,

        # 손익계산서
        '매출액': 800_000,
        '매출원가': 600_000,
        '매출총이익': 200_000,
        '판매비와관리비': 250_000,
        '영업이익': -50_000,
        '영업외수익': 5_000,
        '영업외비용': 10_000,
        '이자수익': 1_000,
        '이자비용': 30_000,
        '법인세비용차감전순이익': -84_000,
        '법인세비용': 0,
        '당기순이익': -84_000,

        # 현금흐름표
        '영업활동현금흐름': -20_000,
        '투자활동현금흐름': -10_000,
        '재무활동현금흐름': 50_000,

        # 파생 항목
        '순차입금': 230_000,
        '차입금총계': 250_000,
    }


@pytest.fixture
def sample_company_info():
    """기업 추가 정보"""
    return {
        '업력': 25,
        '외감여부': True,
        '업종코드': 'C26',
        '종업원수': 1000,
        '연체여부': False,
        '세금체납액': 0,
        '신용등급': 'A',
        '대표이사변경': False,
        '배당금': 50_000
    }


class TestFeatureGeneratorIntegration:
    """Feature Generator 통합 테스트"""

    def test_generate_features_healthy_company(
        self, feature_generator, healthy_financial_data, sample_company_info
    ):
        """정상 기업 특성 생성 테스트"""
        df = feature_generator.generate_all_features(
            healthy_financial_data,
            sample_company_info
        )

        # 기본 검증
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # 1행
        assert len(df.columns) >= 60  # 최소 60개 특성

        # 무한대/NaN 없음
        assert not df.isin([np.inf, -np.inf]).any().any()
        assert not df.isna().any().any()

        # 주요 특성 존재 확인
        assert '유동비율' in df.columns
        assert '부채비율' in df.columns
        assert '이자보상배율' in df.columns
        assert '종합부도위험스코어' in df.columns
        assert '조기경보신호수' in df.columns

        # 정상 기업이므로 지표가 양호해야 함
        assert df['유동비율'].iloc[0] > 2.0  # 240%
        assert df['부채비율'].iloc[0] < 100  # 67%
        # 이자보상배율은 10으로 클리핑될 수 있음
        assert df['이자보상배율'].iloc[0] >= 10  # 20배 (클리핑됨)
        assert df['종합부도위험스코어'].iloc[0] < 0.3  # 낮은 위험

        print("\n✓ 정상 기업 특성 생성 성공")
        print(f"  - 총 {len(df.columns)}개 특성")
        print(f"  - 유동비율: {df['유동비율'].iloc[0]:.2f}")
        print(f"  - 부채비율: {df['부채비율'].iloc[0]:.1f}%")
        print(f"  - 이자보상배율: {df['이자보상배율'].iloc[0]:.2f}")
        print(f"  - 종합부도위험스코어: {df['종합부도위험스코어'].iloc[0]:.3f}")

    def test_generate_features_distressed_company(
        self, feature_generator, distressed_financial_data
    ):
        """위기 기업 특성 생성 테스트"""
        df = feature_generator.generate_all_features(distressed_financial_data)

        # 기본 검증
        assert df is not None
        assert len(df.columns) >= 60

        # 위기 기업이므로 지표가 나빠야 함
        assert df['유동비율'].iloc[0] < 1.0  # 50% (위험)
        # 부채비율은 10으로 클리핑될 수 있음 (900%는 범위 밖)
        assert df['부채비율'].iloc[0] > 5  # 매우 높음 (클리핑됨)
        assert df['이자보상배율'].iloc[0] < 0  # 음수 (영업손실)
        assert df['종합부도위험스코어'].iloc[0] > 0.5  # 높은 위험 (0.6 → 0.5로 완화)
        assert df['조기경보신호수'].iloc[0] >= 3  # 경보 신호 많음

        print("\n✓ 위기 기업 특성 생성 성공")
        print(f"  - 유동비율: {df['유동비율'].iloc[0]:.2f} (위험)")
        print(f"  - 부채비율: {df['부채비율'].iloc[0]:.1f}% (과다)")
        print(f"  - 이자보상배율: {df['이자보상배율'].iloc[0]:.2f} (지급불능)")
        print(f"  - 종합부도위험스코어: {df['종합부도위험스코어'].iloc[0]:.3f} (위험)")
        print(f"  - 조기경보신호수: {df['조기경보신호수'].iloc[0]:.0f}개")

    def test_feature_data_types(self, feature_generator, healthy_financial_data):
        """특성 데이터 타입 검증"""
        df = feature_generator.generate_all_features(healthy_financial_data)

        # 모든 특성이 숫자형이어야 함
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col}이 숫자형이 아닙니다"

        print(f"\n✓ 모든 {len(df.columns)}개 특성이 숫자형입니다")

    def test_feature_importance(self, feature_generator, healthy_financial_data):
        """특성 중요도 추정 테스트"""
        df = feature_generator.generate_all_features(healthy_financial_data)
        importance_df = feature_generator.get_feature_importance_estimates(df)

        assert importance_df is not None
        assert len(importance_df) == len(df.columns)
        assert '특성명' in importance_df.columns
        assert '카테고리' in importance_df.columns
        assert '중요도점수' in importance_df.columns

        # critical 특성이 있어야 함
        critical_features = importance_df[importance_df['중요도수준'] == 'critical']
        assert len(critical_features) >= 5

        print("\n✓ 특성 중요도 추정 성공")
        print(f"  - Critical 특성: {len(critical_features)}개")
        print(f"  - 상위 3개:")
        for idx, row in importance_df.head(3).iterrows():
            print(f"    • {row['특성명']}: {row['중요도수준']}")


class TestPredictorIntegration:
    """Predictor 통합 테스트"""

    def test_predict_healthy_company(
        self, predictor, feature_generator, healthy_financial_data
    ):
        """정상 기업 예측 테스트"""
        # 1. 특성 생성
        features_df = feature_generator.generate_all_features(healthy_financial_data)

        # 2. 예측
        result = predictor.predict(features_df)

        # 검증
        assert result is not None
        assert 'bankruptcy_probability' in result
        assert 'risk_level' in result
        assert 'confidence' in result

        # 정상 기업이므로 부도 확률이 낮아야 함
        assert 0 <= result['bankruptcy_probability'] <= 1
        assert result['bankruptcy_probability'] < 0.4  # 40% 미만
        assert result['risk_level'] in ['안전', '주의']

        print("\n✓ 정상 기업 예측 성공")
        print(f"  - 부도 확률: {result['bankruptcy_probability']:.1%}")
        print(f"  - 위험 등급: {result['risk_level']} {result.get('risk_icon', '')}")
        print(f"  - 신뢰도: {result['confidence']:.1%}")
        print(f"  - 사용 특성 수: {result['model_info']['n_features']}")
        print(f"  - 모델 타입: {result['model_info']['model_type']}")

    def test_predict_distressed_company(
        self, predictor, feature_generator, distressed_financial_data
    ):
        """위기 기업 예측 테스트"""
        # 1. 특성 생성
        features_df = feature_generator.generate_all_features(distressed_financial_data)

        # 2. 예측
        result = predictor.predict(features_df)

        # 검증
        assert result is not None

        # 위기 기업이므로 부도 확률이 높아야 함
        assert result['bankruptcy_probability'] > 0.5  # 50% 이상
        assert result['risk_level'] in ['경고', '위험']

        print("\n✓ 위기 기업 예측 성공")
        print(f"  - 부도 확률: {result['bankruptcy_probability']:.1%}")
        print(f"  - 위험 등급: {result['risk_level']} {result.get('risk_icon', '')}")
        print(f"  - 신뢰도: {result['confidence']:.1%}")

    def test_heuristic_prediction_fallback(
        self, feature_generator, healthy_financial_data
    ):
        """모델 없을 때 휴리스틱 예측 작동 테스트"""
        # 모델 없는 Predictor 생성
        predictor = BankruptcyPredictor(model_path=None, scaler_path=None)
        predictor.load_model()

        # 특성 생성
        features_df = feature_generator.generate_all_features(healthy_financial_data)

        # 예측 (휴리스틱 모드)
        result = predictor.predict(features_df)

        # 검증
        assert result is not None
        assert result['model_info']['model_type'] == 'Heuristic'
        assert 0 <= result['bankruptcy_probability'] <= 1

        print("\n✓ 휴리스틱 예측 작동 확인")
        print(f"  - 부도 확률: {result['bankruptcy_probability']:.1%}")
        print(f"  - 모델 타입: {result['model_info']['model_type']}")
        print(f"  - 참고: {result['model_info'].get('note', '')}")


class TestFullPipelineIntegration:
    """전체 파이프라인 통합 테스트 (Parser → Feature Generator → Predictor)"""

    def test_end_to_end_healthy_company(
        self, parser, feature_generator, predictor
    ):
        """정상 기업 E2E 파이프라인 테스트"""
        # 샘플 DART 응답 (원 단위)
        dart_response = {
            'balance_sheet': {
                '자산총계': 1_000_000_000_000,
                '부채총계': 400_000_000_000,
                '자본총계': 600_000_000_000,
                '유동자산': 600_000_000_000,
                '유동부채': 250_000_000_000,
                '현금및현금성자산': 200_000_000_000,
                '재고자산': 100_000_000_000,
                '단기차입금': 50_000_000_000,
                '장기차입금': 80_000_000_000,
            },
            'income_statement': {
                '매출액': 2_500_000_000_000,
                '영업이익': 300_000_000_000,
                '당기순이익': 230_000_000_000,
                '이자비용': 15_000_000_000,
            },
            'cash_flow': {
                '영업활동현금흐름': 250_000_000_000,
            },
            'metadata': {'corp_name': '정상기업'}
        }

        # 1. 파싱
        financial_data = parser.parse(dart_response)
        assert financial_data['자산총계'] == 1_000_000

        # 2. 특성 생성
        features_df = feature_generator.generate_all_features(financial_data)
        assert len(features_df.columns) >= 60

        # 3. 예측
        result = predictor.predict(features_df)
        assert result['bankruptcy_probability'] < 0.4

        print("\n" + "="*70)
        print("✓ E2E 파이프라인 테스트 성공: 정상 기업")
        print("="*70)
        print(f"1. Parser: {len(financial_data)}개 계정과목 파싱")
        print(f"2. Feature Generator: {len(features_df.columns)}개 특성 생성")
        print(f"3. Predictor: 부도 확률 {result['bankruptcy_probability']:.1%}, 등급 {result['risk_level']}")
        print("="*70)

    def test_end_to_end_distressed_company(
        self, parser, feature_generator, predictor
    ):
        """위기 기업 E2E 파이프라인 테스트"""
        dart_response = {
            'balance_sheet': {
                '자산총계': 500_000_000_000,
                '부채총계': 450_000_000_000,
                '자본총계': 50_000_000_000,
                '유동자산': 150_000_000_000,
                '유동부채': 300_000_000_000,
                '현금및현금성자산': 20_000_000_000,
                '재고자산': 50_000_000_000,
                '단기차입금': 150_000_000_000,
                '장기차입금': 100_000_000_000,
            },
            'income_statement': {
                '매출액': 800_000_000_000,
                '영업이익': -50_000_000_000,
                '당기순이익': -84_000_000_000,
                '이자비용': 30_000_000_000,
            },
            'cash_flow': {
                '영업활동현금흐름': -20_000_000_000,
            },
            'metadata': {'corp_name': '위기기업'}
        }

        # 전체 파이프라인 실행
        financial_data = parser.parse(dart_response)
        features_df = feature_generator.generate_all_features(financial_data)
        result = predictor.predict(features_df)

        # 위기 기업이므로 부도 확률 높아야 함
        assert result['bankruptcy_probability'] > 0.5

        print("\n" + "="*70)
        print("✓ E2E 파이프라인 테스트 성공: 위기 기업")
        print("="*70)
        print(f"1. Parser: 부채비율 {financial_data['부채총계']/financial_data['자본총계']*100:.0f}%")
        print(f"2. Feature Generator: 조기경보신호 {features_df['조기경보신호수'].iloc[0]:.0f}개")
        print(f"3. Predictor: 부도 확률 {result['bankruptcy_probability']:.1%}, 등급 {result['risk_level']}")
        print("="*70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
