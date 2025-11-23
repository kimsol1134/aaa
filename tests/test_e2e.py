"""
End-to-End Tests

전체 플로우가 정상 동작하는지 테스트
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
from src.utils.helpers import get_risk_level, identify_critical_risks, identify_warnings
from src.visualization.charts import create_risk_gauge, create_shap_waterfall_real


class TestEndToEndFlow:
    """전체 플로우 E2E 테스트"""

    @pytest.fixture
    def normal_company_data(self):
        """정상 기업 데이터"""
        return {
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

    @pytest.fixture
    def risk_company_data(self):
        """위험 기업 데이터"""
        return {
            '자산총계': 1_000_000, '부채총계': 950_000, '자본총계': 50_000,
            '유동자산': 300_000, '비유동자산': 700_000,
            '유동부채': 500_000, '비유동부채': 450_000,
            '현금및현금성자산': 20_000, '단기금융상품': 5_000,
            '매출채권': 150_000, '재고자산': 80_000,
            '유형자산': 500_000, '무형자산': 100_000,
            '단기차입금': 250_000, '장기차입금': 400_000,
            '매출액': 1_000_000, '매출원가': 800_000, '매출총이익': 200_000,
            '판매비와관리비': 180_000, '영업이익': 20_000,
            '이자비용': 80_000, '당기순이익': -50_000,
            '영업활동현금흐름': 10_000, '매입채무': 150_000,
        }

    def test_complete_flow_normal_company(self, normal_company_data):
        """정상 기업 전체 플로우 테스트"""
        print("\n" + "=" * 60)
        print("정상 기업 E2E 테스트")
        print("=" * 60)

        # Step 1: 특성 생성
        print("\n[Step 1] 특성 생성 중...")
        generator = DomainFeatureGenerator()
        features_df = generator.generate_all_features(normal_company_data)
        assert features_df is not None
        assert len(features_df) == 1
        print(f"  ✓ {len(features_df.columns)}개 특성 생성 완료")

        # Step 2: 예측
        print("\n[Step 2] 부도 위험 예측 중...")
        predictor = BankruptcyPredictor()
        predictor.load_model()
        result = predictor.predict(features_df)
        assert result is not None
        print(f"  ✓ 부도 확률: {result['bankruptcy_probability']:.2%}")
        print(f"  ✓ 위험 등급: {result['risk_level']} {result['risk_icon']}")

        # Step 3: 위험 요인 분석
        print("\n[Step 3] 위험 요인 분석 중...")
        critical_risks = identify_critical_risks(features_df)
        warnings = identify_warnings(features_df)
        print(f"  ✓ Critical 리스크: {len(critical_risks)}개")
        print(f"  ✓ Warning: {len(warnings)}개")

        # Step 4: 비즈니스 가치 계산
        print("\n[Step 4] 비즈니스 가치 계산 중...")
        calc = BusinessValueCalculator()
        value = calc.calculate_single_company(result['bankruptcy_probability'])
        print(f"  ✓ 예상 손실: {value['expected_loss']:,.0f}원")
        print(f"  ✓ 예상 수익: {value['expected_profit']:,.0f}원")
        print(f"  ✓ 순 기대값: {value['net']:,.0f}원")

        # Step 5: SHAP 값 검증
        print("\n[Step 5] SHAP 값 검증 중...")
        if result.get('shap_values'):
            print(f"  ✓ SHAP 값 계산 완료 ({len(result['shap_values'])}개 특성)")

            # SHAP Waterfall 차트 생성
            fig = create_shap_waterfall_real(
                shap_values=np.array(result['shap_values']),
                feature_values=features_df.iloc[0],
                feature_names=result['feature_names'],
                base_value=result['shap_base_value']
            )
            assert fig is not None
            print("  ✓ SHAP Waterfall 차트 생성 완료")
        else:
            print("  ⚠ SHAP 값 없음 (모델 미로드)")

        # Step 6: 시각화
        print("\n[Step 6] 시각화 생성 중...")
        gauge_fig = create_risk_gauge(result['bankruptcy_probability'])
        assert gauge_fig is not None
        print("  ✓ Risk Gauge 차트 생성 완료")

        print("\n" + "=" * 60)
        print("정상 기업 E2E 테스트 완료")
        print("=" * 60)

    def test_complete_flow_risk_company(self, risk_company_data):
        """위험 기업 전체 플로우 테스트"""
        print("\n" + "=" * 60)
        print("위험 기업 E2E 테스트")
        print("=" * 60)

        # Step 1: 특성 생성
        print("\n[Step 1] 특성 생성 중...")
        generator = DomainFeatureGenerator()
        features_df = generator.generate_all_features(risk_company_data)
        assert features_df is not None
        print(f"  ✓ {len(features_df.columns)}개 특성 생성 완료")

        # Step 2: 예측
        print("\n[Step 2] 부도 위험 예측 중...")
        predictor = BankruptcyPredictor()
        predictor.load_model()
        result = predictor.predict(features_df)
        assert result is not None
        print(f"  ✓ 부도 확률: {result['bankruptcy_probability']:.2%}")
        print(f"  ✓ 위험 등급: {result['risk_level']} {result['risk_icon']}")

        # 위험 기업은 높은 부도 확률을 가져야 함
        # (모델 없을 때는 휴리스틱 사용)
        print(f"  → 위험 등급 확인: {result['risk_level']}")

        # Step 3: 위험 요인 분석
        print("\n[Step 3] 위험 요인 분석 중...")
        critical_risks = identify_critical_risks(features_df)
        warnings = identify_warnings(features_df)
        print(f"  ✓ Critical 리스크: {len(critical_risks)}개")
        for risk in critical_risks[:3]:  # 상위 3개만 출력
            print(f"    - {risk['name']}: {risk['value']:.2f}")
        print(f"  ✓ Warning: {len(warnings)}개")

        # Step 4: 비즈니스 가치 계산
        print("\n[Step 4] 비즈니스 가치 계산 중...")
        calc = BusinessValueCalculator()
        value = calc.calculate_single_company(result['bankruptcy_probability'])
        print(f"  ✓ 예상 손실: {value['expected_loss']:,.0f}원")
        print(f"  ✓ 예상 수익: {value['expected_profit']:,.0f}원")
        print(f"  ✓ 순 기대값: {value['net']:,.0f}원")

        # 위험 기업은 순 기대값이 낮아야 함
        if result['bankruptcy_probability'] > 0.5:
            assert value['net'] < 0, "고위험 기업의 순 기대값이 음수여야 함"

        print("\n" + "=" * 60)
        print("위험 기업 E2E 테스트 완료")
        print("=" * 60)

    def test_multiple_companies_batch(self):
        """여러 기업 배치 처리 테스트"""
        print("\n" + "=" * 60)
        print("배치 처리 E2E 테스트")
        print("=" * 60)

        companies = {
            '정상기업': {
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
            },
            '주의기업': {
                '자산총계': 1_000_000, '부채총계': 700_000, '자본총계': 300_000,
                '유동자산': 400_000, '비유동자산': 600_000,
                '유동부채': 400_000, '비유동부채': 300_000,
                '현금및현금성자산': 50_000, '단기금융상품': 20_000,
                '매출채권': 180_000, '재고자산': 100_000,
                '유형자산': 400_000, '무형자산': 100_000,
                '단기차입금': 150_000, '장기차입금': 250_000,
                '매출액': 1_500_000, '매출원가': 1_000_000, '매출총이익': 500_000,
                '판매비와관리비': 350_000, '영업이익': 150_000,
                '이자비용': 50_000, '당기순이익': 80_000,
                '영업활동현금흐름': 100_000, '매입채무': 120_000,
            }
        }

        generator = DomainFeatureGenerator()
        predictor = BankruptcyPredictor()
        predictor.load_model()
        calc = BusinessValueCalculator()

        results = []

        for company_name, financial_data in companies.items():
            print(f"\n[{company_name}] 분석 중...")

            # 특성 생성
            features_df = generator.generate_all_features(financial_data)

            # 예측
            result = predictor.predict(features_df)

            # 비즈니스 가치
            value = calc.calculate_single_company(result['bankruptcy_probability'])

            results.append({
                'company': company_name,
                'probability': result['bankruptcy_probability'],
                'risk_level': result['risk_level'],
                'net_value': value['net']
            })

            print(f"  ✓ 부도 확률: {result['bankruptcy_probability']:.2%}")
            print(f"  ✓ 위험 등급: {result['risk_level']}")
            print(f"  ✓ 순 기대값: {value['net']:,.0f}원")

        # 포트폴리오 분석
        print("\n[포트폴리오 분석]")
        probabilities = [r['probability'] for r in results]
        portfolio_value = calc.calculate_portfolio(probabilities)
        print(f"  ✓ 총 기업 수: {portfolio_value['num_companies']}개")
        print(f"  ✓ 총 순 기대값: {portfolio_value['total_net']:,.0f}원")
        print(f"  ✓ 기업당 평균: {portfolio_value['avg_profit_per_company'] - portfolio_value['avg_loss_per_company']:,.0f}원")

        print("\n" + "=" * 60)
        print("배치 처리 E2E 테스트 완료")
        print("=" * 60)


class TestDataValidation:
    """데이터 검증 테스트"""

    def test_edge_cases(self):
        """극단적인 케이스 테스트"""
        print("\n" + "=" * 60)
        print("Edge Case 테스트")
        print("=" * 60)

        generator = DomainFeatureGenerator()
        predictor = BankruptcyPredictor()
        predictor.load_model()

        # 케이스 1: 매우 작은 값
        print("\n[Case 1] 매우 작은 값")
        small_data = {
            '자산총계': 1, '부채총계': 0, '자본총계': 1,
            '유동자산': 1, '비유동자산': 0,
            '유동부채': 0, '비유동부채': 0,
            '현금및현금성자산': 1, '단기금융상품': 0,
            '매출채권': 0, '재고자산': 0,
            '유형자산': 0, '무형자산': 0,
            '단기차입금': 0, '장기차입금': 0,
            '매출액': 10, '매출원가': 5, '매출총이익': 5,
            '판매비와관리비': 2, '영업이익': 3,
            '이자비용': 0, '당기순이익': 3,
            '영업활동현금흐름': 3, '매입채무': 0,
        }

        try:
            features_df = generator.generate_all_features(small_data)
            result = predictor.predict(features_df)
            print(f"  ✓ 처리 성공: 부도 확률 {result['bankruptcy_probability']:.2%}")
        except Exception as e:
            print(f"  ✗ 에러 발생: {e}")
            raise

        # 케이스 2: 0으로 나누기 방지
        print("\n[Case 2] 제로 값 처리")
        zero_data = {
            '자산총계': 1_000_000, '부채총계': 0, '자본총계': 1_000_000,
            '유동자산': 1_000_000, '비유동자산': 0,
            '유동부채': 0, '비유동부채': 0,
            '현금및현금성자산': 1_000_000, '단기금융상품': 0,
            '매출채권': 0, '재고자산': 0,
            '유형자산': 0, '무형자산': 0,
            '단기차입금': 0, '장기차입금': 0,
            '매출액': 1_000_000, '매출원가': 0, '매출총이익': 1_000_000,
            '판매비와관리비': 0, '영업이익': 1_000_000,
            '이자비용': 0, '당기순이익': 1_000_000,
            '영업활동현금흐름': 1_000_000, '매입채무': 0,
        }

        try:
            features_df = generator.generate_all_features(zero_data)
            result = predictor.predict(features_df)
            print(f"  ✓ 처리 성공: 부도 확률 {result['bankruptcy_probability']:.2%}")
        except Exception as e:
            print(f"  ✗ 에러 발생: {e}")
            raise

        print("\n" + "=" * 60)
        print("Edge Case 테스트 완료")
        print("=" * 60)


def run_all_e2e_tests():
    """모든 E2E 테스트 실행"""
    print("\n" + "=" * 80)
    print(" " * 20 + "E2E Tests 실행 중...")
    print("=" * 80)

    # pytest 실행
    exit_code = pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-s',  # stdout 출력
        '-p', 'no:warnings'
    ])

    return exit_code


if __name__ == "__main__":
    exit_code = run_all_e2e_tests()
    sys.exit(exit_code)
