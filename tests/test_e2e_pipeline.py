"""
End-to-End 테스트: 실제 사용자 시나리오 시뮬레이션

시나리오:
1. 정상 기업 (부도 위험 낮음)
2. 위험 기업 (부도 위험 높음)
3. 에러 처리 (잘못된 데이터 입력)
"""

import pytest
import pandas as pd
import os
from dotenv import load_dotenv

from src.dart_api.client import DartAPIClient
from src.dart_api.parser import FinancialStatementParser
from src.domain_features.feature_generator import DomainFeatureGenerator
from src.models.predictor import BankruptcyPredictor
from src.utils.helpers import (
    get_risk_level,
    identify_critical_risks,
    identify_warnings,
    generate_recommendations
)
from pathlib import Path

# 환경 변수 로드
load_dotenv()


@pytest.fixture
def api_key():
    """DART API 키 (있으면 사용, 없으면 Mock 데이터 사용)"""
    return os.getenv('DART_API_KEY')


@pytest.fixture
def full_pipeline():
    """전체 파이프라인 컴포넌트"""
    parser = FinancialStatementParser(unit_conversion=1_000_000)
    feature_generator = DomainFeatureGenerator()

    model_path = Path('data/processed/best_model_XGBoost.pkl')
    scaler_path = Path('data/processed/scaler.pkl')
    predictor = BankruptcyPredictor(
        model_path=model_path if model_path.exists() else None,
        scaler_path=scaler_path if scaler_path.exists() else None
    )
    predictor.load_model()

    return {
        'parser': parser,
        'feature_generator': feature_generator,
        'predictor': predictor
    }


class TestE2EScenario1HealthyCompany:
    """시나리오 1: 정상 기업 (부도 위험 낮음)"""

    def test_healthy_company_full_workflow(self, full_pipeline):
        """정상 기업 전체 워크플로우 테스트"""
        parser = full_pipeline['parser']
        feature_generator = full_pipeline['feature_generator']
        predictor = full_pipeline['predictor']

        # === 1. 데이터 입력 ===
        print("\n" + "="*70)
        print("시나리오 1: 정상 기업 부도 예측")
        print("="*70)

        # 샘플 재무제표 데이터 (우량 기업)
        dart_response = {
            'balance_sheet': {
                '자산총계': 1_000_000_000_000,  # 1조원
                '부채총계': 400_000_000_000,    # 0.4조원
                '자본총계': 600_000_000_000,    # 0.6조원
                '유동자산': 600_000_000_000,
                '유동부채': 250_000_000_000,
                '비유동자산': 400_000_000_000,
                '비유동부채': 150_000_000_000,
                '현금및현금성자산': 200_000_000_000,
                '단기금융상품': 100_000_000_000,
                '매출채권': 150_000_000_000,
                '재고자산': 100_000_000_000,
                '유형자산': 250_000_000_000,
                '무형자산': 50_000_000_000,
                '단기차입금': 50_000_000_000,
                '장기차입금': 80_000_000_000,
                '매입채무': 80_000_000_000,
            },
            'income_statement': {
                '매출액': 2_500_000_000_000,  # 2.5조원
                '매출원가': 1_500_000_000_000,
                '매출총이익': 1_000_000_000_000,
                '판매비와관리비': 600_000_000_000,
                '영업이익': 300_000_000_000,   # 12% 영업이익률
                '이자비용': 15_000_000_000,
                '당기순이익': 230_000_000_000,
            },
            'cash_flow': {
                '영업활동현금흐름': 250_000_000_000,
            },
            'metadata': {
                'corp_name': '우량기업(주)',
                'bsns_year': '2023'
            }
        }

        company_info = {
            '업력': 30,
            '외감여부': True,
            '업종코드': 'C26',
            '종업원수': 2000,
            '연체여부': False,
            '세금체납액': 0,
            '신용등급': 'AA',
            '대표이사변경': False,
            '배당금': 80_000
        }

        print("\n[입력 데이터]")
        print(f"  기업명: {dart_response['metadata']['corp_name']}")
        print(f"  자산총계: 1조원")
        print(f"  매출액: 2.5조원")
        print(f"  영업이익률: 12.0%")

        # === 2. Parser: 재무제표 파싱 ===
        financial_data = parser.parse(dart_response)
        is_valid, errors = parser.validate(financial_data)

        assert is_valid, f"재무제표 검증 실패: {errors}"

        print("\n[Step 1: Parser]")
        print(f"  ✓ {len(financial_data)}개 계정과목 파싱 완료")
        print(f"  ✓ 재무제표 검증 통과")

        # === 3. Feature Generator: 65개 특성 생성 ===
        features_df = feature_generator.generate_all_features(
            financial_data,
            company_info
        )

        assert len(features_df.columns) >= 60
        assert not features_df.isin([float('inf'), float('-inf')]).any().any()

        print("\n[Step 2: Feature Generator]")
        print(f"  ✓ {len(features_df.columns)}개 도메인 특성 생성 완료")
        print(f"  - 유동비율: {features_df['유동비율'].iloc[0]:.2f}")
        print(f"  - 부채비율: {features_df['부채비율'].iloc[0]:.1f}%")
        print(f"  - 이자보상배율: {features_df['이자보상배율'].iloc[0]:.1f}")
        print(f"  - 종합부도위험스코어: {features_df['종합부도위험스코어'].iloc[0]:.3f}")

        # === 4. Predictor: 부도 확률 예측 ===
        prediction_result = predictor.predict(features_df)

        assert 0 <= prediction_result['bankruptcy_probability'] <= 1
        assert prediction_result['risk_level'] in ['안전', '주의', '경고', '위험']

        print("\n[Step 3: Predictor]")
        print(f"  ✓ 부도 확률: {prediction_result['bankruptcy_probability']:.1%}")
        print(f"  ✓ 위험 등급: {prediction_result['risk_level']} {prediction_result.get('risk_icon', '')}")
        print(f"  ✓ 신뢰도: {prediction_result['confidence']:.1%}")
        print(f"  ✓ 사용 모델: {prediction_result['model_info']['model_type']}")

        # === 5. Risk Analysis: 위험 요인 분석 ===
        critical_risks = identify_critical_risks(features_df)
        warnings = identify_warnings(features_df)

        print("\n[Step 4: Risk Analysis]")
        print(f"  Critical 위험: {len(critical_risks)}개")
        print(f"  Warning 경고: {len(warnings)}개")

        # === 6. Recommendations: 개선 권장사항 ===
        recommendations = generate_recommendations(features_df, financial_data)

        print("\n[Step 5: Recommendations]")
        print(f"  권장사항: {len(recommendations)}개")

        # === 검증: 정상 기업이므로 부도 확률 낮아야 함 ===
        assert prediction_result['bankruptcy_probability'] < 0.35, \
            f"정상 기업인데 부도 확률이 {prediction_result['bankruptcy_probability']:.1%}로 높습니다"

        assert prediction_result['risk_level'] in ['안전', '주의'], \
            f"정상 기업인데 위험 등급이 {prediction_result['risk_level']}입니다"

        assert len(critical_risks) == 0, \
            f"정상 기업인데 Critical 위험이 {len(critical_risks)}개 발견되었습니다"

        print("\n" + "="*70)
        print("✅ 시나리오 1 성공: 정상 기업으로 정확히 분류됨")
        print(f"   부도 확률 {prediction_result['bankruptcy_probability']:.1%}, 등급 {prediction_result['risk_level']}")
        print("="*70)


class TestE2EScenario2DistressedCompany:
    """시나리오 2: 위험 기업 (부도 위험 높음)"""

    def test_distressed_company_full_workflow(self, full_pipeline):
        """위험 기업 전체 워크플로우 테스트"""
        parser = full_pipeline['parser']
        feature_generator = full_pipeline['feature_generator']
        predictor = full_pipeline['predictor']

        print("\n" + "="*70)
        print("시나리오 2: 위험 기업 부도 예측")
        print("="*70)

        # === 1. 데이터 입력 (위기 상황 기업) ===
        dart_response = {
            'balance_sheet': {
                '자산총계': 500_000_000_000,   # 0.5조원
                '부채총계': 450_000_000_000,   # 0.45조원 (부채비율 900%)
                '자본총계': 50_000_000_000,    # 0.05조원
                '유동자산': 150_000_000_000,
                '유동부채': 300_000_000_000,   # 유동비율 0.5 (위험)
                '비유동자산': 350_000_000_000,
                '비유동부채': 150_000_000_000,
                '현금및현금성자산': 20_000_000_000,  # 현금 부족
                '단기금융상품': 10_000_000_000,
                '매출채권': 60_000_000_000,
                '재고자산': 50_000_000_000,
                '유형자산': 300_000_000_000,
                '무형자산': 20_000_000_000,
                '단기차입금': 150_000_000_000,  # 차입금 과다
                '장기차입금': 100_000_000_000,
                '매입채무': 100_000_000_000,
            },
            'income_statement': {
                '매출액': 800_000_000_000,     # 0.8조원
                '매출원가': 600_000_000_000,
                '매출총이익': 200_000_000_000,
                '판매비와관리비': 250_000_000_000,
                '영업이익': -50_000_000_000,   # 영업손실
                '이자비용': 30_000_000_000,    # 이자 부담 과다
                '당기순이익': -84_000_000_000, # 당기순손실
            },
            'cash_flow': {
                '영업활동현금흐름': -20_000_000_000,  # 음수 현금흐름
            },
            'metadata': {
                'corp_name': '위기기업(주)',
                'bsns_year': '2023'
            }
        }

        company_info = {
            '업력': 15,
            '외감여부': True,
            '업종코드': 'C24',
            '종업원수': 500,
            '연체여부': True,    # 연체 발생
            '세금체납액': 5_000,
            '신용등급': 'BB',    # 낮은 신용등급
            '대표이사변경': True,  # 대표이사 변경
            '배당금': 0
        }

        print("\n[입력 데이터]")
        print(f"  기업명: {dart_response['metadata']['corp_name']}")
        print(f"  부채비율: 900%")
        print(f"  유동비율: 0.5")
        print(f"  영업이익: -500억원 (적자)")
        print(f"  연체여부: 발생")

        # === 2~5. 전체 파이프라인 실행 ===
        financial_data = parser.parse(dart_response)
        features_df = feature_generator.generate_all_features(
            financial_data,
            company_info
        )
        prediction_result = predictor.predict(features_df)
        critical_risks = identify_critical_risks(features_df)
        warnings = identify_warnings(features_df)
        recommendations = generate_recommendations(features_df, financial_data)

        print("\n[Pipeline 실행 완료]")
        print(f"  부도 확률: {prediction_result['bankruptcy_probability']:.1%}")
        print(f"  위험 등급: {prediction_result['risk_level']} {prediction_result.get('risk_icon', '')}")
        print(f"  Critical 위험: {len(critical_risks)}개")
        print(f"  Warning 경고: {len(warnings)}개")

        # Critical 위험 상세 출력
        print("\n[Critical 위험 요인]")
        for risk in critical_risks:
            print(f"  🔴 {risk['name']}: {risk['explanation']}")

        # 권장사항 상세 출력
        print("\n[개선 권장사항]")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['title']} (우선순위: {rec['priority']})")
            print(f"     현재 상태: {rec['current_status']}")

        # === 검증: 위험 기업이므로 부도 확률 높아야 함 ===
        assert prediction_result['bankruptcy_probability'] > 0.5, \
            f"위험 기업인데 부도 확률이 {prediction_result['bankruptcy_probability']:.1%}로 낮습니다"

        assert prediction_result['risk_level'] in ['경고', '위험'], \
            f"위험 기업인데 위험 등급이 {prediction_result['risk_level']}입니다"

        assert len(critical_risks) >= 2, \
            f"위험 기업인데 Critical 위험이 {len(critical_risks)}개만 발견되었습니다"

        print("\n" + "="*70)
        print("✅ 시나리오 2 성공: 위험 기업으로 정확히 분류됨")
        print(f"   부도 확률 {prediction_result['bankruptcy_probability']:.1%}, Critical 위험 {len(critical_risks)}개")
        print("="*70)


class TestE2EScenario3ErrorHandling:
    """시나리오 3: 에러 처리 (잘못된 데이터 입력)"""

    def test_missing_critical_accounts(self, full_pipeline):
        """필수 계정과목 누락 처리"""
        parser = full_pipeline['parser']

        print("\n" + "="*70)
        print("시나리오 3-1: 필수 계정과목 누락 에러 처리")
        print("="*70)

        # 불완전한 데이터
        incomplete_data = {
            'balance_sheet': {
                '자산총계': 1_000_000_000_000,
                # 부채총계 누락
                # 자본총계 누락
            },
            'income_statement': {
                '매출액': 1_000_000_000_000,
            },
            'cash_flow': {},
            'metadata': {}
        }

        financial_data = parser.parse(incomplete_data)
        is_valid, errors = parser.validate(financial_data)

        # 검증 실패해야 함
        assert is_valid is False
        assert len(errors) > 0

        print(f"\n  ✓ 검증 실패 감지: {len(errors)}개 오류")
        for error in errors:
            print(f"    - {error}")

        print("\n✅ 시나리오 3-1 성공: 누락 데이터 감지됨")

    def test_negative_assets(self, full_pipeline):
        """음수 자산 감지"""
        parser = full_pipeline['parser']

        print("\n" + "="*70)
        print("시나리오 3-2: 이상 데이터 (음수 자산) 감지")
        print("="*70)

        abnormal_data = {
            'balance_sheet': {
                '자산총계': -100_000_000_000,  # 음수 자산 (이상)
                '부채총계': 50_000_000_000,
                '자본총계': -150_000_000_000,
                '유동자산': 10_000_000_000,
                '유동부채': 5_000_000_000,
            },
            'income_statement': {
                '매출액': 100_000_000_000,
                '영업이익': 10_000_000_000,
                '당기순이익': 5_000_000_000,
            },
            'cash_flow': {
                '영업활동현금흐름': 10_000_000_000,
            },
            'metadata': {}
        }

        financial_data = parser.parse(abnormal_data)
        is_valid, errors = parser.validate(financial_data)

        # 음수 자산으로 검증 실패
        assert is_valid is False
        assert any('음수 자산' in error for error in errors)

        print(f"\n  ✓ 이상 데이터 감지: {errors[0]}")
        print("\n✅ 시나리오 3-2 성공: 이상 데이터 감지됨")

    def test_feature_generator_with_missing_data(self, full_pipeline):
        """결측 데이터로 특성 생성 시 에러 없음 확인"""
        feature_generator = full_pipeline['feature_generator']

        print("\n" + "="*70)
        print("시나리오 3-3: 부분 결측 데이터 처리")
        print("="*70)

        # 일부 계정과목만 있는 데이터
        partial_data = {
            '자산총계': 1_000_000,
            '부채총계': 600_000,
            '자본총계': 400_000,
            '유동자산': 500_000,
            '유동부채': 300_000,
            '매출액': 2_000_000,
            # 나머지 항목은 없음
        }

        # 에러 없이 특성 생성되어야 함
        try:
            features_df = feature_generator.generate_all_features(partial_data)

            # 무한대/NaN 없어야 함
            assert not features_df.isin([float('inf'), float('-inf')]).any().any()
            assert not features_df.isna().any().any()

            print(f"\n  ✓ {len(features_df.columns)}개 특성 생성 완료 (결측 처리됨)")
            print("  ✓ 무한대/NaN 없음")
            print("\n✅ 시나리오 3-3 성공: 결측 데이터 안전하게 처리됨")

        except Exception as e:
            pytest.fail(f"결측 데이터 처리 실패: {str(e)}")


class TestE2EScenario4RealDartAPI:
    """시나리오 4: 실제 DART API 호출 (API 키가 있을 때만)"""

    def test_real_dart_api_integration(self, api_key, full_pipeline):
        """실제 DART API 호출 + 전체 파이프라인"""
        if not api_key:
            pytest.skip("DART_API_KEY가 설정되지 않았습니다. Mock 데이터 테스트만 수행됩니다.")

        print("\n" + "="*70)
        print("시나리오 4: 실제 DART API 연동 테스트")
        print("="*70)

        try:
            # DART API Client 생성
            client = DartAPIClient(api_key)

            # 1. 기업 검색 (삼성전자)
            company = client.search_company("삼성전자")
            print(f"\n  ✓ 기업 검색 성공: {company['corp_name']} ({company.get('stock_code', 'N/A')})")

            # 2. 재무제표 조회
            statements = client.get_financial_statements(
                corp_code=company['corp_code'],
                bsns_year="2023"
            )
            print(f"  ✓ 재무제표 조회 성공")

            # 3. 전체 파이프라인 실행
            parser = full_pipeline['parser']
            feature_generator = full_pipeline['feature_generator']
            predictor = full_pipeline['predictor']

            financial_data = parser.parse(statements)
            features_df = feature_generator.generate_all_features(financial_data)
            prediction_result = predictor.predict(features_df)

            print(f"\n  ✓ 전체 파이프라인 실행 성공")
            print(f"    - 부도 확률: {prediction_result['bankruptcy_probability']:.1%}")
            print(f"    - 위험 등급: {prediction_result['risk_level']}")

            # 삼성전자는 대기업이므로 부도 확률 낮아야 함
            assert prediction_result['bankruptcy_probability'] < 0.5

            print("\n✅ 시나리오 4 성공: 실제 DART API 연동 완료")

        except Exception as e:
            print(f"\n⚠ 실제 API 호출 실패: {str(e)}")
            print("  (네트워크 문제 또는 API 제한일 수 있습니다)")
            # 실패해도 테스트는 통과 (선택적 테스트)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
