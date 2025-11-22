"""
Integration Test 1: Parser 모듈 테스트

Parser가 샘플 DART 응답을 정상적으로 파싱하는지 검증
"""

import pytest
import pandas as pd
from src.dart_api.parser import FinancialStatementParser


@pytest.fixture
def parser():
    """FinancialStatementParser fixture"""
    return FinancialStatementParser(unit_conversion=1_000_000)


@pytest.fixture
def healthy_company_response():
    """정상 기업 샘플 DART 응답 (유동성 양호, 수익성 좋음)"""
    return {
        'balance_sheet': {
            '자산총계': 1_000_000_000_000,  # 1조원
            '부채총계': 400_000_000_000,
            '자본총계': 600_000_000_000,
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
            '자본금': 100_000_000_000,
            '이익잉여금': 450_000_000_000,
        },
        'income_statement': {
            '매출액': 2_500_000_000_000,  # 2.5조원
            '매출원가': 1_500_000_000_000,
            '매출총이익': 1_000_000_000_000,
            '판매비와관리비': 600_000_000_000,
            '영업이익': 300_000_000_000,
            '영업외수익': 20_000_000_000,
            '영업외비용': 30_000_000_000,
            '이자수익': 10_000_000_000,
            '이자비용': 15_000_000_000,
            '법인세비용차감전순이익': 290_000_000_000,
            '법인세비용': 60_000_000_000,
            '당기순이익': 230_000_000_000,
        },
        'cash_flow': {
            '영업활동현금흐름': 250_000_000_000,
            '투자활동현금흐름': -100_000_000_000,
            '재무활동현금흐름': -50_000_000_000,
        },
        'metadata': {
            'corp_code': '00000001',
            'corp_name': '정상기업',
            'bsns_year': '2023',
            'reprt_code': '11011',
            'fs_div': 'CFS'
        }
    }


@pytest.fixture
def distressed_company_response():
    """위기 기업 샘플 DART 응답 (유동성 부족, 적자)"""
    return {
        'balance_sheet': {
            '자산총계': 500_000_000_000,  # 0.5조원
            '부채총계': 450_000_000_000,
            '자본총계': 50_000_000_000,  # 자본 매우 적음
            '유동자산': 150_000_000_000,
            '유동부채': 300_000_000_000,  # 유동비율 0.5 (위험)
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
            '자본금': 50_000_000_000,
            '이익잉여금': 0,
        },
        'income_statement': {
            '매출액': 800_000_000_000,  # 0.8조원
            '매출원가': 600_000_000_000,
            '매출총이익': 200_000_000_000,
            '판매비와관리비': 250_000_000_000,
            '영업이익': -50_000_000_000,  # 영업손실
            '영업외수익': 5_000_000_000,
            '영업외비용': 10_000_000_000,
            '이자수익': 1_000_000_000,
            '이자비용': 30_000_000_000,  # 이자 부담 과다
            '법인세비용차감전순이익': -84_000_000_000,
            '법인세비용': 0,
            '당기순이익': -84_000_000_000,  # 당기순손실
        },
        'cash_flow': {
            '영업활동현금흐름': -20_000_000_000,  # 음수 현금흐름
            '투자활동현금흐름': -10_000_000_000,
            '재무활동현금흐름': 50_000_000_000,
        },
        'metadata': {
            'corp_code': '00000002',
            'corp_name': '위기기업',
            'bsns_year': '2023',
            'reprt_code': '11011',
            'fs_div': 'CFS'
        }
    }


class TestParserIntegration:
    """Parser 통합 테스트"""

    def test_parse_healthy_company(self, parser, healthy_company_response):
        """정상 기업 파싱 테스트"""
        result = parser.parse(healthy_company_response)

        # 기본 검증
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) > 30  # 최소 30개 항목 (메타데이터 포함)

        # 금액 단위 변환 확인 (원 → 백만원)
        assert result['자산총계'] == 1_000_000  # 1조원 → 100만 백만원
        assert result['매출액'] == 2_500_000  # 2.5조원
        assert result['당기순이익'] == 230_000

        # 필수 계정과목 존재 확인
        assert '자산총계' in result
        assert '부채총계' in result
        assert '자본총계' in result
        assert '유동자산' in result
        assert '유동부채' in result
        assert '매출액' in result
        assert '영업이익' in result
        assert '당기순이익' in result

        # 파생 항목 계산 확인
        assert '당좌자산' in result
        assert '순차입금' in result
        assert '차입금총계' in result

        # 메타데이터 확인
        assert '_metadata' in result
        assert result['_metadata']['corp_name'] == '정상기업'

        print("\n✓ 정상 기업 파싱 성공")
        print(f"  - 총 {len(result)}개 항목 파싱")
        print(f"  - 자산총계: {result['자산총계']:,.0f} 백만원")
        print(f"  - 매출액: {result['매출액']:,.0f} 백만원")
        print(f"  - 당기순이익: {result['당기순이익']:,.0f} 백만원")

    def test_parse_distressed_company(self, parser, distressed_company_response):
        """위기 기업 파싱 테스트"""
        result = parser.parse(distressed_company_response)

        # 기본 검증
        assert result is not None
        assert result['자산총계'] == 500_000
        assert result['부채총계'] == 450_000
        assert result['자본총계'] == 50_000

        # 음수 값 처리 확인
        assert result['영업이익'] == -50_000  # 영업손실
        assert result['당기순이익'] == -84_000  # 당기순손실
        assert result['영업활동현금흐름'] == -20_000  # 음수 현금흐름

        print("\n✓ 위기 기업 파싱 성공")
        print(f"  - 부채비율: {result['부채총계'] / result['자본총계'] * 100:.1f}%")
        print(f"  - 영업이익: {result['영업이익']:,.0f} 백만원 (적자)")
        print(f"  - 당기순손실: {result['당기순이익']:,.0f} 백만원")

    def test_validate_healthy_company(self, parser, healthy_company_response):
        """정상 기업 검증 테스트"""
        result = parser.parse(healthy_company_response)
        is_valid, errors = parser.validate(result)

        assert is_valid is True
        assert len(errors) == 0

        print("\n✓ 정상 기업 검증 성공")

    def test_validate_distressed_company(self, parser, distressed_company_response):
        """위기 기업 검증 테스트 (유동성 문제 감지)"""
        result = parser.parse(distressed_company_response)
        is_valid, errors = parser.validate(result)

        # 재무상태표 항등식은 성립해야 함 (부도 위기여도 회계는 맞아야 함)
        assets = result['자산총계']
        liabilities = result['부채총계']
        equity = result['자본총계']

        balance_diff = abs(assets - (liabilities + equity))
        assert balance_diff < assets * 0.01  # 1% 이내 허용

        print("\n✓ 위기 기업 검증 완료")
        print(f"  - 검증 결과: {'통과' if is_valid else '경고'}")
        if not is_valid:
            print(f"  - 경고 {len(errors)}개:")
            for err in errors:
                print(f"    • {err}")

    def test_get_summary_healthy_company(self, parser, healthy_company_response):
        """정상 기업 요약 정보 생성"""
        result = parser.parse(healthy_company_response)
        summary = parser.get_summary(result)

        assert summary is not None
        assert summary['기업규모'] == '대기업'  # 2.5조 매출
        assert summary['영업이익률'] > 10  # 12% (300/2500)
        assert summary['부채비율'] < 100  # 약 67% (400/600)
        assert summary['유동비율'] > 200  # 240% (600/250)

        print("\n✓ 정상 기업 요약 정보 생성 성공")
        print(f"  - 기업규모: {summary['기업규모']}")
        print(f"  - 영업이익률: {summary['영업이익률']:.2f}%")
        print(f"  - 부채비율: {summary['부채비율']:.2f}%")
        print(f"  - 유동비율: {summary['유동비율']:.2f}%")

    def test_get_summary_distressed_company(self, parser, distressed_company_response):
        """위기 기업 요약 정보 생성"""
        result = parser.parse(distressed_company_response)
        summary = parser.get_summary(result)

        assert summary is not None
        assert summary['영업이익률'] < 0  # 적자
        assert summary['부채비율'] > 500  # 900% (450/50)
        assert summary['유동비율'] < 100  # 50% (150/300)

        print("\n✓ 위기 기업 요약 정보 생성 성공")
        print(f"  - 기업규모: {summary['기업규모']}")
        print(f"  - 영업이익률: {summary['영업이익률']:.2f}% (적자)")
        print(f"  - 부채비율: {summary['부채비율']:.2f}% (위험)")
        print(f"  - 유동비율: {summary['유동비율']:.2f}% (부족)")

    def test_account_mapping_completeness(self, parser):
        """계정과목 매핑 완전성 테스트"""
        # 다양한 이름 변형 테스트
        test_data = {
            'balance_sheet': {
                '자산총액': 1000_000_000_000,  # '총계' 대신 '총액'
                '부채총액': 600_000_000_000,
                '자본총액': 400_000_000_000,
                '당좌자산': 500_000_000_000,
                '유동부채': 300_000_000_000,
                '매출채권 및 기타채권': 100_000_000_000,  # 변형된 이름
            },
            'income_statement': {
                '수익(매출액)': 2_000_000_000_000,  # 변형된 이름
                '영업이익(손실)': 200_000_000_000,
                '당기순이익(손실)': 150_000_000_000,
            },
            'cash_flow': {
                '영업활동으로 인한 현금흐름': 180_000_000_000,
            },
            'metadata': {}
        }

        result = parser.parse(test_data)

        # 표준 이름으로 매핑되었는지 확인
        assert result['자산총계'] == 1_000_000  # '총액' → '총계'
        assert result['매출액'] == 2_000_000  # '수익(매출액)' → '매출액'
        assert result['매출채권'] == 100_000  # '매출채권 및 기타채권' → '매출채권'

        print("\n✓ 계정과목 매핑 완전성 검증 성공")
        print(f"  - {len(parser.ACCOUNT_MAPPING)}개 표준 계정과목 매핑 가능")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
