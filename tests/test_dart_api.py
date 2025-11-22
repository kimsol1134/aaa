"""
DART API 모듈 테스트

주의: 실제 DART API 키가 필요합니다.
.env 파일에 DART_API_KEY를 설정하세요.
"""

import os
import pytest
from dotenv import load_dotenv
from src.dart_api.client import DartAPIClient
from src.dart_api.parser import FinancialStatementParser

# 환경 변수 로드
load_dotenv()


@pytest.fixture
def api_key():
    """DART API 키 fixture"""
    key = os.getenv('DART_API_KEY')
    if not key:
        pytest.skip("DART_API_KEY가 설정되지 않았습니다.")
    return key


@pytest.fixture
def client(api_key):
    """DartAPIClient fixture"""
    return DartAPIClient(api_key)


@pytest.fixture
def parser():
    """FinancialStatementParser fixture"""
    return FinancialStatementParser()


class TestDartAPIClient:
    """DART API 클라이언트 테스트"""

    def test_init_without_api_key(self):
        """API 키 없이 초기화 시 ValueError 발생"""
        with pytest.raises(ValueError, match="DART API 키"):
            DartAPIClient(api_key="")

    def test_search_company_samsung(self, client):
        """삼성전자 검색 테스트"""
        result = client.search_company("삼성전자")

        assert result is not None
        assert result['corp_code'] is not None
        assert result['corp_name'] == '삼성전자'
        assert '005930' in result.get('stock_code', '')

        print(f"\n✓ 기업 검색 성공: {result}")

    def test_search_company_not_found(self, client):
        """존재하지 않는 기업 검색 시 에러"""
        with pytest.raises(ValueError, match="기업을 찾을 수 없습니다"):
            client.search_company("존재하지않는기업명123456")

    def test_get_financial_statements(self, client):
        """재무제표 조회 테스트 (삼성전자 2023년)"""
        # 삼성전자 corp_code: 00126380
        corp_code = "00126380"
        statements = client.get_financial_statements(
            corp_code=corp_code,
            bsns_year="2023"
        )

        assert statements is not None
        assert 'balance_sheet' in statements
        assert 'income_statement' in statements
        assert 'cash_flow' in statements
        assert 'metadata' in statements

        # 재무상태표에 주요 항목 존재 확인
        bs = statements['balance_sheet']
        assert len(bs) > 0
        assert '자산총계' in bs or '자산총액' in bs

        # 손익계산서에 주요 항목 존재 확인
        is_stmt = statements['income_statement']
        assert len(is_stmt) > 0

        print(f"\n✓ 재무제표 조회 성공:")
        print(f"  - 재무상태표: {len(bs)}개 항목")
        print(f"  - 손익계산서: {len(is_stmt)}개 항목")
        print(f"  - 현금흐름표: {len(statements['cash_flow'])}개 항목")

    def test_get_multi_year_statements(self, client):
        """다년도 재무제표 조회 테스트"""
        corp_code = "00126380"  # 삼성전자
        years = ["2022", "2023"]

        results = client.get_multi_year_statements(corp_code, years)

        assert len(results) == 2
        assert "2022" in results
        assert "2023" in results

        for year, data in results.items():
            if data:  # None이 아닌 경우
                assert 'balance_sheet' in data

        print(f"\n✓ 다년도 조회 성공: {list(results.keys())}")


class TestFinancialStatementParser:
    """재무제표 파싱 테스트"""

    @pytest.fixture
    def sample_dart_response(self):
        """샘플 DART API 응답"""
        return {
            'balance_sheet': {
                '자산총계': 1_000_000_000_000,  # 1조원
                '부채총계': 600_000_000_000,
                '자본총계': 400_000_000_000,
                '유동자산': 500_000_000_000,
                '유동부채': 300_000_000_000,
                '현금및현금성자산': 100_000_000_000,
                '재고자산': 50_000_000_000,
                '단기차입금': 50_000_000_000,
                '장기차입금': 100_000_000_000,
            },
            'income_statement': {
                '매출액': 2_000_000_000_000,  # 2조원
                '매출원가': 1_200_000_000_000,
                '영업이익': 200_000_000_000,
                '당기순이익': 150_000_000_000,
                '이자비용': 20_000_000_000,
            },
            'cash_flow': {
                '영업활동현금흐름': 180_000_000_000,
            },
            'metadata': {
                'corp_code': '00000000',
                'bsns_year': '2023',
                'reprt_code': '11011'
            }
        }

    def test_parse_basic(self, parser, sample_dart_response):
        """기본 파싱 테스트"""
        result = parser.parse(sample_dart_response)

        assert result is not None
        assert '자산총계' in result
        assert '매출액' in result
        assert '영업활동현금흐름' in result

        # 백만원 단위로 변환 확인
        assert result['자산총계'] == 1_000_000  # 1조원 → 100만 백만원
        assert result['매출액'] == 2_000_000

        print(f"\n✓ 파싱 성공: {len(result)}개 항목")
        print(f"  자산총계: {result['자산총계']:,.0f} 백만원")
        print(f"  매출액: {result['매출액']:,.0f} 백만원")

    def test_account_mapping(self, parser, sample_dart_response):
        """계정과목 매핑 테스트"""
        result = parser.parse(sample_dart_response)

        # 표준 계정명으로 매핑되었는지 확인
        assert '자산총계' in result
        assert '유동자산' in result
        assert '매출액' in result

    def test_derived_accounts(self, parser, sample_dart_response):
        """파생 계정과목 계산 테스트"""
        result = parser.parse(sample_dart_response)

        # 당좌자산 = 유동자산 - 재고자산
        expected_quick_assets = (500_000_000_000 - 50_000_000_000) / 1_000_000
        assert abs(result['당좌자산'] - expected_quick_assets) < 1

        # 순차입금 = 차입금 - 현금
        expected_net_debt = (50_000_000_000 + 100_000_000_000 - 100_000_000_000) / 1_000_000
        assert abs(result['순차입금'] - expected_net_debt) < 1

        print(f"\n✓ 파생 항목 계산 성공:")
        print(f"  당좌자산: {result['당좌자산']:,.0f} 백만원")
        print(f"  순차입금: {result['순차입금']:,.0f} 백만원")

    def test_validate_success(self, parser, sample_dart_response):
        """검증 성공 케이스"""
        result = parser.parse(sample_dart_response)
        is_valid, errors = parser.validate(result)

        assert is_valid is True
        assert len(errors) == 0

        print("\n✓ 재무제표 검증 성공")

    def test_validate_missing_accounts(self, parser):
        """필수 계정과목 누락 시 검증 실패"""
        incomplete_data = {
            '자산총계': 1000,
            # 부채총계 누락
            # 자본총계 누락
        }

        is_valid, errors = parser.validate(incomplete_data)

        assert is_valid is False
        assert len(errors) > 0
        assert any('부채총계' in error for error in errors)

        print(f"\n✓ 검증 실패 감지: {len(errors)}개 오류")

    def test_get_summary(self, parser, sample_dart_response):
        """요약 정보 생성 테스트"""
        result = parser.parse(sample_dart_response)
        summary = parser.get_summary(result)

        assert summary is not None
        assert '매출액' in summary
        assert '기업규모' in summary
        assert '영업이익률' in summary
        assert '부채비율' in summary

        # 기업 규모 판정 (2조원 → 대기업)
        assert summary['기업규모'] == '대기업'

        # 영업이익률 = 200 / 2000 * 100 = 10%
        assert abs(summary['영업이익률'] - 10.0) < 0.1

        print(f"\n✓ 요약 정보 생성 성공:")
        print(f"  기업규모: {summary['기업규모']}")
        print(f"  영업이익률: {summary['영업이익률']:.2f}%")
        print(f"  부채비율: {summary['부채비율']:.2f}%")


class TestIntegration:
    """통합 테스트 (실제 API 호출 + 파싱)"""

    def test_end_to_end_samsung(self, client, parser):
        """End-to-End 테스트: 삼성전자 조회 → 파싱 → 검증"""
        # 1. 기업 검색
        company = client.search_company("삼성전자")
        assert company is not None

        # 2. 재무제표 조회
        statements = client.get_financial_statements(
            corp_code=company['corp_code'],
            bsns_year="2023"
        )
        assert statements is not None

        # 3. 파싱
        parsed = parser.parse(statements)
        assert parsed is not None
        assert len(parsed) > 20  # 최소 20개 항목

        # 4. 검증
        is_valid, errors = parser.validate(parsed)
        if not is_valid:
            print(f"\n⚠ 검증 경고: {errors}")
        # 삼성전자는 대기업이므로 대부분 검증 통과 예상

        # 5. 요약
        summary = parser.get_summary(parsed)
        assert summary['기업규모'] == '대기업'

        print("\n" + "="*60)
        print("✓ End-to-End 테스트 성공: 삼성전자")
        print("="*60)
        print(f"기업명: {company['corp_name']}")
        print(f"종목코드: {company['stock_code']}")
        print(f"자산총계: {parsed['자산총계']:,.0f} 백만원")
        print(f"매출액: {parsed['매출액']:,.0f} 백만원")
        print(f"영업이익률: {summary.get('영업이익률', 0):.2f}%")
        print(f"부채비율: {summary.get('부채비율', 0):.2f}%")
        print("="*60)


if __name__ == "__main__":
    # pytest 실행
    pytest.main([__file__, "-v", "-s"])
