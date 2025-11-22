"""
DART Open API 클라이언트

주요 기능:
1. 기업 검색 (회사명 → 종목코드 변환)
2. 재무제표 조회 (단일 회계연도)
3. 다년도 재무제표 조회 (추이 분석용)
4. API Rate Limit 관리
5. 에러 핸들링 및 재시도 로직

필수 API 엔드포인트:
- 공시검색: /api/company.json (기업 기본정보)
- 재무제표: /api/fnlttSinglAcntAll.json (단일회사 전체 재무제표)
"""

import requests
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DartAPIClient:
    """DART API 클라이언트"""

    BASE_URL = "https://opendart.fss.or.kr/api"
    RATE_LIMIT_DELAY = 1.0  # 초당 1회 제한

    def __init__(self, api_key: str):
        """
        Args:
            api_key: DART API 키 (환경 변수에서 로드)
        """
        if not api_key:
            raise ValueError("DART API 키가 제공되지 않았습니다. .env 파일에 DART_API_KEY를 설정하세요.")

        self.api_key = api_key
        self.last_request_time = 0
        self.session = requests.Session()

    def _handle_rate_limit(self):
        """API Rate Limit 관리 (1초당 1회 제한)"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.RATE_LIMIT_DELAY:
            sleep_time = self.RATE_LIMIT_DELAY - time_since_last_request
            logger.debug(f"Rate limit 대기: {sleep_time:.2f}초")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(
        self,
        endpoint: str,
        params: Dict,
        max_retries: int = 3
    ) -> Dict:
        """
        API 요청 실행 (에러 핸들링 및 재시도 포함)

        Args:
            endpoint: API 엔드포인트
            params: 요청 파라미터
            max_retries: 최대 재시도 횟수

        Returns:
            API 응답 JSON

        Raises:
            requests.exceptions.Timeout: 타임아웃
            requests.exceptions.HTTPError: HTTP 에러
            ValueError: 잘못된 응답
        """
        url = f"{self.BASE_URL}/{endpoint}"
        params['crtfc_key'] = self.api_key

        for attempt in range(max_retries):
            try:
                self._handle_rate_limit()

                logger.info(f"API 요청: {endpoint}, 시도: {attempt + 1}/{max_retries}")
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()

                data = response.json()

                # DART API 에러 체크
                if data.get('status') == '000':
                    return data
                elif data.get('status') == '013':
                    raise ValueError(f"검색 결과가 없습니다: {params}")
                elif data.get('status') == '020':
                    raise ValueError("API 키가 유효하지 않습니다.")
                else:
                    error_msg = data.get('message', '알 수 없는 에러')
                    raise ValueError(f"DART API 에러 ({data.get('status')}): {error_msg}")

            except requests.exceptions.Timeout:
                logger.warning(f"타임아웃 발생 (시도 {attempt + 1})")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # 지수 백오프

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    logger.warning("Rate limit 초과, 재시도 중...")
                    time.sleep(5)
                else:
                    raise

        raise Exception("최대 재시도 횟수 초과")

    def search_company(self, company_name: str) -> Dict:
        """
        기업명으로 종목코드 검색

        Args:
            company_name: 기업명 (예: "삼성전자")

        Returns:
            {
                'corp_code': '00126380',
                'corp_name': '삼성전자',
                'stock_code': '005930',
                'modify_date': '20231201'
            }

        Raises:
            ValueError: 기업을 찾을 수 없음
        """
        # 회사명으로 검색 (corpCode.xml 사용)
        # 실제 구현에서는 DART에서 제공하는 corpCode.xml을 파싱해야 합니다
        # 여기서는 단순화를 위해 API endpoint 사용

        endpoint = "company.json"
        params = {
            'corp_name': company_name
        }

        try:
            data = self._make_request(endpoint, params)

            if not data.get('list'):
                raise ValueError(f"'{company_name}' 기업을 찾을 수 없습니다.")

            # 가장 유사한 결과 반환
            result = data['list'][0]

            logger.info(f"기업 검색 성공: {result.get('corp_name')} ({result.get('stock_code')})")

            return {
                'corp_code': result.get('corp_code'),
                'corp_name': result.get('corp_name'),
                'stock_code': result.get('stock_code'),
                'modify_date': result.get('modify_date')
            }

        except Exception as e:
            logger.error(f"기업 검색 실패: {str(e)}")
            raise

    def get_financial_statements(
        self,
        corp_code: str,
        bsns_year: str,
        reprt_code: str = "11011"  # 사업보고서
    ) -> Dict:
        """
        재무제표 조회

        Args:
            corp_code: 고유번호 (8자리)
            bsns_year: 사업연도 (YYYY)
            reprt_code: 보고서 코드
                - 11011: 사업보고서
                - 11012: 반기보고서
                - 11013: 1분기보고서
                - 11014: 3분기보고서

        Returns:
            {
                'balance_sheet': {...},      # 재무상태표
                'income_statement': {...},   # 손익계산서
                'cash_flow': {...}           # 현금흐름표
            }

        Raises:
            ValueError: 재무제표를 찾을 수 없음
        """
        endpoint = "fnlttSinglAcntAll.json"
        params = {
            'corp_code': corp_code,
            'bsns_year': bsns_year,
            'reprt_code': reprt_code,
            'fs_div': 'CFS'  # CFS: 연결재무제표, OFS: 개별재무제표
        }

        try:
            data = self._make_request(endpoint, params)

            if not data.get('list'):
                # 연결재무제표가 없으면 개별재무제표 시도
                logger.warning("연결재무제표 없음, 개별재무제표 조회 중...")
                params['fs_div'] = 'OFS'
                data = self._make_request(endpoint, params)

                if not data.get('list'):
                    raise ValueError(f"재무제표를 찾을 수 없습니다: {corp_code}, {bsns_year}")

            # 계정과목별로 분류
            statements = {
                'balance_sheet': {},
                'income_statement': {},
                'cash_flow': {},
                'metadata': {
                    'corp_code': corp_code,
                    'bsns_year': bsns_year,
                    'reprt_code': reprt_code,
                    'fs_div': params['fs_div']
                }
            }

            for item in data['list']:
                account_nm = item.get('account_nm')  # 계정명
                thstrm_amount = item.get('thstrm_amount')  # 당기금액
                sj_div = item.get('sj_div')  # 재무제표구분 (BS: 재무상태표, IS: 손익계산서, CF: 현금흐름표)

                if not account_nm or not thstrm_amount:
                    continue

                # 숫자로 변환 (쉼표 제거)
                try:
                    amount = float(thstrm_amount.replace(',', ''))
                except (ValueError, AttributeError):
                    amount = 0

                # 재무제표 구분에 따라 저장
                if sj_div == 'BS':
                    statements['balance_sheet'][account_nm] = amount
                elif sj_div == 'IS':
                    statements['income_statement'][account_nm] = amount
                elif sj_div == 'CF':
                    statements['cash_flow'][account_nm] = amount

            logger.info(f"재무제표 조회 성공: {bsns_year}년")
            logger.info(f"  - 재무상태표: {len(statements['balance_sheet'])}개 항목")
            logger.info(f"  - 손익계산서: {len(statements['income_statement'])}개 항목")
            logger.info(f"  - 현금흐름표: {len(statements['cash_flow'])}개 항목")

            return statements

        except Exception as e:
            logger.error(f"재무제표 조회 실패: {str(e)}")
            raise

    def get_multi_year_statements(
        self,
        corp_code: str,
        years: List[str]
    ) -> Dict[str, Dict]:
        """
        다년도 재무제표 조회 (추이 분석용)

        Args:
            corp_code: 고유번호
            years: 사업연도 리스트 (예: ['2021', '2022', '2023'])

        Returns:
            {
                '2021': {...},
                '2022': {...},
                '2023': {...}
            }
        """
        results = {}

        for year in years:
            try:
                statements = self.get_financial_statements(corp_code, year)
                results[year] = statements
            except Exception as e:
                logger.warning(f"{year}년 재무제표 조회 실패: {str(e)}")
                results[year] = None

        return results
