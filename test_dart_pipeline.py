"""
DART API를 사용한 Part3 파이프라인 테스트

실제 기업 데이터를 불러와서 파이프라인 전체 흐름을 검증합니다.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# src 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.dart_api.client import DartAPIClient
from src.dart_api.parser import FinancialStatementParser
from src.domain_features.feature_generator import DomainFeatureGenerator
from src.models.predictor import BankruptcyPredictor

def main():
    print("=" * 80)
    print("🚀 DART API + Part3 파이프라인 통합 테스트")
    print("=" * 80)

    # 1. DART API 클라이언트 초기화
    print("\n📡 1. DART API 클라이언트 초기화...")
    api_key = os.getenv('DART_API_KEY')
    if not api_key:
        logger.warning("⚠️  DART_API_KEY가 .env 파일에 없습니다!")
        print("   → 더미 데이터로 파이프라인을 테스트합니다.\n")

        # 더미 데이터로 테스트
        financial_data = create_dummy_financial_data()
        company_info_dict = {
            '업력': 50,
            '외감여부': True,
            '업종코드': 'C26',
            '종업원수': 100000,
            '연체여부': False,
            '신용등급': 'AAA'
        }
        test_with_dummy_data(financial_data, company_info_dict)
        return

    dart_client = DartAPIClient(api_key)
    print(f"   ✅ API Key: {api_key[:10]}...")

    # 2. 테스트 기업 선택 (삼성전자)
    print("\n🏢 2. 테스트 기업 정보 조회...")
    corp_code = "00126380"  # 삼성전자
    corp_name = "삼성전자"

    try:
        # 기업 개황 조회
        company_info = dart_client.get_company_info(corp_code)
        print(f"   기업명: {company_info.get('corp_name', corp_name)}")
        print(f"   대표자: {company_info.get('ceo_nm', 'N/A')}")
        print(f"   업종: {company_info.get('induty_code', 'N/A')}")
    except Exception as e:
        logger.warning(f"기업 개황 조회 실패: {e}")
        company_info = {'corp_name': corp_name}

    # 3. 재무제표 조회
    print("\n📊 3. 재무제표 조회...")
    year = 2023
    report_code = "11011"  # 사업보고서

    try:
        # 재무상태표
        bs = dart_client.get_financial_statement(
            corp_code=corp_code,
            bsns_year=str(year),
            reprt_code=report_code,
            fs_div="OFS"  # 재무상태표
        )

        # 손익계산서
        is_ = dart_client.get_financial_statement(
            corp_code=corp_code,
            bsns_year=str(year),
            reprt_code=report_code,
            fs_div="OFS"  # 손익계산서
        )

        print(f"   ✅ {year}년 재무제표 조회 완료")
        print(f"   재무상태표 항목 수: {len(bs) if bs else 0}")
        print(f"   손익계산서 항목 수: {len(is_) if is_ else 0}")

    except Exception as e:
        logger.error(f"재무제표 조회 실패: {e}")
        print("\n⚠️  실제 DART 데이터 조회 실패. 더미 데이터로 테스트합니다.")

        # 더미 재무 데이터 생성
        financial_data = create_dummy_financial_data()
        company_info_dict = {
            '업력': 50,
            '외감여부': True,
            '업종코드': 'C26',
            '종업원수': 100000,
            '연체여부': False,
            '신용등급': 'AAA'
        }

        # 특성 생성으로 바로 점프
        test_with_dummy_data(financial_data, company_info_dict)
        return

    # 4. 재무제표 파싱
    print("\n🔍 4. 재무제표 파싱...")
    parser = FinancialStatementParser()

    try:
        # BS와 IS 합치기
        all_statements = bs + is_ if (bs and is_) else []

        if not all_statements:
            raise ValueError("재무제표 데이터가 비어있습니다.")

        # DataFrame 변환 및 파싱
        df = pd.DataFrame(all_statements)
        financial_data = parser.parse_financial_statement(df)

        print(f"   ✅ 파싱 완료: {len(financial_data)}개 항목")

        # 주요 항목 출력
        key_items = ['유동자산', '유동부채', '매출액', '당기순이익', '총자산']
        print("\n   주요 재무 항목:")
        for item in key_items:
            value = financial_data.get(item, 0)
            if value:
                print(f"   - {item}: {value:,.0f}")

    except Exception as e:
        logger.error(f"재무제표 파싱 실패: {e}")
        financial_data = create_dummy_financial_data()

    # 5. 도메인 특성 생성
    print("\n🔧 5. 도메인 특성 생성 (65개 특성)...")
    feature_generator = DomainFeatureGenerator()

    # 기업 추가 정보
    company_info_dict = {
        '업력': company_info.get('est_dt', 50),
        '외감여부': True,
        '업종코드': company_info.get('induty_code', 'C26'),
        '종업원수': company_info.get('emp_no', 100000),
        '연체여부': False,
        '신용등급': 'AAA'
    }

    try:
        features_df = feature_generator.generate_all_features(
            financial_data=financial_data,
            company_info=company_info_dict
        )

        print(f"   ✅ 특성 생성 완료: {features_df.shape[1]}개")
        print(f"\n   생성된 특성 샘플 (첫 10개):")
        for col in list(features_df.columns)[:10]:
            print(f"   - {col}: {features_df[col].iloc[0]:.4f}")

    except Exception as e:
        logger.error(f"특성 생성 실패: {e}")
        return

    # 6. 예측 모델 테스트
    print("\n🎯 6. Part3 파이프라인 예측 테스트...")

    # 6-1. 파이프라인 모델 테스트 (우선)
    test_pipeline_model(features_df, corp_name)

    # 6-2. 휴리스틱 모델 테스트 (비교용)
    test_heuristic_model(features_df, corp_name)

    print("\n" + "=" * 80)
    print("✅ 테스트 완료!")
    print("=" * 80)


def test_pipeline_model(features_df: pd.DataFrame, corp_name: str):
    """Part3 파이프라인 모델로 예측"""
    print("\n   [A] Part3 파이프라인 모델 테스트")

    # 모델 경로 확인
    pipeline_path = Path('data/processed/발표_Part3_v3_최종모델.pkl')

    if not pipeline_path.exists():
        print(f"   ⚠️  파이프라인 모델이 없습니다: {pipeline_path}")
        print("   → train_final_model.py를 먼저 실행하세요.")
        return

    try:
        predictor = BankruptcyPredictor(
            pipeline_path=pipeline_path,
            use_pipeline=True
        )
        predictor.load_model()

        result = predictor.predict(features_df)

        print(f"\n   📊 {corp_name} 부도 예측 결과:")
        print(f"   - 부도 확률: {result['bankruptcy_probability']:.2%}")
        print(f"   - 위험 등급: {result['risk_icon']} {result['risk_level']}")
        print(f"   - 신뢰도: {result['confidence']:.2%}")
        print(f"   - 모델: {result['model_info']['model_type']}")
        print(f"   - 사용 특성 수: {result['model_info']['n_features']}개")

        if 'shap_values' in result:
            print(f"   - SHAP 분석: ✅ 완료")

    except Exception as e:
        logger.error(f"파이프라인 예측 실패: {e}", exc_info=True)


def test_heuristic_model(features_df: pd.DataFrame, corp_name: str):
    """휴리스틱 모델로 예측 (비교용)"""
    print("\n   [B] 휴리스틱 모델 테스트 (모델 없음)")

    try:
        predictor = BankruptcyPredictor()  # 모델 없음

        result = predictor.predict(features_df)

        print(f"\n   📊 {corp_name} 부도 예측 결과 (휴리스틱):")
        print(f"   - 부도 확률: {result['bankruptcy_probability']:.2%}")
        print(f"   - 위험 등급: {result['risk_icon']} {result['risk_level']}")
        print(f"   - 신뢰도: {result['confidence']:.2%}")
        print(f"   - 모델: {result['model_info']['model_type']}")

    except Exception as e:
        logger.error(f"휴리스틱 예측 실패: {e}", exc_info=True)


def create_dummy_financial_data():
    """테스트용 더미 재무 데이터 생성"""
    return {
        # 자산
        '유동자산': 100_000_000,
        '재고자산': 20_000_000,
        '매출채권': 30_000_000,
        '현금및현금성자산': 40_000_000,
        '비유동자산': 150_000_000,
        '총자산': 250_000_000,

        # 부채
        '유동부채': 60_000_000,
        '단기차입금': 20_000_000,
        '비유동부채': 40_000_000,
        '총부채': 100_000_000,

        # 자본
        '자본총계': 150_000_000,

        # 손익
        '매출액': 300_000_000,
        '매출원가': 200_000_000,
        '매출총이익': 100_000_000,
        '영업이익': 50_000_000,
        '당기순이익': 30_000_000,
        '법인세비용': 10_000_000,
        '이자비용': 5_000_000,

        # 현금흐름
        '영업활동현금흐름': 40_000_000,
        '투자활동현금흐름': -20_000_000,
        '재무활동현금흐름': -10_000_000,
    }


def test_with_dummy_data(financial_data: dict, company_info: dict):
    """더미 데이터로 전체 파이프라인 테스트"""
    print("\n🔧 더미 데이터로 특성 생성 중...")

    feature_generator = DomainFeatureGenerator()

    try:
        features_df = feature_generator.generate_all_features(
            financial_data=financial_data,
            company_info=company_info
        )

        print(f"   ✅ 특성 생성 완료: {features_df.shape[1]}개")

        # 예측 테스트
        print("\n🎯 예측 테스트...")
        test_pipeline_model(features_df, "테스트 기업")
        test_heuristic_model(features_df, "테스트 기업")

    except Exception as e:
        logger.error(f"더미 데이터 테스트 실패: {e}", exc_info=True)


if __name__ == "__main__":
    main()
