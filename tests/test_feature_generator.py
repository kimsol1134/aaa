"""
도메인 특성 생성 모듈 테스트
"""

import pytest
import pandas as pd
from src.domain_features.feature_generator import DomainFeatureGenerator


@pytest.fixture
def sample_financial_data():
    """샘플 재무제표 데이터"""
    return {
        # 재무상태표
        '자산총계': 1_000_000,  # 백만원
        '부채총계': 600_000,
        '자본총계': 400_000,
        '유동자산': 500_000,
        '비유동자산': 500_000,
        '유동부채': 300_000,
        '비유동부채': 300_000,
        '현금및현금성자산': 100_000,
        '단기금융상품': 50_000,
        '매출채권': 100_000,
        '재고자산': 50_000,
        '유형자산': 300_000,
        '무형자산': 50_000,
        '단기차입금': 50_000,
        '장기차입금': 100_000,
        '매입채무': 50_000,

        # 손익계산서
        '매출액': 2_000_000,
        '매출원가': 1_200_000,
        '매출총이익': 800_000,
        '판매비와관리비': 500_000,
        '영업이익': 200_000,
        '이자비용': 20_000,
        '당기순이익': 150_000,

        # 현금흐름표
        '영업활동현금흐름': 180_000,
    }


@pytest.fixture
def sample_company_info():
    """샘플 기업 정보"""
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


@pytest.fixture
def generator():
    """DomainFeatureGenerator fixture"""
    return DomainFeatureGenerator()


class TestDomainFeatureGenerator:
    """도메인 특성 생성기 테스트"""

    def test_generate_all_features(self, generator, sample_financial_data, sample_company_info):
        """모든 특성 생성 테스트"""
        df = generator.generate_all_features(sample_financial_data, sample_company_info)

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # 1행
        assert len(df.columns) >= 60  # 최소 60개 특성

        print(f"\n✓ 총 {len(df.columns)}개 특성 생성")
        print(f"  특성 목록: {list(df.columns[:10])}...")

    def test_liquidity_features(self, generator, sample_financial_data):
        """유동성 특성 생성 테스트"""
        df = generator.generate_all_features(sample_financial_data)

        # 주요 유동성 특성 존재 확인
        assert '유동비율' in df.columns
        assert '현금비율' in df.columns
        assert '현금소진일수' in df.columns
        assert '유동성위기지수' in df.columns

        # 값 검증
        assert df['유동비율'].iloc[0] > 0
        assert df['현금비율'].iloc[0] > 0

        print("\n✓ 유동성 특성 생성 성공")
        print(f"  유동비율: {df['유동비율'].iloc[0]:.2f}")
        print(f"  현금소진일수: {df['현금소진일수'].iloc[0]:.1f}일")

    def test_insolvency_features(self, generator, sample_financial_data):
        """지급불능 특성 생성 테스트"""
        df = generator.generate_all_features(sample_financial_data)

        # 주요 지급불능 특성 존재 확인
        assert '부채비율' in df.columns
        assert '이자보상배율' in df.columns
        assert '지급불능위험지수' in df.columns

        # 값 검증
        assert df['부채비율'].iloc[0] > 0
        assert df['이자보상배율'].iloc[0] > 0

        print("\n✓ 지급불능 특성 생성 성공")
        print(f"  부채비율: {df['부채비율'].iloc[0]:.1f}%")
        print(f"  이자보상배율: {df['이자보상배율'].iloc[0]:.2f}")

    def test_composite_features(self, generator, sample_financial_data):
        """복합 리스크 특성 생성 테스트"""
        df = generator.generate_all_features(sample_financial_data)

        # 복합 특성 존재 확인
        assert '종합부도위험스코어' in df.columns
        assert '조기경보신호수' in df.columns
        assert '재무건전성지수' in df.columns

        # 값 범위 검증
        assert 0 <= df['종합부도위험스코어'].iloc[0] <= 1
        assert df['조기경보신호수'].iloc[0] >= 0

        print("\n✓ 복합 리스크 특성 생성 성공")
        print(f"  종합부도위험스코어: {df['종합부도위험스코어'].iloc[0]:.3f}")
        print(f"  조기경보신호수: {df['조기경보신호수'].iloc[0]:.0f}")
        print(f"  재무건전성지수: {df['재무건전성지수'].iloc[0]:.1f}")

    def test_no_inf_or_nan(self, generator, sample_financial_data):
        """무한대/NaN 값이 없는지 테스트"""
        df = generator.generate_all_features(sample_financial_data)

        # 무한대 확인
        assert not df.isin([float('inf'), float('-inf')]).any().any()

        # NaN 확인
        assert not df.isna().any().any()

        print("\n✓ 무한대/NaN 값 없음")

    def test_feature_importance_estimates(self, generator, sample_financial_data):
        """특성 중요도 추정 테스트"""
        df = generator.generate_all_features(sample_financial_data)
        importance_df = generator.get_feature_importance_estimates(df)

        assert importance_df is not None
        assert len(importance_df) == len(df.columns)
        assert '특성명' in importance_df.columns
        assert '카테고리' in importance_df.columns
        assert '중요도점수' in importance_df.columns

        # 상위 5개 중요 특성 출력
        print("\n✓ 특성 중요도 추정 성공")
        print("  상위 5개 중요 특성:")
        for idx, row in importance_df.head(5).iterrows():
            print(f"    {row['특성명']}: {row['중요도수준']} ({row['카테고리']})")

    def test_summary_stats(self, generator, sample_financial_data):
        """요약 통계 테스트"""
        df = generator.generate_all_features(sample_financial_data)
        summary = generator.get_summary_stats(df)

        assert summary is not None
        assert '총특성수' in summary
        assert '카테고리별특성수' in summary
        assert '통계' in summary

        print(f"\n✓ 요약 통계:")
        print(f"  총 특성 수: {summary['총특성수']}")
        print(f"  카테고리별 특성 수: {summary['카테고리별특성수']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
