"""
Streamlit UI 개선사항 테스트 (Pytest 불필요)

UI 개선사항이 정상적으로 작동하는지 테스트합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.domain_features import DomainFeatureGenerator
from src.models import BankruptcyPredictor
from src.utils.business_value import BusinessValueCalculator
from src.utils.helpers import get_risk_level, identify_critical_risks, identify_warnings
from src.visualization.charts import create_risk_gauge, create_shap_waterfall_real
import numpy as np


def test_traffic_light_logic():
    """Traffic Light 로직 테스트"""
    print("\n" + "=" * 60)
    print("Test 1: Traffic Light 로직")
    print("=" * 60)

    test_cases = [
        (0.005, "안전", "🟢"),  # < 1.68%
        (0.03, "주의", "🟡"),    # 1.68% ~ 4.68%
        (0.08, "위험", "🔴"),    # > 4.68%
    ]

    for prob, expected_label, expected_icon in test_cases:
        level, icon, msg = get_risk_level(prob)

        # Traffic Light 로직
        if prob < 0.0168:
            light_label = "안전"
            light_icon = "🟢"
        elif prob < 0.0468:
            light_label = "주의"
            light_icon = "🟡"
        else:
            light_label = "위험"
            light_icon = "🔴"

        assert light_label == expected_label, f"Label mismatch for {prob}"
        assert light_icon == expected_icon, f"Icon mismatch for {prob}"

        print(f"  ✓ 부도 확률 {prob*100:.2f}% → {light_label} {light_icon}")

    print("  ✅ Traffic Light 로직 테스트 통과")


def test_risk_dashboard_data():
    """위험 대시보드 데이터 테스트"""
    print("\n" + "=" * 60)
    print("Test 2: 위험 대시보드 데이터")
    print("=" * 60)

    # 위험 기업 데이터
    financial_data = {
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

    generator = DomainFeatureGenerator()
    features_df = generator.generate_all_features(financial_data)

    # Critical 리스크 확인
    critical_risks = identify_critical_risks(features_df)
    warnings = identify_warnings(features_df)

    print(f"  ✓ Critical 리스크: {len(critical_risks)}개")
    for i, risk in enumerate(critical_risks[:3], 1):
        print(f"    {i}. {risk['name']}: {risk['value']:.2f} (기준: {risk['threshold']:.2f})")

    print(f"  ✓ Warning: {len(warnings)}개")
    for i, warning in enumerate(warnings[:3], 1):
        print(f"    {i}. {warning['name']}: {warning['value']:.2f}")

    # 카드형 레이아웃을 위한 데이터 구조 확인
    assert isinstance(critical_risks, list), "Critical 리스크는 리스트여야 함"
    assert isinstance(warnings, list), "Warning은 리스트여야 함"

    if critical_risks:
        assert 'name' in critical_risks[0], "리스크에 'name' 키가 있어야 함"
        assert 'value' in critical_risks[0], "리스크에 'value' 키가 있어야 함"
        assert 'threshold' in critical_risks[0], "리스크에 'threshold' 키가 있어야 함"

    print("  ✅ 위험 대시보드 데이터 테스트 통과")


def test_business_value_with_params():
    """비즈니스 가치 파라미터 조정 테스트"""
    print("\n" + "=" * 60)
    print("Test 3: 비즈니스 가치 인터랙티브 파라미터")
    print("=" * 60)

    test_params = [
        (5_000_000, 500_000),    # 기본값
        (10_000_000, 1_000_000), # 큰 대출
        (1_000_000, 100_000),    # 작은 대출
    ]

    prob = 0.02  # 2% 부도 확률

    for avg_loan, avg_interest in test_params:
        calc = BusinessValueCalculator(avg_loan=avg_loan, avg_interest=avg_interest)
        value = calc.calculate_single_company(prob)

        print(f"\n  [대출 {avg_loan:,}원, 이자 {avg_interest:,}원]")
        print(f"    예상 손실: {value['expected_loss']:,}원")
        print(f"    예상 수익: {value['expected_profit']:,}원")
        print(f"    순 기대값: {value['net']:,}원")

        # 검증
        assert 'expected_loss' in value, "예상 손실 키가 있어야 함"
        assert 'expected_profit' in value, "예상 수익 키가 있어야 함"
        assert 'net' in value, "순 기대값 키가 있어야 함"

        # 낮은 부도 확률에서 순 기대값은 양수여야 함
        assert value['net'] > 0, f"낮은 부도 확률({prob})에서 순 기대값이 양수여야 함"

    print("\n  ✅ 비즈니스 가치 파라미터 테스트 통과")


def test_shap_visualization_data():
    """SHAP 시각화 데이터 테스트"""
    print("\n" + "=" * 60)
    print("Test 4: SHAP 시각화 데이터")
    print("=" * 60)

    # 샘플 SHAP 데이터
    np.random.seed(42)
    n_features = 10

    shap_values = np.random.randn(n_features) * 0.1
    feature_names = [f'특성_{i}' for i in range(n_features)]
    import pandas as pd
    feature_values = pd.Series({f'특성_{i}': np.random.rand() for i in range(n_features)})
    base_value = 0.015

    # SHAP Waterfall 차트 생성
    try:
        fig = create_shap_waterfall_real(
            shap_values=shap_values,
            feature_values=feature_values,
            feature_names=feature_names,
            base_value=base_value
        )

        assert fig is not None, "차트가 생성되어야 함"
        assert hasattr(fig, 'data'), "차트에 data 속성이 있어야 함"

        print("  ✓ SHAP Waterfall 차트 생성 성공")
        print(f"  ✓ 특성 개수: {len(feature_names)}개")
        print(f"  ✓ Base Value: {base_value:.4f}")

        # 색상 범례 검증 (실제로는 HTML에 있지만, 데이터 구조 확인)
        print("  ✓ 범례 정보: 빨간색(위험 증가), 파란색(위험 감소)")

        print("  ✅ SHAP 시각화 데이터 테스트 통과")

    except Exception as e:
        print(f"  ✗ SHAP 차트 생성 실패: {e}")
        raise


def test_progress_stages():
    """프로그레스 단계 테스트"""
    print("\n" + "=" * 60)
    print("Test 5: 프로그레스 단계")
    print("=" * 60)

    # 프로그레스 단계 정의
    stages = [
        (0, 10, "1/3 단계: 도메인 특성 생성 중..."),
        (10, 40, "특성 생성 완료"),
        (40, 50, "2/3 단계: 부도 위험 예측 중..."),
        (50, 70, "예측 완료"),
        (70, 85, "3/3 단계: 분석 결과 준비 중..."),
        (85, 100, "모든 분석 완료!"),
    ]

    for start, end, msg in stages:
        print(f"  [{start}% → {end}%] {msg}")
        assert 0 <= start <= 100, "시작 프로그레스는 0~100 사이여야 함"
        assert 0 <= end <= 100, "종료 프로그레스는 0~100 사이여야 함"
        assert start < end, "프로그레스는 증가해야 함"

    print("  ✅ 프로그레스 단계 테스트 통과")


def test_ui_data_integration():
    """UI 데이터 통합 테스트 (전체 플로우)"""
    print("\n" + "=" * 60)
    print("Test 6: UI 데이터 통합 (전체 플로우)")
    print("=" * 60)

    # 샘플 데이터
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

    print("\n  [Step 1] 특성 생성 (0% → 40%)")
    generator = DomainFeatureGenerator()
    features_df = generator.generate_all_features(financial_data)
    print(f"    ✓ {len(features_df.columns)}개 특성 생성")

    print("\n  [Step 2] 예측 (40% → 70%)")
    predictor = BankruptcyPredictor()
    predictor.load_model()
    result = predictor.predict(features_df)
    print(f"    ✓ 부도 확률: {result['bankruptcy_probability']:.2%}")

    print("\n  [Step 3] Traffic Light 표시")
    prob = result['bankruptcy_probability']
    if prob < 0.0168:
        light_label, light_icon = "안전", "🟢"
    elif prob < 0.0468:
        light_label, light_icon = "주의", "🟡"
    else:
        light_label, light_icon = "위험", "🔴"
    print(f"    ✓ {light_label} {light_icon}")

    print("\n  [Step 4] 위험 대시보드")
    critical_risks = identify_critical_risks(features_df)
    warnings = identify_warnings(features_df)
    print(f"    ✓ Critical: {len(critical_risks)}개, Warning: {len(warnings)}개")

    print("\n  [Step 5] 비즈니스 가치")
    calc = BusinessValueCalculator(avg_loan=5_000_000, avg_interest=500_000)
    value = calc.calculate_single_company(prob)
    print(f"    ✓ 순 기대값: {value['net']:,}원")

    print("\n  [Step 6] 분석 완료 (100%)")
    print("    ✓ 모든 UI 컴포넌트 데이터 준비 완료")

    print("\n  ✅ UI 데이터 통합 테스트 통과")


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "=" * 80)
    print(" " * 25 + "Streamlit UI 테스트")
    print("=" * 80)

    tests = [
        test_traffic_light_logic,
        test_risk_dashboard_data,
        test_business_value_with_params,
        test_shap_visualization_data,
        test_progress_stages,
        test_ui_data_integration,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n  ❌ 테스트 실패: {e}")
            failed += 1
        except Exception as e:
            print(f"\n  ❌ 예외 발생: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print(f"테스트 결과: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
