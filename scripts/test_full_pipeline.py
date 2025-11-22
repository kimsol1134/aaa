#!/usr/bin/env python3
"""
실제 사용 시뮬레이션 스크립트

전체 부도 예측 파이프라인을 Streamlit 없이 Python 스크립트로 재현

실행 방법:
    python scripts/test_full_pipeline.py
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.dart_api.parser import FinancialStatementParser
from src.domain_features.feature_generator import DomainFeatureGenerator
from src.models.predictor import BankruptcyPredictor
from src.utils.helpers import (
    get_risk_level,
    format_korean_number,
    identify_critical_risks,
    identify_warnings,
    generate_recommendations
)

# 환경 변수 로드
load_dotenv()


def print_section_header(title):
    """섹션 헤더 출력"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_subsection(title):
    """서브섹션 출력"""
    print(f"\n📌 {title}")
    print("-" * 80)


def simulate_healthy_company():
    """정상 기업 시뮬레이션"""
    print_section_header("시뮬레이션 1: 정상 기업 부도 예측")

    # === 1. 컴포넌트 초기화 ===
    print_subsection("Step 1: 컴포넌트 초기화")

    parser = FinancialStatementParser(unit_conversion=1_000_000)
    feature_generator = DomainFeatureGenerator()

    model_path = project_root / 'data' / 'processed' / 'best_model_XGBoost.pkl'
    scaler_path = project_root / 'data' / 'processed' / 'scaler.pkl'

    predictor = BankruptcyPredictor(
        model_path=model_path if model_path.exists() else None,
        scaler_path=scaler_path if scaler_path.exists() else None
    )
    predictor.load_model()

    print(f"  ✓ Parser 초기화 완료")
    print(f"  ✓ Feature Generator 초기화 완료")
    print(f"  ✓ Predictor 초기화 완료 (모델: {predictor.model_path or 'Heuristic'})")

    # === 2. 샘플 재무 데이터 준비 ===
    print_subsection("Step 2: 샘플 재무 데이터 준비")

    # 우량 기업 재무제표 (원 단위)
    dart_response = {
        'balance_sheet': {
            '자산총계': 1_000_000_000_000,
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
        },
        'income_statement': {
            '매출액': 2_500_000_000_000,
            '매출원가': 1_500_000_000_000,
            '매출총이익': 1_000_000_000_000,
            '판매비와관리비': 600_000_000_000,
            '영업이익': 300_000_000_000,
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

    print(f"  기업명: {dart_response['metadata']['corp_name']}")
    print(f"  자산총계: {format_korean_number(1_000_000)}")
    print(f"  매출액: {format_korean_number(2_500_000)}")
    print(f"  부채비율: 66.7%")

    # === 3. Parser: 재무제표 파싱 ===
    print_subsection("Step 3: 재무제표 파싱")

    import time
    start_time = time.time()

    financial_data = parser.parse(dart_response)
    is_valid, errors = parser.validate(financial_data)

    parse_time = time.time() - start_time

    print(f"  ✓ {len(financial_data)}개 계정과목 파싱 완료 ({parse_time:.3f}초)")
    print(f"  ✓ 검증 결과: {'통과' if is_valid else '실패'}")

    if not is_valid:
        for error in errors:
            print(f"    ⚠ {error}")

    # === 4. Feature Generator: 특성 생성 ===
    print_subsection("Step 4: 도메인 특성 생성 (65개)")

    start_time = time.time()

    features_df = feature_generator.generate_all_features(
        financial_data,
        company_info
    )

    feature_time = time.time() - start_time

    print(f"  ✓ {len(features_df.columns)}개 특성 생성 완료 ({feature_time:.3f}초)")
    print(f"\n  주요 특성:")
    print(f"    • 유동비율: {features_df['유동비율'].iloc[0]:.2f}")
    print(f"    • 부채비율: {features_df['부채비율'].iloc[0]:.1f}%")
    print(f"    • 이자보상배율: {features_df['이자보상배율'].iloc[0]:.1f}")
    print(f"    • 영업이익률: {features_df['영업이익률'].iloc[0]:.2%}")
    print(f"    • 종합부도위험스코어: {features_df['종합부도위험스코어'].iloc[0]:.3f}")

    # === 5. Predictor: 부도 확률 예측 ===
    print_subsection("Step 5: 부도 확률 예측")

    start_time = time.time()

    prediction_result = predictor.predict(features_df)

    predict_time = time.time() - start_time

    print(f"  ✓ 예측 완료 ({predict_time:.3f}초)")
    print(f"\n  📊 예측 결과:")
    print(f"    • 부도 확률: {prediction_result['bankruptcy_probability']:.1%}")
    print(f"    • 위험 등급: {prediction_result['risk_level']} {prediction_result.get('risk_icon', '')}")
    print(f"    • 신뢰도: {prediction_result['confidence']:.1%}")
    print(f"    • 사용 모델: {prediction_result['model_info']['model_type']}")

    # === 6. Risk Analysis: 위험 요인 분석 ===
    print_subsection("Step 6: 위험 요인 분석")

    critical_risks = identify_critical_risks(features_df)
    warnings = identify_warnings(features_df)

    print(f"  Critical 위험: {len(critical_risks)}개")
    if critical_risks:
        for risk in critical_risks:
            print(f"    🔴 {risk['name']}: {risk['explanation']}")
    else:
        print(f"    ✅ Critical 위험 없음")

    print(f"\n  Warning 경고: {len(warnings)}개")
    if warnings:
        for warning in warnings[:3]:
            print(f"    🟡 {warning['name']}: {warning['explanation']}")
    else:
        print(f"    ✅ Warning 경고 없음")

    # === 7. Recommendations: 개선 권장사항 ===
    print_subsection("Step 7: 개선 권장사항")

    recommendations = generate_recommendations(features_df, financial_data)

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n  권장사항 {i}: {rec['title']}")
            print(f"    우선순위: {rec['priority']}")
            print(f"    현재 상태: {rec['current_status']}")
            print(f"    예상 효과: {rec['expected_impact']}")
    else:
        print(f"  ✅ 개선이 필요한 사항 없음")

    # === 8. 전체 실행 시간 ===
    total_time = parse_time + feature_time + predict_time
    print_subsection("전체 파이프라인 실행 시간")
    print(f"  Parser: {parse_time:.3f}초")
    print(f"  Feature Generator: {feature_time:.3f}초")
    print(f"  Predictor: {predict_time:.3f}초")
    print(f"  ──────────────────")
    print(f"  합계: {total_time:.3f}초")

    return prediction_result


def simulate_distressed_company():
    """위기 기업 시뮬레이션"""
    print_section_header("시뮬레이션 2: 위기 기업 부도 예측")

    # 컴포넌트 초기화
    parser = FinancialStatementParser(unit_conversion=1_000_000)
    feature_generator = DomainFeatureGenerator()

    model_path = project_root / 'data' / 'processed' / 'best_model_XGBoost.pkl'
    scaler_path = project_root / 'data' / 'processed' / 'scaler.pkl'

    predictor = BankruptcyPredictor(
        model_path=model_path if model_path.exists() else None,
        scaler_path=scaler_path if scaler_path.exists() else None
    )
    predictor.load_model()

    # 위기 기업 재무제표
    dart_response = {
        'balance_sheet': {
            '자산총계': 500_000_000_000,
            '부채총계': 450_000_000_000,
            '자본총계': 50_000_000_000,
            '유동자산': 150_000_000_000,
            '유동부채': 300_000_000_000,
            '비유동자산': 350_000_000_000,
            '비유동부채': 150_000_000_000,
            '현금및현금성자산': 20_000_000_000,
            '단기금융상품': 10_000_000_000,
            '매출채권': 60_000_000_000,
            '재고자산': 50_000_000_000,
            '유형자산': 300_000_000_000,
            '무형자산': 20_000_000_000,
            '단기차입금': 150_000_000_000,
            '장기차입금': 100_000_000_000,
            '매입채무': 100_000_000_000,
        },
        'income_statement': {
            '매출액': 800_000_000_000,
            '매출원가': 600_000_000_000,
            '매출총이익': 200_000_000_000,
            '판매비와관리비': 250_000_000_000,
            '영업이익': -50_000_000_000,
            '이자비용': 30_000_000_000,
            '당기순이익': -84_000_000_000,
        },
        'cash_flow': {
            '영업활동현금흐름': -20_000_000_000,
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
        '연체여부': True,
        '세금체납액': 5_000,
        '신용등급': 'BB',
        '대표이사변경': True,
        '배당금': 0
    }

    print(f"\n  기업명: {dart_response['metadata']['corp_name']}")
    print(f"  부채비율: 900%")
    print(f"  유동비율: 0.5")
    print(f"  영업손실: {format_korean_number(50_000)}")

    # 전체 파이프라인 실행
    financial_data = parser.parse(dart_response)
    features_df = feature_generator.generate_all_features(financial_data, company_info)
    prediction_result = predictor.predict(features_df)
    critical_risks = identify_critical_risks(features_df)
    warnings = identify_warnings(features_df)
    recommendations = generate_recommendations(features_df, financial_data)

    # 결과 출력
    print_subsection("예측 결과")
    print(f"  📊 부도 확률: {prediction_result['bankruptcy_probability']:.1%}")
    print(f"  📊 위험 등급: {prediction_result['risk_level']} {prediction_result.get('risk_icon', '')}")

    print_subsection("Critical 위험 요인")
    for risk in critical_risks:
        print(f"  🔴 {risk['name']}")
        print(f"     현재값: {risk['value']:.2f} (기준: {risk['threshold']})")
        print(f"     설명: {risk['explanation']}\n")

    print_subsection("개선 권장사항")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n  {i}. {rec['title']} (우선순위: {rec['priority']})")
        print(f"     {rec['problem']}")
        print(f"     예상 효과: {rec['expected_impact']}")

    return prediction_result


def main():
    """메인 실행 함수"""
    print("\n")
    print("█"*80)
    print("█                                                                              █")
    print("█         한국 기업 부도 예측 시스템 - 전체 파이프라인 시뮬레이션             █")
    print("█                                                                              █")
    print("█"*80)

    try:
        # 1. 정상 기업 시뮬레이션
        result1 = simulate_healthy_company()

        # 2. 위기 기업 시뮬레이션
        result2 = simulate_distressed_company()

        # 3. 요약
        print_section_header("시뮬레이션 요약")
        print(f"\n  ✅ 정상 기업: 부도 확률 {result1['bankruptcy_probability']:.1%}, 등급 {result1['risk_level']}")
        print(f"  ✅ 위기 기업: 부도 확률 {result2['bankruptcy_probability']:.1%}, 등급 {result2['risk_level']}")

        print("\n  🎯 검증:")
        if result1['bankruptcy_probability'] < 0.4:
            print(f"  ✅ 정상 기업 예측 정확 (부도 확률 < 40%)")
        else:
            print(f"  ❌ 정상 기업 예측 부정확 (부도 확률 {result1['bankruptcy_probability']:.1%})")

        if result2['bankruptcy_probability'] > 0.5:
            print(f"  ✅ 위기 기업 예측 정확 (부도 확률 > 50%)")
        else:
            print(f"  ❌ 위기 기업 예측 부정확 (부도 확률 {result2['bankruptcy_probability']:.1%})")

        print("\n")
        print("█"*80)
        print("█                                                                              █")
        print("█                     ✅ 전체 파이프라인 시뮬레이션 성공                       █")
        print("█                                                                              █")
        print("█"*80)
        print("\n")

        return 0

    except Exception as e:
        print("\n")
        print("█"*80)
        print("█                                                                              █")
        print("█                     ❌ 시뮬레이션 실패                                       █")
        print("█                                                                              █")
        print("█"*80)
        print(f"\n에러: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
