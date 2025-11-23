"""
한국 기업 부도 예측 시스템 - Streamlit 앱

DART API 연동 및 실시간 부도 위험 분석
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import *
from src.dart_api import DartAPIClient, FinancialStatementParser
from src.domain_features import DomainFeatureGenerator
from src.models import BankruptcyPredictor
from src.visualization.charts import create_risk_gauge, create_shap_waterfall, create_shap_waterfall_real, create_radar_chart
from src.utils.helpers import (
    get_risk_level, format_korean_number,
    identify_critical_risks, identify_warnings, generate_recommendations
)
from src.utils.business_value import BusinessValueCalculator
import numpy as np

# 페이지 설정
st.set_page_config(**PAGE_CONFIG)

# 한글 폰트 설정
import matplotlib.pyplot as plt
plt.rc('font', family=KOREAN_FONT)
plt.rc('axes', unicode_minus=False)


# ========== 캐시된 리소스 ==========

@st.cache_resource
def load_predictor():
    """모델 로딩 (캐시)"""
    predictor = BankruptcyPredictor(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH
    )
    predictor.load_model()
    return predictor


@st.cache_data(ttl=3600)
def fetch_dart_data(company_name: str, year: str):
    """DART API 데이터 조회 (1시간 캐시)"""
    if not DART_API_KEY:
        st.error("❌ DART API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return None, None

    try:
        client = DartAPIClient(DART_API_KEY)

        # 기업 검색
        with st.spinner(f"'{company_name}' 검색 중..."):
            company = client.search_company(company_name)

        st.success(f"✓ {company['corp_name']} ({company['stock_code']}) 검색 완료")

        # 재무제표 조회
        with st.spinner(f"{year}년 재무제표 조회 중..."):
            statements = client.get_financial_statements(
                corp_code=company['corp_code'],
                bsns_year=year
            )

        st.success(f"✓ {year}년 재무제표 조회 완료")

        return company, statements

    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
        return None, None


# ========== 메인 앱 ==========

def main():
    """메인 앱"""

    # 헤더
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown("---")

    # 사이드바 - 입력 방식 선택
    st.sidebar.header("📋 입력 방식 선택")

    input_method = st.sidebar.radio(
        "데이터 입력 방법",
        [
            "🔍 DART API 검색 (상장기업)",
            "📝 재무제표 직접 입력",
            "📂 샘플 데이터 사용"
        ]
    )

    # 변수 초기화
    company_info = None
    financial_data = None
    company_name = None
    year = None

    # ===== 입력 모드 1: DART API 검색 =====
    if input_method == "🔍 DART API 검색 (상장기업)":
        st.header("🔍 DART API 기업 검색")

        col1, col2 = st.columns([3, 1])

        with col1:
            company_name = st.text_input(
                "기업명 또는 종목코드",
                value="삼성전자",
                help="예: 삼성전자, SK하이닉스, 005930"
            )

        with col2:
            # 동적으로 회계연도 생성 (현재 년도부터 과거 5년)
            from datetime import datetime
            current_year = datetime.now().year
            year_options = [str(current_year - i) for i in range(6)]  # 2024, 2023, 2022, 2021, 2020, 2019

            year = st.selectbox(
                "회계연도",
                options=year_options,
                index=0
            )

        if st.button("🚀 조회 및 분석 시작", type="primary"):
            # DART API 조회
            company, statements = fetch_dart_data(company_name, year)

            if company and statements:
                # 파싱
                parser = FinancialStatementParser()
                financial_data = parser.parse(statements)

                company_info = {
                    'corp_name': company['corp_name'],
                    'stock_code': company['stock_code'],
                    'year': year
                }

                # 분석 실행
                run_analysis(financial_data, company_info)

    # ===== 입력 모드 2: 직접 입력 =====
    elif input_method == "📝 재무제표 직접 입력":
        st.header("📝 재무제표 직접 입력")

        st.info("주요 재무 항목을 입력하세요 (단위: 백만원)")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("재무상태표")
            자산총계 = st.number_input("자산총계 (백만원)", value=1_000_000, step=10_000)
            부채총계 = st.number_input("부채총계 (백만원)", value=600_000, step=10_000)
            자본총계 = st.number_input("자본총계 (백만원)", value=400_000, step=10_000)
            유동자산 = st.number_input("유동자산 (백만원)", value=500_000, step=10_000)
            유동부채 = st.number_input("유동부채 (백만원)", value=300_000, step=10_000)
            현금 = st.number_input("현금및현금성자산 (백만원)", value=100_000, step=10_000)

        with col2:
            st.subheader("손익계산서")
            매출액 = st.number_input("매출액 (백만원)", value=2_000_000, step=10_000)
            매출원가 = st.number_input("매출원가 (백만원)", value=1_200_000, step=10_000)
            영업이익 = st.number_input("영업이익 (백만원)", value=200_000, step=10_000)
            당기순이익 = st.number_input("당기순이익 (백만원)", value=150_000, step=10_000)
            이자비용 = st.number_input("이자비용 (백만원)", value=20_000, step=1_000)
            영업활동현금흐름 = st.number_input("영업활동현금흐름 (백만원)", value=180_000, step=10_000)

        if st.button("🚀 분석 시작", type="primary"):
            financial_data = {
                '자산총계': 자산총계,
                '부채총계': 부채총계,
                '자본총계': 자본총계,
                '유동자산': 유동자산,
                '비유동자산': 자산총계 - 유동자산,
                '유동부채': 유동부채,
                '비유동부채': 부채총계 - 유동부채,
                '현금및현금성자산': 현금,
                '매출액': 매출액,
                '매출원가': 매출원가,
                '매출총이익': 매출액 - 매출원가,
                '영업이익': 영업이익,
                '당기순이익': 당기순이익,
                '이자비용': 이자비용,
                '영업활동현금흐름': 영업활동현금흐름,
                # 기타 기본값
                '단기금융상품': 0,
                '매출채권': 유동자산 * 0.2,
                '재고자산': 유동자산 * 0.1,
                '유형자산': (자산총계 - 유동자산) * 0.6,
                '무형자산': (자산총계 - 유동자산) * 0.1,
                '단기차입금': 유동부채 * 0.3,
                '장기차입금': (부채총계 - 유동부채) * 0.5,
                '판매비와관리비': 매출액 * 0.2,
                '매입채무': 유동부채 * 0.2,
            }

            company_info = {
                'corp_name': '직접입력 기업',
                'year': '2023'
            }

            run_analysis(financial_data, company_info)

    # ===== 입력 모드 3: 샘플 데이터 =====
    else:
        st.header("📂 샘플 데이터")

        st.info("샘플 기업 데이터로 시스템을 테스트해보세요.")

        sample_type = st.selectbox(
            "샘플 유형 선택",
            [
                "정상 기업 (부도 위험 낮음)",
                "주의 기업 (일부 위험 요소)",
                "위험 기업 (부도 위험 높음)"
            ]
        )

        if st.button("📊 샘플 분석", type="primary"):
            if "정상" in sample_type:
                financial_data = create_sample_data("normal")
                company_info = {'corp_name': '정상 샘플 기업', 'year': '2023'}
            elif "주의" in sample_type:
                financial_data = create_sample_data("caution")
                company_info = {'corp_name': '주의 샘플 기업', 'year': '2023'}
            else:
                financial_data = create_sample_data("risk")
                company_info = {'corp_name': '위험 샘플 기업', 'year': '2023'}

            run_analysis(financial_data, company_info)


def run_analysis(financial_data: dict, company_info: dict):
    """
    분석 실행 및 결과 표시

    Args:
        financial_data: 재무제표 데이터
        company_info: 기업 정보
    """
    st.markdown("---")
    st.header(f"📊 분석 결과: {company_info.get('corp_name', '기업')}")

    # 프로그레스 바 초기화
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 1. 특성 생성 (0% → 40%)
    status_text.text("🔄 1/3 단계: 도메인 특성 생성 중...")
    progress_bar.progress(10)

    generator = DomainFeatureGenerator()
    features_df = generator.generate_all_features(financial_data)
    progress_bar.progress(40)
    status_text.text(f"✓ 특성 생성 완료 ({len(features_df.columns)}개)")

    # 2. 예측 (40% → 70%)
    status_text.text("🔄 2/3 단계: 부도 위험 예측 중...")
    progress_bar.progress(50)

    predictor = load_predictor()
    result = predictor.predict(features_df)
    progress_bar.progress(70)
    status_text.text("✓ 예측 완료")

    # 3. 분석 준비 (70% → 100%)
    status_text.text("🔄 3/3 단계: 분석 결과 준비 중...")
    progress_bar.progress(85)

    # 잠시 대기 (UX 개선)
    import time
    time.sleep(0.3)

    progress_bar.progress(100)
    status_text.text("✅ 모든 분석 완료!")

    # 프로그레스 바 제거
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    # 완료 메시지
    st.success(f"✓ 분석 완료: {len(features_df.columns)}개 특성 생성, 부도 확률 {result['bankruptcy_probability']*100:.2f}%")

    # ========== 섹션 1: 종합 평가 ==========
    display_overall_assessment(result, features_df, financial_data)

    # ========== 섹션 2: 위험 요인 분석 ==========
    display_risk_analysis(result, features_df)

    # ========== 섹션 3: 비즈니스 가치 분석 ==========
    display_business_value(result)

    # ========== 섹션 4: 개선 권장사항 ==========
    display_recommendations(features_df, financial_data)

    # ========== 섹션 5: 상세 특성 ==========
    display_detailed_features(features_df)

    # ========== 섹션 6: 재무제표 원본 ==========
    display_financial_statements(financial_data)


def display_overall_assessment(result: dict, features_df: pd.DataFrame, financial_data: dict):
    """섹션 1: 종합 평가"""
    st.markdown("## 🎯 종합 부도 위험 평가")

    risk_prob = result['bankruptcy_probability']

    # 대형 Traffic Light 인디케이터
    if risk_prob < 0.0168:  # 안전 (< 1.68%)
        light_color = "#4caf50"  # 초록
        light_icon = "🟢"
        light_label = "안전"
        light_desc = f"부도 확률 {risk_prob*100:.2f}% (기준: < 1.68%)"
    elif risk_prob < 0.0468:  # 주의 (1.68% ~ 4.68%)
        light_color = "#ffeb3b"  # 노랑
        light_icon = "🟡"
        light_label = "주의"
        light_desc = f"부도 확률 {risk_prob*100:.2f}% (기준: 1.68% ~ 4.68%)"
    else:  # 위험 (> 4.68%)
        light_color = "#f44336"  # 빨강
        light_icon = "🔴"
        light_label = "위험"
        light_desc = f"부도 확률 {risk_prob*100:.2f}% (기준: > 4.68%)"

    # 대형 신호등 HTML
    st.markdown(f"""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px;">
        <div style="font-size: 100px; margin-bottom: 10px;">{light_icon}</div>
        <h1 style="color: white; margin: 10px 0; font-size: 48px;">{light_label}</h1>
        <p style="color: white; font-size: 20px; margin: 5px 0;">{light_desc}</p>
        <p style="color: rgba(255,255,255,0.9); font-size: 16px; margin-top: 15px;">{result['risk_message']}</p>
    </div>
    """, unsafe_allow_html=True)

    # 핵심 지표 (4개 컬럼)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="부도 확률",
            value=f"{risk_prob*100:.2f}%",
            delta=f"{result['risk_level']}"
        )

    with col2:
        건전성지수 = features_df.get('재무건전성지수', pd.Series([50])).iloc[0]
        delta_건전성 = "양호" if 건전성지수 >= 60 else "취약"
        st.metric(
            label="재무 건전성",
            value=f"{건전성지수:.0f}점",
            delta=delta_건전성
        )

    with col3:
        경보신호수 = int(features_df.get('조기경보신호수', pd.Series([0])).iloc[0])
        delta_경보 = "정상" if 경보신호수 == 0 else f"{경보신호수}개 감지"
        st.metric(
            label="조기경보신호",
            value=f"{경보신호수}개",
            delta=delta_경보,
            delta_color="inverse" if 경보신호수 > 0 else "normal"
        )

    with col4:
        종합위험스코어 = features_df.get('종합부도위험스코어', pd.Series([50])).iloc[0]
        delta_위험 = "낮음" if 종합위험스코어 < 30 else ("보통" if 종합위험스코어 < 60 else "높음")
        st.metric(
            label="종합위험스코어",
            value=f"{종합위험스코어:.0f}점",
            delta=delta_위험,
            delta_color="inverse" if 종합위험스코어 >= 60 else "normal"
        )

    # 게이지 차트
    st.plotly_chart(create_risk_gauge(risk_prob), use_container_width=True)


def display_risk_analysis(result: dict, features_df: pd.DataFrame):
    """섹션 2: 위험 요인 분석"""
    st.markdown("---")
    st.markdown("## 🔍 위험 요인 상세 분석")

    critical_risks = identify_critical_risks(features_df)
    warnings = identify_warnings(features_df)

    # Critical 리스크 (상단 전체 너비로 강조)
    st.markdown("### 🚨 Critical 리스크 (즉시 조치 필요)")

    if critical_risks:
        for risk in critical_risks:
            st.markdown(f"""
            <div style="background: #ffebee; border-left: 5px solid #f44336; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 24px; margin-right: 10px;">🚨</span>
                    <h4 style="color: #c62828; margin: 0; font-size: 20px;">{risk['name']}</h4>
                </div>
                <div style="margin: 10px 0; padding: 10px; background: white; border-radius: 5px;">
                    <p style="margin: 5px 0; font-size: 16px;"><strong>현재값:</strong> <span style="color: #f44336; font-size: 18px; font-weight: bold;">{risk['value']:.2f}</span></p>
                    <p style="margin: 5px 0; font-size: 16px;"><strong>위험 기준:</strong> {risk['threshold']:.2f}</p>
                </div>
                <p style="color: #555; font-size: 15px; margin: 10px 0; line-height: 1.5;">💡 {risk['explanation']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #e8f5e9; border-left: 5px solid #4caf50; padding: 15px; margin: 15px 0; border-radius: 8px;">
            <p style="color: #2e7d32; font-size: 16px; margin: 0;">✅ Critical 리스크가 발견되지 않았습니다.</p>
        </div>
        """, unsafe_allow_html=True)

    # Warning (2 컬럼 그리드)
    st.markdown("### ⚠️ Warning (개선 권장)")

    if warnings:
        # 2개씩 묶어서 표시
        for i in range(0, len(warnings), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(warnings):
                    warning = warnings[i + j]
                    with col:
                        st.markdown(f"""
                        <div style="background: #fffde7; border-left: 4px solid #ffeb3b; padding: 15px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                <span style="font-size: 20px; margin-right: 8px;">⚠️</span>
                                <h5 style="color: #f57f17; margin: 0; font-size: 16px;">{warning['name']}</h5>
                            </div>
                            <p style="margin: 5px 0; font-size: 14px;"><strong>현재:</strong> <span style="color: #f57f17;">{warning['value']:.2f}</span></p>
                            <p style="margin: 5px 0; font-size: 14px;"><strong>권장:</strong> {warning['threshold']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #e8f5e9; border-left: 4px solid #4caf50; padding: 15px; margin: 15px 0; border-radius: 8px;">
            <p style="color: #2e7d32; font-size: 16px; margin: 0;">✅ 모든 지표가 권장 범위 내에 있습니다.</p>
        </div>
        """, unsafe_allow_html=True)

    # SHAP Waterfall 차트
    st.markdown("---")
    st.markdown("### 📊 주요 위험 요인 기여도 (SHAP 분석)")
    st.caption("각 특성이 부도 확률에 미치는 영향을 분석합니다. 빨간색은 위험 증가, 파란색은 위험 감소를 의미합니다.")

    if result.get('shap_values'):
        # 실제 SHAP 값 사용
        fig_shap = create_shap_waterfall_real(
            shap_values=np.array(result['shap_values']),
            feature_values=features_df.iloc[0],
            feature_names=result['feature_names'],
            base_value=result['shap_base_value']
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # 범례 추가
        st.markdown("""
        <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin-top: 10px;">
            <p style="margin: 5px 0;"><span style="color: #f44336;">■</span> <strong>빨간색:</strong> 부도 위험을 증가시키는 요인</p>
            <p style="margin: 5px 0;"><span style="color: #2196F3;">■</span> <strong>파란색:</strong> 부도 위험을 감소시키는 요인 (보호 요인)</p>
            <p style="margin: 5px 0;"><strong>Base Value:</strong> 모든 기업의 평균 부도 확률</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # SHAP 값 없으면 간소화 버전 사용
        fig_shap = create_shap_waterfall(features_df.iloc[0])
        st.plotly_chart(fig_shap, use_container_width=True)
        st.info("ℹ️ 모델 로드 실패로 간소화된 분석을 표시합니다.")


def display_business_value(result: dict):
    """섹션 3: 비즈니스 가치 분석"""
    st.markdown("---")
    st.markdown("## 💰 비즈니스 가치 분석")

    # 인터랙티브 파라미터 조정
    st.markdown("### 🎛️ 대출 조건 설정")

    col1, col2 = st.columns(2)

    with col1:
        avg_loan = st.slider(
            "평균 대출 금액 (백만원)",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
            help="기업당 평균 대출 금액을 설정하세요"
        ) * 1_000_000

    with col2:
        avg_interest = st.slider(
            "평균 이자 수익 (백만원)",
            min_value=0.1,
            max_value=10.0,
            value=0.5,
            step=0.1,
            help="대출 건당 예상 이자 수익을 설정하세요"
        ) * 1_000_000

    # 실시간 계산
    calc = BusinessValueCalculator(avg_loan=avg_loan, avg_interest=avg_interest)
    value = calc.calculate_single_company(result['bankruptcy_probability'])

    # 결과 표시
    st.markdown("### 📈 예상 수익/손실")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "예상 손실",
            f"{value['expected_loss']:,.0f}원",
            delta=f"부도 확률 {result['bankruptcy_probability']*100:.2f}%",
            delta_color="inverse"
        )

    with col2:
        st.metric(
            "예상 수익",
            f"{value['expected_profit']:,.0f}원",
            delta=f"정상 확률 {(1-result['bankruptcy_probability'])*100:.2f}%",
            delta_color="normal"
        )

    with col3:
        delta_color = "normal" if value['net'] > 0 else "inverse"
        st.metric(
            "순 기대값",
            f"{value['net']:,.0f}원",
            delta="대출 승인 권장" if value['net'] > 0 else "대출 거절 권장",
            delta_color=delta_color
        )

    # 의사결정 권장사항
    if value['net'] > 0:
        st.success(f"✅ **의사결정:** 대출 승인 권장 (순 기대값: {value['net']:,.0f}원 > 0)")
    else:
        st.error(f"❌ **의사결정:** 대출 거절 권장 (순 기대값: {value['net']:,.0f}원 < 0)")

    # 모델 성능 통계
    st.markdown("---")
    st.markdown("### 📊 모델 성능 (Test Set 기준)")
    perf = calc.get_model_performance_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("ROI", perf['roi'], delta="투자 대비 수익률")

    with col2:
        st.metric("Payback Period", f"{perf['payback_months']}개월", delta="투자 회수 기간")

    with col3:
        st.metric("연간 절감액", perf['annual_savings_krw'], delta="비용 절감")

    with col4:
        st.metric("F2-Score", f"{perf['f2_score']:.2f}", delta="모델 정확도")

    # 해석 가이드
    with st.expander("💡 비즈니스 가치 해석 가이드"):
        st.markdown("""
        #### ROI (Return on Investment)
        - **920%**: 모델 도입 투자 대비 9배 이상의 수익 창출
        - 부도 기업을 사전에 감지하여 손실 방지

        #### Payback Period
        - **1.3개월**: 모델 투자 비용을 1.3개월 내 회수
        - 매우 빠른 투자 회수로 비즈니스 리스크 최소화

        #### 연간 절감액
        - **460M KRW**: 잘못된 대출 결정 방지로 연간 4.6억원 절감
        - Type II Error (부도 미탐지) 감소 효과

        #### F2-Score
        - 재현율(Recall)을 중시하는 평가 지표
        - 부도 기업을 놓치지 않는 것이 중요하므로 F2 사용
        """)

    # 시나리오 분석
    st.markdown("---")
    st.markdown("### 🔮 시나리오 분석")

    scenario_col1, scenario_col2 = st.columns(2)

    with scenario_col1:
        st.markdown("""
        <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3;">
            <h5 style="color: #1976d2; margin-top: 0;">✅ 승인 시 (Approve)</h5>
            <p><strong>정상 기업일 경우:</strong></p>
            <p style="margin-left: 20px;">→ 이자 수익: {avg_interest:,.0f}원</p>
            <p><strong>부도 기업일 경우:</strong></p>
            <p style="margin-left: 20px;">→ 손실: {avg_loan:,.0f}원</p>
        </div>
        """.format(avg_interest=avg_interest, avg_loan=avg_loan), unsafe_allow_html=True)

    with scenario_col2:
        st.markdown("""
        <div style="background: #fce4ec; padding: 15px; border-radius: 8px; border-left: 4px solid #e91e63;">
            <h5 style="color: #c2185b; margin-top: 0;">❌ 거절 시 (Reject)</h5>
            <p><strong>정상 기업일 경우:</strong></p>
            <p style="margin-left: 20px;">→ 기회 손실: 이자 수익 포기</p>
            <p><strong>부도 기업일 경우:</strong></p>
            <p style="margin-left: 20px;">→ 손실 방지: 대출금 회수 불능 방지</p>
        </div>
        """, unsafe_allow_html=True)


def display_recommendations(features_df: pd.DataFrame, financial_data: dict):
    """섹션 4: 개선 권장사항"""
    st.markdown("---")
    st.markdown("## 💡 실행 가능한 개선 권장사항")

    recommendations = generate_recommendations(features_df, financial_data)

    for i, rec in enumerate(recommendations, 1):
        with st.expander(
            f"권장사항 {i}: {rec['title']} (우선순위: {rec['priority']})",
            expanded=(i == 1)
        ):
            st.markdown(f"**현재 상태:**\n{rec['current_status']}")
            st.markdown(f"**문제점:**\n{rec['problem']}")
            st.markdown(f"**개선 방안:**{rec['solution']}")
            st.markdown(f"**예상 효과:**\n{rec['expected_impact']}")


def display_detailed_features(features_df: pd.DataFrame):
    """섹션 5: 상세 특성"""
    st.markdown("---")
    with st.expander("📋 생성된 특성 상세 보기"):
        st.markdown(f"총 {len(features_df.columns)}개 특성이 생성되었습니다.")

        # 카테고리별로 분류
        categories = {
            '유동성': [col for col in features_df.columns if any(kw in col for kw in ['유동', '현금', '운전자본'])],
            '지급불능': [col for col in features_df.columns if any(kw in col for kw in ['부채', '자본', '이자', '레버리지'])],
            '재무조작': [col for col in features_df.columns if any(kw in col for kw in ['발생액', '채권', '재고', '조작', '이익의질'])],
            '복합리스크': [col for col in features_df.columns if any(kw in col for kw in ['위험', '지수', '신호', '건전성'])]
        }

        for cat_name, cols in categories.items():
            if cols:
                st.markdown(f"**{cat_name} 특성 ({len(cols)}개)**")
                cat_df = features_df[cols].T
                cat_df.columns = ['값']
                st.dataframe(cat_df, use_container_width=True)


def display_financial_statements(financial_data: dict):
    """섹션 6: 재무제표 원본"""
    st.markdown("---")
    with st.expander("📋 재무제표 원본 데이터 보기"):
        # 재무상태표
        st.markdown("### 재무상태표")
        bs_data = {
            '항목': ['자산총계', '유동자산', '비유동자산', '부채총계', '유동부채', '비유동부채', '자본총계'],
            '금액 (백만원)': [
                financial_data.get('자산총계', 0),
                financial_data.get('유동자산', 0),
                financial_data.get('비유동자산', 0),
                financial_data.get('부채총계', 0),
                financial_data.get('유동부채', 0),
                financial_data.get('비유동부채', 0),
                financial_data.get('자본총계', 0)
            ]
        }
        st.dataframe(pd.DataFrame(bs_data), use_container_width=True)

        # 손익계산서
        st.markdown("### 손익계산서")
        is_data = {
            '항목': ['매출액', '매출원가', '매출총이익', '영업이익', '당기순이익'],
            '금액 (백만원)': [
                financial_data.get('매출액', 0),
                financial_data.get('매출원가', 0),
                financial_data.get('매출총이익', 0),
                financial_data.get('영업이익', 0),
                financial_data.get('당기순이익', 0)
            ]
        }
        st.dataframe(pd.DataFrame(is_data), use_container_width=True)


def create_sample_data(sample_type: str) -> dict:
    """샘플 데이터 생성"""
    if sample_type == "normal":
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
    elif sample_type == "caution":
        return {
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
    else:  # risk
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


# ========== 앱 실행 ==========

if __name__ == "__main__":
    main()
