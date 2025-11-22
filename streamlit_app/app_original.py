"""
🚨 한국 기업 부도 위험 예측 시스템

도메인 지식 기반 AI 모델로 기업의 부도 위험을 실시간 평가합니다.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import joblib
import os

# 페이지 설정
st.set_page_config(
    page_title="부도위험 예측 시스템",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 한글 폰트 설정
import matplotlib.pyplot as plt
import platform
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)


@st.cache_resource
def load_models():
    """모델 및 스케일러 로딩"""
    try:
        model_dir = '../data/processed/'

        # 베스트 모델 로딩
        model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_')]
        if model_files:
            model_path = os.path.join(model_dir, model_files[0])
            model = joblib.load(model_path)
        else:
            st.error("모델 파일을 찾을 수 없습니다.")
            return None, None, None

        # 스케일러 로딩
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        scaler = joblib.load(scaler_path)

        # 특성 메타데이터
        features_path = os.path.join(model_dir, 'selected_features.csv')
        df_sample = pd.read_csv(features_path, nrows=1)
        feature_names = [col for col in df_sample.columns if col != '모형개발용Performance(향후1년내부도여부)']

        return model, scaler, feature_names
    except Exception as e:
        st.error(f"모델 로딩 오류: {str(e)}")
        return None, None, None


def create_risk_gauge(risk_score):
    """리스크 스코어 게이지 차트"""
    risk_percent = risk_score * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "부도 위험도", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkred" if risk_percent > 70 else "orange" if risk_percent > 40 else "lightblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#90EE90'},
                {'range': [30, 60], 'color': '#FFD700'},
                {'range': [60, 100], 'color': '#FF6B6B'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_percent
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def get_risk_level(risk_score):
    """위험 등급 반환"""
    if risk_score < 0.3:
        return "안전", "🟢", "부도 위험이 낮습니다"
    elif risk_score < 0.6:
        return "주의", "🟡", "일부 재무 지표 개선 필요"
    elif risk_score < 0.8:
        return "경고", "🟠", "부도 위험이 높습니다"
    else:
        return "위험", "🔴", "즉시 조치가 필요합니다"


def create_feature_importance_plot(importances, feature_names, top_n=10):
    """특성 중요도 차트"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    fig = go.Figure(go.Bar(
        y=importance_df['feature'].values[::-1],
        x=importance_df['importance'].values[::-1],
        orientation='h',
        marker_color='lightcoral'
    ))

    fig.update_layout(
        title=f'주요 위험 요인 (상위 {top_n}개)',
        xaxis_title='중요도',
        yaxis_title='특성',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


# 메인 앱
def main():
    st.title("🚨 한국 기업 부도 위험 예측 시스템")
    st.markdown("---")

    # 모델 로딩
    model, scaler, feature_names = load_models()

    if model is None:
        st.error("⚠️ 모델을 로딩할 수 없습니다. 노트북을 먼저 실행해주세요.")
        st.stop()

    # 사이드바 - 데이터 입력 방식 선택
    st.sidebar.title("📊 데이터 입력")
    input_method = st.sidebar.radio(
        "입력 방식 선택",
        ["수동 입력", "CSV 업로드", "샘플 데이터"]
    )

    # 입력 데이터
    input_data = None

    if input_method == "수동 입력":
        st.sidebar.markdown("### 주요 재무 지표 입력")

        with st.sidebar.expander("📈 재무상태표", expanded=True):
            유동자산 = st.number_input("유동자산 (백만원)", value=1000, step=100)
            유동부채 = st.number_input("유동부채 (백만원)", value=500, step=100)
            자산총계 = st.number_input("자산총계 (백만원)", value=2000, step=100)
            부채총계 = st.number_input("부채총계 (백만원)", value=1000, step=100)

        with st.sidebar.expander("💰 손익계산서"):
            매출액 = st.number_input("매출액 (백만원)", value=3000, step=100)
            영업이익 = st.number_input("영업이익 (백만원)", value=200, step=10)
            당기순이익 = st.number_input("당기순이익 (백만원)", value=150, step=10)

        with st.sidebar.expander("💵 현금흐름표"):
            영업현금흐름 = st.number_input("영업활동현금흐름 (백만원)", value=180, step=10)

        # 간단한 특성 계산 (실제로는 모든 특성 필요)
        자본총계 = 자산총계 - 부채총계

        # 주요 비율 계산
        유동비율 = 유동자산 / (유동부채 + 1) if 유동부채 > 0 else 0
        부채비율 = 부채총계 / (자본총계 + 1) if 자본총계 > 0 else 0
        ROA = 당기순이익 / (자산총계 + 1) if 자산총계 > 0 else 0

        # 더미 데이터 생성 (실제로는 모든 특성 필요)
        input_data = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

        # 계산된 값 일부 채우기
        if '유동비율' in feature_names:
            input_data.loc[0, '유동비율'] = 유동비율
        if '부채비율' in feature_names:
            input_data.loc[0, '부채비율'] = 부채비율
        if 'ROA' in feature_names:
            input_data.loc[0, 'ROA'] = ROA

    elif input_method == "CSV 업로드":
        uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=['csv'])
        if uploaded_file:
            input_data = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ {len(input_data)}개 기업 데이터 로딩 완료")

    else:  # 샘플 데이터
        try:
            sample_path = '../data/processed/selected_features.csv'
            sample_data = pd.read_csv(sample_path, nrows=5)
            target_col = '모형개발용Performance(향후1년내부도여부)'
            if target_col in sample_data.columns:
                input_data = sample_data.drop(columns=[target_col])
            else:
                input_data = sample_data
            st.sidebar.success("✅ 샘플 데이터 5개 로딩 완료")
        except:
            st.sidebar.warning("샘플 데이터를 찾을 수 없습니다.")

    # 예측 실행
    if input_data is not None and st.sidebar.button("🔍 부도 위험 분석", type="primary"):
        try:
            # 결측치 처리
            input_filled = input_data.fillna(input_data.median())
            input_filled = input_filled.replace([np.inf, -np.inf], 0)

            # 스케일링
            input_scaled = scaler.transform(input_filled)

            # 예측
            risk_proba = model.predict_proba(input_scaled)[:, 1]

            # 단일 기업 분석
            if len(input_data) == 1:
                risk_score = risk_proba[0]
                risk_level, risk_icon, risk_msg = get_risk_level(risk_score)

                # 대시보드 레이아웃
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    st.markdown(f"### {risk_icon} 위험 등급: **{risk_level}**")
                    st.metric("부도 확률", f"{risk_score*100:.1f}%",
                             delta=f"{(risk_score-0.5)*100:.1f}%p")
                    st.info(risk_msg)

                with col2:
                    fig_gauge = create_risk_gauge(risk_score)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col3:
                    st.markdown("### 📊 종합 평가")
                    st.metric("유동비율", f"{유동비율:.2f}",
                             delta="정상" if 유동비율 > 1 else "경고",
                             delta_color="normal" if 유동비율 > 1 else "inverse")
                    st.metric("부채비율", f"{부채비율:.0f}%",
                             delta="정상" if 부채비율 < 200 else "경고",
                             delta_color="normal" if 부채비율 < 200 else "inverse")

                # 상세 분석
                st.markdown("---")
                st.markdown("## 🔍 상세 위험 분석")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 💡 주요 재무 지표")
                    metrics_df = pd.DataFrame({
                        '지표': ['유동비율', '부채비율', 'ROA', '매출액'],
                        '값': [f"{유동비율:.2f}", f"{부채비율:.0f}%", f"{ROA*100:.2f}%", f"{매출액:,.0f}M"],
                        '상태': [
                            "✅" if 유동비율 > 1 else "⚠️",
                            "✅" if 부채비율 < 200 else "⚠️",
                            "✅" if ROA > 0 else "⚠️",
                            "✅"
                        ]
                    })
                    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

                with col2:
                    st.markdown("### 🎯 개선 권장사항")
                    recommendations = []

                    if 유동비율 < 1:
                        recommendations.append("⚠️ 단기 유동성 개선 필요")
                    if 부채비율 > 200:
                        recommendations.append("⚠️ 부채 비율 축소 권장")
                    if ROA < 0:
                        recommendations.append("⚠️ 수익성 개선 시급")
                    if 영업현금흐름 < 0:
                        recommendations.append("⚠️ 현금흐름 관리 필요")

                    if not recommendations:
                        st.success("✅ 전반적으로 양호한 재무 상태입니다")
                    else:
                        for rec in recommendations:
                            st.warning(rec)

            # 다중 기업 분석
            else:
                st.markdown("## 📊 다중 기업 부도 위험 분석")

                results_df = pd.DataFrame({
                    '기업 ID': range(1, len(risk_proba) + 1),
                    '부도 확률': risk_proba,
                    '위험 등급': [get_risk_level(p)[0] for p in risk_proba]
                })

                # 위험 분포
                col1, col2 = st.columns(2)

                with col1:
                    fig_hist = px.histogram(
                        results_df, x='부도 확률',
                        nbins=20,
                        title='부도 확률 분포',
                        color_discrete_sequence=['lightcoral']
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                with col2:
                    risk_counts = results_df['위험 등급'].value_counts()
                    fig_pie = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title='위험 등급 분포',
                        color_discrete_sequence=px.colors.sequential.RdYlGn_r
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                # 결과 테이블
                st.markdown("### 🏢 기업별 분석 결과")
                st.dataframe(
                    results_df.sort_values('부도 확률', ascending=False),
                    hide_index=True,
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"⚠️ 예측 중 오류 발생: {str(e)}")
            st.exception(e)

    # 푸터
    st.markdown("---")
    st.markdown("""
    ### 📌 사용 안내
    - **수동 입력**: 주요 재무 지표를 직접 입력하여 분석
    - **CSV 업로드**: 여러 기업의 데이터를 한번에 분석
    - **샘플 데이터**: 테스트용 샘플 데이터로 기능 확인

    ### ⚠️ 주의사항
    - 이 시스템은 참고용이며, 최종 의사결정은 전문가와 상담 필요
    - 모델은 역사적 데이터 기반으로 학습되었습니다
    - 정기적인 모델 업데이트가 필요합니다

    ---
    🤖 Powered by AI | 도메인 지식 기반 부도 예측 모델
    """)


if __name__ == "__main__":
    main()
