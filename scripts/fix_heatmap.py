"""
히트맵 가독성 개선 스크립트
"""
import json
import os

# 개선된 히트맵 코드 (옵션 3가지)

# 옵션 1: 텍스트 크기 줄이고 그래프 크기 키우기
option1 = """# 업종별 유동성 히트맵 (개선 버전)
if len(available_liquidity) >= 2:
    # 정상기업 기준 업종별 유동성 지표
    normal_liquidity = df_industry[df_industry[target_col] == 0].groupby('대분류명')[available_liquidity].median()

    # 표준화 (0-1 스케일)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    normal_liquidity_scaled = pd.DataFrame(
        scaler.fit_transform(normal_liquidity),
        columns=normal_liquidity.columns,
        index=normal_liquidity.index
    )

    fig = go.Figure(data=go.Heatmap(
        z=normal_liquidity_scaled.T.values,
        x=normal_liquidity_scaled.index,
        y=normal_liquidity_scaled.columns,
        colorscale='RdYlGn',
        text=normal_liquidity.T.values,
        texttemplate='%{text:.0f}',  # 정수로 표시
        textfont={"size": 7},  # 폰트 크기 축소
        hovertemplate='%{y}<br>%{x}<br>값: %{text:.1f}<extra></extra>'  # hover에 상세 정보
    ))

    fig.update_layout(
        title='업종별 유동성 지표 히트맵 (정상기업 중앙값, 표준화)',
        xaxis_title='산업 대분류',
        yaxis_title='유동성 지표',
        height=600,  # 높이 증가
        width=1400,  # 너비 증가
        font=dict(family='Malgun Gothic', size=10),
        xaxis={'tickangle': -45, 'tickfont': {'size': 9}},
        yaxis={'tickfont': {'size': 9}}
    )

    fig.show()"""

# 옵션 2: 텍스트 제거하고 hover만 표시
option2 = """# 업종별 유동성 히트맵 (텍스트 제거 버전)
if len(available_liquidity) >= 2:
    # 정상기업 기준 업종별 유동성 지표
    normal_liquidity = df_industry[df_industry[target_col] == 0].groupby('대분류명')[available_liquidity].median()

    # 표준화 (0-1 스케일)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    normal_liquidity_scaled = pd.DataFrame(
        scaler.fit_transform(normal_liquidity),
        columns=normal_liquidity.columns,
        index=normal_liquidity.index
    )

    fig = go.Figure(data=go.Heatmap(
        z=normal_liquidity_scaled.T.values,
        x=normal_liquidity_scaled.index,
        y=normal_liquidity_scaled.columns,
        colorscale='RdYlGn',
        text=normal_liquidity.T.values,
        hovertemplate='<b>%{y}</b><br>%{x}<br>중앙값: %{text:,.1f}<br>표준화: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title='업종별 유동성 지표 히트맵 (정상기업 중앙값, 표준화)<br><sub>마우스를 올리면 상세 값을 확인할 수 있습니다</sub>',
        xaxis_title='산업 대분류',
        yaxis_title='유동성 지표',
        height=500,
        width=1200,
        font=dict(family='Malgun Gothic', size=11),
        xaxis={'tickangle': -45},
        margin=dict(l=150, r=50, t=100, b=150)  # 여백 조정
    )

    fig.show()"""

# 옵션 3: 조건부 텍스트 표시 (큰 값만)
option3 = """# 업종별 유동성 히트맵 (조건부 텍스트 표시)
if len(available_liquidity) >= 2:
    # 정상기업 기준 업종별 유동성 지표
    normal_liquidity = df_industry[df_industry[target_col] == 0].groupby('대분류명')[available_liquidity].median()

    # 표준화 (0-1 스케일)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    normal_liquidity_scaled = pd.DataFrame(
        scaler.fit_transform(normal_liquidity),
        columns=normal_liquidity.columns,
        index=normal_liquidity.index
    )

    # 큰 값만 텍스트로 표시 (상위 20%)
    text_values = normal_liquidity.T.values.copy()
    threshold = np.percentile(text_values, 80)
    text_display = np.where(text_values >= threshold,
                           np.round(text_values, 0).astype(int).astype(str),
                           '')

    fig = go.Figure(data=go.Heatmap(
        z=normal_liquidity_scaled.T.values,
        x=normal_liquidity_scaled.index,
        y=normal_liquidity_scaled.columns,
        colorscale='RdYlGn',
        text=text_display,
        texttemplate='%{text}',
        textfont={"size": 8},
        hovertemplate='<b>%{y}</b><br>%{x}<br>중앙값: %{customdata:,.1f}<extra></extra>',
        customdata=normal_liquidity.T.values
    ))

    fig.update_layout(
        title='업종별 유동성 지표 히트맵 (정상기업 중앙값, 표준화)<br><sub>상위 20% 값만 표시</sub>',
        xaxis_title='산업 대분류',
        yaxis_title='유동성 지표',
        height=550,
        width=1300,
        font=dict(family='Malgun Gothic', size=10),
        xaxis={'tickangle': -45},
        margin=dict(l=150)
    )

    fig.show()"""

print("=== 히트맵 개선 옵션 ===\n")
print("옵션 1: 텍스트 크기 줄이고 그래프 크기 키우기")
print("  - 모든 값을 정수로 표시")
print("  - 폰트 크기 7로 축소")
print("  - 그래프 크기 1400x600\n")

print("옵션 2: 텍스트 제거하고 hover만 표시 (권장)")
print("  - 셀에 숫자 표시 안 함")
print("  - 마우스 올리면 상세 정보 표시")
print("  - 깔끔한 비주얼\n")

print("옵션 3: 조건부 텍스트 표시")
print("  - 상위 20% 값만 표시")
print("  - 중요한 값만 강조\n")

# 사용자가 선택할 수 있도록
selected_option = option2  # 기본값: 옵션 2 (권장)

# 노트북 업데이트
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
notebook_path = os.path.join(base_dir, 'notebooks', '01_도메인_기반_부도원인_분석.ipynb')

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 히트맵 셀 찾아서 교체
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if '업종별 유동성 히트맵' in source and 'go.Heatmap' in source:
            cell['source'] = selected_option.split('\n')
            print(f"✅ 셀 {i} 업데이트 완료 (옵션 2 적용)")
            break

# 저장
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✅ 노트북 업데이트 완료: {notebook_path}")
print("\n💡 다른 옵션을 원하시면 말씀해주세요!")
