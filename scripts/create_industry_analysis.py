"""
노트북에 추가할 업종별 분석 셀 생성 스크립트
"""
import json

# 추가할 셀들
cells_to_add = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4.5 업종별 리스크 분석\n",
            "\n",
            "한국표준산업분류(10차)를 활용한 업종별 부도 패턴 및 재무 리스크 분석"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "# 산업분류 매핑 테이블 로드\n",
            "mapping = pd.read_csv('../data/ksic_mapping.csv')\n",
            "\n",
            "# 기업 데이터와 업종 매핑\n",
            "df_industry = df.merge(mapping, left_on='업종(중분류)', right_on='업종코드', how='left')\n",
            "\n",
            "print(\"=== 업종 매핑 결과 ===\")\n",
            "print(f\"매핑 성공: {df_industry['대분류코드'].notna().sum():,}건 ({df_industry['대분류코드'].notna().sum()/len(df)*100:.1f}%)\")\n",
            "print(f\"매핑 실패: {df_industry['대분류코드'].isna().sum():,}건\")\n",
            "\n",
            "print(\"\\n=== 대분류별 기업 수 ===\")\n",
            "print(df_industry['대분류명'].value_counts())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.5.1 업종별 부도율 분석"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "import plotly.graph_objects as go\n",
            "from plotly.subplots import make_subplots\n",
            "\n",
            "# 대분류별 부도율\n",
            "major_default = df_industry.groupby('대분류명').agg({\n",
            "    target_col: ['sum', 'count', 'mean']\n",
            "}).round(4)\n",
            "major_default.columns = ['부도기업수', '전체기업수', '부도율']\n",
            "major_default = major_default.sort_values('부도율', ascending=False)\n",
            "\n",
            "print(\"=== 대분류별 부도율 ===\")\n",
            "print(major_default)\n",
            "\n",
            "# 중분류별 부도율 (상위 20개)\n",
            "minor_default = df_industry.groupby(['대분류명', '중분류명']).agg({\n",
            "    target_col: ['sum', 'count', 'mean']\n",
            "}).round(4)\n",
            "minor_default.columns = ['부도기업수', '전체기업수', '부도율']\n",
            "minor_default = minor_default[minor_default['전체기업수'] >= 50]  # 최소 50개 기업 이상\n",
            "minor_default = minor_default.sort_values('부도율', ascending=False)\n",
            "\n",
            "print(\"\\n=== 중분류별 부도율 (상위 20개, 최소 50개 기업) ===\")\n",
            "print(minor_default.head(20))"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "# 대분류별 부도율 시각화\n",
            "fig = go.Figure()\n",
            "\n",
            "fig.add_trace(go.Bar(\n",
            "    x=major_default.index,\n",
            "    y=major_default['부도율'] * 100,\n",
            "    text=[f\"{v:.2f}%\" for v in major_default['부도율'] * 100],\n",
            "    textposition='outside',\n",
            "    marker_color='indianred',\n",
            "    name='부도율'\n",
            "))\n",
            "\n",
            "fig.update_layout(\n",
            "    title='대분류별 부도율',\n",
            "    xaxis_title='산업 대분류',\n",
            "    yaxis_title='부도율 (%)',\n",
            "    height=500,\n",
            "    font=dict(family='Malgun Gothic', size=12),\n",
            "    xaxis={'tickangle': -45}\n",
            ")\n",
            "\n",
            "fig.show()"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "# 중분류별 부도율 시각화 (상위 20개)\n",
            "top_20_minor = minor_default.head(20)\n",
            "\n",
            "# 인덱스를 문자열로 변환 (대분류명 + 중분류명)\n",
            "labels = [f\"{major}\\n{minor}\" for major, minor in top_20_minor.index]\n",
            "\n",
            "fig = go.Figure()\n",
            "\n",
            "fig.add_trace(go.Bar(\n",
            "    x=labels,\n",
            "    y=top_20_minor['부도율'] * 100,\n",
            "    text=[f\"{v:.2f}%\" for v in top_20_minor['부도율'] * 100],\n",
            "    textposition='outside',\n",
            "    marker_color='coral',\n",
            "    name='부도율'\n",
            "))\n",
            "\n",
            "fig.update_layout(\n",
            "    title='중분류별 부도율 Top 20 (최소 50개 기업 이상)',\n",
            "    xaxis_title='산업 중분류',\n",
            "    yaxis_title='부도율 (%)',\n",
            "    height=600,\n",
            "    font=dict(family='Malgun Gothic', size=10),\n",
            "    xaxis={'tickangle': -45}\n",
            ")\n",
            "\n",
            "fig.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.5.2 업종별 재무지표 리스크 분석"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "# 주요 재무지표별 업종 평균\n",
            "key_ratios = [\n",
            "    '재무비율_부채비율',\n",
            "    '재무비율_유동비율',\n",
            "    '재무비율_자기자본비율',\n",
            "    '재무비율_영업이익율',\n",
            "    '재무비율_당기순이익율',\n",
            "    '재무비율_자기자본이익률(ROE)',\n",
            "    '당좌비율',\n",
            "    '이자보상배율'\n",
            "]\n",
            "\n",
            "# 대분류별 재무지표 평균 (부도기업 vs 정상기업)\n",
            "industry_risk = {}\n",
            "\n",
            "for ratio in key_ratios:\n",
            "    if ratio in df_industry.columns:\n",
            "        industry_comparison = df_industry.groupby(['대분류명', target_col])[ratio].agg(['mean', 'median', 'std']).round(2)\n",
            "        industry_risk[ratio] = industry_comparison\n",
            "        \n",
            "        print(f\"\\n=== {ratio} - 대분류별 비교 ===\")\n",
            "        print(industry_comparison)"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "# 업종별 부채비율 비교 (부도 vs 정상)\n",
            "if '재무비율_부채비율' in df_industry.columns:\n",
            "    debt_ratio_by_industry = df_industry.groupby(['대분류명', target_col])['재무비율_부채비율'].median().unstack()\n",
            "    \n",
            "    fig = go.Figure()\n",
            "    \n",
            "    fig.add_trace(go.Bar(\n",
            "        name='정상기업',\n",
            "        x=debt_ratio_by_industry.index,\n",
            "        y=debt_ratio_by_industry[0],\n",
            "        marker_color='lightblue'\n",
            "    ))\n",
            "    \n",
            "    fig.add_trace(go.Bar(\n",
            "        name='부도기업',\n",
            "        x=debt_ratio_by_industry.index,\n",
            "        y=debt_ratio_by_industry[1],\n",
            "        marker_color='darkred'\n",
            "    ))\n",
            "    \n",
            "    fig.update_layout(\n",
            "        title='업종별 부채비율 비교 (중앙값)',\n",
            "        xaxis_title='산업 대분류',\n",
            "        yaxis_title='부채비율 (%)',\n",
            "        barmode='group',\n",
            "        height=500,\n",
            "        font=dict(family='Malgun Gothic', size=12),\n",
            "        xaxis={'tickangle': -45}\n",
            "    )\n",
            "    \n",
            "    fig.show()"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "# 업종별 유동비율 비교 (부도 vs 정상)\n",
            "if '재무비율_유동비율' in df_industry.columns:\n",
            "    current_ratio_by_industry = df_industry.groupby(['대분류명', target_col])['재무비율_유동비율'].median().unstack()\n",
            "    \n",
            "    fig = go.Figure()\n",
            "    \n",
            "    fig.add_trace(go.Bar(\n",
            "        name='정상기업',\n",
            "        x=current_ratio_by_industry.index,\n",
            "        y=current_ratio_by_industry[0],\n",
            "        marker_color='lightgreen'\n",
            "    ))\n",
            "    \n",
            "    fig.add_trace(go.Bar(\n",
            "        name='부도기업',\n",
            "        x=current_ratio_by_industry.index,\n",
            "        y=current_ratio_by_industry[1],\n",
            "        marker_color='orange'\n",
            "    ))\n",
            "    \n",
            "    fig.update_layout(\n",
            "        title='업종별 유동비율 비교 (중앙값)',\n",
            "        xaxis_title='산업 대분류',\n",
            "        yaxis_title='유동비율 (%)',\n",
            "        barmode='group',\n",
            "        height=500,\n",
            "        font=dict(family='Malgun Gothic', size=12),\n",
            "        xaxis={'tickangle': -45}\n",
            "    )\n",
            "    \n",
            "    fig.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.5.3 업종별 유동성 위기 분석"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "# 유동성 지표\n",
            "liquidity_cols = [\n",
            "    '재무비율_유동비율',\n",
            "    '당좌비율',\n",
            "    '순운전자본',\n",
            "    '현금',\n",
            "    '현금등가물',\n",
            "    '현금성자산'\n",
            "]\n",
            "\n",
            "available_liquidity = [col for col in liquidity_cols if col in df_industry.columns]\n",
            "\n",
            "if available_liquidity:\n",
            "    # 대분류별 유동성 지표 평균\n",
            "    liquidity_by_industry = df_industry.groupby(['대분류명', target_col])[available_liquidity].median().round(2)\n",
            "    \n",
            "    print(\"=== 업종별 유동성 지표 (중앙값) ===\")\n",
            "    print(liquidity_by_industry)"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "# 업종별 유동성 히트맵\n",
            "if len(available_liquidity) >= 2:\n",
            "    # 정상기업 기준 업종별 유동성 지표\n",
            "    normal_liquidity = df_industry[df_industry[target_col] == 0].groupby('대분류명')[available_liquidity].median()\n",
            "    \n",
            "    # 표준화 (0-1 스케일)\n",
            "    from sklearn.preprocessing import MinMaxScaler\n",
            "    scaler = MinMaxScaler()\n",
            "    normal_liquidity_scaled = pd.DataFrame(\n",
            "        scaler.fit_transform(normal_liquidity),\n",
            "        columns=normal_liquidity.columns,\n",
            "        index=normal_liquidity.index\n",
            "    )\n",
            "    \n",
            "    fig = go.Figure(data=go.Heatmap(\n",
            "        z=normal_liquidity_scaled.T.values,\n",
            "        x=normal_liquidity_scaled.index,\n",
            "        y=normal_liquidity_scaled.columns,\n",
            "        colorscale='RdYlGn',\n",
            "        text=normal_liquidity.T.values,\n",
            "        texttemplate='%{text:.1f}',\n",
            "        textfont={\"size\": 10}\n",
            "    ))\n",
            "    \n",
            "    fig.update_layout(\n",
            "        title='업종별 유동성 지표 히트맵 (정상기업 중앙값, 표준화)',\n",
            "        xaxis_title='산업 대분류',\n",
            "        yaxis_title='유동성 지표',\n",
            "        height=400,\n",
            "        font=dict(family='Malgun Gothic', size=10),\n",
            "        xaxis={'tickangle': -45}\n",
            "    )\n",
            "    \n",
            "    fig.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.5.4 업종별 리스크 스코어"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "# 업종별 종합 리스크 스코어 계산\n",
            "industry_risk_score = pd.DataFrame()\n",
            "\n",
            "# 1. 부도율 (높을수록 위험)\n",
            "industry_risk_score['부도율'] = major_default['부도율']\n",
            "\n",
            "# 2. 평균 부채비율 (높을수록 위험)\n",
            "if '재무비율_부채비율' in df_industry.columns:\n",
            "    industry_risk_score['평균부채비율'] = df_industry.groupby('대분류명')['재무비율_부채비율'].median()\n",
            "\n",
            "# 3. 평균 유동비율 (낮을수록 위험, 역수 취함)\n",
            "if '재무비율_유동비율' in df_industry.columns:\n",
            "    avg_current = df_industry.groupby('대분류명')['재무비율_유동비율'].median()\n",
            "    industry_risk_score['유동비율역수'] = 1 / (avg_current / 100 + 0.01)  # 0으로 나누기 방지\n",
            "\n",
            "# 4. 평균 영업이익률 (낮을수록 위험, 음수면 더 위험)\n",
            "if '재무비율_영업이익율' in df_industry.columns:\n",
            "    industry_risk_score['영업이익률'] = -df_industry.groupby('대분류명')['재무비율_영업이익율'].median()  # 음수 취함\n",
            "\n",
            "# 표준화 및 종합 스코어\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "scaler = StandardScaler()\n",
            "risk_scaled = pd.DataFrame(\n",
            "    scaler.fit_transform(industry_risk_score.fillna(0)),\n",
            "    columns=industry_risk_score.columns,\n",
            "    index=industry_risk_score.index\n",
            ")\n",
            "\n",
            "# 종합 리스크 스코어 (평균)\n",
            "industry_risk_score['종합리스크스코어'] = risk_scaled.mean(axis=1)\n",
            "industry_risk_score = industry_risk_score.sort_values('종합리스크스코어', ascending=False)\n",
            "\n",
            "print(\"=== 업종별 종합 리스크 스코어 ===\")\n",
            "print(industry_risk_score.round(3))"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "# 종합 리스크 스코어 시각화\n",
            "fig = go.Figure()\n",
            "\n",
            "colors = ['red' if score > 0 else 'green' for score in industry_risk_score['종합리스크스코어']]\n",
            "\n",
            "fig.add_trace(go.Bar(\n",
            "    x=industry_risk_score.index,\n",
            "    y=industry_risk_score['종합리스크스코어'],\n",
            "    marker_color=colors,\n",
            "    text=[f\"{v:.2f}\" for v in industry_risk_score['종합리스크스코어']],\n",
            "    textposition='outside'\n",
            "))\n",
            "\n",
            "fig.update_layout(\n",
            "    title='업종별 종합 리스크 스코어 (표준화)',\n",
            "    xaxis_title='산업 대분류',\n",
            "    yaxis_title='종합 리스크 스코어',\n",
            "    height=500,\n",
            "    font=dict(family='Malgun Gothic', size=12),\n",
            "    xaxis={'tickangle': -45}\n",
            ")\n",
            "\n",
            "fig.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.5.5 업종별 인사이트 요약"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"📊 업종별 리스크 분석 - 핵심 인사이트\")\n",
            "print(\"=\"*80)\n",
            "\n",
            "# 1. 부도율 최고/최저 업종\n",
            "print(\"\\n1️⃣ 부도율 분석\")\n",
            "print(f\"   - 최고 부도율: {major_default.index[0]} ({major_default['부도율'].iloc[0]*100:.2f}%)\")\n",
            "print(f\"   - 최저 부도율: {major_default.index[-1]} ({major_default['부도율'].iloc[-1]*100:.2f}%)\")\n",
            "print(f\"   - 전체 평균: {major_default['부도율'].mean()*100:.2f}%\")\n",
            "\n",
            "# 2. 리스크 스코어 최고/최저 업종\n",
            "print(\"\\n2️⃣ 종합 리스크 스코어\")\n",
            "print(f\"   - 최고 위험: {industry_risk_score.index[0]} (스코어: {industry_risk_score['종합리스크스코어'].iloc[0]:.3f})\")\n",
            "print(f\"   - 최저 위험: {industry_risk_score.index[-1]} (스코어: {industry_risk_score['종합리스크스코어'].iloc[-1]:.3f})\")\n",
            "\n",
            "# 3. 업종별 기업 분포\n",
            "print(\"\\n3️⃣ 업종별 기업 분포 (상위 3개)\")\n",
            "top_3_industries = df_industry['대분류명'].value_counts().head(3)\n",
            "for idx, (industry, count) in enumerate(top_3_industries.items(), 1):\n",
            "    pct = count / len(df_industry) * 100\n",
            "    print(f\"   {idx}. {industry}: {count:,}개 기업 ({pct:.1f}%)\")\n",
            "\n",
            "# 4. 재무지표 특성\n",
            "if '재무비율_부채비율' in df_industry.columns:\n",
            "    print(\"\\n4️⃣ 재무 특성\")\n",
            "    high_debt = df_industry.groupby('대분류명')['재무비율_부채비율'].median().idxmax()\n",
            "    high_debt_val = df_industry.groupby('대분류명')['재무비율_부채비율'].median().max()\n",
            "    print(f\"   - 부채비율 최고 업종: {high_debt} ({high_debt_val:.1f}%)\")\n",
            "    \n",
            "if '재무비율_유동비율' in df_industry.columns:\n",
            "    low_liquidity = df_industry.groupby('대분류명')['재무비율_유동비율'].median().idxmin()\n",
            "    low_liquidity_val = df_industry.groupby('대분류명')['재무비율_유동비율'].median().min()\n",
            "    print(f\"   - 유동비율 최저 업종: {low_liquidity} ({low_liquidity_val:.1f}%)\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*80)"
        ]
    }
]

import os

# 노트북 파일 로드
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
notebook_path = os.path.join(base_dir, 'notebooks', '01_도메인_기반_부도원인_분석.ipynb')

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 섹션 4와 5 사이에 삽입 (인덱스 27 이전)
insert_position = 27

# 셀 삽입
for i, cell in enumerate(cells_to_add):
    nb['cells'].insert(insert_position + i, cell)

# 저장
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"✅ 노트북에 {len(cells_to_add)}개 셀 추가 완료 (위치: {insert_position})")
print("📍 추가된 섹션: 4.5 업종별 리스크 분석")
print(f"📁 파일: {notebook_path}")
