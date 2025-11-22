#!/usr/bin/env python3
"""
노트북 02_고급_도메인_특성공학.ipynb에 누락된 함수들을 추가하는 스크립트
- 성장성 지표 함수 및 호출
- 수익성/활동성 함수 및 호출
- 복합 리스크 지표 호출
- 상호작용/비선형 특성 호출
"""

import json
import sys

def add_missing_functions(notebook_path):
    """노트북에 누락된 함수 추가"""

    # 노트북 로드
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook['cells']
    print(f"원본 셀 개수: {len(cells)}")

    # 1. 섹션 6.5 (성장성 지표) 다음에 함수 및 호출 코드 추가
    print("\n1. 성장성 지표 함수 추가...")

    # 섹션 6.5의 인덱스 찾기
    section_6_5_idx = None
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell.get('source', []))
            if '## 6.5 성장성 지표 (Growth Features)' in source:
                section_6_5_idx = i
                break

    if section_6_5_idx is not None:
        growth_function_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def create_growth_features(df):\n",
                "    \"\"\"성장성 지표 생성 (Growth Features)\"\"\"\n",
                "    \n",
                "    features = pd.DataFrame(index=df.index)\n",
                "    \n",
                "    # 1. 매출 성장률 (Sales Growth Rate)\n",
                "    if '매출액증가율' in df.columns:\n",
                "        features['매출증가율_YoY'] = df['매출액증가율']\n",
                "    elif '매출액' in df.columns and '전기매출액' in df.columns:\n",
                "        features['매출증가율_YoY'] = (\n",
                "            (df['매출액'] - df['전기매출액']) / (df['전기매출액'] + 1)\n",
                "        ) * 100\n",
                "    \n",
                "    # 2. 영업이익 성장률\n",
                "    if '영업이익증가율' in df.columns:\n",
                "        features['영업이익증가율_YoY'] = df['영업이익증가율']\n",
                "    elif '영업이익' in df.columns and '전기영업이익' in df.columns:\n",
                "        features['영업이익증가율_YoY'] = (\n",
                "            (df['영업이익'] - df['전기영업이익']) / (df['전기영업이익'].abs() + 1)\n",
                "        ) * 100\n",
                "    \n",
                "    # 3. 당기순이익 성장률\n",
                "    if '당기순이익증가율' in df.columns:\n",
                "        features['당기순이익증가율_YoY'] = df['당기순이익증가율']\n",
                "    elif '당기순이익' in df.columns and '전기당기순이익' in df.columns:\n",
                "        features['당기순이익증가율_YoY'] = (\n",
                "            (df['당기순이익'] - df['전기당기순이익']) / (df['전기당기순이익'].abs() + 1)\n",
                "        ) * 100\n",
                "    \n",
                "    # 4. 총자산 성장률\n",
                "    if '총자산증가율' in df.columns:\n",
                "        features['총자산증가율_YoY'] = df['총자산증가율']\n",
                "    elif '총자산' in df.columns and '전기총자산' in df.columns:\n",
                "        features['총자산증가율_YoY'] = (\n",
                "            (df['총자산'] - df['전기총자산']) / (df['전기총자산'] + 1)\n",
                "        ) * 100\n",
                "    \n",
                "    # 5. 자본 성장률 (자본잠식 여부 확인)\n",
                "    if '자본증가율' in df.columns:\n",
                "        features['자본증가율_YoY'] = df['자본증가율']\n",
                "    elif '자기자본' in df.columns and '전기자기자본' in df.columns:\n",
                "        features['자본증가율_YoY'] = (\n",
                "            (df['자기자본'] - df['전기자기자본']) / (df['전기자기자본'].abs() + 1)\n",
                "        ) * 100\n",
                "    \n",
                "    # 6. 성장 둔화 신호 (매출 감소 + 영업이익 감소)\n",
                "    features['성장둔화신호'] = (\n",
                "        (features.get('매출증가율_YoY', 0) < -10).astype(int) +\n",
                "        (features.get('영업이익증가율_YoY', 0) < -30).astype(int)\n",
                "    )\n",
                "    \n",
                "    # 7. 역성장 위험 (음의 성장률)\n",
                "    features['역성장위험'] = (\n",
                "        (features.get('매출증가율_YoY', 0) < 0).astype(int) * 2 +\n",
                "        (features.get('영업이익증가율_YoY', 0) < 0).astype(int) * 3 +\n",
                "        (features.get('당기순이익증가율_YoY', 0) < 0).astype(int)\n",
                "    )\n",
                "    \n",
                "    print(f\"✅ 성장성 지표 {features.shape[1]}개 생성 완료\")\n",
                "    return features\n",
                "\n",
                "growth_features = create_growth_features(df)\n",
                "print(\"\\n생성된 성장성 지표:\")\n",
                "print(growth_features.columns.tolist())\n"
            ]
        }

        cells.insert(section_6_5_idx + 1, growth_function_cell)
        print(f"   - 성장성 지표 함수 추가 완료 (인덱스 {section_6_5_idx + 1})")

    # 2. 섹션 6.6 (수익성/활동성) 다음에 함수 및 호출 코드 추가
    print("\n2. 수익성/활동성 지표 함수 추가...")

    # 섹션 6.6의 인덱스 찾기 (성장성 함수 추가 후 인덱스가 변경됨)
    section_6_6_idx = None
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell.get('source', []))
            if '## 6.6 수익성 및 활동성 보완 지표' in source:
                section_6_6_idx = i
                break

    if section_6_6_idx is not None:
        profitability_function_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def create_profitability_activity_features(df):\n",
                "    \"\"\"수익성 및 활동성 지표 생성\"\"\"\n",
                "    \n",
                "    features = pd.DataFrame(index=df.index)\n",
                "    \n",
                "    # 1. 수익성 지표 (Profitability)\n",
                "    \n",
                "    # 영업이익률 (Operating Profit Margin)\n",
                "    if '영업이익' in df.columns and '매출액' in df.columns:\n",
                "        features['영업이익률'] = (df['영업이익'] / (df['매출액'] + 1)) * 100\n",
                "    \n",
                "    # 순이익률 (Net Profit Margin)\n",
                "    if '당기순이익' in df.columns and '매출액' in df.columns:\n",
                "        features['순이익률'] = (df['당기순이익'] / (df['매출액'] + 1)) * 100\n",
                "    \n",
                "    # 매출총이익률 (Gross Profit Margin)\n",
                "    if '매출총이익' in df.columns and '매출액' in df.columns:\n",
                "        features['매출총이익률'] = (df['매출총이익'] / (df['매출액'] + 1)) * 100\n",
                "    elif '매출원가' in df.columns and '매출액' in df.columns:\n",
                "        features['매출총이익률'] = ((df['매출액'] - df['매출원가']) / (df['매출액'] + 1)) * 100\n",
                "    \n",
                "    # ROA (Return on Assets)\n",
                "    if '당기순이익' in df.columns and '총자산' in df.columns:\n",
                "        features['ROA'] = (df['당기순이익'] / (df['총자산'] + 1)) * 100\n",
                "    \n",
                "    # ROE (Return on Equity)\n",
                "    if '당기순이익' in df.columns and '자기자본' in df.columns:\n",
                "        features['ROE'] = (df['당기순이익'] / (df['자기자본'].abs() + 1)) * 100\n",
                "    \n",
                "    # 이익의 질 (Accruals Quality)\n",
                "    # 이익의 질 = 영업현금흐름 / 당기순이익 (1에 가까울수록 양호)\n",
                "    if '영업활동현금흐름' in df.columns and '당기순이익' in df.columns:\n",
                "        features['이익의질'] = df['영업활동현금흐름'] / (df['당기순이익'].abs() + 1)\n",
                "        # 음수 이익의 질 = 현금흐름 < 이익 (의심)\n",
                "        features['이익품질_이상신호'] = (features['이익의질'] < 0.5).astype(int)\n",
                "    \n",
                "    # 2. 활동성 지표 (Activity / Efficiency)\n",
                "    \n",
                "    # 총자산회전율 (Total Asset Turnover)\n",
                "    if '매출액' in df.columns and '총자산' in df.columns:\n",
                "        features['총자산회전율'] = df['매출액'] / (df['총자산'] + 1)\n",
                "    \n",
                "    # 재고자산회전율 (Inventory Turnover)\n",
                "    if '매출원가' in df.columns and '재고자산' in df.columns:\n",
                "        features['재고자산회전율'] = df['매출원가'] / (df['재고자산'] + 1)\n",
                "        # 낮은 재고회전율 = 재고 적체 위험\n",
                "        features['재고적체위험'] = (features['재고자산회전율'] < 2).astype(int)\n",
                "    \n",
                "    # 매출채권회전율 (Accounts Receivable Turnover)\n",
                "    if '매출액' in df.columns and '매출채권' in df.columns:\n",
                "        features['매출채권회전율'] = df['매출액'] / (df['매출채권'] + 1)\n",
                "        # 낮은 매출채권회전율 = 회수 지연\n",
                "        features['매출채권회수지연'] = (features['매출채권회전율'] < 5).astype(int)\n",
                "    \n",
                "    # 3. 유동성_효율성 교차 지표\n",
                "    # 낮은 재고회전율 + 낮은 유동비율 = 위험\n",
                "    if '유동비율' in df.columns and '재고자산회전율' in features.columns:\n",
                "        features['유동성_효율성_교차위험'] = (\n",
                "            ((df['유동비율'] < 100).astype(int) +\n",
                "             (features['재고자산회전율'] < 2).astype(int))\n",
                "        )\n",
                "    \n",
                "    print(f\"✅ 수익성 및 활동성 지표 {features.shape[1]}개 생성 완료\")\n",
                "    return features\n",
                "\n",
                "profitability_activity_features = create_profitability_activity_features(df)\n",
                "print(\"\\n생성된 수익성/활동성 지표:\")\n",
                "print(profitability_activity_features.columns.tolist())\n"
            ]
        }

        cells.insert(section_6_6_idx + 1, profitability_function_cell)
        print(f"   - 수익성/활동성 지표 함수 추가 완료 (인덱스 {section_6_6_idx + 1})")

    # 3. 섹션 7 (복합 리스크 지표) 함수 다음에 호출 코드 추가
    print("\n3. 복합 리스크 지표 호출 코드 추가...")

    # 섹션 7의 복합 리스크 함수 다음 찾기
    section_7_function_idx = None
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'def create_composite_risk_features' in source:
                section_7_function_idx = i
                break

    if section_7_function_idx is not None:
        composite_call_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 복합 리스크 지표 생성 (앞서 생성한 특성들을 딕셔너리로 전달)\n",
                "features_dict = {\n",
                "    'liquidity': liquidity_features,\n",
                "    'insolvency': insolvency_features,\n",
                "    'manipulation': manipulation_features,\n",
                "    'korean': korean_features,\n",
                "    'stakeholder': profitability_activity_features  # 수익성 지표 포함\n",
                "}\n",
                "\n",
                "composite_features = create_composite_risk_features(df, features_dict)\n",
                "print(f\"\\n✅ 복합 리스크 지표 {composite_features.shape[1]}개 생성 완료\")\n",
                "print(\"\\n생성된 복합 리스크 지표:\")\n",
                "print(composite_features.columns.tolist())\n"
            ]
        }

        cells.insert(section_7_function_idx + 1, composite_call_cell)
        print(f"   - 복합 리스크 지표 호출 코드 추가 완료 (인덱스 {section_7_function_idx + 1})")

    # 4. 섹션 8 (상호작용/비선형) 함수 다음에 호출 코드 추가
    print("\n4. 상호작용/비선형 특성 호출 코드 추가...")

    # 섹션 8의 상호작용 함수 다음 찾기
    section_8_function_idx = None
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'def create_interaction_features' in source:
                section_8_function_idx = i
                break

    if section_8_function_idx is not None:
        interaction_call_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 상호작용 및 비선형 특성 생성\n",
                "interaction_features_dict = {\n",
                "    'liquidity': liquidity_features,\n",
                "    'insolvency': insolvency_features,\n",
                "    'profitability': profitability_activity_features\n",
                "}\n",
                "\n",
                "interaction_features = create_interaction_features(df, interaction_features_dict)\n",
                "print(f\"\\n✅ 상호작용/비선형 특성 {interaction_features.shape[1]}개 생성 완료\")\n",
                "print(\"\\n생성된 상호작용/비선형 특성:\")\n",
                "print(interaction_features.columns.tolist())\n"
            ]
        }

        cells.insert(section_8_function_idx + 1, interaction_call_cell)
        print(f"   - 상호작용/비선형 특성 호출 코드 추가 완료 (인덱스 {section_8_function_idx + 1})")

    # 수정된 노트북 저장
    notebook['cells'] = cells

    print(f"\n수정 후 셀 개수: {len(cells)}")

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print(f"\n✅ 노트북 함수 추가 완료: {notebook_path}")

    return True


if __name__ == "__main__":
    notebook_path = "../notebooks/02_고급_도메인_특성공학.ipynb"

    try:
        add_missing_functions(notebook_path)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
