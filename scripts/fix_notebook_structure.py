#!/usr/bin/env python3
"""
노트북 02_고급_도메인_특성공학.ipynb의 구조를 수정하는 스크립트
- 섹션 6.5와 6.6의 순서 교정
- 빈 마크다운 셀 제거
- 중복된 특성 저장 섹션 제거
- 섹션 7에 복합 리스크 지표 생성 함수 추가
"""

import json
import sys

def fix_notebook_structure(notebook_path):
    """노트북 구조 수정"""

    # 노트북 로드
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook['cells']

    print(f"원본 셀 개수: {len(cells)}")

    # 1. 섹션 6.5 (인덱스 16)와 6.6 (인덱스 13) 순서 교환
    print("\n1. 섹션 6.5와 6.6 순서 교환...")
    cell_6_5 = cells[16]  # 6.5 성장성 지표
    cell_6_6 = cells[13]  # 6.6 수익성 및 활동성

    # 6.5가 6.6보다 먼저 와야 함
    # 현재 순서: 12(섹션6) → 13(6.6) → 14(코드) → 15(빈셀) → 16(6.5) → 17(섹션7)
    # 목표 순서: 12(섹션6) → 13(6.5) → 14(6.6) → 15(코드) → 16(섹션7)

    cells[13] = cell_6_5  # 6.5를 13번 위치로
    cells[16] = cell_6_6  # 6.6을 16번 위치로

    # 2. 빈 마크다운 셀 (인덱스 15) 제거
    print("2. 빈 마크다운 셀 제거...")
    # 15번 셀이 빈 셀인지 확인
    if cells[15]['cell_type'] == 'markdown':
        source = ''.join(cells[15].get('source', []))
        if not source or source.strip() == '':
            cells.pop(15)
            print("   - 빈 셀 제거 완료")

    # 3. 섹션 7에 복합 리스크 지표 생성 함수 추가
    print("3. 섹션 7에 복합 리스크 지표 생성 함수 추가...")

    # 섹션 7의 인덱스 찾기 (빈 셀 제거 후 인덱스가 변경됨)
    section_7_idx = None
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell.get('source', []))
            if '## 7. 복합 리스크 지표 생성' in source:
                section_7_idx = i
                break

    if section_7_idx is not None:
        # 섹션 7 다음에 함수 정의 코드 셀 추가
        composite_risk_function = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def create_composite_risk_features(df, features_dict):\n",
                "    \"\"\"복합 리스크 지표 생성 (여러 도메인 특성의 조합)\"\"\"\n",
                "    \n",
                "    features = pd.DataFrame(index=df.index)\n",
                "    \n",
                "    # 1. 재무건전성지수 (Financial Health Index)\n",
                "    # 유동성 + 수익성 + 지급능력의 평균\n",
                "    liquidity_cols = ['즉각지급능력', '현금소진일수', '운전자본비율']\n",
                "    profitability_cols = ['영업이익률', '순이익률']\n",
                "    solvency_cols = ['이자보상배율', '부채상환년수']\n",
                "    \n",
                "    # 각 도메인별 정규화된 점수 계산\n",
                "    from sklearn.preprocessing import StandardScaler\n",
                "    scaler = StandardScaler()\n",
                "    \n",
                "    liquidity_score = features_dict['liquidity'][liquidity_cols].fillna(0)\n",
                "    liquidity_score = pd.DataFrame(\n",
                "        scaler.fit_transform(liquidity_score), \n",
                "        index=liquidity_score.index, \n",
                "        columns=liquidity_score.columns\n",
                "    ).mean(axis=1)\n",
                "    \n",
                "    profitability_score = features_dict['stakeholder'][profitability_cols].fillna(0)\n",
                "    profitability_score = pd.DataFrame(\n",
                "        scaler.fit_transform(profitability_score), \n",
                "        index=profitability_score.index, \n",
                "        columns=profitability_score.columns\n",
                "    ).mean(axis=1)\n",
                "    \n",
                "    solvency_score = features_dict['insolvency'][solvency_cols].fillna(0)\n",
                "    solvency_score = pd.DataFrame(\n",
                "        scaler.fit_transform(solvency_score), \n",
                "        index=solvency_score.index, \n",
                "        columns=solvency_score.columns\n",
                "    ).mean(axis=1)\n",
                "    \n",
                "    features['재무건전성지수'] = (liquidity_score + profitability_score + solvency_score) / 3\n",
                "    \n",
                "    # 2. 유동성스트레스지수\n",
                "    features['유동성스트레스지수'] = (\n",
                "        -features_dict['liquidity']['운전자본비율'].fillna(0) + \n",
                "        features_dict['insolvency']['단기부채비중'].fillna(0)\n",
                "    ) / 2\n",
                "    \n",
                "    # 3. 지급불능위험지수\n",
                "    features['지급불능위험지수'] = (\n",
                "        features_dict['insolvency']['자본잠식도'].fillna(0) + \n",
                "        features_dict['insolvency']['부채상환년수'].fillna(0)\n",
                "    ) / 2\n",
                "    \n",
                "    # 4. 시장포지션지수 (규모 정규화)\n",
                "    if '총자산' in df.columns:\n",
                "        features['시장포지션지수'] = pd.qcut(\n",
                "            df['총자산'].fillna(0), \n",
                "            q=10, \n",
                "            labels=False, \n",
                "            duplicates='drop'\n",
                "        )\n",
                "    else:\n",
                "        features['시장포지션지수'] = 0\n",
                "    \n",
                "    # 5. 종합부도위험스코어 (가중 평균)\n",
                "    weights = {\n",
                "        'liquidity': 0.3,\n",
                "        'insolvency': 0.3,\n",
                "        'manipulation': 0.2,\n",
                "        'stakeholder': 0.2\n",
                "    }\n",
                "    \n",
                "    risk_components = []\n",
                "    for domain, weight in weights.items():\n",
                "        domain_features = features_dict[domain]\n",
                "        # 각 도메인의 평균 리스크 점수 계산\n",
                "        domain_score = domain_features.fillna(0).mean(axis=1)\n",
                "        risk_components.append(domain_score * weight)\n",
                "    \n",
                "    features['종합부도위험스코어'] = sum(risk_components)\n",
                "    \n",
                "    # 6. 조기경보신호수 (위험 신호 개수)\n",
                "    warning_signals = []\n",
                "    \n",
                "    # 자본잠식\n",
                "    if '자본잠식도' in features_dict['insolvency'].columns:\n",
                "        warning_signals.append(\n",
                "            (features_dict['insolvency']['자본잠식도'] > 0.5).astype(int)\n",
                "        )\n",
                "    \n",
                "    # 연체 여부\n",
                "    if '연체여부' in features_dict['stakeholder'].columns:\n",
                "        warning_signals.append(\n",
                "            features_dict['stakeholder']['연체여부']\n",
                "        )\n",
                "    \n",
                "    # 현금 고갈 위험\n",
                "    if '현금소진일수' in features_dict['liquidity'].columns:\n",
                "        warning_signals.append(\n",
                "            (features_dict['liquidity']['현금소진일수'] < 30).astype(int)\n",
                "        )\n",
                "    \n",
                "    # 이자보상배율 < 1\n",
                "    if '이자보상배율' in features_dict['insolvency'].columns:\n",
                "        warning_signals.append(\n",
                "            (features_dict['insolvency']['이자보상배율'] < 1).astype(int)\n",
                "        )\n",
                "    \n",
                "    features['조기경보신호수'] = sum(warning_signals) if warning_signals else 0\n",
                "    \n",
                "    # 7. 위험경보등급 (4단계 분류)\n",
                "    def classify_risk(score):\n",
                "        if score < -1:\n",
                "            return 0  # 정상\n",
                "        elif score < 0:\n",
                "            return 1  # 주의\n",
                "        elif score < 1:\n",
                "            return 2  # 경고\n",
                "        else:\n",
                "            return 3  # 위험\n",
                "    \n",
                "    features['위험경보등급'] = features['종합부도위험스코어'].apply(classify_risk)\n",
                "    \n",
                "    return features\n"
            ]
        }

        # 섹션 7 다음에 함수 삽입 (기존 통합 코드 전)
        cells.insert(section_7_idx + 1, composite_risk_function)
        print(f"   - 복합 리스크 지표 함수 추가 완료 (인덱스 {section_7_idx + 1})")

    # 4. 섹션 9 제목 수정
    print("4. 섹션 9 제목 수정...")
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell.get('source', []))
            if '## 9. 모든 특성 통합' in source:
                # 제목을 더 명확하게 수정
                cells[i]['source'] = [
                    "## 9. 특성 저장 및 메타데이터 생성\n",
                    "\n",
                    "### 💾 생성된 특성 저장\n",
                    "\n",
                    "모든 도메인 특성을 CSV 파일로 저장하고, 각 특성의 메타데이터를 기록합니다.\n"
                ]
                print(f"   - 섹션 9 제목 수정 완료")
                break

    # 5. 중복된 섹션 11 제거
    print("5. 중복된 특성 저장 섹션 제거...")
    indices_to_remove = []
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell.get('source', []))
            if '## 11. 특성 저장' in source:
                # 이 섹션과 다음 코드 셀을 제거 대상으로 표시
                indices_to_remove.append(i)
                if i + 1 < len(cells) and cells[i + 1]['cell_type'] == 'code':
                    indices_to_remove.append(i + 1)
                break

    # 역순으로 제거 (인덱스 변경 방지)
    for idx in sorted(indices_to_remove, reverse=True):
        cells.pop(idx)
        print(f"   - 인덱스 {idx} 셀 제거")

    # 6. 섹션 번호 재정렬 (11 제거 후 12 → 11로 변경)
    print("6. 섹션 번호 재정렬...")
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell.get('source', []))
            if '## 12. 핵심 인사이트 정리' in source:
                cells[i]['source'][0] = '## 11. 핵심 인사이트 정리\n'
                print(f"   - 섹션 12 → 11로 변경")
                break

    # 수정된 노트북 저장
    notebook['cells'] = cells

    print(f"\n수정 후 셀 개수: {len(cells)}")

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print(f"\n✅ 노트북 수정 완료: {notebook_path}")

    return True


if __name__ == "__main__":
    notebook_path = "../notebooks/02_고급_도메인_특성공학.ipynb"

    try:
        fix_notebook_structure(notebook_path)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
