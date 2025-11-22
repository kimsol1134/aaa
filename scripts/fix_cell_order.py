#!/usr/bin/env python3
"""
노트북의 셀 순서를 수정하는 스크립트
- 이해관계자 행동 패턴 함수를 올바른 위치로 이동
"""

import json
import sys

def fix_cell_order(notebook_path):
    """셀 순서 수정"""

    # 노트북 로드
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook['cells']
    print(f"원본 셀 개수: {len(cells)}")

    # 현재 구조:
    # 12: 섹션 6 (이해관계자 행동 패턴)
    # 13: 섹션 6.5 (성장성)
    # 14: 성장성 함수
    # 15: 이해관계자 행동 패턴 함수 ❌
    # 16: 섹션 6.6 (수익성/활동성)
    # 17: 수익성/활동성 함수

    # 목표 구조:
    # 12: 섹션 6 (이해관계자 행동 패턴)
    # 13: 이해관계자 행동 패턴 함수 ✅
    # 14: 섹션 6.5 (성장성)
    # 15: 성장성 함수
    # 16: 섹션 6.6 (수익성/활동성)
    # 17: 수익성/활동성 함수

    print("\n셀 순서 재배치...")

    # 셀 15 (이해관계자 행동 패턴 함수)를 제거하고 섹션 6 다음으로 이동
    stakeholder_cell = cells.pop(15)

    # 섹션 6의 인덱스 찾기
    section_6_idx = None
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell.get('source', []))
            if '## 6. 이해관계자 행동 패턴 특성' in source:
                section_6_idx = i
                break

    if section_6_idx is not None:
        # 섹션 6 다음에 삽입
        cells.insert(section_6_idx + 1, stakeholder_cell)
        print(f"   - 이해관계자 행동 패턴 함수를 인덱스 {section_6_idx + 1}로 이동")

    # 수정된 노트북 저장
    notebook['cells'] = cells

    print(f"\n수정 후 셀 개수: {len(cells)}")

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print(f"\n✅ 셀 순서 수정 완료: {notebook_path}")

    return True


if __name__ == "__main__":
    notebook_path = "../notebooks/02_고급_도메인_특성공학.ipynb"

    try:
        fix_cell_order(notebook_path)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
