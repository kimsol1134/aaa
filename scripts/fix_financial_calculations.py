#!/usr/bin/env python3
"""
금융 전문가 피드백 기반 노트북 수정 스크립트

주요 개선사항:
1. 부채비율 계산 - 자본잠식 기업 별도 처리
2. 이자보상배율 계산 - 이자비용 0/음수 처리
3. 좀비 기업 용어 변경
4. 당좌비율 계산 주석 추가
5. VIF 분석 개선
"""

import json
import re
from pathlib import Path

def fix_debt_ratio_calculation(cell_source):
    """
    부채비율 계산 로직 개선

    Before: df['부채비율'] = df['부채총계'] / (df['자본총계'] + 1) * 100
    After: 자본총계 <= 0인 경우 별도 처리
    """
    old_pattern = r"df\['부채비율'\]\s*=\s*df\['부채총계'\]\s*/\s*\(df\['자본총계'\]\s*\+\s*1\)\s*\*\s*100"

    new_code = """# 부채비율 계산 (개선됨 - 자본잠식 기업 별도 처리)
    # 문제점: 자본총계가 음수인 경우 부채비율도 음수가 되어 데이터 왜곡
    # 해결책: 자본총계가 0 이하인 경우 최댓값으로 Cap 처리
    if '부채총계' in df.columns and '자본총계' in df.columns:
        # 정상 기업만 부채비율 계산
        df['부채비율'] = np.where(
            df['자본총계'] > 0,
            df['부채총계'] / df['자본총계'] * 100,
            np.nan  # 자본잠식 기업은 NaN 처리
        )

        # 자본잠식 기업에 대해 최댓값으로 Cap (데이터셋 내 99 백분위수 또는 9999%)
        max_ratio = df[df['자본총계'] > 0]['부채비율'].quantile(0.99)
        max_ratio = min(max_ratio, 9999)  # 최대 9999%로 제한

        df['부채비율'] = df['부채비율'].fillna(max_ratio)

        print(f"✅ 부채비율 계산 완료 (자본잠식 기업 {(df['자본총계'] <= 0).sum()}개는 {max_ratio:.0f}%로 Cap 처리)")"""

    if re.search(old_pattern, cell_source):
        # 기존 패턴 찾아서 교체
        new_source = re.sub(
            r"(\s*)#\s*부채비율 계산.*?\n\s*if '부채총계'.*?\n\s*df\['부채비율'\].*?100",
            new_code,
            cell_source,
            flags=re.DOTALL
        )
        return new_source

    return None


def fix_icr_calculation(cell_source):
    """
    이자보상배율(ICR) 계산 로직 개선

    Before: df['이자보상배율_ICR'] = df[operating_income] / (df[interest_expense] + 1)
    After: 이자비용 0/음수인 경우 별도 처리
    """
    old_pattern = r"df\['이자보상배율_ICR'\]\s*=\s*df\[operating_income\]\s*/\s*\(df\[interest_expense\]\s*\+\s*1\)"

    new_code = """# 이자보상배율(ICR) 계산 (개선됨)
        # 문제점: 이자비용이 0이거나 음수인 경우 비율의 스케일이 왜곡됨
        # 해결책: 이자비용 0/음수는 별도 카테고리로 분류

        # 1. 이자비용이 양수인 경우만 ICR 계산
        df['이자보상배율_ICR'] = np.where(
            df[interest_expense] > 0,
            df[operating_income] / df[interest_expense],
            np.nan
        )

        # 2. 이자비용이 0 또는 음수인 경우 처리
        # - 이자비용 0: 무차입 경영 (매우 높은 ICR로 간주, 예: 100)
        # - 이자비용 < 0: 이자수익 > 이자비용 (매우 높은 ICR로 간주, 예: 100)
        df['이자보상배율_ICR'] = df['이자보상배율_ICR'].fillna(100)

        # 3. 극단값 클리핑 (ICR > 100은 100으로 제한)
        df['이자보상배율_ICR'] = df['이자보상배율_ICR'].clip(upper=100)

        print(f"✅ 이자보상배율 계산 완료")
        print(f"   - 이자비용 0 또는 음수 기업: {(df[interest_expense] <= 0).sum()}개 (ICR=100으로 처리)")"""

    if re.search(old_pattern, cell_source):
        new_source = re.sub(
            r"(\s*)df\['이자보상배율_ICR'\].*?interest_expense.*?\+.*?1\)",
            new_code,
            cell_source
        )
        return new_source

    return None


def fix_zombie_company_terminology(cell_source):
    """
    좀비 기업 용어 변경

    - "좀비 기업" → "이자보상배율 1 미만 기업" 또는 "잠재적 한계기업"
    - 주석 추가: 금융권 표준은 3년 연속 ICR < 1.0이나, 단일 시점 데이터로 인해 판단 제한적
    """
    replacements = [
        (r'좀비 기업 \(<1\.0\)', '이자보상배율 1 미만 기업'),
        (r"'좀비 기업'", "'이자보상배율 1 미만 기업'"),
        (r'"좀비 기업"', '"이자보상배율 1 미만 기업"'),
        (r'좀비기업', '한계기업(ICR<1.0)'),
    ]

    # 주석 추가 패턴
    icr_category_pattern = r"(def categorize_icr.*?:)"
    comment = r'\1\n    """\n    이자보상배율(ICR) 기반 기업 분류\n    \n    주의: 금융권 표준 한계기업(좀비기업) 정의는 "3년 연속 ICR < 1.0"이나,\n    현재 데이터는 단일 시점(2021년 8월)이므로 "잠재적 한계기업"으로 해석해야 함.\n    """'

    modified = cell_source
    for old, new in replacements:
        modified = re.sub(old, new, modified)

    # 주석 추가
    if re.search(icr_category_pattern, modified):
        modified = re.sub(icr_category_pattern, comment, modified)

    return modified if modified != cell_source else None


def fix_quick_ratio_comment(cell_source):
    """
    당좌비율 계산에 주석 추가

    - 엄밀한 당좌비율 정의와의 차이점 명시
    """
    old_pattern = r"(df\['당좌비율'\]\s*=\s*\(df\['유동자산'\]\s*-\s*df\['재고자산'\]\)\s*/\s*\(df\['유동부채'\]\s*\+\s*1\))"

    new_code = r"""# 당좌비율 계산 (약식)
        # 주의: 엄밀한 회계적 정의는 "당좌자산 / 유동부채"로,
        #       유동자산에서 재고자산뿐만 아니라 선급금, 선급비용 등 비현금성 자산도 차감해야 함.
        #       본 분석에서는 재고자산만 차감한 약식 당좌비율을 사용함.
        \1"""

    if re.search(old_pattern, cell_source):
        new_source = re.sub(old_pattern, new_code, cell_source)
        return new_source

    return None


def process_notebook(notebook_path, fixes):
    """
    노트북 파일을 읽어서 수정사항 적용

    Args:
        notebook_path: 노트북 파일 경로
        fixes: 적용할 수정 함수 리스트
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    modified_cells = 0

    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue

        cell_source = ''.join(cell['source'])

        for fix_func in fixes:
            new_source = fix_func(cell_source)
            if new_source:
                cell['source'] = new_source.split('\n')
                # 줄바꿈 처리
                cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line
                                 for i, line in enumerate(cell['source'])]
                modified_cells += 1
                print(f"✅ {notebook_path.name}: {fix_func.__name__} 적용")
                break

    if modified_cells > 0:
        # 백업 생성
        backup_path = notebook_path.with_suffix('.ipynb.backup')
        notebook_path.rename(backup_path)
        print(f"📁 백업 생성: {backup_path}")

        # 수정된 노트북 저장
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, ensure_ascii=False, indent=1)

        print(f"💾 수정 완료: {notebook_path.name} ({modified_cells}개 셀 수정)\n")
    else:
        print(f"ℹ️ {notebook_path.name}: 수정사항 없음\n")


def main():
    """메인 실행 함수"""
    notebooks_dir = Path(__file__).parent.parent / 'notebooks'

    # 수정할 노트북 파일 목록
    notebook_files = [
        '01_도메인_기반_부도원인_분석_Part1_데이터로딩_및_기본분석.ipynb',
        '01_심화_재무_분석.ipynb',
    ]

    print("=" * 80)
    print("금융 전문가 피드백 기반 노트북 수정 시작")
    print("=" * 80)
    print()

    for notebook_file in notebook_files:
        notebook_path = notebooks_dir / notebook_file

        if not notebook_path.exists():
            print(f"⚠️ 파일을 찾을 수 없습니다: {notebook_path}")
            continue

        print(f"📖 처리 중: {notebook_file}")
        print("-" * 80)

        # 적용할 수정사항
        fixes = [
            fix_debt_ratio_calculation,
            fix_icr_calculation,
            fix_zombie_company_terminology,
            fix_quick_ratio_comment,
        ]

        process_notebook(notebook_path, fixes)

    print("=" * 80)
    print("✅ 모든 노트북 수정 완료")
    print("=" * 80)


if __name__ == '__main__':
    main()
