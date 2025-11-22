#!/usr/bin/env python3
"""
상세 노트북 비교 스크립트

원본 노트북의 모든 셀을 텍스트로 출력하여
요약 문서와 직접 비교할 수 있게 합니다.
"""

import json
from pathlib import Path
from typing import Dict, List


def extract_code(source) -> str:
    """소스 코드 추출"""
    if isinstance(source, list):
        return ''.join(source)
    return str(source)


def extract_output_text(output: Dict) -> str:
    """출력 텍스트 추출"""
    text_parts = []
    output_type = output.get('output_type', '')

    if output_type == 'stream':
        text = output.get('text', '')
        if isinstance(text, list):
            text = ''.join(text)
        text_parts.append(text)

    elif output_type == 'execute_result':
        data = output.get('data', {})
        if 'text/plain' in data:
            text = data['text/plain']
            if isinstance(text, list):
                text = ''.join(text)
            text_parts.append(text)

    elif output_type == 'display_data':
        data = output.get('data', {})
        if 'text/plain' in data:
            text = data['text/plain']
            if isinstance(text, list):
                text = ''.join(text)
            text_parts.append(text)
        if 'image/png' in data:
            text_parts.append("[이미지 출력]")

    elif output_type == 'error':
        ename = output.get('ename', '')
        evalue = output.get('evalue', '')
        text_parts.append(f"ERROR: {ename}: {evalue}")

    return '\n'.join(text_parts)


def print_notebook_content(notebook_path: str, output_file: str):
    """노트북 내용을 텍스트 파일로 출력"""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook.get('cells', [])

    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(f"# 노트북 내용: {Path(notebook_path).name}\n\n")
        out.write(f"총 셀 개수: {len(cells)}\n")
        out.write("=" * 80 + "\n\n")

        markdown_count = 0
        code_count = 0

        for idx, cell in enumerate(cells, 1):
            cell_type = cell.get('cell_type', '')

            if cell_type == 'markdown':
                markdown_count += 1
                source = extract_code(cell.get('source', ''))

                out.write(f"## [마크다운 #{markdown_count}]\n\n")
                out.write(source)
                out.write("\n\n" + "-" * 80 + "\n\n")

            elif cell_type == 'code':
                code_count += 1
                source = extract_code(cell.get('source', ''))

                # 빈 셀은 건너뛰기
                if not source.strip():
                    continue

                out.write(f"## [코드 #{code_count}]\n\n")
                out.write("```python\n")
                out.write(source)
                out.write("\n```\n\n")

                # 출력 결과
                outputs = cell.get('outputs', [])
                if outputs:
                    out.write("**출력:**\n\n")
                    for output in outputs:
                        output_text = extract_output_text(output)
                        if output_text:
                            out.write("```\n")
                            # 출력이 너무 길면 자르기
                            lines = output_text.split('\n')
                            if len(lines) > 50:
                                out.write('\n'.join(lines[:30]))
                                out.write(f"\n\n... ({len(lines) - 45}줄 생략) ...\n\n")
                                out.write('\n'.join(lines[-15:]))
                            else:
                                out.write(output_text)
                            out.write("\n```\n\n")

                out.write("-" * 80 + "\n\n")

        out.write(f"\n\n총계: 마크다운 {markdown_count}개, 코드 {code_count}개\n")


def compare_with_summary(notebook_output: str, summary_path: str):
    """노트북 출력과 요약 문서 비교"""

    with open(notebook_output, 'r', encoding='utf-8') as f:
        notebook_content = f.read()

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary_content = f.read()

    print(f"\n비교 분석:")
    print(f"  노트북 출력 크기: {len(notebook_content):,} 문자")
    print(f"  요약 문서 크기: {len(summary_content):,} 문자")
    print(f"  압축률: {len(summary_content) / len(notebook_content) * 100:.1f}%")

    # 함수 정의 확인
    notebook_functions = set()
    summary_functions = set()

    for line in notebook_content.split('\n'):
        if line.strip().startswith('def '):
            func_name = line.strip()[4:line.strip().index('(')].strip()
            notebook_functions.add(func_name)

    for line in summary_content.split('\n'):
        if line.strip().startswith('def '):
            func_name = line.strip()[4:line.strip().index('(')].strip()
            summary_functions.add(func_name)

    missing_functions = notebook_functions - summary_functions
    if missing_functions:
        print(f"\n  ⚠️  요약에 빠진 함수: {missing_functions}")
    else:
        print(f"\n  ✅ 모든 함수가 요약에 포함됨 ({len(notebook_functions)}개)")

    # 중요 키워드 확인
    important_keywords = [
        'pd.read_csv',
        'to_csv',
        'joblib.dump',
        'plt.figure',
        'sns.',
    ]

    print(f"\n  중요 키워드 출현 빈도 비교:")
    for keyword in important_keywords:
        nb_count = notebook_content.count(keyword)
        sum_count = summary_content.count(keyword)
        if nb_count > 0:
            coverage = sum_count / nb_count * 100
            status = "✅" if coverage >= 80 else "⚠️ "
            print(f"    {status} '{keyword}': 노트북 {nb_count}회, 요약 {sum_count}회 ({coverage:.0f}%)")


def main():
    """메인 실행"""

    project_root = Path(__file__).parent.parent
    notebooks_dir = project_root / 'notebooks'
    summaries_dir = project_root / 'docs' / 'notebook_summaries'
    output_dir = project_root / 'docs' / 'notebook_analysis'
    output_dir.mkdir(exist_ok=True)

    notebooks = [
        ('02_고급_도메인_특성공학.ipynb', '02_고급_도메인_특성공학_summary.md'),
        ('01_심화_재무_분석.ipynb', '01_심화_재무_분석_summary.md'),
        ('01_도메인_기반_부도원인_분석.ipynb', '01_도메인_기반_부도원인_분석_summary.md'),
    ]

    print("=" * 80)
    print("노트북 상세 비교 분석")
    print("=" * 80)

    for notebook_name, summary_name in notebooks:
        notebook_path = notebooks_dir / notebook_name
        summary_path = summaries_dir / summary_name

        if not notebook_path.exists() or not summary_path.exists():
            continue

        print(f"\n{'='*80}")
        print(f"📓 분석 중: {notebook_name}")
        print(f"{'='*80}")

        output_file = output_dir / notebook_name.replace('.ipynb', '_full_content.txt')

        # 노트북 내용 추출
        print(f"  노트북 내용 추출 중...")
        print_notebook_content(str(notebook_path), str(output_file))
        print(f"  ✅ 저장됨: {output_file}")

        # 요약과 비교
        compare_with_summary(str(output_file), str(summary_path))

    print(f"\n{'='*80}")
    print(f"✅ 분석 완료!")
    print(f"📁 출력 위치: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
