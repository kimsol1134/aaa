#!/usr/bin/env python3
"""
노트북 요약 문서 생성 스크립트

큰 Jupyter 노트북 파일을 Claude Code가 읽을 수 있도록
중요한 로직과 출력 결과를 포함한 마크다운 요약 문서로 변환합니다.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any


def extract_text_from_output(output: Dict[str, Any]) -> str:
    """출력 셀에서 텍스트 추출"""
    text_parts = []

    output_type = output.get('output_type', '')

    if output_type == 'stream':
        # print() 출력
        text = output.get('text', '')
        if isinstance(text, list):
            text = ''.join(text)
        text_parts.append(text.strip())

    elif output_type == 'execute_result':
        # 셀 실행 결과 (마지막 표현식)
        data = output.get('data', {})
        if 'text/plain' in data:
            text = data['text/plain']
            if isinstance(text, list):
                text = ''.join(text)
            text_parts.append(text.strip())
        if 'text/html' in data:
            # HTML 테이블 등
            html = data['text/html']
            if isinstance(html, list):
                html = ''.join(html)
            # HTML 간단히 표시
            text_parts.append(f"<HTML 출력: {len(html)} 문자>")

    elif output_type == 'display_data':
        # 그래프, 이미지 등
        data = output.get('data', {})
        if 'text/plain' in data:
            text = data['text/plain']
            if isinstance(text, list):
                text = ''.join(text)
            text_parts.append(text.strip())
        if 'image/png' in data:
            text_parts.append("[이미지: PNG 차트/그래프]")
        if 'text/html' in data:
            html = data['text/html']
            if isinstance(html, list):
                html = ''.join(html)
            text_parts.append(f"<HTML 출력: {len(html)} 문자>")

    elif output_type == 'error':
        # 에러 메시지
        ename = output.get('ename', '')
        evalue = output.get('evalue', '')
        text_parts.append(f"ERROR: {ename}: {evalue}")

    return '\n'.join(text_parts)


def summarize_output(output_text: str, max_lines: int = 50) -> str:
    """출력이 너무 길면 요약"""
    lines = output_text.split('\n')

    if len(lines) <= max_lines:
        return output_text

    # 처음 30줄 + ... + 마지막 15줄
    head = '\n'.join(lines[:30])
    tail = '\n'.join(lines[-15:])
    omitted = len(lines) - 45

    return f"{head}\n\n... ({omitted}줄 생략) ...\n\n{tail}"


def extract_code(source: Any) -> str:
    """소스 코드 추출"""
    if isinstance(source, list):
        return ''.join(source)
    return str(source)


def is_important_cell(cell: Dict[str, Any]) -> bool:
    """중요한 셀인지 판단 (간단한 휴리스틱)"""
    cell_type = cell.get('cell_type', '')

    # 마크다운은 모두 중요
    if cell_type == 'markdown':
        return True

    if cell_type != 'code':
        return False

    source = extract_code(cell.get('source', ''))

    # 빈 셀은 제외
    if not source.strip():
        return False

    # 주석만 있는 셀은 제외
    lines = [line.strip() for line in source.split('\n') if line.strip()]
    if all(line.startswith('#') for line in lines):
        return False

    # 간단한 import만 있는 셀 (중요하지만 간략히)
    if len(lines) <= 3 and all(
        line.startswith('import ') or line.startswith('from ') or line.startswith('#')
        for line in lines
    ):
        return True

    return True


def create_notebook_summary(notebook_path: str, output_path: str, max_output_lines: int = 50):
    """노트북을 요약 마크다운 문서로 변환"""

    # 노트북 로드
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook.get('cells', [])
    notebook_name = Path(notebook_path).name

    # 마크다운 생성
    md_lines = []
    md_lines.append(f"# {notebook_name} - 요약 문서\n")
    md_lines.append(f"> 원본: `{notebook_path}`\n")
    md_lines.append(f"> 자동 생성됨: 중요한 로직과 출력 결과 포함\n")
    md_lines.append("---\n")

    code_cell_count = 0
    markdown_cell_count = 0

    for idx, cell in enumerate(cells):
        cell_type = cell.get('cell_type', '')

        if not is_important_cell(cell):
            continue

        if cell_type == 'markdown':
            markdown_cell_count += 1
            source = extract_code(cell.get('source', ''))
            md_lines.append(f"{source}\n")

        elif cell_type == 'code':
            code_cell_count += 1
            source = extract_code(cell.get('source', ''))

            # 코드 블록
            md_lines.append(f"### 코드 셀 #{code_cell_count}\n")
            md_lines.append("```python")
            md_lines.append(source)
            md_lines.append("```\n")

            # 출력 결과
            outputs = cell.get('outputs', [])
            if outputs:
                md_lines.append("**출력:**\n")

                for output in outputs:
                    output_text = extract_text_from_output(output)
                    if output_text:
                        # 출력이 너무 길면 요약
                        summarized = summarize_output(output_text, max_output_lines)
                        md_lines.append("```")
                        md_lines.append(summarized)
                        md_lines.append("```\n")

            md_lines.append("---\n")

    # 파일 저장
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    print(f"✅ 요약 문서 생성 완료: {output_path}")
    print(f"   - 마크다운 셀: {markdown_cell_count}개")
    print(f"   - 코드 셀: {code_cell_count}개")
    print(f"   - 파일 크기: {os.path.getsize(output_path) / 1024:.1f} KB")


def main():
    """메인 실행 함수"""

    # 프로젝트 루트 디렉토리
    project_root = Path(__file__).parent.parent
    notebooks_dir = project_root / 'notebooks'
    summaries_dir = project_root / 'docs' / 'notebook_summaries'

    # 요약할 노트북 목록
    notebooks_to_summarize = [
        '01_도메인_기반_부도원인_분석.ipynb',
        '01_심화_재무_분석.ipynb',
        '02_고급_도메인_특성공학.ipynb',
    ]

    print("=" * 80)
    print("노트북 요약 문서 생성 시작")
    print("=" * 80)

    for notebook_name in notebooks_to_summarize:
        notebook_path = notebooks_dir / notebook_name

        if not notebook_path.exists():
            print(f"⚠️  파일을 찾을 수 없습니다: {notebook_path}")
            continue

        # 출력 파일명
        output_name = notebook_name.replace('.ipynb', '_summary.md')
        output_path = summaries_dir / output_name

        print(f"\n📓 처리 중: {notebook_name}")
        print(f"   원본 크기: {os.path.getsize(notebook_path) / 1024 / 1024:.1f} MB")

        try:
            create_notebook_summary(
                str(notebook_path),
                str(output_path),
                max_output_lines=50
            )
        except Exception as e:
            print(f"❌ 에러 발생: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"✅ 모든 요약 문서 생성 완료!")
    print(f"📁 저장 위치: {summaries_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
