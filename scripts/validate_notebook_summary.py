#!/usr/bin/env python3
"""
노트북 요약 문서 검증 스크립트

원본 노트북과 생성된 요약 문서를 비교하여
빠진 내용이나 문제점을 찾아냅니다.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_notebook(notebook_path: str) -> Dict:
    """노트북 파일 로드"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_summary(summary_path: str) -> str:
    """요약 문서 로드"""
    with open(summary_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_code(source) -> str:
    """소스 코드 추출"""
    if isinstance(source, list):
        return ''.join(source)
    return str(source)


def analyze_notebook(notebook: Dict) -> Dict:
    """노트북 분석"""
    cells = notebook.get('cells', [])

    stats = {
        'total_cells': len(cells),
        'markdown_cells': 0,
        'code_cells': 0,
        'empty_code_cells': 0,
        'code_with_output': 0,
        'code_without_output': 0,
        'cells_with_images': 0,
        'cells_with_html': 0,
        'cells_with_errors': 0,
        'important_functions': [],
        'important_classes': [],
        'data_files_loaded': [],
        'data_files_saved': [],
    }

    for cell in cells:
        cell_type = cell.get('cell_type', '')

        if cell_type == 'markdown':
            stats['markdown_cells'] += 1

        elif cell_type == 'code':
            stats['code_cells'] += 1
            source = extract_code(cell.get('source', ''))

            # 빈 셀 체크
            if not source.strip():
                stats['empty_code_cells'] += 1
                continue

            # 출력 체크
            outputs = cell.get('outputs', [])
            if outputs:
                stats['code_with_output'] += 1

                # 출력 타입 분석
                for output in outputs:
                    output_type = output.get('output_type', '')

                    if output_type == 'error':
                        stats['cells_with_errors'] += 1

                    data = output.get('data', {})
                    if 'image/png' in data:
                        stats['cells_with_images'] += 1
                    if 'text/html' in data:
                        stats['cells_with_html'] += 1
            else:
                stats['code_without_output'] += 1

            # 중요한 함수/클래스 찾기
            lines = source.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('def ') and '(' in line:
                    func_name = line[4:line.index('(')].strip()
                    if not func_name.startswith('_'):
                        stats['important_functions'].append(func_name)
                elif line.startswith('class ') and ':' in line:
                    class_name = line[6:line.index(':')].strip()
                    stats['important_classes'].append(class_name)

            # 데이터 파일 로딩/저장 찾기
            if 'pd.read_csv' in source or 'pd.read_excel' in source:
                # 파일명 추출 시도
                for line in lines:
                    if 'read_csv' in line or 'read_excel' in line:
                        stats['data_files_loaded'].append(line.strip()[:100])

            if '.to_csv' in source or '.to_excel' in source or 'joblib.dump' in source:
                for line in lines:
                    if '.to_csv' in line or '.to_excel' in line or 'joblib.dump' in line:
                        stats['data_files_saved'].append(line.strip()[:100])

    return stats


def check_summary_coverage(notebook_stats: Dict, summary: str) -> List[str]:
    """요약 문서가 중요한 내용을 커버하는지 확인"""
    issues = []

    # 함수 커버리지 확인
    missing_functions = []
    for func_name in notebook_stats['important_functions']:
        if f"def {func_name}" not in summary:
            missing_functions.append(func_name)

    if missing_functions:
        issues.append(f"❌ 빠진 함수: {', '.join(missing_functions[:10])}")
        if len(missing_functions) > 10:
            issues.append(f"   ... 외 {len(missing_functions) - 10}개 더")

    # 클래스 커버리지 확인
    missing_classes = []
    for class_name in notebook_stats['important_classes']:
        if f"class {class_name}" not in summary:
            missing_classes.append(class_name)

    if missing_classes:
        issues.append(f"❌ 빠진 클래스: {', '.join(missing_classes)}")

    # 데이터 파일 로딩 확인
    if notebook_stats['data_files_loaded']:
        if 'read_csv' not in summary and 'read_excel' not in summary:
            issues.append(f"⚠️  데이터 로딩 코드가 요약에 없을 수 있음")

    # 데이터 파일 저장 확인
    if notebook_stats['data_files_saved']:
        found_save = False
        for save_line in notebook_stats['data_files_saved']:
            if any(keyword in summary for keyword in ['.to_csv', '.to_excel', 'joblib.dump']):
                found_save = True
                break
        if not found_save:
            issues.append(f"⚠️  데이터 저장 코드가 요약에 없을 수 있음")

    # 이미지/차트 확인
    if notebook_stats['cells_with_images'] > 0:
        image_mentions = summary.count('[이미지:') + summary.count('PNG')
        if image_mentions < notebook_stats['cells_with_images'] * 0.5:
            issues.append(f"⚠️  이미지/차트 출력 수: 원본 {notebook_stats['cells_with_images']}개, 요약에서 ~{image_mentions}개 언급")

    return issues


def validate_notebook_summary(notebook_path: str, summary_path: str) -> Tuple[Dict, List[str]]:
    """노트북과 요약 문서 검증"""

    # 노트북 분석
    notebook = load_notebook(notebook_path)
    notebook_stats = analyze_notebook(notebook)

    # 요약 문서 로드
    summary = load_summary(summary_path)

    # 커버리지 확인
    issues = check_summary_coverage(notebook_stats, summary)

    return notebook_stats, issues


def main():
    """메인 실행 함수"""

    project_root = Path(__file__).parent.parent
    notebooks_dir = project_root / 'notebooks'
    summaries_dir = project_root / 'docs' / 'notebook_summaries'

    # 검증할 노트북 목록
    notebooks_to_validate = [
        ('02_고급_도메인_특성공학.ipynb', '02_고급_도메인_특성공학_summary.md'),
        ('01_심화_재무_분석.ipynb', '01_심화_재무_분석_summary.md'),
        ('01_도메인_기반_부도원인_분석.ipynb', '01_도메인_기반_부도원인_분석_summary.md'),
    ]

    print("=" * 80)
    print("노트북 요약 문서 검증")
    print("=" * 80)

    for notebook_name, summary_name in notebooks_to_validate:
        notebook_path = notebooks_dir / notebook_name
        summary_path = summaries_dir / summary_name

        if not notebook_path.exists():
            print(f"\n⚠️  노트북 파일을 찾을 수 없음: {notebook_path}")
            continue

        if not summary_path.exists():
            print(f"\n⚠️  요약 파일을 찾을 수 없음: {summary_path}")
            continue

        print(f"\n{'='*80}")
        print(f"📓 검증 중: {notebook_name}")
        print(f"{'='*80}")

        try:
            stats, issues = validate_notebook_summary(str(notebook_path), str(summary_path))

            # 통계 출력
            print(f"\n📊 원본 노트북 통계:")
            print(f"   - 전체 셀: {stats['total_cells']}개")
            print(f"   - 마크다운 셀: {stats['markdown_cells']}개")
            print(f"   - 코드 셀: {stats['code_cells']}개")
            print(f"   - 출력 있는 코드 셀: {stats['code_with_output']}개")
            print(f"   - 출력 없는 코드 셀: {stats['code_without_output']}개")
            print(f"   - 빈 코드 셀: {stats['empty_code_cells']}개")
            print(f"   - 이미지/차트 출력: {stats['cells_with_images']}개")
            print(f"   - HTML 출력: {stats['cells_with_html']}개")
            print(f"   - 에러 출력: {stats['cells_with_errors']}개")

            if stats['important_functions']:
                print(f"\n🔧 정의된 함수: {len(stats['important_functions'])}개")
                if stats['important_functions'][:5]:
                    print(f"   예시: {', '.join(stats['important_functions'][:5])}")

            if stats['important_classes']:
                print(f"\n📦 정의된 클래스: {len(stats['important_classes'])}개")
                print(f"   {', '.join(stats['important_classes'])}")

            if stats['data_files_loaded']:
                print(f"\n📥 데이터 로딩: {len(stats['data_files_loaded'])}회")

            if stats['data_files_saved']:
                print(f"\n💾 데이터 저장: {len(stats['data_files_saved'])}회")

            # 이슈 출력
            if issues:
                print(f"\n⚠️  발견된 이슈:")
                for issue in issues:
                    print(f"   {issue}")
            else:
                print(f"\n✅ 이슈 없음! 요약 문서가 원본을 잘 커버하고 있습니다.")

        except Exception as e:
            print(f"❌ 에러 발생: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("검증 완료!")
    print("=" * 80)


if __name__ == '__main__':
    main()
