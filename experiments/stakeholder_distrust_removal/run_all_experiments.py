"""
전체 실험 실행 스크립트
Week 1, 2, 3 실험을 순차적으로 실행하고 결과 종합
"""

import sys
sys.path.append('/home/user/aaa/experiments/stakeholder_distrust_removal/scripts')

import subprocess
from pathlib import Path
from datetime import datetime
import json

# 실험 디렉토리
BASE_DIR = Path('/home/user/aaa/experiments/stakeholder_distrust_removal')

# 실험 스크립트 목록
EXPERIMENTS = {
    'week1': [
        'week1_diagnosis/exp1_kfold_cv.py',
        'week1_diagnosis/exp2_distribution_comparison.py',
        'week1_diagnosis/exp3_smote_ablation.py'
    ],
    'week2': [
        'week2_feature_engineering/exp1_credit_rating_transformation.py',
        'week2_feature_engineering/exp2_vif_based_removal.py'
    ]
}


def run_experiment(script_path):
    """개별 실험 실행"""
    print(f"\n{'='*100}")
    print(f"실행: {script_path}")
    print(f"{'='*100}\n")

    result = subprocess.run(
        ['python3', str(script_path)],
        cwd=str(BASE_DIR),
        capture_output=False,
        text=True
    )

    return result.returncode == 0


def main():
    """전체 실험 실행"""
    print("=" * 100)
    print("이해관계자_불신지수 제거 모델 - 전체 실험 실행")
    print("=" * 100)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    results_summary = []
    total_experiments = sum(len(exps) for exps in EXPERIMENTS.values())
    completed = 0
    failed = 0

    for week, scripts in EXPERIMENTS.items():
        print(f"\n{'='*100}")
        print(f"{week.upper()} 실험 시작")
        print(f"{'='*100}")

        for script in scripts:
            script_path = BASE_DIR / script
            experiment_name = script_path.stem

            if script_path.exists():
                success = run_experiment(script_path)

                if success:
                    completed += 1
                    status = '✅ 성공'
                else:
                    failed += 1
                    status = '❌ 실패'

                results_summary.append({
                    'week': week,
                    'experiment': experiment_name,
                    'status': status,
                    'script': str(script)
                })
            else:
                print(f"⚠️ 스크립트 없음: {script_path}")
                results_summary.append({
                    'week': week,
                    'experiment': experiment_name,
                    'status': '⚠️ 스크립트 없음',
                    'script': str(script)
                })

    # 결과 요약
    print(f"\n\n{'='*100}")
    print("전체 실험 결과 요약")
    print(f"{'='*100}\n")

    print(f"총 실험 수: {total_experiments}개")
    print(f"완료: {completed}개")
    print(f"실패: {failed}개")
    print(f"성공률: {completed/total_experiments*100:.1f}%\n")

    print("실험별 결과:")
    for result in results_summary:
        print(f"  {result['status']} {result['week']:6s} | {result['experiment']}")

    print(f"\n종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}")

    # JSON 저장
    summary_path = BASE_DIR / 'results' / f'experiment_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 실험 요약 저장: {summary_path}")


if __name__ == '__main__':
    main()
