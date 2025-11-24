"""
실제 로드되는 모델 파일 확인
"""
import joblib
from pathlib import Path

model_path = Path("data/processed/final_model.pkl")

if model_path.exists():
    print(f"✓ 모델 파일 존재: {model_path}")
    print(f"  크기: {model_path.stat().st_size / 1024 / 1024:.2f} MB")

    model = joblib.load(model_path)

    print(f"\n모델 타입: {type(model)}")
    print(f"모델 클래스: {model.__class__.__name__}")

    # VotingClassifier인 경우
    if hasattr(model, 'estimators_'):
        print(f"\n✓ VotingClassifier 확인:")
        print(f"  estimators_ 개수: {len(model.estimators_)}")

        if hasattr(model, 'estimators'):
            print(f"  estimators:")
            for name, est in model.estimators:
                print(f"    - {name}: {type(est).__name__}")

        print(f"\n  fitted estimators_:")
        for i, est in enumerate(model.estimators_):
            print(f"    - [{i}] {type(est).__name__}")

    # Pipeline인 경우
    elif hasattr(model, 'steps'):
        print(f"\n✓ Pipeline 확인:")
        print(f"  steps:")
        for name, step in model.steps:
            print(f"    - {name}: {type(step).__name__}")

    # 단일 모델인 경우
    else:
        print(f"\n✓ 단일 모델")
        if hasattr(model, 'get_params'):
            params = model.get_params()
            print(f"  주요 파라미터:")
            for key in list(params.keys())[:10]:
                print(f"    - {key}: {params[key]}")
else:
    print(f"❌ 모델 파일 없음: {model_path}")

    # 다른 모델 파일 찾기
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        print(f"\ndata/processed 폴더의 .pkl 파일:")
        for pkl_file in processed_dir.glob("*.pkl"):
            size_mb = pkl_file.stat().st_size / 1024 / 1024
            print(f"  - {pkl_file.name}: {size_mb:.2f} MB")
