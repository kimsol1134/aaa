"""
Part3 모델이 기대하는 feature 이름 확인
"""
import joblib
from pathlib import Path

model_path = Path("data/processed/발표_Part3_v3_최종모델.pkl")

print("=" * 80)
print("Part3 v3 모델 검사")
print("=" * 80)

model = joblib.load(model_path)

print(f"\n모델 타입: {type(model)}")

if hasattr(model, 'steps'):
    print(f"\nPipeline 단계: {len(model.steps)}개")
    for step_name, step in model.steps:
        print(f"  - {step_name}: {type(step).__name__}")

    # 마지막 단계 (CatBoost) 확인
    final_step = model.steps[-1][1]
    print(f"\n최종 분류기: {type(final_step).__name__}")

    # Feature names 확인
    if hasattr(final_step, 'feature_names_'):
        print(f"\n✓ 학습 시 사용한 Feature 이름 ({len(final_step.feature_names_)}개):")
        for i, fname in enumerate(final_step.feature_names_, 1):
            print(f"  {i:2d}. {fname}")
    else:
        print("\n⚠️ feature_names_ 속성 없음")

    # 다른 속성 확인
    if hasattr(final_step, 'get_feature_names'):
        print("\n✓ get_feature_names() 메서드 있음")
        try:
            names = final_step.get_feature_names()
            print(f"  Feature 개수: {len(names)}")
        except:
            print("  호출 실패")

    if hasattr(final_step, 'feature_importances_'):
        print(f"\n✓ Feature importances: {len(final_step.feature_importances_)}개")
else:
    print("\n단일 모델 (Pipeline 아님)")
