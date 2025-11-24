import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add project root to path
project_root = Path('/Users/solkim/Dev/junwoo/deployment')
sys.path.insert(0, str(project_root))

from src.models.predictor import BankruptcyPredictor
import src.models.predictor
print(f"PREDICTOR MODULE FILE: {src.models.predictor.__file__}")
from src.domain_features.feature_generator import DomainFeatureGenerator
from config import PIPELINE_PATH, MODEL_PATH, SCALER_PATH
# Import transformers to ensure they are available for unpickling
from src.preprocessing.transformers import InfiniteHandler, LogTransformer, Winsorizer

# Patch __main__ to include these classes if necessary (sometimes needed for pickle)
import __main__
__main__.InfiniteHandler = InfiniteHandler
__main__.LogTransformer = LogTransformer
__main__.Winsorizer = Winsorizer

def test_full_flow():
    print("Testing full flow with real Samsung data...")
    
    # 1. Mock Samsung 2024 Data (based on fetched data)
    financial_data = {
        '유동자산': 218805227.0,
        '유동부채': 93326299.0,
        '현금및현금성자산': 97391652.0,
        '재고자산': 55255996.0,
        '단기금융상품': 81708933.0,
        '자산총계': 514531948.0,
        '매출원가': 186562268.0,
        '판매비와관리비': 81582674.0,
        '영업활동현금흐름': 72982621.0,
        '당기순이익': 34451351.0,
        '부채총계': 112339878.0,
        '자본총계': 402192070.0,
        '비유동자산': 295726721.0,
        '비유동부채': 19013579.0,
        '영업이익': 32725961.0,
        '이자비용': 0.0,
        '단기차입금': 13172504.0,
        '장기차입금': 3935860.0,
        '매출액': 300870903.0,
        '매출총이익': 114308635.0,
        '매출채권': 40000000.0,
        '매입채무': 12370177.0,
    }
    
    company_info = {
        '업력': 55,
        '외감여부': True,
        '업종코드': 'C26',
        '종업원수': 100000,
        '연체여부': False,
        '신용등급': 'AAA',
        '총연체건수': 0,       # Added for 연체심각도
        '세금체납여부': 0,     # Added for 공공정보리스크
    }

    # 2. Generate Features
    print("Generating features...")
    generator = DomainFeatureGenerator()
    features_df = generator.generate_all_features(financial_data, company_info)
    
    print(f"Generated {len(features_df.columns)} features.")

    # 3. Load Model and Predict
    print("\nLoading model and predicting...")
    try:
        predictor = BankruptcyPredictor(
            pipeline_path=PIPELINE_PATH,
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            use_pipeline=True
        )
        predictor.load_model()
        
        result = predictor.predict(features_df)
        print("\nPrediction Result:")
        # Print keys to check if 'shap_values' is present
        print(result.keys())
        
        if 'shap_values' in result:
            print("✓ SHAP values present")
            print(f"SHAP base value: {result.get('shap_base_value')}")
            # Check type of shap values
            shap_vals = result['shap_values']
            print(f"SHAP values type: {type(shap_vals)}")
            if isinstance(shap_vals, list):
                 print(f"First SHAP value: {shap_vals[0]} (Type: {type(shap_vals[0])})")
        else:
            print("✗ SHAP values MISSING")

        if result['model_info']['model_type'] == 'Heuristic':
            print("\nWARNING: Prediction fell back to Heuristic!")
        else:
            print("\nSUCCESS: Prediction used the ML model.")
            
    except Exception as e:
        print(f"\nFAILURE: Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_flow()
