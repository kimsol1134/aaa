"""
DART API → 예측 전체 파이프라인 단계별 디버깅

삼성전자 2024 데이터로 각 단계를 검증하고 문제 파악
"""

import sys
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

from config import DART_API_KEY
from src.dart_api import DartAPIClient, FinancialStatementParser
from src.domain_features.feature_generator import DomainFeatureGenerator
from src.models.predictor import BankruptcyPredictor
from config import PIPELINE_PATH

print("=" * 80)
print("🔍 DART API → 예측 파이프라인 디버깅")
print("=" * 80)

# ========== 1단계: DART API 조회 ==========
print("\n" + "=" * 80)
print("1단계: DART API 조회")
print("=" * 80)

client = DartAPIClient(DART_API_KEY)
company = client.search_company("삼성전자")
print(f"✓ 기업: {company['corp_name']} ({company['stock_code']})")

statements = client.get_financial_statements(
    corp_code=company['corp_code'],
    bsns_year='2024'
)

print(f"✓ 재무상태표: {len(statements['balance_sheet'])}개 항목")
print(f"✓ 손익계산서: {len(statements['income_statement'])}개 항목")
print(f"✓ 현금흐름표: {len(statements['cash_flow'])}개 항목")

# ========== 2단계: 재무제표 파싱 ==========
print("\n" + "=" * 80)
print("2단계: 재무제표 파싱")
print("=" * 80)

parser = FinancialStatementParser()
financial_data = parser.parse(statements)

print(f"✓ 파싱된 항목: {len(financial_data)}개")
print("\n주요 재무 항목:")
key_items = ['자산총계', '부채총계', '자본총계', '매출액', '영업이익', '당기순이익', '영업활동현금흐름']
for item in key_items:
    value = financial_data.get(item, 0)
    if value != 0:
        print(f"  {item}: {value:,.0f} (백만원)")
    else:
        print(f"  {item}: ❌ 없음")

# ========== 3단계: 기업 정보 조회 ==========
print("\n" + "=" * 80)
print("3단계: 기업 정보 조회 (DART company.json)")
print("=" * 80)

dart_company_info = client.get_company_info(company['corp_code'])

company_info = {
    'corp_name': company['corp_name'],
    'stock_code': company['stock_code'],
    'year': '2024',
    '업종코드': dart_company_info.get('업종코드', ''),
    '업력': dart_company_info.get('업력', 10),
    '종업원수': dart_company_info.get('종업원수', 100),
    '외감여부': dart_company_info.get('외감여부', True),
}

print(f"✓ 기업명: {company_info['corp_name']}")
print(f"  업종코드: {company_info['업종코드'] or '❌ 없음 (기본값 사용)'}")
print(f"  업력: {company_info['업력']}년")
print(f"  종업원수: {company_info['종업원수']}명")
print(f"  외감여부: {company_info['외감여부']}")

# ========== 4단계: 도메인 특성 생성 ==========
print("\n" + "=" * 80)
print("4단계: 도메인 특성 생성 (80개)")
print("=" * 80)

generator = DomainFeatureGenerator()
features_df = generator.generate_all_features(financial_data, company_info)

print(f"✓ 생성된 특성: {len(features_df.columns)}개")
print("\n주요 특성 샘플 (20개):")
sample_features = [
    '유동비율', '당좌비율', '현금소진일수', '부채비율', '이자보상배율',
    '매출총이익률', '영업이익률', 'ROA', 'ROE', '재고회전율',
    '유동성위기지수', '지급불능위험지수', '재무조작위험지수', '한국시장리스크지수',
    '종합부도위험스코어', '조기경보신호수', '재무건전성지수', '업력', '제조업여부', '외감여부'
]

for feature in sample_features:
    if feature in features_df.columns:
        value = features_df[feature].iloc[0]
        print(f"  {feature}: {value:.4f}")
    else:
        print(f"  {feature}: ❌ 없음")

# ========== 5단계: Feature 매핑 (80개 → 27개) ==========
print("\n" + "=" * 80)
print("5단계: Feature 매핑 확인")
print("=" * 80)

predictor = BankruptcyPredictor(pipeline_path=PIPELINE_PATH)
predictor.load_model()

# _prepare_features 직접 호출하여 매핑 확인
X_prepared = predictor._prepare_features(features_df.copy())

print(f"✓ 모델 입력 특성: {len(X_prepared.columns)}개")
print("\n27개 특성:")
for i, col in enumerate(X_prepared.columns, 1):
    value = X_prepared[col].iloc[0]
    print(f"  {i:2d}. {col:20s}: {value:10.4f}")

# ========== 6단계: 모델 예측 ==========
print("\n" + "=" * 80)
print("6단계: 모델 예측")
print("=" * 80)

result = predictor.predict(features_df)

print(f"✓ 부도 확률: {result['bankruptcy_probability']:.1%}")
print(f"  위험 등급: {result['risk_level']} {result.get('risk_icon', '')}")
print(f"  신뢰도: {result['confidence']:.1%}")
print(f"  모델 타입: {result['model_info']['model_type']}")
print(f"  사용 특성 수: {result['model_info']['n_features']}개")

if 'shap_values' in result:
    print(f"  SHAP: ✓ 계산됨")
else:
    print(f"  SHAP: ❌ 계산 실패")

# ========== 7단계: 문제 진단 ==========
print("\n" + "=" * 80)
print("7단계: 문제 진단")
print("=" * 80)

issues = []

# 1. 재무제표 항목 누락 확인
missing_items = []
for item in key_items:
    if financial_data.get(item, 0) == 0:
        missing_items.append(item)

if missing_items:
    issues.append(f"⚠️ 재무제표 누락 항목: {', '.join(missing_items)}")

# 2. 기업 정보 기본값 사용 확인
if not company_info['업종코드']:
    issues.append("⚠️ 업종코드 없음 → 한국시장특화 특성 부정확")

# 3. 특성 값 이상 확인
if features_df['종합부도위험스코어'].iloc[0] > 0.5:
    issues.append(f"⚠️ 종합부도위험스코어 높음: {features_df['종합부도위험스코어'].iloc[0]:.2f}")

# 4. 삼성전자인데 고위험 판정
if result['bankruptcy_probability'] > 0.1 and company['stock_code'] == '005930':
    issues.append(f"❌ 삼성전자가 {result['bankruptcy_probability']:.1%} 부도 확률 → 명백한 오류")

# 5. SHAP 실패
if 'shap_values' not in result:
    issues.append("⚠️ SHAP 계산 실패 → VotingClassifier 문제")

if issues:
    print("발견된 문제:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("✓ 문제 없음")

print("\n" + "=" * 80)
print("디버깅 완료")
print("=" * 80)
