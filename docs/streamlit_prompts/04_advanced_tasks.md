# Part 4: 고급 작업 (SHAP, 비즈니스 가치)

> **읽기 시간**: 20분 | **난이도**: ⭐⭐⭐ 고급

---

## Task 4: 실제 SHAP 값 계산 ⭐⭐⭐⭐

### 목표
하드코딩된 "SHAP-style" 차트를 실제 SHAP 라이브러리 기반으로 교체

### Step 1: SHAP 설치
```bash
echo "shap>=0.41.0" >> requirements.txt
pip install shap
```

### Step 2: `src/models/predictor.py` 수정

**`predict` 메서드에 SHAP 계산 추가**:
```python
def predict(self, features_df: pd.DataFrame) -> Dict:
    # ... (기존 예측 로직) ...

    # 🆕 SHAP 값 계산
    try:
        import shap
        explainer = shap.TreeExplainer(self.model)
        shap_values_result = explainer.shap_values(X_scaled)

        # CatBoost는 리스트 반환 → 부도(1) 클래스만 사용
        if isinstance(shap_values_result, list):
            shap_values = shap_values_result[1][0]
        else:
            shap_values = shap_values_result[0]

        result['shap_values'] = shap_values.tolist()
        result['shap_base_value'] = float(explainer.expected_value)
        result['feature_names'] = list(X.columns)
    except Exception as e:
        logger.warning(f"SHAP 계산 실패: {e}")
        result['shap_values'] = None

    return result
```

### Step 3: `src/visualization/charts.py` - 실제 SHAP Waterfall

**기존 `create_shap_waterfall` 삭제 후 새로 작성**:
```python
def create_shap_waterfall_real(
    shap_values: np.ndarray,
    feature_values: pd.Series,
    feature_names: List[str],
    base_value: float,
    max_display: int = 10
) -> go.Figure:
    """실제 SHAP 값 기반 Waterfall"""

    # 절대값 기준 상위 N개
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[-max_display:][::-1]

    top_features = [feature_names[i] for i in top_indices]
    top_shap_values = [shap_values[i] for i in top_indices]

    # Waterfall 차트 생성
    fig = go.Figure(go.Waterfall(
        x=["기준값"] + top_features + ["최종값"],
        y=[base_value] + top_shap_values + [sum(shap_values)],
        measure=["absolute"] + ["relative"] * max_display + ["total"],
        increasing={"marker": {"color": "#FF6B6B"}},  # 빨강
        decreasing={"marker": {"color": "#51CF66"}},  # 초록
    ))

    return fig
```

### Step 4: `streamlit_app/app.py`에서 호출

```python
def display_risk_analysis(result, features_df):
    if result.get('shap_values'):
        fig = create_shap_waterfall_real(
            shap_values=np.array(result['shap_values']),
            feature_values=features_df.iloc[0],
            feature_names=result['feature_names'],
            base_value=result['shap_base_value']
        )
        st.plotly_chart(fig)
    else:
        st.warning("SHAP 값 없음")
```

---

## Task 5: 비즈니스 가치 분석 추가 ⭐⭐⭐

### 목표
Part 4 노트북의 ROI, Confusion Matrix를 앱에 추가

### Step 1: 새 모듈 생성

**파일**: `src/utils/business_value.py`

```python
class BusinessValueCalculator:
    def __init__(
        self,
        avg_loan=5_000_000,
        avg_interest=500_000,
        recovery_rate=0.3
    ):
        self.avg_loan = avg_loan
        self.avg_interest = avg_interest
        self.recovery_rate = recovery_rate

    def calculate_single_company(self, prob: float):
        expected_loss = prob * self.avg_loan * (1 - self.recovery_rate)
        expected_profit = (1 - prob) * self.avg_interest
        return {
            'expected_loss': expected_loss,
            'expected_profit': expected_profit,
            'net': expected_profit - expected_loss
        }
```

### Step 2: 앱에 섹션 추가

**파일**: `streamlit_app/app.py`

```python
def display_business_value(result, features_df):
    st.markdown("## 💰 비즈니스 가치 분석")

    calc = BusinessValueCalculator()
    value = calc.calculate_single_company(result['bankruptcy_probability'])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("예상 손실", f"{value['expected_loss']:,.0f}원")
    with col2:
        st.metric("예상 수익", f"{value['expected_profit']:,.0f}원")
    with col3:
        st.metric("순 기대값", f"{value['net']:,.0f}원")

    # Part 4 노트북 결과 표시
    st.markdown("""
    ### 📊 모델 성능 (Test Set)
    - ROI: **920%**
    - Payback: **1.3개월**
    - 연간 절감: **460M KRW**
    """)
```

---

## 📋 전체 검증 체크리스트

### ✅ Critical Tasks
- [ ] Task 1: 모델 파일 `발표_Part3_v3_최종모델.pkl` 로드 성공
- [ ] Task 2: Traffic Light 임계값 0.0168, 0.0468 적용
- [ ] Task 3: 51개 특성 생성 확인

### ✅ Advanced Tasks
- [ ] Task 4: 실제 SHAP Waterfall 차트 표시
- [ ] Task 5: 비즈니스 가치 분석 섹션 추가

### ✅ 전체 플로우
- [ ] DART API로 "삼성전자" 조회 성공
- [ ] 재무제표 → 51개 특성 생성
- [ ] CatBoost 모델 예측
- [ ] Traffic Light 등급 정확
- [ ] SHAP 차트 실제 값 표시
- [ ] 에러 없이 완료

---

**작성일**: 2025-11-23
