# 02_고급_도메인_특성공학.ipynb - 요약 문서

> 원본: `/home/user/aaa/notebooks/02_고급_도메인_특성공학.ipynb`

> 자동 생성됨: 중요한 로직과 출력 결과 포함

---

# 🔧 고급 도메인 특성 공학

## 목표: "왜 기업이 부도가 나는가?"에 대한 도메인 지식을 특성으로 구현

이 노트북에서는 단순한 통계적 특성이 아닌, 부도의 근본 원인을 포착하는 도메인 특화 특성을 생성합니다.

## 📌 Why: 왜 도메인 기반 특성이 필요한가?

### 1️⃣ 문제 인식: 원본 데이터의 한계

**원본 데이터 (159개 변수)의 문제점:**

- ❌ **정적 스냅샷에 불과**: 재무제표 항목 중심 (자산, 부채, 매출 등) → 특정 시점의 재무 상태만 보여줌
- ❌ **부도의 "원인"을 직접 설명하지 못함**: "유동자산 = 1억원"이라는 정보만으로는 기업이 위험한지 알 수 없음
- ❌ **한국 시장 특성 미반영**: 외부감사 의무, 제조업 중심 산업구조, 대기업 의존도 등 한국 특유의 리스크 요인 누락
- ❌ **신용 행동과 재무정보의 단절**: 연체 이력, 신용등급 등 행동 패턴 정보가 재무 지표와 따로 존재

**예시로 보는 한계:**
```
기업 A: 유동자산 1억원, 유동부채 5천만원
→ 이것만으로는 안전한지 위험한지 판단 불가
→ 유동비율(200%)을 계산해야 함 → 하지만 이것도 부족
→ 현금비율, 현금소진일수, 운전자본 회전율 등 추가 지표 필요
```

**결론: 원본 데이터는 "재료"일 뿐, "부도 위험"을 직접 측정하는 "지표"가 아님**

---

### 2️⃣ 도메인 지식: 기업이 부도나는 3가지 경로

**학계 및 실무 연구 기반 (Altman 1968; Ohlson 1980; 한국은행 2020)**

#### 🔴 경로 1: 유동성 위기 (Liquidity Crisis) - **부도의 70%**

**정의:** 현금이 고갈되어 단기 채무를 갚지 못하는 상황

**특징:**
- 장부상 흑자여도 발생 가능 (**흑자도산**)
- 매출은 있지만 현금 회수가 늦어지면 부도
- 부도 발생 **3개월 전**에 급격히 악화되는 지표들

**실제 사례:**
```
중소 제조업체 B사:
- 연간 매출 100억원, 영업이익 10억원 (흑자)
- 문제: 대기업 납품대금 회수가 3개월 소요
- 결과: 급여·임대료 지급 불가 → 흑자도산
```

**위험 신호:**
- 현금소진일수 < 30일 (한 달도 못 버팀)
- 유동비율 < 100% (단기 부채가 유동자산보다 많음)
- 운전자본 음수 (유동부채 > 유동자산)

#### 🟠 경로 2: 지급불능 (Insolvency) - **부도의 20%**

**정의:** 부채가 자산을 초과하여 구조적으로 회생 불가능한 상황

**특징:**
- 자본잠식 (자본총계 < 0)
- 영업이익으로 이자도 갚을 수 없음
- 회복에 3년 이상 소요되는 구조적 문제

**실제 사례:**
```
건설업체 C사:
- 자본금 50억원, 자본잠식 80% (자본총계 10억원)
- 연간 영업이익 5억원, 이자비용 8억원
- 결과: 이자조차 갚을 수 없음 → 구조조정 불가피
```

**위험 신호:**
- 자본잠식도 > 50% (자본의 절반 이상 손실)
- 이자보상배율 < 1.0 (영업이익 < 이자비용)
- 부채상환년수 > 10년 (현금흐름으로 부채 상환 불가)

#### 🟡 경로 3: 신뢰 상실 (Loss of Confidence) - **부도의 10%**

**정의:** 연체·체납 이력으로 금융기관과 거래처가 자금줄을 차단

**특징:**
- 갑작스러운 신용경색
- 한 번의 연체가 도미노처럼 확산
- 재무제표보다 "행동 패턴"이 더 중요

**실제 사례:**
```
유통업체 D사:
- 재무지표 양호 (부채비율 120%, ROA 5%)
- 문제: 세금 체납 1회 발생 → 신용등급 BB로 하락
- 결과: 은행이 운전자금 대출 거부 → 3개월 내 부도
```

**위험 신호:**
- 연체 이력 1회 이상
- 세금 체납 발생
- 신용등급 BB 이하 (등급 5 이상)

---

### 3️⃣ 특성 공학 전략: 경로별 조기 감지 지표 개발

**목표:** 부도 3~6개월 전에 미리 예측할 수 있는 신호 포착

#### 📊 설계한 특성 체계

| 카테고리 | 특성 수 | 목적 | 대표 지표 | 비즈니스 질문 |
|----------|---------|------|-----------|--------------|
| **유동성 위기** | 10개 | 단기 생존 가능성 | 현금소진일수, 운전자본비율 | "3개월 내 살아남을 수 있는가?" |
| **지급불능** | 8개 | 장기 회생 가능성 | 자본잠식도, 부채상환년수 | "구조적으로 회생 가능한가?" |
| **재무조작 탐지** | 15개 | 회계 신뢰성 검증 | M-Score, 발생액 품질 | "재무제표를 신뢰할 수 있는가?" |
| **한국 시장 특화** | 13개 | 한국 기업 특성 | 외감 여부, 제조업 리스크 | "한국 시장의 위험을 반영했는가?" |
| **이해관계자 행동** | 9개 | 신용 행동 패턴 | 연체, 신용등급 | "이 기업을 신뢰할 수 있는가?" |
| **성장성 지표** | 5개 | 성장 둔화 감지 | 매출/영업이익 증가율 | "성장이 멈추고 있는가?" |
| **수익성/활동성** | 5개 | 수익성과 효율성 | 영업이익률, 총자산회전율 | "돈을 얼마나 벌고 자산을 활용하는가?" |
| **복합 리스크** | 7개 | 종합 위험도 | 종합부도위험스코어 | "전체적인 위험도는 얼마나 되는가?" |
| **비선형/상호작용** | 3개 | 복잡한 관계 | 레버리지×수익성 | "복합적 위험 요인이 있는가?" |

**총 75개 특성 생성**

#### 🎯 왜 통계적 특성이 아닌 도메인 특성인가?

**통계적 접근의 문제:**
- PCA, 군집화 등은 수학적으로 데이터를 압축할 뿐
- **해석 불가능**: "주성분 1"이 무엇을 의미하는지 설명 불가
- 실무 적용 어려움: 심사역에게 "왜 대출을 거절했는가?" 설명 불가

**도메인 접근의 장점:**
- ✅ **해석 가능**: "현금소진일수가 15일이라 위험합니다"
- ✅ **실무 적용**: 심사 기준으로 직접 활용 가능
- ✅ **논리적 설득력**: "왜 이 지표가 중요한가?"에 대한 이론적 근거 존재

---

### 4️⃣ 기대 효과

#### 📈 정량적 효과

1. **예측 정확도 향상**
   - 기존 (원본 159개 변수): ROC-AUC 0.75
   - 개선 (도메인 75개 특성): ROC-AUC **0.82+** (목표)
   - PR-AUC 0.12+ (불균형 데이터에서 실질적 성능)

2. **조기 경보 능력**
   - 부도 **3~6개월 전** 신호 포착
   - Type II Error < 20% (부도 기업을 놓치는 비율 최소화)

#### 💡 정성적 효과

1. **해석 가능성 (Explainability)**
   - 각 특성이 명확한 비즈니스 의미를 가짐
   - "왜 이 기업이 위험한가?"를 구체적으로 설명 가능

2. **한국 시장 적합성**
   - 외부감사 의무, 제조업 중심, 대기업 의존도 등 반영
   - 글로벌 모델을 단순 이식하는 것보다 실효성 높음

3. **실무 활용성**
   - 신용평가사, 은행, 투자사에서 즉시 사용 가능
   - 각 특성이 심사 기준으로 활용 가능

---

### 5️⃣ 이론적 근거 (학술 배경)

**이 특성들은 임의로 만든 것이 아니라, 50년 이상의 학술 연구와 실무 경험을 기반으로 함:**

| 이론/모델 | 연도 | 핵심 기여 | 본 프로젝트 반영 |
|-----------|------|-----------|------------------|
| **Altman Z-Score** | 1968 | 유동성, 레버리지, 수익성 통합 | 유동성 위기 특성, 지급불능 특성 |
| **Ohlson O-Score** | 1980 | 로지스틱 회귀 기반 부도 예측 | 확률적 접근, 비선형 변환 |
| **Beneish M-Score** | 1999 | 재무조작 탐지 | 한국형 M-Score (15개 특성) |
| **Zmijewski Score** | 1984 | 현금흐름 중시 | OCF 기반 유동성 지표 |
| **한국은행 연구** | 2020 | 한국 기업 부도 실증 분석 | 외감, 제조업, 신용 행동 특성 |

---

### 6️⃣ 요약: 왜 이 75개 특성이 필요한가?

**핵심 메시지:**

1. **원본 데이터는 재료일 뿐, 부도 위험을 직접 측정하지 못함**
2. **기업 부도의 3가지 경로 (유동성/지급불능/신뢰상실)를 사전에 포착하기 위함**
3. **통계적 특성이 아닌 도메인 지식 기반 특성으로 해석 가능성과 실무 적용성 확보**
4. **50년 이상의 학술 연구와 한국 시장 실증 분석을 기반으로 이론적 타당성 확보**
5. **부도 3~6개월 전 조기 경보를 통해 실질적 손실 예방 가능**
6. **성장성 지표 추가로 성장 둔화라는 조기 경보 신호 포착 가능**

**다음 섹션부터 각 카테고리별로 "왜 이 지표가 필요한가?"를 구체적으로 설명하며 특성을 생성합니다.**

---

## 1. 환경 설정 및 데이터 로딩

### 📦 필요한 라이브러리

- **pandas**: 데이터 처리 및 특성 생성
- **numpy**: 수치 계산 및 무한대 값 처리
- **sklearn**: StandardScaler (정규화), RandomForestClassifier (특성 중요도)

### 📊 원본 데이터

- 파일: `data/기업신용평가정보_210801.csv`
- 기업 수: 50,105개
- 변수 수: 159개 (재무/신용 변수)
- 타겟 변수: `모형개발용Performance(향후1년내부도여부)`
- 부도율: ~1.5% (불균형 데이터)


### 코드 셀 #1

```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# 한글 폰트 설정
import platform
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)

# 데이터 로딩
df = pd.read_csv('../data/기업신용평가정보_210801.csv', encoding='utf-8')
target_col = '모형개발용Performance(향후1년내부도여부)'

print(f"✅ 데이터 로딩 완료: {df.shape[0]:,} 기업, {df.shape[1]:,} 변수")
print(f"✅ 부도율: {df[target_col].mean()*100:.2f}%")
```

**출력:**

```
✅ 데이터 로딩 완료: 50,105 기업, 159 변수
✅ 부도율: 1.51%
```

---

## 2. 유동성 위기 조기 감지 특성

### 💡 왜 유동성이 가장 중요한가?

**경제적 가설: "부도는 지급불능이 아닌 유동성 위기로 시작된다"**

**학술적 배경 (Whitaker 1999):**
- 부도 기업의 **67%는 흑자**였음 (장부상 이익 발생)
- 하지만 **현금이 없어서** 급여/세금/이자를 지급하지 못함
- 유동성 위기는 부도 **3~6개월 전**에 나타남

**실무 사례:**
```
건설업체 A사:
- 영업이익: 100억 (흑자) ✅
- 현금및현금성자산: 5억 ❌
- 단기차입금: 200억 (3개월 내 만기) ❌
- 즉각지급능력 = 5 / 200 = 0.025 (2.5%) → 위험!
- 결과: 2개월 후 부도
```

### 생성할 10개 유동성 특성

| 특성 | 의미 | 임계값 |
|------|------|--------|
| 즉각지급능력 | 현금 / 단기부채 | < 0.1 위험 |
| 현금소진일수 | 현금 / 일평균지출 | < 30일 위험 |
| 운전자본비율 | (유동자산-유동부채) / 총자산 | < 0 위험 |
| 당좌비율 | (유동자산-재고) / 유동부채 | < 1.0 주의 |
| 현금비율 | 현금 / 유동부채 | < 0.2 주의 |
| 유동성갭 | 유동자산 - 유동부채 | < 0 위험 |
| 단기부채비중 | 단기부채 / 총부채 | > 0.5 주의 |
| 현금흐름대비부채 | 총부채 / 영업현금흐름 | > 5 위험 |
| 긴급유동성 | 초유동자산 / 유동부채 | < 0.5 주의 |
| 운전자본회전율 | 매출 / 운전자본 | < 1 주의 |


### 코드 셀 #2

```python
def create_liquidity_crisis_features(df):
    """유동성 위기를 조기에 감지하는 특성 생성"""
    
    features = pd.DataFrame(index=df.index)
    
    # 1. 즉각적 지급능력
    if '현금' in df.columns and '유동부채' in df.columns:
        features['즉각지급능력'] = (df['현금'] + df.get('현금성자산', 0)) / (df['유동부채'] + 1)
        features['현금소진일수'] = (df['현금'] + df.get('현금성자산', 0)) / (df.get('영업비용', df['매출원가']) / 365 + 1)
    
    # 2. 운전자본 건전성
    if '유동자산' in df.columns and '유동부채' in df.columns:
        features['운전자본'] = df['유동자산'] - df['유동부채']
        features['운전자본비율'] = features['운전자본'] / (df.get('매출액', 1) + 1)
        features['운전자본_대_자산'] = features['운전자본'] / (df.get('자산총계', 1) + 1)
    
    # 3. 긴급 자금조달 여력
    if '매출채권' in df.columns and '단기차입금' in df.columns:
        features['긴급유동성'] = (df['현금'] + df.get('현금성자산', 0) + df['매출채권'] * 0.8) / (df['단기차입금'] + 1)
    
    # 4. 유동성 압박 지표
    if '유동부채' in df.columns and '부채총계' in df.columns:
        features['단기부채비중'] = df['유동부채'] / (df['부채총계'] + 1)
        features['유동성압박지수'] = (df['유동부채'] / (df['유동자산'] + 1)) * (df['부채총계'] / (df['자산총계'] + 1))
    
    # 5. 현금흐름 기반 유동성
    if '영업활동현금흐름' in df.columns:
        features['OCF_대_유동부채'] = df['영업활동현금흐름'] / (df.get('유동부채', 1) + 1)
        features['현금창출능력'] = df['영업활동현금흐름'] / (df.get('매출액', 1) + 1)
    
    print(f"✅ 유동성 위기 특성 {features.shape[1]}개 생성 완료")
    return features

liquidity_features = create_liquidity_crisis_features(df)
print("\n생성된 유동성 특성:")
print(liquidity_features.columns.tolist())
```

**출력:**

```
✅ 유동성 위기 특성 10개 생성 완료

생성된 유동성 특성:
['즉각지급능력', '현금소진일수', '운전자본', '운전자본비율', '운전자본_대_자산', '긴급유동성', '단기부채비중', '유동성압박지수', 'OCF_대_유동부채', '현금창출능력']
```

---

## 3. 지급불능 패턴 특성

### 💡 유동성 위기 vs 지급불능

**차이점:**
- **유동성 위기**: 일시적 현금 부족 (단기 문제)
- **지급불능**: 구조적 부채 초과 (장기 문제)

**경제적 가설: "자본잠식 + 과다부채 = 회생 불가능"**

**실무 사례:**
```
제조업체 B사:
- 총자산: 500억
- 총부채: 600억 (부채비율 120%)
- 자본: -100억 (완전자본잠식) ❌
- 이자비용: 30억/년
- 영업이익: 10억/년
- 이자보상배율 = 10 / 30 = 0.33 < 1 ❌
- 결과: 이자도 갚지 못함 → 6개월 후 부도
```

### 생성할 8개 지급불능 특성

| 특성 | 의미 | 임계값 |
|------|------|--------|
| 자본잠식도 | 음의 자본 비율 | > 0 위험 |
| 이자보상배율 | 영업이익 / 이자비용 | < 1 위험 |
| 부채상환년수 | 총부채 / 영업현금흐름 | > 10년 위험 |
| 재무레버리지 | 총자산 / 자기자본 | > 5 주의 |
| 단기부채비중 | 단기부채 / 총부채 | > 0.7 주의 |
| 고정장기적합률 | 비유동자산 / 장기자본 | > 1.2 주의 |
| 부채비율 | 총부채 / 자기자본 | > 200% 주의 |
| 차입금의존도 | 차입금 / 총자산 | > 0.5 주의 |


### 코드 셀 #3

```python
def create_insolvency_features(df):
    """지급불능 위험을 포착하는 특성 생성"""
    
    features = pd.DataFrame(index=df.index)
    
    # 1. 자본 잠식도
    if '자본총계' in df.columns:
        features['자본잠식여부'] = (df['자본총계'] < 0).astype(int)
        features['자본잠식도'] = np.where(df.get('납입자본금', 1) > 0, 
                                       np.maximum(0, 1 - df['자본총계'] / df.get('납입자본금', 1)), 0)
    
    # 2. 차입금 의존도
    if '단기차입금' in df.columns and '장기차입금' in df.columns:
        features['총차입금'] = df['단기차입금'] + df['장기차입금']
        features['차입금의존도'] = features['총차입금'] / (df.get('자산총계', 1) + 1)
        features['차입금_대_매출'] = features['총차입금'] / (df.get('매출액', 1) + 1)
    
    # 3. 이자보상능력
    if '영업손익' in df.columns and '금융비용' in df.columns:
        features['이자보상배율'] = (df['영업손익'] + df.get('감가상각비', 0)) / (df['금융비용'] + 1)
        features['이자부담률'] = df['금융비용'] / (df.get('매출액', 1) + 1)
    
    # 4. 부채 상환 능력
    if '당기순이익' in df.columns and '부채총계' in df.columns:
        features['부채상환년수'] = df['부채총계'] / (df['당기순이익'] + df.get('감가상각비', 0) + 1)
        features['순부채비율'] = (df['부채총계'] - df.get('현금', 0)) / (df.get('자본총계', 1) + 1)
    
    # 5. 레버리지 위험
    if '자산총계' in df.columns and '자본총계' in df.columns:
        features['재무레버리지'] = df['자산총계'] / (df['자본총계'].abs() + 1)
        features['부채레버리지'] = df.get('부채총계', 0) / (df['자본총계'].abs() + 1)
    
    print(f"✅ 지급불능 패턴 특성 {features.shape[1]}개 생성 완료")
    return features

insolvency_features = create_insolvency_features(df)
print("\n생성된 지급불능 특성:")
print(insolvency_features.columns.tolist())
```

**출력:**

```
✅ 지급불능 패턴 특성 8개 생성 완료

생성된 지급불능 특성:
['자본잠식여부', '자본잠식도', '이자보상배율', '이자부담률', '부채상환년수', '순부채비율', '재무레버리지', '부채레버리지']
```

---

## 4. 재무조작 탐지 특성 (Beneish M-Score 한국형)

### 💡 왜 재무조작 탐지가 필요한가?

**경제적 가설: "부도 직전 기업은 실적을 부풀린다"**

**학술적 배경 (Beneish 1999):**
- **M-Score**: 재무제표 조작 가능성을 수치화한 지표
- 8개 재무 비율의 가중합으로 계산
- M-Score > -2.22: 조작 의심 (76% 정확도)

**한국형 M-Score 조정:**
- 외부감사 여부 반영 (한국 시장 특성)
- 중소기업 특성 반영 (회계 품질 차이)

**실제 사례 (대우조선해양 2015):**
```
- 매출채권 급증 (DSRI↑): 가공 매출 의심
- 매출총이익률 급증 (GMI↑): 원가 조작 의심
- 발생액 비정상 (TATA↑): 현금 없는 이익 의심
- M-Score > -2.22 → 조작 의심
- 결과: 2015년 분식회계 적발
```

### Beneish M-Score 8개 구성 요소

| 지표 | 의미 | 조작 신호 |
|------|------|----------|
| DSRI | 매출채권 / 매출 증가율 | 높을수록 가공매출 의심 |
| GMI | 매출총이익률 변화 | 감소 시 조작 가능성 |
| AQI | 자산 품질 지수 | 높을수록 자산 부풀리기 의심 |
| SGI | 매출 성장률 | 과도한 성장 시 의심 |
| DEPI | 감가상각률 변화 | 감소 시 이익 부풀리기 의심 |
| SGAI | 판관비 / 매출 변화 | 증가 시 비효율 의심 |
| LVGI | 레버리지 증가율 | 증가 시 재무위험 증가 |
| TATA | 발생액 / 총자산 | 높을수록 현금 없는 이익 의심 |

**M-Score 계산식:**
```
M-Score = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI 
          + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

해석:
- M-Score > -2.22: 조작 가능성 높음
- M-Score ≤ -2.22: 정상
```


### 코드 셀 #4

```python
def create_manipulation_detection_features(df):
    """재무조작 가능성을 탐지하는 특성 생성"""
    
    features = pd.DataFrame(index=df.index)
    
    # 공통 변수 안전하게 확보 (부채비율이 없으면 계산)
    if '부채비율' in df.columns:
        부채비율 = df['부채비율']
    elif '부채총계' in df.columns and '자본총계' in df.columns:
        부채비율 = df['부채총계'] / (df['자본총계'].abs() + 1) * 100
    else:
        부채비율 = 100  # 기본값 (데이터 부족 시)

    # 1. 매출채권 이상 증가
    if '매출채권' in df.columns and '매출액' in df.columns:
        features['매출채권회전율'] = df['매출액'] / (df['매출채권'] + 1)
        features['매출채권비율'] = df['매출채권'] / (df['매출액'] + 1)
        # 수정된 부분: df['부채비율'] 대신 위에서 확보한 변수 사용
        features['매출채권_이상지표'] = features['매출채권비율'] * (부채비율 / 100)
    
    # 2. 재고자산 이상 적체
    if '재고자산' in df.columns and '매출원가' in df.columns:
        features['재고회전율'] = df['매출원가'] / (df['재고자산'] + 1)
        features['재고보유일수'] = 365 / (features['재고회전율'] + 0.1)
        features['재고_이상지표'] = df['재고자산'] / (df.get('자산총계', 1) + 1) * 100
    
    # 3. 발생액(Accruals) 품질
    if '당기순이익' in df.columns and '영업활동현금흐름' in df.columns:
        features['총발생액'] = df['당기순이익'] - df['영업활동현금흐름']
        features['발생액비율'] = features['총발생액'] / (df.get('자산총계', 1) + 1)
        features['현금흐름품질'] = df['영업활동현금흐름'] / (df['당기순이익'] + 1)
    
    # 4. 비용 자본화 의심
    if '무형자산' in df.columns and '영업비용' in df.columns:
        features['무형자산비율'] = df['무형자산'] / (df.get('자산총계', 1) + 1)
        features['비용자본화지표'] = df['무형자산'] / (df['영업비용'] + 1)
    
    # 5. 매출총이익률 악화
    if '매출총이익' in df.columns and '매출액' in df.columns:
        features['매출총이익률'] = df['매출총이익'] / (df['매출액'] + 1) * 100
        features['영업레버리지'] = df.get('영업손익', 0) / (df['매출총이익'] + 1)
    
    # 6. 판관비 이상 증가
    if '판매비와관리비' in df.columns and '매출액' in df.columns:
        features['판관비율'] = df['판매비와관리비'] / (df['매출액'] + 1) * 100
        features['판관비효율성'] = df.get('영업손익', 0) / (df['판매비와관리비'] + 1)
    
    # 7. M-Score 종합
    m_score = 0
    if '매출채권비율' in features.columns:
        m_score += features['매출채권비율'] * 0.92
    if '재고_이상지표' in features.columns:
        m_score += features['재고_이상지표'] * 0.528
    if '발생액비율' in features.columns:
        m_score += features['발생액비율'] * 4.679
    if '무형자산비율' in features.columns:
        m_score += features['무형자산비율'] * 0.404
    
    features['M_Score_한국형'] = m_score - 2.22  # 한국 시장 조정
    features['재무조작위험'] = (features['M_Score_한국형'] > 0).astype(int)
    
    print(f"✅ 재무조작 탐지 특성 {features.shape[1]}개 생성 완료")
    return features

manipulation_features = create_manipulation_detection_features(df)
print("\n생성된 재무조작 탐지 특성:")
print(manipulation_features.columns.tolist())
```

**출력:**

```
✅ 재무조작 탐지 특성 15개 생성 완료

생성된 재무조작 탐지 특성:
['매출채권회전율', '매출채권비율', '매출채권_이상지표', '재고회전율', '재고보유일수', '재고_이상지표', '총발생액', '발생액비율', '현금흐름품질', '매출총이익률', '영업레버리지', '판관비율', '판관비효율성', 'M_Score_한국형', '재무조작위험']
```

---

## 5. 한국 시장 특화 특성

### 💡 재무 전문가 관점: 한국 기업 생태계의 특수성

**한국 시장의 구조적 특징 (한국은행 2020; 중소기업연구원 2019)**

한국 기업 생태계는 글로벌 시장과 다른 독특한 특성을 가지며, 이를 반영하지 않으면 부도 예측 정확도가 낮아집니다.

**한국 시장 3대 특징:**

#### 1️⃣ 외부감사 의무화 (K-GAAP / K-IFRS)

```
외감기업 = 자산 120억원 이상 OR 매출 70억원 이상 OR 부채 70억원 이상
비외감기업 = 위 기준 미달 중소기업
```

**왜 중요한가?**
- **외감기업**: 회계법인 감사 → 재무제표 신뢰도 ↑
- **비외감기업**: 자체 작성 → 분식 가능성 ↑, 부도율 2.5배 높음

**실무 통계 (2020년 기준):**
- 전체 기업: 100만개
- 외감기업: 3만개 (3%)
- **비외감 기업 부도율: 2.4%**
- **외감 기업 부도율: 0.9%**

**리스크 페널티:**
```python
비외감_리스크 = (1 - 외감기업) * 2  # 비외감은 2배 가중치
```

#### 2️⃣ 제조업 중심 경제 (수출 의존)

**한국 산업 구조:**
- 제조업 비중: **27%** (OECD 평균 15%)
- 수출 의존도: **GDP의 40%** (미국 12%, 일본 18%)
- 제조업 기업 수: 전체의 **35%**

**제조업의 위험 요인:**
1. **재고 리스크**: 재고자산 비중 높음 (평균 25%)
2. **설비 투자**: 고정자산 비중 높음 → 부채비율 ↑
3. **대기업 의존도**: 납품 대금 회수 지연 (평균 90일)
4. **환율 민감도**: 수출 비중 높음

**실제 데이터:**
```
제조업 평균:
- 부채비율: 180% (서비스업 80%)
- 유동비율: 120% (서비스업 150%)
- 재고자산비율: 25% (서비스업 5%)
→ 제조업이 구조적으로 더 위험
```

**리스크 지표:**
```python
제조업_레버리지위험 = 제조업여부 * (부채비율 / 100)
제조업_재고위험 = 제조업여부 * (재고자산 / 자산총계)
```

#### 3️⃣ 대기업 의존도 (하청 구조)

**한국 경제의 구조적 문제:**
- 삼성/현대 등 10대 그룹 매출: GDP의 **80%**
- 중소기업의 **70%가 대기업 하청**
- 대기업 부도 → 중소기업 연쇄 부도

**납품 대금 회수 지연:**
- 대기업 → 중소기업 평균 결제 기간: **90일**
- 글로벌 평균: **30일**
- 회수 지연 → 운전자본 부족 → 흑자도산

**측정 지표:**
```
매출채권집중도 = 매출채권 / 매출액
거래처리스크 = 매출채권집중도 * (부채비율 / 100)
```

**실무 판단:**
- 매출채권집중도 > 30% → 대기업 의존도 높음
- 거래처리스크 > 0.5 → 대기업 부도 시 연쇄 위험

**실제 사례: 대우조선해양 사태 (2016)**
```
- 대우조선 워크아웃 신청
→ 하청업체 350개 연쇄 부도
→ 고용 25,000명 영향
→ 지역 경제 붕괴
```

#### 4️⃣ 업력 (경기 사이클 경험)

**Startup vs 성숙 기업:**
- 업력 < 3년: **신생기업 리스크** (부도율 5.2%)
- 업력 3~10년: 성장기 (부도율 2.1%)
- 업력 > 10년: 성숙기 (부도율 0.8%)

**왜 업력이 중요한가?**
- 경기 사이클 1회 (약 7년) 경험 여부
- 신용 history 부족
- 고객 기반 미확보

#### 5️⃣ 기업 규모 (자금 조달 능력)

**한국 중소기업 정의:**
- 자산 5,000억원 미만
- 매출 1,500억원 미만
- 직원 1,000명 미만

**규모별 리스크:**
- **대기업**: 은행 신용등급 우대, 자금 조달 용이
- **중소기업**: 고금리 대출, 담보 필요, 유동성 위기 취약

**실무 통계:**
```
기업 규모별 부도율 (2020):
- 자산 100억 미만: 2.8%
- 자산 100~500억: 1.5%
- 자산 500억 이상: 0.6%
→ 규모가 작을수록 5배 위험
```

---

### 📊 생성할 13개 한국 시장 특화 지표 요약

| 지표 | 목적 | 임계값 (위험) | 근거 |
|------|------|---------------|------|
| 외감기업 | 재무제표 신뢰도 | 0 (비외감) | 금융감독원 |
| 비외감_리스크 | 비외감 페널티 | 2 (비외감) | 실증 분석 |
| 제조업_레버리지위험 | 제조업 부채 | > 1.5 | 업종 평균 |
| 제조업_재고위험 | 제조업 재고 | > 0.25 | 업종 평균 |
| 매출채권집중도 | 대기업 의존도 | > 30% | 실무 기준 |
| 거래처리스크 | 연쇄 부도 | > 0.5 | 실무 기준 |
| 유형자산비중 | 자산 경직성 | > 60% | 제조업 평균 |
| 자산유동성위험 | 유형자산 × 부채 | > 1.0 | 복합 지표 |
| 업력 | 생존 기간 | < 3년 | 통계청 |
| 신생기업리스크 | 업력 페널티 | 3 (신생) | 실증 분석 |
| 업력_안정성 | log(업력) | < 1 | 복합 지표 |
| 기업규모_로그 | 규모 효과 | < 18 (100억) | 중기청 |
| 중소기업리스크 | 규모 페널티 | 1 (중소) | 실증 분석 |

**핵심 메시지:**
> "글로벌 모델은 한국에서 작동하지 않는다" - 시장 특수성의 중요성

### 코드 셀 #5

```python
def create_korean_market_features(df):
    """한국 시장 특화 리스크 특성 생성"""
    
    features = pd.DataFrame(index=df.index)
    
    # 1. 대기업 의존도 (매출 집중도 프록시)
    if '매출채권' in df.columns and '매출액' in df.columns:
        features['매출채권집중도'] = df['매출채권'] / (df['매출액'] + 1)
        features['거래처리스크'] = features['매출채권집중도'] * df.get('부채비율', 100) / 100
    
    # 2. 수출 민감도 (제조업 특화)
    if '업종(중분류)' in df.columns:
        제조업여부 = df['업종(중분류)'].str.contains('제조', na=False).astype(int)
        features['제조업_레버리지위험'] = 제조업여부 * df.get('부채비율', 0) / 100
        features['제조업_재고위험'] = 제조업여부 * df.get('재고자산', 0) / (df.get('자산총계', 1) + 1)
    
    # 3. 부동산 연계 리스크
    if '유형자산' in df.columns and '자산총계' in df.columns:
        features['유형자산비중'] = df['유형자산'] / (df['자산총계'] + 1)
        features['자산유동성위험'] = features['유형자산비중'] * df.get('부채비율', 100) / 100
    
    # 4. 외감 기업 여부
    if '외감구분' in df.columns:
        features['외감기업'] = (df['외감구분'] == 'Y').astype(int)
        features['비외감_리스크'] = (1 - features['외감기업']) * 2  # 비외감 기업 페널티
    
    # 5. 업력 리스크
    if '설립일자' in df.columns:
        try:
            설립연도 = pd.to_datetime(df['설립일자'], errors='coerce').dt.year
            features['업력'] = 2021 - 설립연도  # 데이터 기준년도 2021
            features['신생기업리스크'] = (features['업력'] < 3).astype(int) * 3
            features['업력_안정성'] = np.log1p(features['업력'])
        except:
            pass
    
    # 6. 규모 리스크
    if '자산총계' in df.columns:
        features['기업규모_로그'] = np.log1p(df['자산총계'])
        # 중소기업 리스크 (자산 100억 미만)
        features['중소기업리스크'] = (df['자산총계'] < 10000000).astype(int)
    
    print(f"✅ 한국 시장 특화 특성 {features.shape[1]}개 생성 완료")
    return features

korean_features = create_korean_market_features(df)
print("\n생성된 한국 시장 특화 특성:")
print(korean_features.columns.tolist())
```

**출력:**

```
✅ 한국 시장 특화 특성 13개 생성 완료

생성된 한국 시장 특화 특성:
['매출채권집중도', '거래처리스크', '제조업_레버리지위험', '제조업_재고위험', '유형자산비중', '자산유동성위험', '외감기업', '비외감_리스크', '업력', '신생기업리스크', '업력_안정성', '기업규모_로그', '중소기업리스크']
```

---

## 6. 이해관계자 행동 패턴 특성

### 💡 재무 전문가 관점: 신용 행동 > 재무 지표

**학술적 배경 (Shumway 2001; Campbell et al. 2008)**

Tyler Shumway(2001)는 **부도 전 6개월~1년 사이 신용 행동 변화**가 재무 비율보다 강력한 예측 변수임을 실증했습니다. 본 프로젝트 분석에서도 **연체 관련 변수가 상관계수 0.32**로 1위를 차지했습니다.

**핵심 원리: 연체는 "마지막 선택"**

```
기업의 선택 우선순위:
1. 영업 비용 지급 (급여, 임대료) → 생존 필수
2. 대출 이자 지급 → 신용도 유지
3. 세금 납부 → 법적 의무
4. 대출 원금 상환 → 마지막

연체 발생 = 위 모든 것을 포기 = 부도 직전
```

**실증 데이터 (본 프로젝트):**
- 연체기관수_로그 상관계수: **0.323** (1위)
- 연체여부_미해제: 부도율 **47.7%** (정상 1.2%의 39배)
- 장기연체(30일+): 부도율 **66.0%** (정상 1.4%의 47배)

### 생성할 9개 이해관계자 행동 지표

| 지표 | 실무 의미 | 부도율 증가 배수 |
|------|-----------|------------------|
| 총연체건수 | 연체 빈도 | 25배 |
| 연체여부 | 현재 연체 중 | 39배 |
| 연체심각도 | 연체 × 부채비율 | 복합 |
| 세금체납건수 | 세금 미납 | 21배 |
| 세금체납리스크 | 체납 페널티 | 21배 |
| 공공정보리스크 | 압류/소송 | 15배 |
| 법적리스크 | 법적 분쟁 | 15배 |
| 신용등급점수 | 신용평가사 판단 | 등급별 차등 |
| 이해관계자_불신지수 | 종합 신뢰도 | 35배 |

**핵심 메시지:**
> "재무제표는 과거, 연체 이력은 현재" - 선행 지표의 힘

### 코드 셀 #6

```python
def create_stakeholder_behavior_features(df):
    """이해관계자 행동 패턴을 포착하는 특성 생성"""
    
    features = pd.DataFrame(index=df.index)
    
    # 1. 신용 행동 패턴
    credit_cols = [col for col in df.columns if '연체' in col]
    if credit_cols:
        features['총연체건수'] = df[credit_cols].sum(axis=1)
        features['연체여부'] = (features['총연체건수'] > 0).astype(int)
        features['연체심각도'] = features['총연체건수'] * df.get('부채비율', 100) / 100
    
    # 2. 세금 체납 리스크
    tax_cols = [col for col in df.columns if '체납' in col or '세금' in col]
    if tax_cols:
        features['세금체납건수'] = df[tax_cols].sum(axis=1)
        features['세금체납리스크'] = (features['세금체납건수'] > 0).astype(int) * 5
    
    # 3. 공공정보 리스크
    public_cols = [col for col in df.columns if any(k in col for k in ['압류', '소송', '공공'])]
    if public_cols:
        features['공공정보리스크'] = df[public_cols].sum(axis=1)
        features['법적리스크'] = (features['공공정보리스크'] > 0).astype(int) * 3
    
    # 4. 신용등급 리스크
    rating_cols = [col for col in df.columns if '신용평가등급' in col]
    if rating_cols:
        # 신용등급을 숫자로 변환 (낮은 등급일수록 높은 점수)
        for col in rating_cols:
            if df[col].dtype == 'object':
                # 등급을 리스크 점수로 매핑
                grade_map = {'AAA': 1, 'AA': 2, 'A': 3, 'BBB': 4, 'BB': 5, 'B': 6, 'CCC': 7, 'CC': 8, 'C': 9, 'D': 10}
                features['신용등급점수'] = df[col].map(grade_map).fillna(10)
            else:
                features['신용등급점수'] = df[col]
            break
    
    # 5. 종합 신뢰도 지표
    features['이해관계자_불신지수'] = (
        features.get('연체여부', 0) * 2 +
        features.get('세금체납리스크', 0) +
        features.get('법적리스크', 0) +
        features.get('신용등급점수', 0) / 2
    )
    
    print(f"✅ 이해관계자 행동 패턴 특성 {features.shape[1]}개 생성 완료")
    return features

stakeholder_features = create_stakeholder_behavior_features(df)
print("\n생성된 이해관계자 행동 특성:")
print(stakeholder_features.columns.tolist())
```

**출력:**

```
✅ 이해관계자 행동 패턴 특성 9개 생성 완료

생성된 이해관계자 행동 특성:
['총연체건수', '연체여부', '연체심각도', '세금체납건수', '세금체납리스크', '공공정보리스크', '법적리스크', '신용등급점수', '이해관계자_불신지수']
```

---

## 6.5 성장성 지표 (Growth Features)

### 💡 왜 성장성 지표가 중요한가?

**경제적 가설: "성장 둔화는 부도의 조기 경보 신호"**

**실증 분석 결과 (참고 데이터 기반):**
- 영업이익증가율_YoY 상관계수: -0.012 → **음의 성장(역성장)이 부도와 연관**
- 성장 둔화는 부도 발생 **3~6개월 전**에 나타남

**비즈니스 로직:**
1. **매출 감소** → 시장 점유율 상실, 경쟁력 약화
2. **영업이익 감소** → 수익성 악화, 비용 통제 실패  
3. **당기순이익 감소** → 최종 수익성 저하, 누적 손실
4. **자산 감소** → 사업 축소, 투자 중단
5. **자본 감소** → 자본잠식 진행, 재무구조 악화

**실제 사례:**
```
IT 서비스 기업 E사:
- 2019년: 매출 200억, 영업이익 20억 (성장세)
- 2020년: 매출 180억(-10%), 영업이익 5억(-75%) ← 급격한 둔화
- 2021년: 매출 150억(-17%), 영업이익 -10억 ← 적자 전환
- 결과: 2021년 6월 부도
```

**임계 기준:**
- 매출 감소율 > 10%: 경고
- 영업이익 감소율 > 30%: 위험
- 당기순이익 적자 전환: 매우 위험

### 코드 셀 #7

```python
def create_growth_features(df):
    """성장성 지표 생성 (Growth Features)"""
    
    features = pd.DataFrame(index=df.index)
    
    # 1. 매출 성장률 (Sales Growth Rate)
    if '매출액증가율' in df.columns:
        features['매출증가율_YoY'] = df['매출액증가율']
    elif '매출액' in df.columns and '전기매출액' in df.columns:
        features['매출증가율_YoY'] = (
            (df['매출액'] - df['전기매출액']) / (df['전기매출액'] + 1)
        ) * 100
    
    # 2. 영업이익 성장률
    if '영업이익증가율' in df.columns:
        features['영업이익증가율_YoY'] = df['영업이익증가율']
    elif '영업이익' in df.columns and '전기영업이익' in df.columns:
        features['영업이익증가율_YoY'] = (
            (df['영업이익'] - df['전기영업이익']) / (df['전기영업이익'].abs() + 1)
        ) * 100
    
    # 3. 당기순이익 성장률
    if '당기순이익증가율' in df.columns:
        features['당기순이익증가율_YoY'] = df['당기순이익증가율']
    elif '당기순이익' in df.columns and '전기당기순이익' in df.columns:
        features['당기순이익증가율_YoY'] = (
            (df['당기순이익'] - df['전기당기순이익']) / (df['전기당기순이익'].abs() + 1)
        ) * 100
    
    # 4. 총자산 성장률
    if '총자산증가율' in df.columns:
        features['총자산증가율_YoY'] = df['총자산증가율']
    elif '총자산' in df.columns and '전기총자산' in df.columns:
        features['총자산증가율_YoY'] = (
            (df['총자산'] - df['전기총자산']) / (df['전기총자산'] + 1)
        ) * 100
    
    # 5. 자본 성장률 (자본잠식 여부 확인)
    if '자본증가율' in df.columns:
        features['자본증가율_YoY'] = df['자본증가율']
    elif '자기자본' in df.columns and '전기자기자본' in df.columns:
        features['자본증가율_YoY'] = (
            (df['자기자본'] - df['전기자기자본']) / (df['전기자기자본'].abs() + 1)
        ) * 100
    
    # 6. 성장 둔화 신호 (매출 감소 + 영업이익 감소)
    features['성장둔화신호'] = (
        (features.get('매출증가율_YoY', 0) < -10).astype(int) +
        (features.get('영업이익증가율_YoY', 0) < -30).astype(int)
    )
    
    # 7. 역성장 위험 (음의 성장률)
    features['역성장위험'] = (
        (features.get('매출증가율_YoY', 0) < 0).astype(int) * 2 +
        (features.get('영업이익증가율_YoY', 0) < 0).astype(int) * 3 +
        (features.get('당기순이익증가율_YoY', 0) < 0).astype(int)
    )
    
    print(f"✅ 성장성 지표 {features.shape[1]}개 생성 완료")
    return features

growth_features = create_growth_features(df)
print("\n생성된 성장성 지표:")
print(growth_features.columns.tolist())

```

**출력:**

```
✅ 성장성 지표 5개 생성 완료

생성된 성장성 지표:
['매출증가율_YoY', '영업이익증가율_YoY', '당기순이익증가율_YoY', '성장둔화신호', '역성장위험']
```

---

## 6.6 수익성 및 활동성 보완 지표

### 💡 왜 수익성과 활동성 지표가 추가로 필요한가?

**경제적 가설: "수익성 악화 + 자산 활용 저하 = 부도 위험"**

**수익성 지표 (Profitability):**
- **영업이익률**: 본업의 수익성 측정 (판관비 통제 능력)
- **순이익률**: 최종 수익성 (이자, 세금 포함)
- **이익의 질 (Accruals)**: 현금이 뒷받침되지 않는 이익 = 분식회계 의심

**활동성 지표 (Activity):**
- **총자산회전율**: 자산을 얼마나 효율적으로 활용하여 매출을 창출하는가?
- **유동성_효율성_교차**: 유동성과 효율성의 상호작용 (낮은 재고회전율 + 낮은 유동비율 = 위험)

**실제 사례:**
```
소매업체 F사:
- 영업이익률: 2% (업계 평균 5%) ← 낮은 수익성
- 총자산회전율: 0.5회 (업계 평균 2회) ← 낮은 효율성
- 이익의 질: -0.05 (음수) ← 현금흐름 < 이익 (의심)
- 결과: 1년 내 부도
```

### 코드 셀 #8

```python
def create_profitability_activity_features(df):
    """수익성 및 활동성 지표 생성"""
    
    features = pd.DataFrame(index=df.index)
    
    # 1. 수익성 지표 (Profitability)
    
    # 영업이익률 (Operating Profit Margin)
    if '영업이익' in df.columns and '매출액' in df.columns:
        features['영업이익률'] = (df['영업이익'] / (df['매출액'] + 1)) * 100
    
    # 순이익률 (Net Profit Margin)
    if '당기순이익' in df.columns and '매출액' in df.columns:
        features['순이익률'] = (df['당기순이익'] / (df['매출액'] + 1)) * 100
    
    # 매출총이익률 (Gross Profit Margin)
    if '매출총이익' in df.columns and '매출액' in df.columns:
        features['매출총이익률'] = (df['매출총이익'] / (df['매출액'] + 1)) * 100
    elif '매출원가' in df.columns and '매출액' in df.columns:
        features['매출총이익률'] = ((df['매출액'] - df['매출원가']) / (df['매출액'] + 1)) * 100
    
    # ROA (Return on Assets)
    if '당기순이익' in df.columns and '총자산' in df.columns:
        features['ROA'] = (df['당기순이익'] / (df['총자산'] + 1)) * 100
    
    # ROE (Return on Equity)
    if '당기순이익' in df.columns and '자기자본' in df.columns:
        features['ROE'] = (df['당기순이익'] / (df['자기자본'].abs() + 1)) * 100
    
    # 이익의 질 (Accruals Quality)
    # 이익의 질 = 영업현금흐름 / 당기순이익 (1에 가까울수록 양호)
    if '영업활동현금흐름' in df.columns and '당기순이익' in df.columns:
        features['이익의질'] = df['영업활동현금흐름'] / (df['당기순이익'].abs() + 1)
        # 음수 이익의 질 = 현금흐름 < 이익 (의심)
        features['이익품질_이상신호'] = (features['이익의질'] < 0.5).astype(int)
    
    # 2. 활동성 지표 (Activity / Efficiency)
    
    # 총자산회전율 (Total Asset Turnover)
    if '매출액' in df.columns and '총자산' in df.columns:
        features['총자산회전율'] = df['매출액'] / (df['총자산'] + 1)
    
    # 재고자산회전율 (Inventory Turnover)
    if '매출원가' in df.columns and '재고자산' in df.columns:
        features['재고자산회전율'] = df['매출원가'] / (df['재고자산'] + 1)
        # 낮은 재고회전율 = 재고 적체 위험
        features['재고적체위험'] = (features['재고자산회전율'] < 2).astype(int)
    
    # 매출채권회전율 (Accounts Receivable Turnover)
    if '매출액' in df.columns and '매출채권' in df.columns:
        features['매출채권회전율'] = df['매출액'] / (df['매출채권'] + 1)
        # 낮은 매출채권회전율 = 회수 지연
        features['매출채권회수지연'] = (features['매출채권회전율'] < 5).astype(int)
    
    # 3. 유동성_효율성 교차 지표
    # 낮은 재고회전율 + 낮은 유동비율 = 위험
    if '유동비율' in df.columns and '재고자산회전율' in features.columns:
        features['유동성_효율성_교차위험'] = (
            ((df['유동비율'] < 100).astype(int) +
             (features['재고자산회전율'] < 2).astype(int))
        )
    
    print(f"✅ 수익성 및 활동성 지표 {features.shape[1]}개 생성 완료")
    return features

profitability_activity_features = create_profitability_activity_features(df)
print("\n생성된 수익성/활동성 지표:")
print(profitability_activity_features.columns.tolist())

```

**출력:**

```
✅ 수익성 및 활동성 지표 8개 생성 완료

생성된 수익성/활동성 지표:
['순이익률', '매출총이익률', '이익의질', '이익품질_이상신호', '재고자산회전율', '재고적체위험', '매출채권회전율', '매출채권회수지연']
```

---

## 7. 복합 리스크 지표 생성

### 💡 재무 전문가 관점: 단일 지표의 함정

**학술적 배경 (Ohlson 1980)**

James Ohlson은 **단일 재무 비율로는 부도를 예측할 수 없다**는 것을 실증했습니다. 대신 **여러 도메인의 특성을 결합한 복합 지표**가 필요합니다.

**복합 지표의 설계 원리:**

1. **재무건전성지수** = 유동성 + 수익성 + 지급능력의 평균
2. **유동성스트레스지수** = 운전자본 부족 + 단기 부채 과다
3. **종합부도위험스코어** = 가중 평균 (도메인별 차등 가중치)
4. **조기경보신호수** = 위험 신호 개수 (자본잠식 + 연체 + 현금 고갈)

### 생성할 7개 복합 리스크 지표

| 지표 | 구성 요소 | 활용 |
|------|-----------|------|
| 재무건전성지수 | 유동성 + 수익성 + 지급능력 | 종합 평가 |
| 유동성스트레스지수 | 운전자본 + 단기부채비중 | 단기 위험 |
| 지급불능위험지수 | 자본잠식 + 부채상환년수 | 장기 위험 |
| 시장포지션지수 | 규모 정규화 | 상대 비교 |
| 종합부도위험스코어 | 전체 가중 평균 | 최종 스코어 |
| 조기경보신호수 | 위험 신호 개수 | 경보 시스템 |
| 위험경보등급 | 4단계 분류 | 등급화 |

**핵심 메시지:**
> "나무를 보지 말고 숲을 보라" - 종합 판단의 중요성

### 코드 셀 #9

```python
def create_composite_risk_features(df, features_dict):
    """복합 리스크 지표 생성 (여러 도메인 특성의 조합)"""
    
    features = pd.DataFrame(index=df.index)
    
    # 1. 재무건전성지수 (Financial Health Index)
    # 유동성 + 수익성 + 지급능력의 평균
    liquidity_cols = ['즉각지급능력', '현금소진일수', '운전자본비율']
    profitability_cols = ['영업이익률', '순이익률']
    solvency_cols = ['이자보상배율', '부채상환년수']
    
    # 각 도메인별 정규화된 점수 계산
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    liquidity_score = features_dict['liquidity'][liquidity_cols].fillna(0)
    liquidity_score = pd.DataFrame(
        scaler.fit_transform(liquidity_score), 
        index=liquidity_score.index, 
        columns=liquidity_score.columns
    ).mean(axis=1)
    
    profitability_score = features_dict['stakeholder'][profitability_cols].fillna(0)
    profitability_score = pd.DataFrame(
        scaler.fit_transform(profitability_score), 
        index=profitability_score.index, 
        columns=profitability_score.columns
    ).mean(axis=1)
    
    solvency_score = features_dict['insolvency'][solvency_cols].fillna(0)
    solvency_score = pd.DataFrame(
        scaler.fit_transform(solvency_score), 
        index=solvency_score.index, 
        columns=solvency_score.columns
    ).mean(axis=1)
    
    features['재무건전성지수'] = (liquidity_score + profitability_score + solvency_score) / 3
    
    # 2. 유동성스트레스지수
    features['유동성스트레스지수'] = (
        -features_dict['liquidity']['운전자본비율'].fillna(0) + 
        features_dict['insolvency']['단기부채비중'].fillna(0)
    ) / 2
    
    # 3. 지급불능위험지수
    features['지급불능위험지수'] = (
        features_dict['insolvency']['자본잠식도'].fillna(0) + 
        features_dict['insolvency']['부채상환년수'].fillna(0)
    ) / 2
    
    # 4. 시장포지션지수 (규모 정규화)
    if '총자산' in df.columns:
        features['시장포지션지수'] = pd.qcut(
            df['총자산'].fillna(0), 
            q=10, 
            labels=False, 
            duplicates='drop'
        )
    else:
        features['시장포지션지수'] = 0
    
    # 5. 종합부도위험스코어 (가중 평균)
    weights = {
        'liquidity': 0.3,
        'insolvency': 0.3,
        'manipulation': 0.2,
        'stakeholder': 0.2
    }
    
    risk_components = []
    for domain, weight in weights.items():
        domain_features = features_dict[domain]
        # 각 도메인의 평균 리스크 점수 계산
        domain_score = domain_features.fillna(0).mean(axis=1)
        risk_components.append(domain_score * weight)
    
    features['종합부도위험스코어'] = sum(risk_components)
    
    # 6. 조기경보신호수 (위험 신호 개수)
    warning_signals = []
    
    # 자본잠식
    if '자본잠식도' in features_dict['insolvency'].columns:
        warning_signals.append(
            (features_dict['insolvency']['자본잠식도'] > 0.5).astype(int)
        )
    
    # 연체 여부
    if '연체여부' in features_dict['stakeholder'].columns:
        warning_signals.append(
            features_dict['stakeholder']['연체여부']
        )
    
    # 현금 고갈 위험
    if '현금소진일수' in features_dict['liquidity'].columns:
        warning_signals.append(
            (features_dict['liquidity']['현금소진일수'] < 30).astype(int)
        )
    
    # 이자보상배율 < 1
    if '이자보상배율' in features_dict['insolvency'].columns:
        warning_signals.append(
            (features_dict['insolvency']['이자보상배율'] < 1).astype(int)
        )
    
    features['조기경보신호수'] = sum(warning_signals) if warning_signals else 0
    
    # 7. 위험경보등급 (4단계 분류)
    def classify_risk(score):
        if score < -1:
            return 0  # 정상
        elif score < 0:
            return 1  # 주의
        elif score < 1:
            return 2  # 경고
        else:
            return 3  # 위험
    
    features['위험경보등급'] = features['종합부도위험스코어'].apply(classify_risk)
    
    return features

```

---

### 코드 셀 #10

```python
# 복합 리스크 지표 생성 (앞서 생성한 특성들을 딕셔너리로 전달)
features_dict = {
    'liquidity': liquidity_features,
    'insolvency': insolvency_features,
    'manipulation': manipulation_features,
    'korean': korean_features,
    'stakeholder': profitability_activity_features  # 수익성 지표 포함
}

composite_features = create_composite_risk_features(df, features_dict)
print(f"\n✅ 복합 리스크 지표 {composite_features.shape[1]}개 생성 완료")
print("\n생성된 복합 리스크 지표:")
print(composite_features.columns.tolist())

```

**출력:**

```
ERROR: KeyError: "['영업이익률'] not in index"
```

---

### 코드 셀 #11

```python
# 모든 특성을 하나의 데이터프레임으로 통합
all_features = pd.concat([
    liquidity_features,
    insolvency_features,
    manipulation_features,
    korean_features,
    stakeholder_features,
    growth_features,
    profitability_activity_features,
    composite_features,
    interaction_features
], axis=1)

print(f"\n✅ 총 {all_features.shape[1]}개의 도메인 기반 특성 생성 완료")
print("\n특성 카테고리별 개수:")
print(f"  - 유동성 위기: {liquidity_features.shape[1]}개")
print(f"  - 지급불능 패턴: {insolvency_features.shape[1]}개")
print(f"  - 재무조작 탐지: {manipulation_features.shape[1]}개")
print(f"  - 한국 시장 특화: {korean_features.shape[1]}개")
print(f"  - 이해관계자 행동: {stakeholder_features.shape[1]}개")
print(f"  - 성장성 지표: {growth_features.shape[1]}개")
print(f"  - 수익성/활동성: {profitability_activity_features.shape[1]}개")
print(f"  - 복합 리스크 지표: {composite_features.shape[1]}개")
print(f"  - 상호작용/비선형: {interaction_features.shape[1]}개")

# 원본 데이터의 기본 정보와 타겟 변수 추가
final_features = pd.concat([
    df[target_col],
    all_features
], axis=1)

# 결측치 비율 확인
missing_ratio = final_features.isnull().mean() * 100
print(f"\n결측치 50% 이상인 특성 수: {(missing_ratio > 50).sum()}개")
```

**출력:**

```
ERROR: NameError: name 'growth_features' is not defined
```

---

## 8. 비선형 관계 및 상호작용 특성

### 💡 왜 비선형 및 상호작용이 중요한가?

**경제적 가설: "위험 요인은 단독이 아닌 결합으로 작동한다"**

**실무 사례:**
```
Case 1: 부채비율 300% + ROA 15%
- 높은 부채지만 수익성 우수 → 안전 ✅

Case 2: 부채비율 300% + ROA -5%
- 높은 부채 + 적자 → 매우 위험 ❌❌
```

**비선형 관계:**
- 부채비율 < 100%: 안전
- 부채비율 100~200%: 주의
- 부채비율 > 200%: 위험 급증 (제곱 효과)

### 생성할 3개 상호작용 특성

| 특성 | 의미 | 해석 |
|------|------|------|
| 레버리지_수익성_상호작용 | 부채비율 × ROA | 부채가 높을 때 수익성이 더 중요 |
| 부채비율_제곱 | (부채비율)² | 과다부채의 비선형 위험 포착 |
| 임계값기반_복합지표 | 임계값 초과 횟수 | 다중 위험 신호 감지 |

**예시: 레버리지_수익성_상호작용**
```
기업 A: 부채비율 50% × ROA 10% = 5
기업 B: 부채비율 300% × ROA 10% = 30
기업 C: 부채비율 300% × ROA -5% = -15 ← 매우 위험!
```


### 코드 셀 #12

```python
def create_interaction_features(df, features_dict):
    """주요 변수 간 상호작용 효과를 포착하는 특성 생성"""
    
    interactions = pd.DataFrame(index=df.index)
    
    # 1. 레버리지 × 수익성 상호작용
    if '부채비율' in df.columns and 'ROE' in df.columns:
        interactions['레버리지_수익성_상호작용'] = (df['부채비율'] / 100) * (df['ROE'] / 100)
        interactions['고레버리지_저수익'] = ((df['부채비율'] > 200) & (df['ROE'] < 0)).astype(int)
    
    # 2. 유동성 × 성장 상호작용
    if '유동비율' in df.columns and '매출액' in df.columns:
        매출성장률 = df['매출액'] / df['매출액'].median() - 1
        interactions['유동성_성장_상호작용'] = df['유동비율'] * 매출성장률
    
    # 3. 규모 × 효율성 상호작용
    if '기업규모_로그' in features_dict['korean'].columns and '총자산회전율' in df.columns:
        interactions['규모_효율성_상호작용'] = features_dict['korean']['기업규모_로그'] * df['총자산회전율']
    
    # 4. 업종 × 재무비율 상호작용
    if '제조업_레버리지위험' in features_dict['korean'].columns:
        interactions['제조업_부채위험'] = features_dict['korean']['제조업_레버리지위험']
    
    # 5. 비선형 변환
    if '부채비율' in df.columns:
        interactions['부채비율_제곱'] = (df['부채비율'] / 100) ** 2
        interactions['부채비율_로그'] = np.log1p(df['부채비율'].abs())
    
    if '현금비율' in df.columns:
        interactions['현금비율_제곱근'] = np.sqrt(df['현금비율'].abs())
    
    # 6. 임계값 기반 특성
    if '이자보상배율' in features_dict['insolvency'].columns:
        interactions['이자보상_임계'] = (features_dict['insolvency']['이자보상배율'] < 1).astype(int)
    
    if '운전자본' in features_dict['liquidity'].columns:
        interactions['운전자본_음수'] = (features_dict['liquidity']['운전자본'] < 0).astype(int)
    
    print(f"✅ 상호작용 및 비선형 특성 {interactions.shape[1]}개 생성 완료")
    return interactions

features_dict = {
    'liquidity': liquidity_features,
    'insolvency': insolvency_features,
    'manipulation': manipulation_features,
    'korean': korean_features,
    'stakeholder': stakeholder_features
}

interaction_features = create_interaction_features(df, features_dict)
```

**출력:**

```
✅ 상호작용 및 비선형 특성 3개 생성 완료
```

---

### 코드 셀 #13

```python
# 상호작용 및 비선형 특성 생성
interaction_features_dict = {
    'liquidity': liquidity_features,
    'insolvency': insolvency_features,
    'profitability': profitability_activity_features
}

interaction_features = create_interaction_features(df, interaction_features_dict)
print(f"\n✅ 상호작용/비선형 특성 {interaction_features.shape[1]}개 생성 완료")
print("\n생성된 상호작용/비선형 특성:")
print(interaction_features.columns.tolist())

```

---

## 9. 특성 저장 및 메타데이터 생성

### 💾 생성된 특성 저장

모든 도메인 특성을 CSV 파일로 저장하고, 각 특성의 메타데이터를 기록합니다.


### 코드 셀 #14

```python
# 생성된 특성을 CSV 파일로 저장
output_path = '../data/features/domain_based_features.csv'
final_features.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 도메인 기반 특성이 {output_path}에 저장되었습니다.")

# 특성 메타데이터 저장
feature_metadata = pd.DataFrame({
    'category': (
        ['liquidity'] * liquidity_features.shape[1] +
        ['insolvency'] * insolvency_features.shape[1] +
        ['manipulation'] * manipulation_features.shape[1] +
        ['korean'] * korean_features.shape[1] +
        ['stakeholder'] * stakeholder_features.shape[1] +
        ['growth'] * growth_features.shape[1] +
        ['profitability_activity'] * profitability_activity_features.shape[1] +
        ['composite'] * composite_features.shape[1] +
        ['interaction'] * interaction_features.shape[1]
    ),
    'feature_name': all_features.columns.tolist(),
    'importance': feature_importance.set_index('feature')['importance'].reindex(all_features.columns).values
})

metadata_path = '../data/features/feature_metadata.csv'
feature_metadata.to_csv(metadata_path, index=False, encoding='utf-8-sig')
print(f"✅ 특성 메타데이터가 {metadata_path}에 저장되었습니다.")
```

---

## 10. 특성 중요도 초기 평가

### 🎯 목적

생성한 65개 도메인 특성 중 **실제로 부도 예측에 유용한 특성**을 식별합니다.

**평가 방법:**
- **RandomForestClassifier**: 트리 기반 특성 중요도 (Gini Importance)
- **불균형 데이터 처리**: class_weight='balanced' 적용
- **Top 30 특성**: 가장 중요한 상위 30개 특성 선별

**특성 중요도 해석:**
- 높은 중요도 → 부도 예측에 강한 영향
- 낮은 중요도 → 제거 고려 (차원 축소)

**다음 단계:**
- `03_상관관계_및_리스크_패턴_분석.ipynb`에서 더 정교한 특성 선택 수행
- 상관관계 분석 → 중복 특성 제거
- 최종적으로 ~40개 특성으로 축소


### 코드 셀 #15

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# 타겟과 특성 분리
X = all_features.copy()
y = df[target_col]

# -------------------------------------------------------------------------
# [수정된 부분] 범주형 변수 처리 (에러 해결 핵심)
# '위험경보등급'이 범주형(Category)이라서 median 계산 및 모델 학습 시 에러 발생
# -> 숫자로 변환 (정상:0, 주의:1, 경고:2, 위험:3)
# -------------------------------------------------------------------------
if '위험경보등급' in X.columns:
    # cat.codes는 -1(NaN), 0, 1, 2... 로 변환됨
    X['위험경보등급'] = X['위험경보등급'].cat.codes
    # 혹시 모를 -1(NaN) 값을 0으로 처리하거나 적절히 대체
    X['위험경보등급'] = X['위험경보등급'].replace(-1, 0)

# 결측치 처리 (중앙값으로 대체)
# 이제 모든 변수가 숫자형이므로 에러가 발생하지 않음
X_filled = X.fillna(X.median())

# 무한대 값 처리
X_filled = X_filled.replace([np.inf, -np.inf], 0)

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_filled, y, test_size=0.2, random_state=42, stratify=y
)

# 간단한 Random Forest로 특성 중요도 평가
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# 특성 중요도 추출
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🎯 상위 20개 중요 특성 (Random Forest 기준)")
print("="*60)
print(feature_importance.head(20).to_string(index=False))

# 시각화
fig = go.Figure()
fig.add_trace(go.Bar(
    y=feature_importance.head(20)['feature'].values[::-1],
    x=feature_importance.head(20)['importance'].values[::-1],
    orientation='h',
    marker_color='lightblue'
))

fig.update_layout(
    title='도메인 기반 특성 중요도 (상위 20개)',
    xaxis_title='Feature Importance',
    yaxis_title='특성명',
    height=600
)
fig.show()

# 간단한 성능 평가
y_pred_proba = rf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)

print(f"\n📊 초기 모델 성능 (도메인 특성만 사용)")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
```

**출력:**

```
🎯 상위 20개 중요 특성 (Random Forest 기준)
============================================================
   feature  importance
이해관계자_불신지수    0.115817
    신용등급점수    0.083416
     연체심각도    0.033175
   조기경보신호수    0.029903
     총연체건수    0.028992
 종합부도위험스코어    0.027431
      연체여부    0.025684
    재무레버리지    0.024257
    부채레버리지    0.023730
    부채상환년수    0.020146
    매출총이익률    0.020110
     순부채비율    0.019389
     이자부담률    0.019175
   기업규모_로그    0.019031
      판관비율    0.018474
      운전자본    0.018116
   시장포지션지수    0.018044
    단기부채비중    0.017896
   유동성압박지수    0.017864
    이자보상배율    0.017796
```

```
📊 초기 모델 성능 (도메인 특성만 사용)
ROC-AUC: 0.8235
PR-AUC: 0.1151
```

---

## 11. 핵심 인사이트 정리


### 코드 셀 #16

```python
print("\n" + "="*80)
print("📌 도메인 기반 특성 공학 - 핵심 인사이트")
print("="*80)

insights = [
    "1. 유동성 위기 조기 감지: 현금소진일수, 운전자본 건전성 등 즉각적 지급능력 특성 생성",
    "2. 지급불능 패턴: 자본잠식도, 부채상환년수 등 장기 지급능력 평가 지표 구현",
    "3. 재무조작 탐지: 한국형 M-Score를 통한 회계 이상 징후 포착",
    "4. 한국 시장 특화: 대기업 의존도, 제조업 특성, 외감 여부 등 한국 기업 특성 반영",
    "5. 이해관계자 신뢰: 연체, 세금체납, 신용등급 등 외부 신뢰도 지표 통합",
    "6. 복합 리스크 지표: 다양한 도메인 지식을 통합한 종합 위험 스코어 개발",
    "7. 비선형 관계: 제곱, 로그, 상호작용 항을 통한 복잡한 패턴 포착"
]

for insight in insights:
    print(insight)

print("\n" + "="*80)
print("🎯 다음 단계 권장사항")
print("="*80)

recommendations = [
    "1. 상관관계 분석을 통한 다중공선성 제거",
    "2. Information Value 계산을 통한 특성 선택",
    "3. SMOTE + Tomek Links 적용 전 특성 스케일링",
    "4. 카테고리별 특성 중요도 기반 가중 앙상블 모델 구축",
    "5. SHAP 분석을 통한 특성 기여도 해석"
]

for rec in recommendations:
    print(rec)
```

**출력:**

```
================================================================================
📌 도메인 기반 특성 공학 - 핵심 인사이트
================================================================================
1. 유동성 위기 조기 감지: 현금소진일수, 운전자본 건전성 등 즉각적 지급능력 특성 생성
2. 지급불능 패턴: 자본잠식도, 부채상환년수 등 장기 지급능력 평가 지표 구현
3. 재무조작 탐지: 한국형 M-Score를 통한 회계 이상 징후 포착
4. 한국 시장 특화: 대기업 의존도, 제조업 특성, 외감 여부 등 한국 기업 특성 반영
5. 이해관계자 신뢰: 연체, 세금체납, 신용등급 등 외부 신뢰도 지표 통합
6. 복합 리스크 지표: 다양한 도메인 지식을 통합한 종합 위험 스코어 개발
7. 비선형 관계: 제곱, 로그, 상호작용 항을 통한 복잡한 패턴 포착

================================================================================
🎯 다음 단계 권장사항
================================================================================
1. 상관관계 분석을 통한 다중공선성 제거
2. Information Value 계산을 통한 특성 선택
3. SMOTE + Tomek Links 적용 전 특성 스케일링
4. 카테고리별 특성 중요도 기반 가중 앙상블 모델 구축
5. SHAP 분석을 통한 특성 기여도 해석
```

---
