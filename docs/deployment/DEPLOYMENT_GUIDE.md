# 🚀 Streamlit Cloud 배포 완벽 가이드

## 📋 배포 전 체크리스트

- [ ] Git LFS 설치 완료
- [ ] DART API 키 발급 완료
- [ ] GitHub 계정 준비
- [ ] 모든 파일 확인

---

## 1️⃣ Git 및 Git LFS 설정

### Git LFS 설치

**Windows:**
```bash
# Git for Windows에 포함됨
git lfs install
```

**macOS:**
```bash
brew install git-lfs
git lfs install
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install git-lfs
git lfs install
```

### Git 초기화

```bash
# deployment 디렉토리로 이동
cd deployment

# Git 초기화
git init

# Git LFS 설정 확인
git lfs track "*.pkl"
git lfs track "*.h5"

# .gitattributes 확인
cat .gitattributes

# 출력:
# *.pkl filter=lfs diff=lfs merge=lfs -text
# *.h5 filter=lfs diff=lfs merge=lfs -text
# ...
```

---

## 2️⃣ 환경 변수 설정

### .env 파일 생성 (로컬 테스트용)

```bash
# .env.example 복사
cp .env.example .env

# 편집기로 .env 열기
nano .env  # 또는 vim, code 등
```

`.env` 파일 내용:
```env
DART_API_KEY=your_actual_dart_api_key_here
```

> ⚠️ **주의**: `.env` 파일은 `.gitignore`에 포함되어 있어 Git에 커밋되지 않습니다.

---

## 3️⃣ 로컬 테스트

### 앱 실행

```bash
# 패키지 설치
pip install -r requirements.txt

# Streamlit 앱 실행
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속 후 테스트:

- [ ] 페이지 로딩 정상
- [ ] 한글 폰트 정상 표시
- [ ] 샘플 데이터 분석 작동
- [ ] DART API 검색 작동 (API 키 입력 시)
- [ ] 에러 없음

---

## 4️⃣ GitHub 레포지토리 생성

### 4.1 GitHub에서 새 레포지토리 생성

1. [GitHub](https://github.com/) 접속 및 로그인
2. 오른쪽 상단 "+" → "New repository" 클릭
3. 레포지토리 정보 입력:
   - **Repository name**: `bankruptcy-prediction-app`
   - **Description**: `한국 기업 부도 예측 시스템 - Streamlit 앱`
   - **Public** (무료) 또는 **Private** (Pro 계정 필요)
   - **Initialize this repository with**: 아무것도 체크하지 않음
4. "Create repository" 클릭

### 4.2 로컬 Git 커밋 및 푸시

```bash
# 모든 파일 추가
git add .

# 커밋
git commit -m "Initial commit: 한국 기업 부도 예측 앱"

# 원격 레포지토리 연결
git remote add origin https://github.com/yourusername/bankruptcy-prediction-app.git

# 푸시
git branch -M main
git push -u origin main
```

### 4.3 Git LFS 파일 확인

GitHub 레포지토리에서:
1. `data/processed/best_model.pkl` 파일 확인
2. 파일 크기 옆에 "Stored with Git LFS" 표시 확인

---

## 5️⃣ Streamlit Cloud 배포

### 5.1 Streamlit Cloud 가입

1. [Streamlit Cloud](https://share.streamlit.io/) 접속
2. "Sign up with GitHub" 클릭
3. GitHub 권한 승인
4. 이메일 인증 완료

### 5.2 새 앱 배포

1. **"New app" 버튼 클릭**

2. **레포지토리 정보 입력**
   ```
   Repository: yourusername/bankruptcy-prediction-app
   Branch: main
   Main file path: app.py
   ```

3. **Advanced settings 클릭**

4. **Python version 선택** (Optional)
   ```
   Python version: 3.10
   ```

5. **Secrets 입력**
   ```toml
   # DART API Key
   DART_API_KEY = "your_actual_dart_api_key_here"
   ```

   > 💡 여러 줄 입력 시 TOML 형식 사용:
   ```toml
   # DART API
   DART_API_KEY = "abc123..."

   # 기타 설정
   LOG_LEVEL = "INFO"
   ```

6. **Deploy! 버튼 클릭**

---

## 6️⃣ 배포 상태 확인

### 배포 로그 모니터링

Streamlit Cloud 대시보드에서 실시간 로그 확인:

```
Cloning repository...
✓ Repository cloned

Installing system packages...
✓ fonts-nanum installed

Installing Python packages...
✓ streamlit 1.29.0 installed
✓ pandas 2.1.3 installed
...

Starting app...
✓ App started successfully

Your app is live at:
https://yourusername-bankruptcy-prediction-app-xxxxx.streamlit.app
```

### 자주 발생하는 에러 및 해결

#### 에러 1: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'xxx'
```
**해결**: `requirements.txt`에 패키지 추가 후 커밋 & 푸시

#### 에러 2: 한글 폰트 없음
```
Font 'NanumGothic' not found
```
**해결**: `packages.txt` 파일 확인 (이미 있음 ✅)

#### 에러 3: Git LFS 파일 다운로드 실패
```
Error downloading object: data/processed/best_model.pkl
```
**해결**: GitHub LFS 대역폭 확인 (무료: 1GB/월)

#### 에러 4: 메모리 부족
```
MemoryError
```
**해결**:
- 무료 티어: RAM 1GB 제한
- 모델 크기 확인 (현재: 103KB ✅)
- Streamlit Pro 고려

#### 에러 5: DART API 키 없음
```
KeyError: 'DART_API_KEY'
```
**해결**: Streamlit Cloud Secrets에 키 추가

---

## 7️⃣ 배포 후 확인사항

### 앱 URL 접속

```
https://yourusername-bankruptcy-prediction-app-xxxxx.streamlit.app
```

### 기능 테스트

- [ ] 페이지 로딩 (2-3초 이내)
- [ ] 한글 폰트 정상 표시
- [ ] 샘플 데이터 분석 작동
- [ ] DART API 검색 작동
- [ ] 그래프/차트 정상 표시
- [ ] 에러 없음

### README 업데이트

배포 완료 후 `README.md` 수정:

```markdown
## 🚀 배포된 앱

> 👉 **[여기서 앱 사용하기](https://yourusername-bankruptcy-prediction-app-xxxxx.streamlit.app)**
```

커밋 및 푸시:
```bash
git add README.md
git commit -m "docs: Update app URL"
git push
```

---

## 8️⃣ 앱 관리

### 재배포 (코드 수정 후)

```bash
# 코드 수정 후
git add .
git commit -m "feat: Add new feature"
git push
```

→ Streamlit Cloud가 **자동으로 재배포** (2-3분 소요)

### 수동 재부팅

Streamlit Cloud 대시보드:
1. 앱 클릭
2. 우측 상단 "⋮" 메뉴
3. "Reboot app" 클릭

### 로그 확인

Streamlit Cloud 대시보드:
- "Manage app" → "Logs" 탭
- 실시간 로그 및 에러 확인

### Secrets 수정

Streamlit Cloud 대시보드:
1. "Manage app" → "Settings" 탭
2. "Secrets" 섹션
3. TOML 형식으로 수정
4. "Save" 클릭 → 자동 재배포

---

## 9️⃣ 성능 최적화 (Optional)

### 캐싱 활용

현재 `app.py`에서 이미 사용 중 ✅:

```python
@st.cache_resource
def load_predictor():
    """모델 로딩 (캐시)"""
    # ...

@st.cache_data(ttl=3600)
def fetch_dart_data(company_name: str, year: str):
    """DART API 데이터 조회 (1시간 캐시)"""
    # ...
```

### 메모리 최적화

모델 크기 줄이기 (필요 시):
```python
# predictor.py에서
model = joblib.load(model_path, mmap_mode='r')  # 메모리 매핑
```

---

## 🔟 Pro 계정 고려 (Optional)

### 무료 vs Pro 비교

| 기능 | 무료 | Pro ($20/월) |
|-----|------|-------------|
| **리소스** | 1 CPU, 1GB RAM | 4 CPU, 8GB RAM |
| **앱 수** | 무제한 | 무제한 |
| **Private 앱** | ❌ | ✅ |
| **Custom domain** | ❌ | ✅ |
| **우선 지원** | ❌ | ✅ |

현재 프로젝트는 **무료 티어로 충분** (모델 103KB, 메모리 효율적)

---

## 📞 도움이 필요하신가요?

### 공식 문서
- [Streamlit Cloud 공식 문서](https://docs.streamlit.io/streamlit-community-cloud)
- [Git LFS 문서](https://git-lfs.github.com/)

### 커뮤니티
- [Streamlit 포럼](https://discuss.streamlit.io/)
- [Streamlit Discord](https://discord.gg/streamlit)

---

## ✅ 최종 체크리스트

배포 완료 확인:

- [ ] Git LFS 설정 완료
- [ ] GitHub 레포지토리 생성 완료
- [ ] Streamlit Cloud 배포 완료
- [ ] 앱 URL 접속 가능
- [ ] 모든 기능 정상 작동
- [ ] README URL 업데이트 완료
- [ ] 에러 없음

🎉 **축하합니다! 배포가 완료되었습니다!** 🎉

---

**다음 단계:**
1. 앱 URL을 SNS/이력서에 공유
2. 사용자 피드백 수집
3. 기능 개선 및 업데이트
4. Star 받기 ⭐
