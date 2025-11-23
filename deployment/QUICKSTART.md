# ⚡ 빠른 시작 가이드 (5분 배포)

## 🎯 목표
이 폴더를 GitHub 레포지토리로 만들고 Streamlit Cloud에 배포하기

---

## 📋 사전 준비 (2분)

### 1. Git LFS 설치

```bash
# macOS
brew install git-lfs

# Linux (Ubuntu/Debian)
sudo apt-get install git-lfs

# Windows
# Git for Windows에 포함됨
```

```bash
# 설치 확인
git lfs install
```

### 2. DART API 키 발급

1. [DART 오픈API](https://opendart.fss.or.kr/) 접속
2. 회원가입 → 로그인
3. "인증키 신청" → 발급
4. API 키 복사 (나중에 사용)

---

## 🚀 배포 단계 (3분)

### Step 1: Git 초기화 (30초)

```bash
# deployment 디렉토리로 이동
cd deployment

# Git 초기화
git init

# 모든 파일 추가
git add .

# 첫 커밋
git commit -m "Initial commit: 한국 기업 부도 예측 앱"
```

---

### Step 2: GitHub 레포지토리 생성 (1분)

1. [GitHub](https://github.com/new) 접속
2. Repository name: `bankruptcy-prediction-app`
3. Public 선택
4. **Initialize 옵션 모두 체크 해제** ⚠️
5. "Create repository" 클릭

---

### Step 3: GitHub에 푸시 (30초)

```bash
# 원격 레포지토리 연결 (yourusername을 본인 계정명으로 변경)
git remote add origin https://github.com/yourusername/bankruptcy-prediction-app.git

# 푸시
git branch -M main
git push -u origin main
```

> ⏳ 푸시 완료까지 10-30초 소요 (Git LFS 업로드)

---

### Step 4: Streamlit Cloud 배포 (1분)

1. **[Streamlit Cloud](https://share.streamlit.io/) 접속**

2. **"Sign up with GitHub" 클릭**

3. **"New app" 클릭**

4. **레포지토리 정보 입력**:
   ```
   Repository: yourusername/bankruptcy-prediction-app
   Branch: main
   Main file path: app.py
   ```

5. **Advanced settings 클릭**

6. **Secrets 입력** (TOML 형식):
   ```toml
   DART_API_KEY = "여기에_발급받은_API_키_입력"
   ```

7. **Deploy! 클릭**

---

### Step 5: 배포 완료 확인 (30초)

배포 로그 확인:
```
✓ Repository cloned
✓ fonts-nanum installed
✓ streamlit installed
✓ App started successfully
```

앱 URL:
```
https://yourusername-bankruptcy-prediction-app-xxxxx.streamlit.app
```

---

## ✅ 배포 완료!

### 앱 테스트

1. 배포된 URL 접속
2. "샘플 데이터 사용" 선택
3. "정상 기업" 선택
4. "샘플 분석" 클릭
5. 결과 확인 ✓

---

## 🔧 배포 후 작업 (Optional)

### README 업데이트

```bash
# README.md 수정 (앱 URL 추가)
nano README.md  # 또는 vim, code 등

# 커밋 및 푸시
git add README.md
git commit -m "docs: Update app URL"
git push
```

### 앱 공유

- SNS에 링크 공유
- 이력서/포트폴리오에 추가
- GitHub README 배지 추가

---

## 🆘 문제 발생 시

### 에러 1: Git LFS 파일 업로드 실패

```bash
# Git LFS 재설치
git lfs install

# 파일 재푸시
git lfs push --all origin main
```

### 에러 2: 한글 깨짐

→ `packages.txt` 파일 확인 (이미 있음 ✅)

### 에러 3: DART API 에러

→ Streamlit Cloud Secrets에 API 키 재확인

### 에러 4: 배포 실패

→ Streamlit Cloud 로그 확인 후 에러 메시지 검색

---

## 📚 더 자세한 가이드

- **전체 배포 가이드**: `DEPLOYMENT_GUIDE.md` 참조
- **폴더 구조 설명**: `STRUCTURE.md` 참조
- **프로젝트 소개**: `README.md` 참조

---

## ⏱️ 전체 소요 시간

- **사전 준비**: 2분
- **Git 초기화**: 30초
- **GitHub 레포 생성**: 1분
- **푸시**: 30초
- **Streamlit 배포**: 1분
- **총**: **약 5분**

---

## 🎉 축하합니다!

이제 여러분의 부도 예측 시스템이 전 세계에 공개되었습니다! 🌍

**다음 단계**:
1. 앱 테스트 및 피드백 수집
2. 기능 추가 및 개선
3. 사용자 가이드 작성
4. GitHub Star 받기 ⭐

---

**행운을 빕니다! 🚀**
