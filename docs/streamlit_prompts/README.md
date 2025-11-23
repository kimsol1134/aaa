# Streamlit 앱 리팩토링 프롬프트 (분할 버전)

> `/notebooks`의 4개 발표 노트북을 기반으로 `/streamlit_app` 로직을 수정하는 상세 가이드

---

## 📁 파일 구조

이 프롬프트는 5개의 파일로 나뉘어져 있습니다. 순서대로 읽거나 필요한 부분만 선택적으로 읽을 수 있습니다.

### 1. `01_overview.md` (필수 읽기)
- 프로젝트 개요 및 목표
- 현재 상황 분석 (노트북 4개, 앱 구조)
- 전체 작업 흐름

### 2. `02_critical_issues.md` (필수 읽기)
- 발견된 8개 주요 차이점
- 각 이슈의 심각도 및 영향 분석
- Issue 1-8 상세 설명

### 3. `03_task_details_priority1.md` (Critical Tasks)
- Priority 1: 시스템 작동 불가 수준의 Critical 이슈
- Task 1: 모델 파일명 수정
- Task 2: Traffic Light 임계값 수정
- Task 3: 특성 개수 일치 확인

### 4. `04_task_details_priority2.md` (High Priority Tasks)
- Priority 2: 기능 부정확성 이슈
- Task 4: 실제 SHAP 값 계산 구현
- Task 5: 비즈니스 가치 분석 추가
- Task 6-8: 기타 High Priority 작업

### 5. `05_checklist_and_resources.md` (참고 자료)
- 프롬프트 엔지니어링 전략
- 최종 체크리스트
- 시작 방법
- 참고 자료 목록

---

## 🚀 빠른 시작 가이드

### Case 1: 처음 시작하는 경우
```
1. 01_overview.md 읽기 (프로젝트 이해)
2. 02_critical_issues.md 읽기 (문제점 파악)
3. 03_task_details_priority1.md 읽고 작업 시작
```

### Case 2: 특정 Task만 수행하는 경우
```
예: Task 4 (SHAP 구현)만 필요한 경우
→ 04_task_details_priority2.md의 "Task 4" 섹션만 읽기
```

### Case 3: 체크리스트만 확인하는 경우
```
→ 05_checklist_and_resources.md 읽기
```

---

## 📊 각 파일 크기 및 내용

| 파일 | 크기 | 예상 읽기 시간 | 주요 내용 |
|------|------|----------------|----------|
| 01_overview.md | ~15KB | 5분 | 프로젝트 개요, 노트북/앱 구조 |
| 02_critical_issues.md | ~25KB | 10분 | 8개 주요 차이점 상세 분석 |
| 03_task_details_priority1.md | ~30KB | 15분 | Task 1-3 상세 코드 및 지시사항 |
| 04_task_details_priority2.md | ~35KB | 20분 | Task 4-5 상세 코드 및 지시사항 |
| 05_checklist_and_resources.md | ~10KB | 5분 | 체크리스트, 참고 자료 |

**전체**: ~115KB (원본 대비 동일하지만 논리적으로 분할됨)

---

## 💡 사용 팁

### Tip 1: 순차 읽기
처음이라면 01 → 02 → 03 → 04 → 05 순서로 읽으세요.

### Tip 2: 선택 읽기
특정 작업만 필요하다면:
- Task 1-3: `03_task_details_priority1.md`만 읽기
- Task 4-5: `04_task_details_priority2.md`만 읽기

### Tip 3: Claude Code와 함께 사용
```
"docs/streamlit_prompts/01_overview.md를 읽고 프로젝트 현황을 요약해줘"
"docs/streamlit_prompts/03_task_details_priority1.md의 Task 1을 수행해줘"
```

---

## 🔄 업데이트 이력

- **v1.0** (2025-11-23): 초기 버전 생성
  - 원본 단일 파일을 5개로 분할
  - 논리적 섹션별 분리
  - 각 파일 독립적으로 읽을 수 있도록 구성

---

**작성일**: 2025-11-23
**프로젝트**: 한국 기업 부도 예측 시스템
**버전**: 1.0
