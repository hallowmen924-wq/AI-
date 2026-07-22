# KB 금융 AI Agent 개발 위키

> KB국민은행·KB국민카드 AI Lab 개발자 과정(5일) 교안을 **재사용 가능한 개발 지식**으로 재정리한 위키입니다.
> 원본은 `강의안(무단배포금지_위반시법적책임)_VER2/` 폴더이며, 각 문서 하단 **출처** 절에 원본 파일 경로를 명시했습니다.

## 한눈에 보기

이 위키는 "LangChain으로 LLM을 부르는 법"에서 시작해 "LangGraph로 통제 가능한 Agent를 만들고, 금융권 규제 아래 운영하는 법"까지의 전체 경로를 다룹니다.
개념 문서 15개 + 실습/프로젝트 문서 3개 + 주제별 MOC 4개로 구성되어 있습니다.

## 🚨 먼저 읽어야 할 것

원본 노트북 11개 파일에 **실제 API 키가 하드코딩되어 있습니다.** 재사용 전 반드시 확인하세요.
→ [오류_환경설정_트러블슈팅.md](오류_환경설정_트러블슈팅.md#1-원본-노트북에-하드코딩된-api-키-즉시-조치-필요)

## 주제별 지도 (MOC)

| MOC | 다루는 범위 |
|---|---|
| [MOC_LLM.md](MOC_LLM.md) | LangChain·LCEL·프롬프트·구조화 출력·Tool Calling·로컬 LLM |
| [MOC_RAG.md](MOC_RAG.md) | 문서 로딩·청킹·임베딩·벡터DB·하이브리드 검색·GraphRAG·평가 |
| [MOC_LangGraph.md](MOC_LangGraph.md) | State/Node/Edge·조건부 분기·ToolNode·루프·HITL·Multi-Agent |
| [MOC_금융_AI.md](MOC_금융_AI.md) | 금융권 특유의 요구사항 — 근거 인용·정직한 실패·PII·거버넌스·내부망 |

## 전체 문서 목록

### 개념 — LLM 기초 (Day1)
- [개념_LangChain_LCEL_기초.md](개념_LangChain_LCEL_기초.md) — 모델 I/O, 프롬프트 템플릿, 파이프 체인, 메모리
- [개념_RAG_파이프라인.md](개념_RAG_파이프라인.md) — 로드→분할→임베딩→Chroma→검색→근거 답변

### 개념 — 검색 품질과 외부 연동 (Day2)
- [개념_문서유형별_전처리.md](개념_문서유형별_전처리.md) — 약관/FAQ/표/스크립트를 유형별로 자르는 법
- [개념_검색품질_개선.md](개념_검색품질_개선.md) — chunk/top-k/메타데이터 필터/임계값/BM25 하이브리드
- [개념_GraphRAG_Vector_융합.md](개념_GraphRAG_Vector_융합.md) — 관계를 따라가는 검색과 평가 자동화
- [개념_Tool_Calling.md](개념_Tool_Calling.md) — LLM이 스스로 함수를 고르게 하는 4단계
- [개념_Text_to_SQL.md](개념_Text_to_SQL.md) — 자연어 → SQL → DB 실행, SELECT 가드레일
- [개념_MCP_구조와_구현.md](개념_MCP_구조와_구현.md) — FastMCP 서버, stdio/HTTP 클라이언트, LangChain 어댑터

### 개념 — LangGraph (Day3)
- [개념_LangGraph_State_Node_Edge.md](개념_LangGraph_State_Node_Edge.md) — 그래프 3요소와 리듀서
- [개념_조건부Edge와_ToolNode.md](개념_조건부Edge와_ToolNode.md) — `tools_condition` + `ToolNode` 표준 패턴
- [개념_Graph_구조_패턴.md](개념_Graph_구조_패턴.md) — Router / Map-Reduce(Send) / Evaluator-Optimizer
- [개념_의도분류_기반_RAG_Agent.md](개념_의도분류_기반_RAG_Agent.md) — 다중 DB 라우팅 + 반복 검색 루프
- [개념_HITL과_검증루프.md](개념_HITL과_검증루프.md) — `interrupt`/`Command`, 3중 검문소
- [개념_Agent_통제력과_루프탈출.md](개념_Agent_통제력과_루프탈출.md) — 메모리·컨텍스트 다이어트·카운터·recursion_limit

### 개념 — Multi-Agent와 서비스화 (Day4)
- [개념_MultiAgent_Supervisor.md](개념_MultiAgent_Supervisor.md) — 고정 경로 vs Supervisor 라우팅
- [개념_FastAPI_Agent_서비스화.md](개념_FastAPI_Agent_서비스화.md) — `/chat` 엔드포인트, 세션=thread_id

### 개념 — 운영 (Day5)
- [개념_Fallback과_Rollback.md](개념_Fallback과_Rollback.md) — 모델 폴백, 검색 폴백, 검증 실패 롤백
- [개념_트레이싱과_감사로그.md](개념_트레이싱과_감사로그.md) — LangSmith / Langfuse / 커스텀 콜백
- [개념_Agent_평가.md](개념_Agent_평가.md) — RAGAS 3대 지표, LLM-as-a-Judge, 회귀 테스트
- [개념_PII보호와_거버넌스.md](개념_PII보호와_거버넌스.md) — 마스킹·인젝션 방어·RBAC·Runbook
- [개념_비용_라우팅_최적화.md](개념_비용_라우팅_최적화.md) — 난이도 라우팅 + 캐시

### 개념 — 내부망
- [개념_LocalLLM_내부망_구축.md](개념_LocalLLM_내부망_구축.md) — HuggingFace 오프라인, Ollama, 반입 절차

### 실습 · 프로젝트 · 문제해결
- [실습_바이브코딩_카탈로그.md](실습_바이브코딩_카탈로그.md) — 28개 실습 앱 색인과 공통 패턴
- [프로젝트_실전과제_가이드.md](프로젝트_실전과제_가이드.md) — 실전 프로젝트 5종과 공통 안전장치
- [오류_환경설정_트러블슈팅.md](오류_환경설정_트러블슈팅.md) — 하드코딩 키, 버전 충돌, 자주 나는 오류

## 추천 학습 순서

1. [개념_LangChain_LCEL_기초.md](개념_LangChain_LCEL_기초.md) → [개념_RAG_파이프라인.md](개념_RAG_파이프라인.md)
2. [개념_문서유형별_전처리.md](개념_문서유형별_전처리.md) → [개념_검색품질_개선.md](개념_검색품질_개선.md)
3. [개념_Tool_Calling.md](개념_Tool_Calling.md) → [개념_LangGraph_State_Node_Edge.md](개념_LangGraph_State_Node_Edge.md) → [개념_조건부Edge와_ToolNode.md](개념_조건부Edge와_ToolNode.md)
4. [개념_Graph_구조_패턴.md](개념_Graph_구조_패턴.md) → [개념_의도분류_기반_RAG_Agent.md](개념_의도분류_기반_RAG_Agent.md) → [개념_HITL과_검증루프.md](개념_HITL과_검증루프.md)
5. [개념_MultiAgent_Supervisor.md](개념_MultiAgent_Supervisor.md) → [개념_FastAPI_Agent_서비스화.md](개념_FastAPI_Agent_서비스화.md)
6. 운영 5종 → [프로젝트_실전과제_가이드.md](프로젝트_실전과제_가이드.md)

## 검증된 실행 환경

Python 3.11 / langchain 1.3.11 / langgraph 1.2.7 / streamlit 1.54 / fastapi 0.139 / mcp 1.28.1

## 출처

- Source: `강의안(무단배포금지_위반시법적책임)_VER2/` (Day1~Day5, LocalLLM_실습, 바이브코딩실습, 실전프로젝트)
