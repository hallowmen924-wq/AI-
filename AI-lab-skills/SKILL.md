---
name: llm-agent-dev-patterns
description: Use when building, reviewing, or explaining LangChain/RAG/LangGraph/Agent code - chain construction (LCEL), RAG pipelines (chunking, embeddings, vector stores, hybrid search, GraphRAG), Tool Calling, Text-to-SQL, MCP servers, multi-agent supervisors, FastAPI agent services, or production concerns like fallback/rollback, tracing, RAGAS evaluation, PII/prompt-injection guardrails, cost routing, or local/offline LLM deployment
---

# LLM Agent Dev Patterns

## Overview
25 reference docs distilled from a 5-day LLM agent development curriculum (LangChain → RAG → LangGraph → Multi-Agent → production operations). Each doc is code-annotated with 핵심 개념(concepts) / 전체 동작 흐름(flow) / 핵심 코드(reusable code) / 주의사항(pitfalls) / 출처(source).

## When to Use
Match the task to a row below, then Read the matching reference file(s) — don't answer from general knowledge alone. These docs capture pitfalls found in the original course material (hardcoded-credential patterns, version-specific API quirks, SQL-injection-style risks in Text-to-SQL, Windows path-quoting issues) that generic knowledge misses.

## Reference Index

| 주제 | 파일 | 다루는 내용 |
|---|---|---|
| LangChain 기초 | references/개념_LangChain_LCEL_기초.md | 모델 I/O, `PromptTemplate`, LCEL(`\|`) 파이프 체인, 대화 메모리 |
| RAG 파이프라인 | references/개념_RAG_파이프라인.md | 로드→분할→임베딩→`Chroma`→검색→근거 답변 |
| 문서유형별 전처리 | references/개념_문서유형별_전처리.md | 약관/FAQ/표/스크립트별 청킹 전략 |
| 검색품질 개선 | references/개념_검색품질_개선.md | chunk_size/top-k/메타데이터 필터/유사도 임계값/BM25 하이브리드 |
| GraphRAG | references/개념_GraphRAG_Vector_융합.md | 지식그래프+벡터 검색 융합, 평가 자동화 |
| Tool Calling | references/개념_Tool_Calling.md | `bind_tools`, `@tool` 데코레이터, LLM이 함수를 스스로 고르는 4단계 루프 |
| Text-to-SQL | references/개념_Text_to_SQL.md | 자연어→SQL 생성→실행, SELECT 전용 가드레일 |
| MCP | references/개념_MCP_구조와_구현.md | `FastMCP` 서버, stdio/HTTP 클라이언트, LangChain 어댑터 |
| LangGraph 기본 | references/개념_LangGraph_State_Node_Edge.md | State/Node/Edge 3요소, 리듀서(Annotated) |
| 조건부Edge/ToolNode | references/개념_조건부Edge와_ToolNode.md | `tools_condition` + `ToolNode` 표준 패턴 |
| Graph 구조 패턴 | references/개념_Graph_구조_패턴.md | Router / Map-Reduce(Send) / Evaluator-Optimizer |
| 의도분류 RAG Agent | references/개념_의도분류_기반_RAG_Agent.md | 다중 DB 라우팅, 반복 검색 루프 |
| HITL/검증루프 | references/개념_HITL과_검증루프.md | `interrupt`/`Command`, 3중 검문소 패턴 |
| Agent 통제력/루프탈출 | references/개념_Agent_통제력과_루프탈출.md | 메모리 관리, 컨텍스트 다이어트, `recursion_limit` |
| Multi-Agent Supervisor | references/개념_MultiAgent_Supervisor.md | 고정 경로 vs Supervisor 동적 라우팅 |
| FastAPI 서비스화 | references/개념_FastAPI_Agent_서비스화.md | `/chat` 엔드포인트 설계, 세션=thread_id |
| Fallback/Rollback | references/개념_Fallback과_Rollback.md | 모델 폴백, 검색 폴백, 검증 실패 시 롤백 |
| 트레이싱/감사로그 | references/개념_트레이싱과_감사로그.md | LangSmith / Langfuse / 커스텀 콜백 |
| Agent 평가 | references/개념_Agent_평가.md | RAGAS 3대 지표, LLM-as-a-Judge, 회귀 테스트 |
| PII/거버넌스 | references/개념_PII보호와_거버넌스.md | 마스킹, 프롬프트 인젝션 방어, RBAC, Runbook |
| 비용/라우팅 최적화 | references/개념_비용_라우팅_최적화.md | 난이도 기반 모델 라우팅, 캐시 |
| Local LLM/내부망 | references/개념_LocalLLM_내부망_구축.md | HuggingFace 오프라인 설치, Ollama, 반입 절차 |
| 실습 앱 카탈로그 | references/실습_바이브코딩_카탈로그.md | 28개 실습 앱 색인과 공통 패턴 |
| 실전 프로젝트 가이드 | references/프로젝트_실전과제_가이드.md | 실전 프로젝트 5종, 공통 안전장치 |
| 전체 개요/학습 순서 | references/README.md | 추천 학습 순서, 검증된 실행 환경, 출처 |

## Common Mistakes
- **Loading every file at once** — pick only the 1-3 files matching the current task; each is a full deep-dive, not a summary.
- **Copying credential-handling code verbatim without checking it** — the original course notebooks had real API keys hardcoded in multiple places (see README.md). When adapting example code, always use env vars / `getpass()`, never hardcode.
- **Known gap**: the source wiki's own README also references `MOC_LLM.md`, `MOC_RAG.md`, `MOC_LangGraph.md`, `MOC_금융_AI.md`, and `오류_환경설정_트러블슈팅.md` — these 5 files were missing from the source folder when this skill was built (2026-07-22) and are not included here. If they turn up later, add them to `references/` and this index.
