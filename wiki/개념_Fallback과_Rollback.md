# 개념: Fallback과 Rollback 설계

## 한눈에 보기

**Fallback(폴백)** 은 "가던 길이 막혔으니 **다른 길로 간다**", **Rollback(롤백)** 은 "잘못 만들었으니 **되돌아가 다시 한다**" 입니다.
LLM 폴백은 `llm.with_fallbacks([backup])` 한 줄, 검색 폴백은 조건부 엣지, 롤백은 검증 실패 시 이전 노드로 되돌리는 엣지로 구현합니다. 금융 SLA를 지키려면 셋 다 필요합니다.

## 핵심 개념

| 구분 | Fallback | Rollback |
|---|---|---|
| 언제 | 실행이 **실패**했을 때 | 실행은 됐는데 **결과가 나쁠 때** |
| 방향 | 옆으로 (대체 수단) | 뒤로 (이전 단계) |
| 예시 | Primary 모델 장애 → Secondary / 벡터 검색 실패 → 키워드 검색 | 검증 불합격 → 초안 폐기 후 재검색 |
| 상한 | 대체 수단 목록 길이 | `MAX_ROLLBACK` 카운터 |

| 용어 | 설명 |
|---|---|
| **SLA** | "1년 중 99.9%는 정상 동작하겠다"는 대외 약속 |
| **가용성** | 실제로 서비스가 살아있는 시간 비율 |
| **`with_fallbacks([...])`** | Primary 실패 시 자동으로 다음 모델 호출 |
| **`with_retry(stop_after_attempt=N)`** | 일시적 오류를 N회까지 자동 재시도 |
| **`request_timeout`** | 응답 제한시간. 무한 대기 방지 |
| **상태머신(stage)** | 지금 어느 단계인지 State에 기록 — 실패 시 "어디서 멈췄는지" 안내 가능 |

**중요**: 폴백도 롤백도 실패하면 **정직하게 사람에게 넘겨야** 합니다. 지어낸 답변보다 "상담원 연결"이 낫습니다. → [[개념_Agent_통제력과_루프탈출]]

## 전체 동작 흐름

```
START → vector_search ──┬─ 근거 있음 ──────────────→ make_draft
                         └─ 없음 → keyword_fallback ─┬→ make_draft      (폴백)
                                                      └→ handle_error   (정직한 실패)
                                    make_draft
                                        ↓
                                    verify ──┬─ 합격 → finalize → END
                                             ├─ 불합격 & rollback<2 → rollback → vector_search   (롤백)
                                             └─ 불합격 & rollback>=2 → handle_error → END
```

## 주요 코드 구성

| 파일 | 역할 |
|---|---|
| `Day5/Day05_01_Fallback설계.ipynb` | LLM 폴백 기초 (Primary → Secondary, retry, timeout) |
| `Day5/Day05_02_Fallback_Rollback_설계.ipynb` | **상태머신 + 검색 폴백 + 검증 롤백 통합** |
| `바이브코딩실습/Day05_01_멈추지않는상담봇/app.py` | Primary 장애 → Secondary 시연 |
| `바이브코딩실습/Day05_02_보고서상태머신/app.py` | 폴백·롤백 시연 (**API 키 없이 동작**) |

## 실행 방법

```bash
pip install -qU langgraph langchain langchain-core langchain-openai langchain-chroma chromadb
$env:OPENAI_API_KEY = "sk-..."

cd 바이브코딩실습/Day05_02_보고서상태머신 && streamlit run app.py
```

## 핵심 코드

### 1. LLM 폴백 — 한 줄 ⭐

```python
primary   = ChatOpenAI(model="gpt-4o-mini-NOT-EXIST", temperature=0, max_retries=0)  # 실패 유도
secondary = ChatOpenAI(model="gpt-4o-mini", temperature=0)

llm_with_fallback = primary.with_fallbacks([secondary])
answer = llm_with_fallback.invoke("정기예금 이자는 언제 지급되나요?")   # Secondary가 응답
```

> `max_retries=0` 을 준 이유: 기본값이면 Primary가 여러 번 재시도하느라 폴백이 늦어집니다. **폴백을 빠르게 하려면 Primary의 재시도를 줄이세요.**

### 2. 타임아웃과 재시도

```python
robust = ChatOpenAI(model="gpt-4o-mini", temperature=0, request_timeout=10, max_retries=0)
robust = robust.with_retry(stop_after_attempt=2)
```

### 3. 어느 모델이 응답했는지 로그 남기기

폴백이 조용히 동작하면 장애를 인지하지 못합니다. **반드시 기록하세요.**

```python
def invoke_with_log(primary_llm, secondary_llm, question):
    try:
        res = primary_llm.invoke(question)
        print("[LOG] 응답 모델: Primary")
        return res
    except Exception as e:
        print(f"[LOG] Primary 실패({type(e).__name__}) → Secondary 전환")
        res = secondary_llm.invoke(question)
        print("[LOG] 응답 모델: Secondary")
        return res
```

### 4. 상태머신 State — stage로 진행 단계 기록

```python
import operator
from typing import Annotated, List, Optional

class ConsultState(TypedDict):
    question: str
    stage: str                                    # 접수→벡터검색→초안작성→검증→완료
    context: List[str]
    search_method: str                            # vector | keyword — 어떤 경로였는지
    draft: str
    answer: str
    verify_passed: Optional[bool]
    rollback_count: Annotated[int, operator.add]
    demo_force_reject: bool                       # 교육용 강제 불합격 스위치
    error: Optional[str]
```

### 5. 검색 폴백 — 벡터 실패 시 키워드로 ⭐

```python
THRESHOLD = 0.4

def vector_search(state):
    hits = store.similarity_search_with_relevance_scores(state["question"], k=2)
    passed = [doc.page_content for doc, score in hits if score >= THRESHOLD]
    return {"stage": "벡터검색", "context": passed, "search_method": "vector"}

def keyword_score(question, text):
    words = set(question.replace("?", " ").replace(".", " ").split())
    return sum(1 for w in words if len(w) >= 2 and w in text)

def keyword_fallback(state):
    best = max(docs, key=lambda d: keyword_score(state["question"], d.page_content))
    if keyword_score(state["question"], best.page_content) == 0:
        return {"stage": "검색실패", "context": [], "search_method": "keyword"}
    return {"stage": "키워드검색", "context": [best.page_content], "search_method": "keyword"}

def route_after_vector(state):
    return "make_draft" if state["context"] else "keyword_fallback"

def route_after_keyword(state):
    return "make_draft" if state["context"] else "handle_error"     # 둘 다 실패 → 정직한 실패
```

### 6. 롤백 — 검증 불합격 시 초안 폐기 후 되돌리기 ⭐⭐

```python
class Verdict(BaseModel):
    passed: bool = Field(description="초안이 근거 문서와 일치하고 (근거: ...) 출처 표기가 있으면 True")
    reason: str = Field(description="판정 이유 한 문장")

verifier = llm.with_structured_output(Verdict)
MAX_ROLLBACK = 2

def verify(state):
    verdict = verifier.invoke(VERIFY_PROMPT.format(
        context="\n".join(state["context"]), draft=state["draft"]))
    return {"stage": "검증", "verify_passed": verdict.passed,
            "error": None if verdict.passed else verdict.reason}

def rollback(state):
    return {"stage": "롤백", "draft": "", "rollback_count": 1}    # ← 초안을 비워 폐기

def route_after_verify(state):
    if state["verify_passed"]:                    return "finalize"
    if state["rollback_count"] >= MAX_ROLLBACK:   return "handle_error"    # 상한
    return "rollback"

builder.add_edge("rollback", "vector_search")     # ← 검색 단계로 되돌아감
```

### 7. 재시도 시 실패 이유를 프롬프트에 되먹이기

같은 실수를 반복하지 않게 하는 요령입니다.

```python
def make_draft(state):
    retry_note = ""
    if state["rollback_count"] > 0:
        retry_note = (f"※ 이전 초안은 검증에서 불합격했습니다(사유: {state['error']}). "
                      "근거 문장을 그대로 인용해 다시 작성하세요.")
    draft = draft_llm.invoke(DRAFT_PROMPT.format(
        retry_note=retry_note, context="\n".join(state["context"]), question=state["question"])).content
    return {"stage": "초안작성", "draft": draft}
```

### 8. 정직한 에스컬레이션 — 어디서 멈췄는지까지 안내

```python
def handle_error(state):
    return {"stage": "오류처리", "answer": (
        "정확한 답변 근거를 확보하지 못했습니다. "
        "잘못된 안내를 드리지 않도록, 상담원 연결로 도와드리겠습니다. "
        f"(경로: {state['stage']} 단계에서 중단)"          # ← 운영자가 원인을 바로 파악
    )}
```

## 폴백 트리거 설계 기준

| 트리거 | 의미 | 대응 |
|---|---|---|
| API 오류 (5xx, 인증 실패) | 모델 서비스 장애 | Secondary 모델 전환 |
| 타임아웃 | 응답 지연 | 재시도 1회 → 실패 시 Secondary |
| 검색 결과 0건 / 임계값 미달 | 근거 없음 | 키워드 검색 폴백 → 실패 시 상담원 |
| 검증 불합격 | 답변 품질 미달 | 롤백 후 재생성 (상한 2회) |
| 비용 임계 초과 | 예산 소진 | 저가 모델 전환 → [[개념_비용_라우팅_최적화]] |

## 주의사항

- 🚨 **`Day05_01` 노트북에 실제 OpenAI 키가 하드코딩되어 있습니다** → [[오류_환경설정_트러블슈팅]]
- **Primary와 Secondary를 같은 벤더로 두면 벤더 장애 시 함께 죽습니다.** 진짜 가용성을 원하면 다른 벤더나 내부망 모델을 예비로 두세요 → [[개념_LocalLLM_내부망_구축]]
- **폴백은 조용히 일어납니다.** 로그와 알림이 없으면 몇 주째 Secondary로만 서비스하고 있어도 모릅니다. `AgentMonitor`와 연결하세요 → [[개념_PII보호와_거버넌스]]
- **Secondary 모델은 품질이 다릅니다.** 폴백 응답에는 "간소화된 답변입니다" 같은 표시를 고려하세요.
- **롤백 상한이 없으면 검증 루프가 무한 반복됩니다.** `MAX_ROLLBACK` 없이 배포하지 마세요.
- **노트북의 `demo_force_reject` 와 `THRESHOLD = 0.99` 조작은 교육용**입니다. 실제 코드에 남기지 마세요.
- **`THRESHOLD`를 전역 변수로 두고 시나리오마다 바꾸는 방식**은 노트북 편의를 위한 것입니다. 서비스 코드에서는 설정으로 주입하세요.
- **키워드 폴백의 `keyword_score`는 단순 부분 문자열 포함 카운트**입니다. 한국어 조사 때문에 정확도가 낮으니, 실무에서는 BM25로 대체하세요 → [[개념_검색품질_개선]]

## 관련 문서

- [[개념_HITL과_검증루프]] — 검증 노드 설계
- [[개념_Agent_통제력과_루프탈출]] — 상한과 정직한 실패
- [[개념_PII보호와_거버넌스]] — Runbook과 장애 대응 절차
- [[개념_비용_라우팅_최적화]] — 비용 기반 폴백
- [[개념_FastAPI_Agent_서비스화]] — 타임아웃 설계

## 출처

- Source: `Day5/Day05_01_Fallback설계.ipynb`
- Source: `Day5/Day05_02_Fallback_Rollback_설계.ipynb`
- Source: `Day5/Day05_01_교안_Fallback설계.pdf`, `Day5/Day05_02_교안_Fallback_Rollback_설계.pdf`
- Source: `바이브코딩실습/Day05_01_멈추지않는상담봇/app.py`, `Day05_02_보고서상태머신/app.py`
