# 개념: Agent 통제력 — 메모리, 컨텍스트 다이어트, 루프 탈출

## 한눈에 보기

Agent는 스스로 도구를 부르고 다음 행동을 고르기 때문에, 잘못 설계하면 **폭주**합니다 — 검색 결과가 없는데 영원히 재검색하거나, 대화가 쌓여 토큰 비용이 폭발합니다.
이 문서는 Agent를 통제하는 4종 세트 — **메모리(Checkpointer), 컨텍스트 다이어트(RemoveMessage), 반복 카운터, recursion_limit** — 와 "정직한 실패" 원칙을 정리합니다.

## 핵심 개념

| 장치 | 막는 문제 | 구현 |
|---|---|---|
| **Checkpointer + thread_id** | 맥락 상실 ("그럼 중도상환수수료는요?" 를 못 알아들음) | `InMemorySaver`, `config={"configurable": {"thread_id": ...}}` |
| **컨텍스트 다이어트** | 대화가 쌓여 매 호출마다 전체 이력 전송 → 비용·지연 폭발 | `RemoveMessage`로 오래된 메시지 제거 |
| **반복 카운터** | 무한 재검색 루프 | `Annotated[int, operator.add]` + 상한 라우팅 |
| **`recursion_limit`** | 카운터를 깜빡한 그래프 | `config={"recursion_limit": N}` → `GraphRecursionError` |
| **정직한 실패** | 근거 없이 지어낸 답변 | `give_up` 노드 → 상담원 이관 안내 |

**핵심 원칙**: 루프 1바퀴 = LLM/검색 호출 = **돈과 시간**입니다. 통제 장치는 안전 문제이자 비용 문제입니다.

**thread_id의 의미**: 같은 `thread_id`면 이전 대화를 이어받고, 다른 `thread_id`면 새 대화입니다. 고객 A와 고객 B의 대화가 섞이지 않는 근거가 이것입니다.

## 전체 동작 흐름

```
① 메모리
   thread_id="cust-001": "주택담보대출 알아봐요" → "중도상환수수료는요?" (주담대 기준으로 답)
   thread_id="cust-002": "중도상환수수료는요?"   (어느 상품인지 되물음)

② 컨텍스트 다이어트
   START → trim(6개 초과분 RemoveMessage) → agent → END

③ 반복 카운터
   START → mock_search ──route_after_search──┬─ 결과 있음 → generate → END
              ↑                               ├─ count >= 3 → give_up → END   ← 탈출
              └───────────────────────────────┘ 결과 없음

④ recursion_limit
   카운터 없는 그래프 → 8회 실행 후 GraphRecursionError 로 강제 종료
```

## 주요 코드 구성

| 파일 | 역할 |
|---|---|
| `Day3/Day03_09_에이전트_통제력_메모리_루프탈출.ipynb` | 4종 세트 전부 |
| `바이브코딩실습/Day03_09_정직한실패검색봇/app.py` | 3회 검색 후 정직한 실패 (**API 키 없이 동작**) |

## 실행 방법

```bash
pip install -qU langgraph langchain langchain-core langchain-openai
$env:OPENAI_API_KEY = "sk-..."

cd 바이브코딩실습/Day03_09_정직한실패검색봇 && streamlit run app.py
```

## 핵심 코드

### 1. 메모리 — Checkpointer + thread_id ⭐

```python
from langgraph.checkpoint.memory import InMemorySaver

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

SYSTEM_PROMPT = (
    "당신은 KB국민은행 대출 상담원입니다. "
    "이전 대화 맥락을 기억하고, 고객이 앞서 말한 상품을 기준으로 이어서 답하세요. "
    "어떤 상품 이야기인지 대화에 없다면, 먼저 상품을 되물어보세요. 답변은 3문장 이내로."
)

def agent(state: ChatState):
    return {"messages": [llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])]}

graph = builder.compile(checkpointer=InMemorySaver())

# 같은 스레드 → 맥락 유지
config_a = {"configurable": {"thread_id": "cust-001"}}
graph.invoke({"messages": [HumanMessage(content="주택담보대출을 알아보고 있어요.")]}, config=config_a)
graph.invoke({"messages": [HumanMessage(content="그럼 중도상환수수료는 어떻게 되나요?")]}, config=config_a)
# → 주택담보대출 기준으로 답변

# 다른 스레드 → 맥락 없음
config_b = {"configurable": {"thread_id": "cust-002"}}
graph.invoke({"messages": [HumanMessage(content="그럼 중도상환수수료는 어떻게 되나요?")]}, config=config_b)
# → "어떤 상품이신가요?" 되물음
```

> 💡 시스템 프롬프트의 **"어떤 상품 이야기인지 대화에 없다면 먼저 되물어보세요"** 가 중요합니다. 이게 없으면 맥락이 없을 때 아무 상품이나 가정해 답합니다.

### 2. 컨텍스트 다이어트 — `RemoveMessage` ⭐

메모리는 공짜가 아닙니다. 대화 100턴이면 **매 호출마다 100개 메시지를 전부 전송**합니다.

```python
from langchain_core.messages import RemoveMessage

MAX_MESSAGES = 6

def trim_history(state: ChatState):
    msgs = state["messages"]
    if len(msgs) <= MAX_MESSAGES:
        return {}                                    # 변경 없음
    to_remove = msgs[: len(msgs) - MAX_MESSAGES]     # 오래된 것부터
    return {"messages": [RemoveMessage(id=m.id) for m in to_remove]}

builder2.add_node("trim", trim_history)
builder2.add_node("agent", agent)
builder2.add_edge(START, "trim")                     # ← agent 앞에 배치
builder2.add_edge("trim", "agent")
builder2.add_edge("agent", END)
```

### 3. 반복 카운터 + 정직한 실패 ⭐⭐ (가장 중요한 패턴)

```python
import operator

class SearchState(TypedDict):
    question: str
    context: list
    answer: str
    retrieval_count: Annotated[int, operator.add]

MAX_RETRIEVAL = 3

def mock_search(state: SearchState):
    return {"retrieval_count": 1, "context": []}     # 결과 없음 (실패 시뮬레이션)

def give_up(state: SearchState):
    return {"answer": (
        f"검색을 {state['retrieval_count']}회 시도했지만 근거 문서를 찾지 못했습니다. "
        "정확한 안내를 위해 상담원 연결을 도와드릴까요?"          # ← 지어내지 않고 넘김
    )}

def route_after_search(state: SearchState):
    if state["context"]:                             # ① 성공
        return "generate"
    if state["retrieval_count"] >= MAX_RETRIEVAL:    # ② 상한 도달 → 정직한 실패
        return "give_up"
    return "mock_search"                             # ③ 재시도

safe_builder.add_conditional_edges("mock_search", route_after_search,
    {"mock_search": "mock_search", "generate": "generate", "give_up": "give_up"})
```

### 4. `recursion_limit` — 최후의 안전벨트

카운터를 깜빡한 그래프도 이걸로 막힙니다. **모든 운영 그래프에 기본으로 넣으세요.**

```python
from langgraph.errors import GraphRecursionError

try:
    unsafe_graph.invoke(
        {"question": "존재하지 않는 상품", "context": [], "answer": "", "retrieval_count": 0},
        config={"recursion_limit": 8},               # 노드 실행 8회 초과 시 예외
    )
except GraphRecursionError:
    print("🚨 recursion_limit이 무한 루프를 강제 종료했습니다.")
    # 운영에서는 여기서 '시스템 오류 안내 + 상담원 연결'로 이어집니다
```

## Agent 통제 장치 4종 세트 — 적용 체크리스트

- [ ] **Checkpointer + thread_id** — 대화형이면 필수. 세션 경계가 명확한가?
- [ ] **컨텍스트 상한** — 대화가 길어질 수 있으면 `trim` 노드 또는 요약 노드
- [ ] **반복 카운터** — 되돌아가는 엣지가 하나라도 있으면 필수
- [ ] **`recursion_limit`** — `invoke`/`stream` 호출마다 config에 포함
- [ ] **give_up 경로** — 실패 시 사람에게 넘어가는 길이 있는가?

## 주의사항

- **`recursion_limit`은 노드 실행 총 횟수**이지 루프 횟수가 아닙니다. 노드가 많은 그래프는 정상 실행에도 수십 번이 필요하므로, 너무 낮게 잡으면 정상 케이스가 죽습니다. 기본값(25)을 확인하고 여유 있게 잡으세요. Day4 Multi-Agent 예제는 `150`을 씁니다.
- **`GraphRecursionError`는 그냥 던져지면 사용자에게 스택 트레이스가 노출됩니다.** 반드시 잡아서 안내 문구로 바꾸세요.
- **컨텍스트를 자르면 앞의 중요한 정보가 사라집니다.** 단순 절삭 대신 (a) 시스템 프롬프트는 항상 유지, (b) 잘린 부분을 요약해 보관하는 전략이 실무에서 쓰입니다.
- **`RemoveMessage`는 `id`가 필요합니다.** 직접 만든 메시지에 id가 없으면 삭제되지 않습니다.
- **`InMemorySaver`는 재시작하면 대화가 전부 사라집니다.** 운영에는 영속 체크포인터를 쓰세요.
- **"정직한 실패"는 UX 설계이기도 합니다.** "모르겠습니다"로 끝내지 말고 **다음 행동**(상담원 연결, 담당 부서 안내)을 함께 제시해야 실제로 쓸 만합니다.
- 실전프로젝트 가이드는 이 원칙을 **모든 프로젝트 필수 안전장치 3종** 중 2개(정직한 실패, 멈추는 장치)로 못 박고 있습니다.

## 관련 문서

- [[개념_HITL과_검증루프]] — 사람 개입과 재시도 상한
- [[개념_조건부Edge와_ToolNode]] — 루프가 생기는 지점
- [[개념_Fallback과_Rollback]] — 실패 시 대체 경로
- [[개념_비용_라우팅_최적화]] — 루프가 비용에 미치는 영향
- [[개념_검색품질_개선]] — "근거 없음" 판정의 검색 측 구현

## 출처

- Source: `Day3/Day03_09_에이전트_통제력_메모리_루프탈출.ipynb`
- Source: `Day3/Day03_09_교안_에이전트_통제력_메모리_루프탈출.pdf`
- Source: `바이브코딩실습/Day03_09_정직한실패검색봇/app.py`
- Source: `실전프로젝트/README.md` (공통 안전장치 3종)
