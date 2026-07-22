# 개념: Multi-Agent와 Supervisor 패턴

## 한눈에 보기

하나의 큰 LLM에 전부 맡기는 대신, **역할을 나눈 여러 에이전트가 협업**하게 만드는 구조입니다.
연결 방식은 두 가지 — 미리 정한 순서로 넘기는 **고정 경로(순차)** 와, 관리자 노드가 매번 다음 담당자를 고르는 **Supervisor**. Supervisor는 `Command(goto=...)` + 구조화 출력 라우팅으로 구현합니다.

## 핵심 개념

| 방식 | 구조 | 장점 | 한계 |
|---|---|---|---|
| **고정 경로** | Amy(검색) → Brad(차트) → ... | 단순, 예측 가능 | 특정 에이전트만 다시 부를 수 없음 |
| **Supervisor** | 모든 에이전트가 supervisor로 복귀 | 유연한 재배정, 부분 수정 가능 | 라우팅 LLM 호출이 매번 추가 |

| 요소 | 설명 |
|---|---|
| **`Command(goto=..., update=...)`** | 노드가 "다음 행선지"와 "State 변경"을 스스로 반환. 이걸 쓰면 `add_edge`가 거의 필요 없음 |
| **`Router` (Pydantic)** | Supervisor의 결정을 `next_worker` + `reason` 구조화 출력으로 받는 "결재 양식" |
| **`name` 태그** | `HumanMessage(content=..., name="researcher")` — 관리자가 "누가 이미 일했는지" 파악하는 근거 |
| **종료 신호** | 고정 경로는 프롬프트 약속(`MISSION COMPLETED`), Supervisor는 `FINISH` → `END` |
| **`create_agent`** | Tool-Calling 기반 ReAct 에이전트를 한 줄로 생성 (`langchain.agents`) |

**핵심 설계**: 관리자와 직원 모두 `Command`로 다음 행선지를 알려주므로, 그래프에 필요한 간선은 **`START → supervisor` 하나뿐**입니다.

## 전체 동작 흐름

### 고정 경로 (Two Agents)
```
START → researcher ─(MISSION COMPLETED?)─┬─ 예 → END
             ↑                            └─ 아니오 → chart_generator ─┬─ 예 → END
             └──────────────────────────────────────────────────────────┘
```

### Supervisor
```
                  ┌──→ product   ─┐
START → supervisor├──→ rule      ─┤ 모두 Command(goto="supervisor")로 복귀
        (라우팅)   ├──→ complaint ─┤
                  ├──→ verify    ─┘
                  └──→ FINISH → END
```

## 주요 코드 구성

| 파일 | 역할 |
|---|---|
| `Day4/Day04_01_Multi-Agent아키텍처.ipynb` | 고정 경로(검색+차트) → Supervisor 전환 |
| `Day4/Day04_02_Supervisor_금융멀티에이전트.ipynb` | **은행 상담팀 Supervisor** (상품·규정·민원·검증 4명) |
| `Day4/charts/` | Python REPL 도구가 생성한 차트 결과물 |
| `바이브코딩실습/Day04_01_협업보고서팀/app.py` | 조사→작성 순차 협업 |
| `바이브코딩실습/Day04_02_Supervisor금융팀/app.py` | Supervisor 데모 |

## 실행 방법

```bash
pip install -q langchain==1.3.11 langchain-openai==1.3.3 langgraph==1.2.7
pip install -U langchain-tavily langchain-experimental      # Day04_01용
$env:OPENAI_API_KEY = "sk-..."

cd 바이브코딩실습/Day04_02_Supervisor금융팀 && streamlit run app.py
```

## 핵심 코드

### 1. 고정 경로 — 종료 신호를 프롬프트로 약속

```python
def make_system_prompt(suffix: str) -> str:
    return f"""당신은 여러 AI 어시스턴트와 협업하는 팀원입니다.
[작업 지침]
1. 완벽한 답변이 어렵다면 가능한 부분까지 진행하세요
2. 나머지는 다른 도구를 가진 팀원이 이어서 작업할 것입니다
3. 다른 팀원에게 넘겨주어야 하는 경우, GO! 를 마지막에 출력하세요.
4. 당신과 다른 팀원의 모든 작업이 완료된 경우 MISSION COMPLETED를 마지막에 출력하세요.
[추가 지침]
{suffix}"""

research_agent = create_agent(llm, tools=[tavily_tool],
    system_prompt=make_system_prompt("""당신의 이름은 Research 전문가 'Amy' 입니다.
[작업] 정보 검색 및 정리를 수행하세요.
[협업] Chart 생성 전문가 'Brad'와 함께 작업 중입니다."""))
```

```python
def get_next_node(last_message, goto: str):
    if "MISSION COMPLETED" in last_message.content:
        return END
    return goto

def research_node(state: State) -> Command[Literal["chart_generator", END]]:
    result = research_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "chart_generator")
    # 다음 에이전트에게는 AIMessage가 아닌 HumanMessage로 전달 (name 태그로 출처 표시)
    result["messages"][-1] = HumanMessage(result["messages"][-1].content, name="researcher")
    return Command(update={"messages": result["messages"]}, goto=goto)

builder.add_node("researcher", research_node)
builder.add_node("chart_generator", chart_node)
builder.add_edge(START, "researcher")        # ← 간선은 이것 하나. 나머지는 Command가 결정
```

> 💡 **AIMessage → HumanMessage 변환**이 중요합니다. 다음 에이전트 입장에서는 앞 에이전트의 결과가 "받은 요청"이어야 자연스럽게 이어받습니다.

### 2. Supervisor — 구조화 출력 라우팅 ⭐⭐

```python
from pydantic import BaseModel, Field
from typing import Literal

WORKERS = ["product", "rule", "complaint", "verify"]

class Route(BaseModel):
    next_worker: Literal["product", "rule", "complaint", "verify", "FINISH"] = Field(
        description="다음 담당자 또는 종료")
    reason: str = Field(description="그 담당자를 고른 이유를 30자 이내로")

router_llm = llm.with_structured_output(Route)

SUPERVISOR_PROMPT = """당신은 은행 상담팀 관리자입니다.
담당자: product(상품조회), rule(규정검색), complaint(민원응대), verify(검증).
[지침]
1. 질문 성격에 맞는 담당자를 한 명 고르세요.
2. 담당자가 답변을 만들면, 반드시 verify에게 검증을 맡기세요.
3. verify까지 끝났으면 FINISH를 고르세요."""

def supervisor(state: State) -> Command[Literal["product", "rule", "complaint", "verify", "__end__"]]:
    decision = router_llm.invoke([SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"])
    print(f"🧭 관리자 결정: {decision.next_worker}  ({decision.reason})")   # 경로 가시화
    goto = END if decision.next_worker == "FINISH" else decision.next_worker
    return Command(goto=goto, update={"next_worker": decision.next_worker})
```

> 💡 `reason` 필드는 기능상 불필요해 보이지만 **디버깅과 시연에서 가장 유용**합니다. 왜 그 담당자를 골랐는지 화면에 보여줄 수 있습니다.

### 3. 워커 노드 — 모두 supervisor로 복귀

```python
def product_agent(state) -> Command[Literal["supervisor"]]:
    q = state["messages"][0].content                  # 원래 질문은 항상 [0]
    info = search_products(q)
    ans = llm.invoke(f"아래 상품 정보를 근거로 답하세요.\n{info}\n\n질문: {q}").content
    return Command(
        goto="supervisor",
        update={"messages": [HumanMessage(content=ans, name="product")]},   # name 태그
    )

def verify_agent(state) -> Command[Literal["supervisor"]]:
    last, q = state["messages"][-1].content, state["messages"][0].content
    check = llm.invoke(f"질문: {q}\n답변: {last}\n\n이 답변이 적절한지 '검증 결과:'로 시작해 한 줄로 평가하세요.").content
    return Command(goto="supervisor", update={"messages": [HumanMessage(content=check, name="verify")]})
```

### 4. 그래프 조립 — 간선 하나

```python
from langgraph.graph import MessagesState, StateGraph, START

class State(MessagesState):        # add_messages 리듀서가 내장되어 있음
    next_worker: str

builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
for name, fn in [("product", product_agent), ("rule", rule_agent),
                 ("complaint", complaint_agent), ("verify", verify_agent)]:
    builder.add_node(name, fn)
builder.add_edge(START, "supervisor")     # ← 이것 하나뿐
graph = builder.compile()
```

### 5. 실행 — recursion_limit 필수 ⭐

Supervisor가 `FINISH`를 결정하지 못하면 무한 순환합니다.

```python
from langgraph.errors import GraphRecursionError

try:
    events = graph.stream(
        {"messages": [HumanMessage(content="월 30만원씩 넣을 적금 상품 알려줘")]},
        {"recursion_limit": 15}, stream_mode="updates")
    for ev in events:
        for node, val in ev.items():
            if val and "messages" in val:
                print(f"[{node}] {val['messages'][-1].content}\n")
except GraphRecursionError:
    print("⚠️ 반복 한도 도달: Supervisor가 FINISH를 결정하지 못해 안전하게 중단했습니다.")
```

### 6. 코드 실행 도구 (Day04_01) — ⚠️ 위험

```python
from langchain_experimental.utilities import PythonREPL
repl = PythonREPL()

@tool
def python_repl_tool(code: Annotated[str, "The python code to execute to generate your chart."]):
    """차트를 그리기 위한 Python 코드를 실행하는 툴입니다."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    # 생성된 matplotlib 그림을 PNG로 저장
    if plt.get_fignums():
        chart_path = os.path.join(CHART_DIR, f"chart_{len(os.listdir(CHART_DIR)) + 1}.png")
        plt.gcf().savefig(chart_path, dpi=120, bbox_inches="tight")
        plt.close("all")
    return result_str + "\n\nIf you have completed all tasks, respond with MISSION COMPLETED."
```

## 주의사항

- 🚨 **`Day04_01`·`Day04_02` 노트북에 실제 OpenAI 키, Tavily 키, LangSmith 키가 하드코딩되어 있습니다** → [[오류_환경설정_트러블슈팅]]
- 🚨 **`PythonREPL`은 임의 코드를 실행합니다.** LLM이 만든 코드가 파일 삭제·네트워크 접근을 할 수 있습니다. 격리된 환경(컨테이너)에서만, 신뢰할 수 있는 입력에만 쓰세요. 고객 대면 서비스에서는 금지입니다.
- 🚨 **Supervisor는 종료를 보장하지 않습니다.** 프롬프트에 "verify까지 끝났으면 FINISH"라고 써도 LLM이 계속 워커를 부를 수 있습니다. `recursion_limit`은 선택이 아니라 필수입니다 → [[개념_Agent_통제력과_루프탈출]]
- **에이전트가 늘수록 비용이 선형 이상으로 증가합니다.** Supervisor 라우팅 1회 + 워커 실행 1회가 한 턴이므로, 4턴이면 LLM 호출 8회입니다. 라우팅에는 저렴한 모델을 쓰는 것이 정석입니다 → [[개념_비용_라우팅_최적화]]
- **`Literal[*routes]` 언패킹 문법은 Python 3.11+** 에서 동작합니다.
- **`MessagesState` 상속 시 `add_messages` 리듀서가 이미 들어 있습니다.** 직접 `Annotated[list, add_messages]`를 다시 쓸 필요 없습니다.
- **워커가 `state["messages"][0]`을 원래 질문으로 가정합니다.** 시스템 메시지를 [0]에 넣으면 깨집니다.
- **Day04_02의 `search_products`/`search_rules`는 파이썬 dict 검색**입니다. 실서비스에서는 이 자리에 벡터 검색이나 DB 조회가 들어갑니다.
- 노트북 상단의 `!sudo apt-get install -y fonts-nanum` 은 Colab 전용입니다. Windows에서는 `Malgun Gothic`을 씁니다.

## 관련 문서

- [[개념_Graph_구조_패턴]] — Router 패턴의 확장
- [[개념_조건부Edge와_ToolNode]] — 워커 내부의 도구 실행
- [[개념_Agent_통제력과_루프탈출]] — recursion_limit
- [[개념_FastAPI_Agent_서비스화]] — 완성된 Agent를 API로
- [[프로젝트_실전과제_가이드]] — 프로젝트 4번(IT장애대응)이 Supervisor 중심

## 출처

- Source: `Day4/Day04_01_Multi-Agent아키텍처.ipynb`
- Source: `Day4/Day04_02_Supervisor_금융멀티에이전트.ipynb`
- Source: `Day4/Day04_01_교안_Multi-Agent아키텍처.pdf`, `Day4/Day04_02_교안_Supervisor_금융멀티에이전트.pdf`
- Source: `바이브코딩실습/Day04_01_협업보고서팀/app.py`, `Day04_02_Supervisor금융팀/app.py`
