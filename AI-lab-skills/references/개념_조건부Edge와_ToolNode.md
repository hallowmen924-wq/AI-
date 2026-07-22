# 개념: 조건부 Edge와 ToolNode

## 한눈에 보기

**조건부 엣지**는 State를 보고 다음 노드를 고르는 갈림길이고, **ToolNode**는 LLM이 요청한 도구를 실제로 실행해 주는 LangGraph 기본 제공 노드입니다.
이 둘을 `tools_condition` 으로 묶은 **`chatbot → tools_condition → ToolNode → chatbot`** 3줄 패턴이 LangGraph Agent의 표준 골격이며, [[개념_Tool_Calling]]에서 손으로 짰던 실행 루프를 대체합니다.

## 핵심 개념

| 요소 | 하는 일 |
|---|---|
| **`add_conditional_edges(출발노드, 판단함수, 매핑)`** | 판단 함수의 반환값을 보고 다음 노드 결정 |
| **판단 함수** | State를 받아 문자열(또는 bool)을 반환. 노드가 아니라 **라우터** |
| **`ToolNode([도구들])`** | 마지막 메시지의 `tool_calls`를 실행하고 `ToolMessage`를 State에 추가 |
| **`tools_condition`** | 미리 만들어진 판단 함수. tool_calls가 있으면 `"tools"`, 없으면 `END` |

**핵심 통찰**: `tools_condition`을 쓰면 매핑 dict를 생략할 수 있습니다. `builder.add_conditional_edges('chatbot', tools_condition)` 한 줄이면 끝입니다.

**왜 `tools` → `chatbot` 으로 되돌아가나**: 도구 결과를 받은 LLM이 (a) 최종 답변을 하거나 (b) 도구를 한 번 더 부를 수 있기 때문입니다. 이 되돌아가는 엣지가 ReAct 루프를 만듭니다. 그래서 반드시 종료 조건 관리가 필요합니다 → [[개념_Agent_통제력과_루프탈출]]

## 전체 동작 흐름

```
START → chatbot ──tools_condition──┬─ tool_calls 있음 → tools(ToolNode)
                                    │                       ↓
                                    │                    chatbot  (결과 보고 다시 판단)
                                    └─ tool_calls 없음 → END
```

## 주요 코드 구성

| 파일 | 역할 |
|---|---|
| `Day3/Day03_03_조건부_Edge와_Tool_Node.ipynb` | 조건부 엣지(멀티턴 대화) → ToolNode 패턴 → 모델 비교 |
| `Day3/개인실습/Day03_04_조건부Edge_ToolNode_Graph구조_개인실습.ipynb` | 빈칸 실습본 |
| `바이브코딩실습/Day03_03_계좌조회챗봇/app.py` | **이 패턴의 정석 데모** — 잔액·거래내역 도구 |
| `Day3/개인실습/Day03_03_계좌조회챗봇/app.py` | 확장판 |

## 실행 방법

```bash
pip install -U langgraph langchain langchain-openai langchain-google-genai langchain-tavily
$env:OPENAI_API_KEY = "sk-..."

cd 바이브코딩실습/Day03_03_계좌조회챗봇 && streamlit run app.py
```

## 핵심 코드

### 1. 조건부 엣지의 기본형 — 종료 조건 만들기

판단 함수가 문자열을 반환하고, 매핑 dict가 그 문자열을 노드 이름으로 바꿉니다.

```python
class State(TypedDict):
    context: Annotated[list, add_messages]
    count: int

def check_end(state):
    return "Done" if state['count'] == 3 else "Resume"     # 라우터 — 노드가 아님

builder.add_edge(START, 'ask_human')
builder.add_edge('ask_human', 'talk')
builder.add_conditional_edges('talk', check_end,
                              {'Done': END, 'Resume': 'ask_human'})   # 되돌아가는 루프
```

bool을 반환해도 됩니다: `{True: END, False: 'simulate'}`

### 2. 도구 정의 — `@tool` 데코레이터

docstring이 곧 도구 설명입니다. LLM이 이걸 읽고 도구를 고릅니다.

```python
from langchain_core.tools import tool

@tool
def multiply(x: int, y: int) -> int:
    "x와 y를 입력받아, x와 y를 곱한 결과를 반환합니다."
    return x * y

@tool
def current_date() -> str:
    "현재 날짜를 %Y-%m-%d 형식으로 반환합니다."
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")
```

### 3. ToolNode 표준 패턴 ⭐ (가장 재사용 가치 높은 코드)

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm_with_tools = llm.bind_tools([multiply, current_date, tavily_search])

def tool_calling_llm(state):
    return {"messages": llm_with_tools.invoke(state["messages"])}

builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node('tools', ToolNode([multiply, current_date, tavily_search]))
builder.add_edge(START, 'tool_calling_llm')
builder.add_conditional_edges('tool_calling_llm', tools_condition)   # ← 매핑 생략 가능
builder.add_edge('tools', 'tool_calling_llm')                        # ← 결과 들고 복귀

graph = builder.compile()
graph.invoke({'messages': [HumanMessage(content='2곱하기4는?')]})
```

> ⚠️ `bind_tools()`에 넘긴 도구 목록과 `ToolNode()`에 넘긴 목록이 **같아야 합니다.** 불일치하면 LLM이 존재하지 않는 도구를 부르고 실행 단계에서 KeyError가 납니다.

### 4. 실행 경로를 화면에 보여주기 (교안 전반의 원칙)

결과만이 아니라 "도구를 거쳤는지"를 사용자에게 보여줍니다.

```python
result = graph.invoke({"messages": [("user", question)]})

used_tools = [m for m in result["messages"] if m.type == "tool"]
if used_tools:
    st.info("🧭 경로: chatbot → **tools(도구 창구)** → chatbot → END")
    for m in used_tools:
        st.code(f"[도구 응답] {m.content}")
else:
    st.info("🧭 경로: chatbot → END (도구 없이 바로 답변)")
```

### 5. 실행 과정 추적

```python
for data in graph.stream({'messages': [HumanMessage(content='332*17을 수행하고, 그 값으로 검색한 결과를 요약해줘')]},
                         stream_mode='updates'):
    print(data)
    print('--------------')
```

## 주의사항

- **모델에 따라 도구 실행 능력 차이가 큽니다.** 원본 노트북이 Gemini와 GPT-5-mini로 같은 그래프를 돌려 비교하는 이유가 이것입니다. sLLM 계열은 도구가 복잡하거나 컨텍스트가 길어지면 실패율이 오릅니다. 대응: 프롬프트에 작동 과정 설명과 예시를 넣습니다.
- **`tools → chatbot` 루프에는 종료 보장이 없습니다.** LLM이 계속 도구를 부르면 무한 루프입니다. `config={"recursion_limit": N}` 을 항상 함께 쓰세요 → [[개념_Agent_통제력과_루프탈출]]
- **`create_react_agent` 로 바꾸지 마세요(교육 목적일 때).** 더 간결하지만, 조건부 엣지와 ToolNode를 직접 조립하는 실습 목적이 사라집니다. Day04_03에서는 반대로 `create_react_agent` 를 씁니다 — 목적이 다르기 때문입니다.
- **`input()` 을 쓰는 노드는 Streamlit·서버 환경에서 동작하지 않습니다.** 원본 노트북은 try/except로 샘플 질문 리스트를 대신 쓰는 방식으로 우회합니다.
- 도구가 되돌릴 수 없는 작업(송금·한도조정)이라면 실행 전에 사람 승인을 넣으세요 → [[개념_HITL과_검증루프]]

## 관련 문서

- [[개념_LangGraph_State_Node_Edge]] — 그래프 기초
- [[개념_Tool_Calling]] — 이 패턴이 자동화하는 원리
- [[개념_Graph_구조_패턴]] — 조건부 엣지의 다른 활용(Router)
- [[개념_Agent_통제력과_루프탈출]] — 루프 안전장치
- [[개념_FastAPI_Agent_서비스화]] — `create_react_agent` 를 쓰는 경우

## 출처

- Source: `Day3/Day03_03_조건부_Edge와_Tool_Node.ipynb`
- Source: `Day3/Day03_03_교안_LangGraph_강의_-_조건부_엣지와_ToolNode.pdf`
- Source: `바이브코딩실습/Day03_03_계좌조회챗봇/app.py`
