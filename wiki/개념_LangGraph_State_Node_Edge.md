# 개념: LangGraph State / Node / Edge

## 한눈에 보기

LangGraph는 애플리케이션의 작동 과정을 **그래프**로 정의하고 실행하는 도구입니다. LCEL 체인이 "일직선 파이프라인"이라면 LangGraph는 **분기·반복·상태 공유**가 가능한 워크플로우입니다.
구성 요소는 단 3개 — 데이터를 담는 **State**, 일을 하는 **Node**, 순서를 정하는 **Edge**. 이 문서는 그 3요소와 State를 다루는 여러 방식(구조화 출력, 메시지 리듀서)을 정리합니다.

## 핵심 개념

| 요소 | 정체 | 규칙 |
|---|---|---|
| **State** | 그래프 전체가 공유하는 데이터 (`TypedDict` 또는 Pydantic) | 노드를 지나며 값이 추가·수정됨 |
| **Node** | State를 받아 **바뀐 부분만 dict로 반환**하는 함수 | 전체 State를 돌려줄 필요 없음 |
| **Edge** | 노드 간 연결. `START`에서 시작해 `END`로 끝남 | 고정 엣지와 조건부 엣지가 있음 |
| **Reducer** | State 값을 **어떻게 합칠지** 정하는 규칙 | `Annotated[list, add_messages]`, `Annotated[int, operator.add]` |
| **compile()** | 빌더를 실행 가능한 Runnable로 변환 | 이후 `.invoke()` / `.stream()` 사용 |

**리듀서가 핵심입니다.** 기본 동작은 **덮어쓰기**입니다. 리스트에 계속 쌓거나 숫자를 누적하고 싶으면 리듀서를 지정해야 합니다.

```python
messages: Annotated[list, add_messages]        # 메시지를 이어붙임 (덮어쓰지 않음)
retrieval_count: Annotated[int, operator.add]  # 노드가 1을 반환하면 누적됨
completed_sections: Annotated[list, operator.add]  # 병렬 노드 결과를 모음
```

**그래프 조립 3단계 (레고와 동일)**

```python
builder = StateGraph(State)          # ① 판 깔기
builder.add_node("이름", 함수)         # ② 블록 얹기
builder.add_edge("A", "B")           # ③ 블록 잇기
graph = builder.compile()            # → 실행 가능
```

## 전체 동작 흐름

```
graph.invoke({'result': 'start!\n'})
        ↓ START → "one"
   first(state)  → return {"result": state['result'] + "첫번째 노드 통과\n"}
        ↓ 반환된 dict가 State에 병합(리듀서 규칙에 따라)
        ↓ "one" → "two"
   second(state) → return {"result": ... + "두번째 노드 통과\n"}
        ↓ "two" → END
   최종 State 반환
```

## 주요 코드 구성

| 파일 | 역할 |
|---|---|
| `Day3/Day03_01_LangGraph핵심개념.ipynb` | State/Node/Edge 기초 + LLM 노드 + 웹검색 노드 |
| `Day3/Day03_02_LangGraph의_다양한_State.ipynb` | 구조화 출력을 State에 담기, 메시지 State, `RemoveMessage` |
| `Day3/개인실습/Day03_02_..._개인실습.ipynb` | 빈칸 실습본 |
| `바이브코딩실습/Day03_01_대출신청흐름그래프/app.py` | 3단계 심사 흐름 데모 (**API 키 없이 동작**) |

## 실행 방법

```bash
pip install -U langgraph langchain langchain-openai langchain-google-genai langchain-tavily langchain-community
$env:OPENAI_API_KEY = "sk-..."

cd 바이브코딩실습/Day03_01_대출신청흐름그래프 && streamlit run app.py
```

## 핵심 코드

### 1. 최소 그래프

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    result: str
    secret: str

def first(state):
    return {"result": state['result'] + "첫번째 노드 통과\n"}   # 바뀐 키만 반환

def second(state):
    return {"result": state['result'] + "두번째 노드 통과\n"}

builder = StateGraph(State)
builder.add_node("one", first)
builder.add_node("two", second)
builder.add_edge(START, "one")
builder.add_edge("one", "two")
builder.add_edge("two", END)

graph = builder.compile()
graph.invoke({'result': 'start!\n', 'secret': 'two'})
```

### 2. 그래프 시각화

```python
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

노트북에서는 `graph` 를 셀 마지막에 두기만 해도 그림이 나옵니다.
Streamlit 앱에서는 `graph.get_graph().draw_mermaid()` 로 Mermaid 텍스트를 얻어 화면에 그립니다.

### 3. 중간 과정 보기 — `invoke` vs `stream`

디버깅에서 가장 자주 쓰는 도구입니다.

```python
graph.invoke(초기상태)                              # 최종 결과만

for data in graph.stream(초기상태, stream_mode='updates'):   # 노드가 바꾼 부분만
    print(data)

for data in graph.stream(초기상태, stream_mode='values'):    # 매 단계의 State 전체
    print(data)
```

| stream_mode | 돌려주는 것 | 언제 |
|---|---|---|
| `updates` | 각 노드가 반환한 dict | "어느 노드가 무엇을 바꿨나" 추적 |
| `values` | 그 시점의 State 전체 | 상태 변화를 통째로 보고 싶을 때 |

### 4. 구조화 출력을 State 값으로 담기 ⭐

Pydantic 객체를 State 필드에 그대로 넣을 수 있습니다. 노드 간에 **형식이 보장된 데이터**를 넘기는 방법입니다.

```python
from pydantic import BaseModel, Field

class Objective(BaseModel):
    instruction: str = Field(description='프롬프트의 지시 사항을 명확히 재구성')
    output_format: str = Field(description='출력 포맷에 대한 설명')
    examples: str = Field(description='예시 출력(1개)')
    notes: str = Field(description='중요한 내용을 4개의 개조식 문장으로')

    @property
    def as_str(self) -> str:                      # 프롬프트에 넣기 좋게 문자열화
        return '\n\n'.join([f'## {key}\n {value}' for key, value in self])

class State(TypedDict):
    instruction: str
    prompt_materials: Objective        # ← Pydantic 객체가 State 값
    full_prompt: str
    result: str

def get_prompt_materials(state):
    chain = prompt | llm.with_structured_output(Objective)
    return {'prompt_materials': chain.invoke({'instruction': state['instruction']})}
```

### 5. 메시지 State와 리듀서 ⭐

대화형 Agent는 대부분 이 형태를 씁니다. `add_messages` 리듀서가 **덮어쓰기 대신 이어붙이기**를 해 줍니다.

```python
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class State(TypedDict):
    context: Annotated[list, add_messages]

def talk(state):
    return {'context': AIMessage(content='AI 메시지 2')}   # 리스트가 아니어도 append 됨
```

### 6. 메시지 삭제 — `RemoveMessage`

컨텍스트가 무한히 커지는 것을 막는 기법입니다. 삭제할 메시지의 `id`를 담은 `RemoveMessage`를 반환합니다.

```python
from langchain_core.messages import RemoveMessage

def delete_message(state):
    messages = state['context']
    return {"context": [RemoveMessage(id=messages[i].id) for i in range(1, 3)]}
```

전체 비우기는 `langgraph.graph.message`의 `REMOVE_ALL_MESSAGES`를 씁니다.
운영에서 쓰는 정리 전략 → [[개념_Agent_통제력과_루프탈출]]

### 7. 실전 예: 웹 검색 3단 그래프

```python
class State(TypedDict):
    question: str
    query: str
    answer: str
    context: list

# get_query: 질문 → 검색어 생성
# tavily_search: 검색 → URL 크롤링 → context에 문서 저장
# answer_question: context 기반 답변 (없으면 "정보가 부족하여 답변할 수 없습니다")

builder.add_edge(START, "get_query")
builder.add_edge("get_query", "tavily_search")
builder.add_edge("tavily_search", "answer_question")
builder.add_edge("answer_question", END)
```

답변 프롬프트에 **"정보가 없다면 '정보가 부족하여 답변할 수 없습니다'를 출력하세요"** 를 넣은 점에 주목하세요. 정직한 실패의 가장 단순한 형태입니다.

## 주의사항

- 🚨 **`Day03_01_LangGraph핵심개념.ipynb` 와 `Day03_04` 에 실제 Google API 키와 Tavily API 키가 하드코딩되어 있습니다.** 재사용 전 반드시 제거하세요 → [[오류_환경설정_트러블슈팅]]
- **노드는 바뀐 키만 반환하면 됩니다.** 전체 State를 반환해도 동작하지만, 의도치 않은 덮어쓰기가 생길 수 있습니다.
- **리듀서를 빠뜨리면 값이 사라집니다.** 리스트를 누적할 생각으로 `list`만 선언하면 마지막 노드의 값만 남습니다.
- **State 필드를 노드 안에서 직접 수정(`state['count'] += 1`)하는 코드가 원본에 있습니다.** 동작은 하지만, 반환 dict로 표현하는 것이 LangGraph의 의도이며 병렬 실행 시 안전합니다.
- **Gemini 사용 시 `InMemoryRateLimiter(requests_per_second=0.167)`** 로 속도를 제한합니다. 무료 티어 한도 회피용이며 실행이 매우 느립니다. OpenAI를 쓰면 제거해도 됩니다.
- **`draw_mermaid_png()` 는 네트워크 호출**을 합니다(Mermaid 렌더링 서비스). 오프라인·내부망에서는 실패하므로 `draw_ascii()` 나 `draw_mermaid()` 텍스트를 쓰세요.

## 관련 문서

- [[개념_LangChain_LCEL_기초]] — 노드 안에서 쓰는 체인 문법
- [[개념_조건부Edge와_ToolNode]] — 분기와 도구 실행
- [[개념_Graph_구조_패턴]] — Router / Map-Reduce / 평가 루프
- [[개념_Agent_통제력과_루프탈출]] — 메모리와 컨텍스트 관리
- [[MOC_LangGraph]]

## 출처

- Source: `Day3/Day03_01_LangGraph핵심개념.ipynb`
- Source: `Day3/Day03_02_LangGraph의_다양한_State.ipynb`
- Source: `Day3/Day03_01_교안_LangGraph_핵심_개념_강의.pdf`
- Source: `바이브코딩실습/Day03_01_대출신청흐름그래프/app.py`, `Day03_02_고객응대문생성기/app.py`
