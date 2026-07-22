# 개념: LangGraph 그래프 구조 패턴 (Router / Map-Reduce / Evaluator-Optimizer)

## 한눈에 보기

기본 그래프를 넘어서면 반복적으로 쓰이는 **3가지 구조 패턴**이 있습니다.
질문을 유형별로 다른 노드에 보내는 **Router**, 일을 나눠 병렬 처리하고 합치는 **Map-Reduce(Send)**, 생성과 평가를 반복하는 **Evaluator-Optimizer**.
이 셋을 알면 이후 의도분류 RAG, Multi-Agent, 검증 루프가 전부 이 조합으로 보입니다.

## 핵심 개념

| 패턴 | 구조 | 대표 용도 |
|---|---|---|
| **Router** | 1 → N 분기, 각자 END | 의도 분류, 민원 자동 배분 |
| **Map-Reduce** | 1 → N 병렬 → 1 통합 | 보고서 챕터별 작성, 대량 문서 요약 |
| **Evaluator-Optimizer** | 생성 ⇄ 평가 루프 | 코드 최적화, 답변 품질 개선, 자가 검증 |

**Router**: `add_conditional_edges` 로 판단 함수의 결과에 따라 서로 다른 노드로 보냅니다. 각 분기는 자기 일만 하고 END로 갑니다.

**Map-Reduce의 핵심은 `Send`**: 리스트 원소 개수만큼 같은 노드를 **동적으로 복제 실행**합니다. 노드 개수를 미리 알 수 없을 때 쓰는 유일한 방법이며, 결과를 모으려면 `Annotated[list, operator.add]` 리듀서가 반드시 필요합니다.

**SubState**: Map 단계의 워커는 전체 State가 아니라 자기 몫만 받습니다. 별도 TypedDict를 만들어 필요한 필드만 노출하면 프롬프트가 가벼워지고 실수가 줄어듭니다.

## 전체 동작 흐름

### Router
```
START → route(분류) ──route_decision──┬→ recommend_recipe → END
                                       ├→ recommend_movie  → END
                                       ├→ counsel          → END
                                       └→ talk             → END
```

### Map-Reduce
```
START → orchestrator (목차 N개 생성)
           ↓ assign_workers → Send("llm_call", {chapter}) × N   ← 병렬
        llm_call × N  → completed_sections에 누적(operator.add)
           ↓
        synthesizer (하나로 합침) → END
```

### Evaluator-Optimizer
```
START → code_generator ──→ code_evaluator ──route_code──┬→ Accepted → END
             ↑                                           │
             └───────── Rejected + Feedback ─────────────┘
```

## 주요 코드 구성

| 파일 | 역할 |
|---|---|
| `Day3/Day03_04_다양한_Graph_구조.ipynb` | 세 패턴 전부 + 각각의 응용 예제 |
| `Day3/Day03_05_통합Agent구현_고급RAG.ipynb` | 패턴 조합 실습 |
| `바이브코딩실습/Day03_04_민원자동배분기/app.py` | Router 데모 (대출/카드/예금팀 분기) |
| `바이브코딩실습/Day04_01_협업보고서팀/app.py` | Map-Reduce 응용 |
| `바이브코딩실습/Day03_07_자가평가답변봇/app.py` | Evaluator-Optimizer 데모 (최대 2회 재시도) |

## 실행 방법

```bash
pip install -U langgraph langchain langchain-core langchain-google-genai langchain-community
$env:OPENAI_API_KEY = "sk-..."

cd 바이브코딩실습/Day03_04_민원자동배분기 && streamlit run app.py
```

## 핵심 코드

### 1. Router — 의도 분류 후 분기 ⭐

분류 노드와 라우팅 함수를 **분리**하는 것이 요령입니다. 분류 결과는 State에 남고, 라우팅 함수는 그 값을 읽기만 합니다.

```python
from typing_extensions import Literal

def route(state: State):
    prompt = ChatPromptTemplate([
        ('system', '''당신의 역할은 사용자의 질문에 대답할 사람을 선택하는 것입니다.
1) 음식 관련 질문: 'FOOD'만 출력하세요.
2) 영화 관련 질문: 'MOVIE'만 출력하세요.
3) 고민 상담: 'COUNSEL'만 출력하세요.
4) 그 외의 대화: 'TALK'만 출력하세요.'''),
        ('user', 'User Query: {query}')
    ])
    return {"classification": (prompt | llm).invoke(state).content}

# 라우팅 함수: State를 읽어 노드 이름만 반환 (반환 타입을 Literal로 명시하면 그래프 그림이 정확해집니다)
def route_decision(state: State) -> Literal['recommend_recipe', 'recommend_movie', 'talk', 'counsel']:
    if "FOOD" in state["classification"]:
        return "recommend_recipe"
    elif "MOVIE" in state["classification"]:
        return "recommend_movie"
    elif "TALK" in state["classification"]:
        return "talk"
    else:
        return "counsel"                     # 기본 경로(fallback)를 반드시 두세요

builder.add_edge(START, 'route')
builder.add_conditional_edges('route', route_decision, {
    'recommend_movie': 'recommend_movie',
    'recommend_recipe': 'recommend_recipe',
    'counsel': 'counsel',
    'talk': 'talk',
})
for n in ['recommend_movie', 'recommend_recipe', 'counsel', 'talk']:
    builder.add_edge(n, END)
```

> 💡 분기마다 **다른 구조화 출력 스키마**를 쓸 수 있습니다. 음식이면 `Recipe`, 영화면 `Movie` — State에 두 필드를 다 두고 해당 분기만 채웁니다.
> 더 견고한 방법은 `with_structured_output`으로 분류 자체를 강제하는 것입니다 → [[개념_의도분류_기반_RAG_Agent]]

### 2. Map-Reduce — `Send` 로 동적 병렬 실행 ⭐⭐

```python
import operator
from typing import Annotated, List
from langgraph.types import Send

class Chapter(BaseModel):
    name: str = Field(description="챕터의 이름")
    outline: str = Field(description="챕터의 주요 내용, 1문장 길이로")

class Contents(BaseModel):
    contents: List[Chapter] = Field(description="전체 리포트의 섹션 구성")

class State(TypedDict):
    topic: str
    contents: List[Chapter]
    completed_sections: Annotated[list, operator.add]   # ← 리듀서 없으면 결과가 덮어써집니다
    final_report: str

class SubState(TypedDict):            # 워커가 받을 최소 State
    chapter: Chapter
    completed_sections: Annotated[list, operator.add]

def orchestrator(state: State):       # ① 계획 수립
    return {"contents": (prompt | llm.with_structured_output(Contents)).invoke(state).contents}

def assign_workers(state: State):     # ② 리스트 길이만큼 노드를 복제 호출
    return [Send("llm_call", {"chapter": s}) for s in state["contents"]]

def llm_call(state: SubState):        # ③ 워커 — 자기 챕터만 작성
    chapter = state['chapter']
    return {"completed_sections": [(prompt | llm).invoke({
        'name': chapter.name, 'outline': chapter.outline}).content]}   # 리스트로 반환

def synthesizer(state: State):        # ④ 통합
    return {"final_report": "\n\n---\n\n".join(state["completed_sections"])}

builder.add_edge(START, "orchestrator")
builder.add_conditional_edges("orchestrator", assign_workers, ["llm_call"])  # ← Send 반환
builder.add_edge("llm_call", "synthesizer")
builder.add_edge("synthesizer", END)
```

`Send("노드이름", 그노드에줄State)` — 조건부 엣지가 문자열 대신 **`Send` 리스트**를 반환하면 병렬 팬아웃이 됩니다.

### 3. Evaluator-Optimizer — 생성·평가 루프 ⭐

평가 결과를 구조화 출력으로 받아 분기하고, 피드백을 다음 생성에 되먹입니다.

```python
class Feedback(BaseModel):
    grade: Literal["optimized", "not optimized"] = Field(description="최적화 여부 판단 결과")
    feedback: str = Field(description="코드 개선이 필요할 경우, 이유 및 방법 설명")

evaluator = llm.with_structured_output(Feedback)

def code_generator(state: State):
    if state.get("feedback"):            # 재시도라면 피드백을 프롬프트에 포함
        result = llm.invoke(f"""다음 문제를 해결하는 파이썬 코드를 작성하세요.
Instruction: {state['instruction']}
다음의 피드백을 고려하세요. Feedback: {state['feedback']}""")
    else:
        result = llm.invoke(f"다음 문제를 해결하는 파이썬 코드를 작성하세요.\nInstruction: {state['instruction']}")
    return {"code": result.content}

def code_evaluator(state: State):
    result = evaluator.invoke(f"...평가 기준...\nSource Code: {state['code']}")
    return {"optimized": result.grade, "feedback": result.feedback}

def route_code(state: State) -> Literal['Accepted', 'Rejected + Feedback']:
    return 'Accepted' if state["optimized"] == "optimized" else 'Rejected + Feedback'

builder.add_edge("code_generator", "code_evaluator")
builder.add_conditional_edges("code_evaluator", route_code, {
    "Accepted": END,
    "Rejected + Feedback": "code_generator",     # ← 되돌아가는 루프
})
```

## 주의사항

- 🚨 **Evaluator-Optimizer 루프에는 원본 코드에 시도 횟수 제한이 없습니다.** 평가자가 계속 불합격을 주면 무한 루프이며 비용이 계속 발생합니다. **반드시 재시도 카운터나 `recursion_limit`을 추가하세요** → [[개념_Agent_통제력과_루프탈출]]
  ```python
  class State(TypedDict):
      retry: Annotated[int, operator.add]
  def route_code(state):
      if state["optimized"] == "optimized": return 'Accepted'
      if state.get("retry", 0) >= 2:       return 'GiveUp'    # 탈출구
      return 'Rejected + Feedback'
  ```
- 🚨 **`Day03_04_다양한_Graph_구조.ipynb` 에 Google API 키가 하드코딩되어 있습니다** → [[오류_환경설정_트러블슈팅]]
- **Router의 분류를 문자열 매칭(`"FOOD" in ...`)으로 하면 취약합니다.** LLM이 "FOOD 관련 질문입니다"처럼 답하면 의도치 않게 매칭될 수 있습니다. `with_structured_output` + `Literal` 로 값 자체를 강제하는 편이 안전합니다.
- **Map-Reduce는 리듀서가 없으면 조용히 망가집니다.** `completed_sections`에 리듀서를 안 붙이면 병렬 노드들이 서로 덮어써 마지막 하나만 남습니다. 에러가 안 나서 발견이 늦습니다.
- **`Send`로 병렬 호출하면 API rate limit에 걸리기 쉽습니다.** 챕터가 20개면 동시에 20번 호출됩니다. `InMemoryRateLimiter` 나 배치 크기 제한을 검토하세요.
- **원본 노트북의 `time.sleep(6)`** 은 rate limit 회피용입니다. 프로덕션 코드에 그대로 옮기지 마세요.

## 관련 문서

- [[개념_LangGraph_State_Node_Edge]] — 리듀서 개념
- [[개념_조건부Edge와_ToolNode]] — 조건부 엣지 기초
- [[개념_의도분류_기반_RAG_Agent]] — Router의 실전 확장
- [[개념_HITL과_검증루프]] — Evaluator-Optimizer의 금융 버전
- [[개념_MultiAgent_Supervisor]] — Router를 에이전트 단위로 확장

## 출처

- Source: `Day3/Day03_04_다양한_Graph_구조.ipynb`
- Source: `Day3/Day03_05_통합Agent구현_고급RAG.ipynb`
- Source: `바이브코딩실습/Day03_04_민원자동배분기/app.py`, `Day03_07_자가평가답변봇/app.py`
