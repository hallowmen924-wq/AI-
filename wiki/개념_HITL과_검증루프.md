# 개념: Human-in-the-Loop와 검증 루프

## 한눈에 보기

**HITL(Human-in-the-Loop)** 은 Agent가 중요한 행동을 하기 직전에 실행을 멈추고 사람의 확인을 받는 구조입니다. LangGraph의 `interrupt()` + `Command(resume=...)` + Checkpointer 조합으로 구현합니다.
**검증 루프**는 답변이 사용자에게 나가기 전 통과해야 하는 **3중 검문소**(입력 유효성 → 근거 검증 → 출력 필터)입니다. 금융권 Agent에서 둘 다 필수입니다.

## 핵심 개념

### HITL

| 요소 | 하는 일 |
|---|---|
| **`interrupt(payload)`** | 그래프를 그 지점에서 일시정지하고 `payload`를 밖으로 내보냄 |
| **Checkpointer** | 멈춘 위치와 State를 저장. **없으면 interrupt가 동작하지 않음** (`InMemorySaver`) |
| **`thread_id`** | 어느 대화를 재개할지 식별. `config={"configurable": {"thread_id": "1"}}` |
| **`Command(resume=값)`** | 사람의 결정을 그래프에 주입해 실행 재개 |
| **`Command(goto=..., update=...)`** | 다음 노드와 State 변경을 한 번에 지정 |

**3가지 사람 결정 유형**

| 액션 | 의미 | 동작 |
|---|---|---|
| `continue` | 그대로 진행 | `Command(goto="run_tool")` |
| `update` | 도구는 유지하되 **인자만 수정** | tool_calls의 args를 교체 후 실행 |
| `feedback` | 도구 사용 중단, **새 지시 전달** | 자연어 피드백을 HumanMessage로 넣고 agent로 복귀 |

### 검증 루프 (3중 검문소)

```
① 입력 유효성  — 빈 질문 / 500자 초과 / 입력 속 PII → 그래프 진입 전 차단
② 근거 검증    — LLM 심사로 "Context에 없는 내용을 지어냈나" 판정 → 재생성
③ 출력 필터    — 금지 표현("수익 보장") / 답변 속 PII → 재생성
   재시도 2회 초과 → give_up (상담원 이관)
```

## 전체 동작 흐름

### HITL
```
START → get_user_input → agent ──route_after_llm──┬─ tool_calls 있음 → human_review
                          ↑                        ├─ 종료 문구 → END
                          │                        └─ 그 외 → get_user_input
                          │                              ↓
                          │                        interrupt() ⏸ 실행 멈춤
                          │                              ↓ Command(resume={"action": ...})
                          └──── run_tool ←── continue/update
                          └──────────────←── feedback
```

### 검증 루프
```
START → validate_input ──┬─ blocked → END
                          └─ ok → generate_answer → validate_grounding → validate_output
                                        ↑                                      │
                                        └──── regenerate ──────────────────────┤
                                              give_up(retry>=2) → END          │
                                              pass → END ──────────────────────┘
```

## 주요 코드 구성

| 파일 | 역할 |
|---|---|
| `Day3/Day03_08_조건분기_검증루프_Human_in_the_loop.ipynb` | 1부 HITL(네이버 블로그 검색) + 2부 검증 루프 |
| `Day3/개인실습/Day03_07_..._개인실습.ipynb` | 빈칸 실습본 |
| `바이브코딩실습/Day03_08_송금승인봇/app.py` | **100만원 초과 시 사람 승인** (API 키 없이 동작) |
| `바이브코딩실습/Day03_07_자가평가답변봇/app.py` | 생성→평가→재시도 루프 |

## 실행 방법

```bash
pip install -U langgraph "langchain[openai]" langchain-openai tavily-python
$env:OPENAI_API_KEY = "sk-..."

cd 바이브코딩실습/Day03_08_송금승인봇 && streamlit run app.py
```

## 핵심 코드

### 1. `human_review` 노드 — HITL의 심장 ⭐

`Command[Literal[...]]` 반환 타입 힌트가 그래프 그림에 경로를 그려 줍니다.

```python
from langgraph.types import Command, interrupt

def human_review(state) -> Command[Literal["agent", "run_tool"]]:
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[-1]

    # ⏸ 여기서 실행이 멈추고, dict가 __interrupt__ 로 밖에 나갑니다
    human_review = interrupt({"question": "이대로 진행할까요?", "tool_call": tool_call})

    review_action = human_review["action"]
    review_data = human_review.get("data")

    if review_action == "continue":
        return Command(goto="run_tool")

    elif review_action == "update":                 # 인자만 교체
        updated_message = {
            "role": "ai",
            "content": last_message.content,
            "tool_calls": [{
                "id": tool_call["id"],
                "name": tool_call["name"],
                "args": review_data,                # ← 사람이 고친 인자
            }],
            "id": last_message.id,                  # 같은 id로 덮어쓰기
        }
        return Command(goto="run_tool", update={"messages": [updated_message]})

    elif review_action == "feedback":               # 도구 취소, 새 지시
        new_human_message = HumanMessage(content=review_data, id=last_message.id)
        return Command(goto="agent", update={"messages": [new_human_message]})
```

### 2. Checkpointer 없으면 동작하지 않습니다

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)     # ← 필수
```

### 3. 중단 확인과 재개

```python
thread = {"configurable": {"thread_id": "1"}}

# 1차 실행 — interrupt에서 멈춤
for event in graph.stream({"messages": []}, thread, stream_mode="updates"):
    print(event)                    # __interrupt__ 키가 보이면 멈춘 상태

print(graph.get_state(thread).next)  # 다음에 실행될 노드 확인

# 사람의 결정으로 재개 — 세 가지 방식
graph.stream(Command(resume={"action": "continue"}), thread, stream_mode="updates")
graph.stream(Command(resume={"action": "update", "data": {"query": "겨울옷코디"}}), thread, ...)
graph.stream(Command(resume={"action": "feedback", "data": "겨울옷 코디로 바꿔주세요"}), thread, ...)
```

### 4. 검증 루프 — ① 입력 유효성 검문소

```python
import re

PII_PATTERNS = {
    "주민등록번호": r"\d{6}-\d{7}",
    "계좌번호":     r"\d{3}-\d{2,6}-\d{2,6}",
    "전화번호":     r"01[016-9]-?\d{3,4}-?\d{4}",
}
FORBIDDEN = ["수익 보장", "원금 보장", "무조건 수익", "손실 없음"]

def find_pii(text):
    return [name for name, p in PII_PATTERNS.items() if re.search(p, text)]

def validate_input(state):
    q = state["question"].strip()
    if not q:               return {"status": "차단: 빈 질문입니다."}
    if len(q) > 500:        return {"status": "차단: 질문이 너무 깁니다. (500자 제한)"}
    pii = find_pii(q)
    if pii:                 return {"status": f"차단: 질문에 개인정보({', '.join(pii)})가 포함되어 있습니다."}
    return {"status": "입력 통과"}
```

### 5. ② 근거 검증 — 환각 탐지 ⭐

```python
from pydantic import BaseModel, Field

class GroundCheck(BaseModel):
    grounded: bool = Field(description="답변이 Context에 근거하면 True, 지어냈으면 False")
    reason: str = Field(description="판단 이유 한 문장")

grader = llm.with_structured_output(GroundCheck)

def validate_grounding(state):
    check = grader.invoke(f"""답변이 Context에 근거하는지 판단하세요.
Context: {state['context']}
답변: {state['answer']}""")
    if check.grounded:
        return {"status": "근거 검증 통과"}
    return {"status": f"재생성 필요: 근거 부족 ({check.reason})", "retry": state["retry"] + 1}
```

### 6. ③ 출력 필터 + 재시도 상한

```python
def validate_output(state):
    if state["status"].startswith("재생성"):     # 앞 검문소에서 이미 불합격이면 통과
        return {}
    for word in FORBIDDEN:                        # 금융 규제 표현
        if word in state["answer"]:
            return {"status": f"재생성 필요: 금지 표현('{word}') 포함", "retry": state["retry"] + 1}
    pii = find_pii(state["answer"])               # 답변에 PII가 새어나갔는지
    if pii:
        return {"status": f"재생성 필요: 답변에 개인정보({', '.join(pii)}) 포함", "retry": state["retry"] + 1}
    return {"status": "최종 통과 ✅"}

def route_output(state):
    if state["status"] == "최종 통과 ✅": return "pass"
    if state["retry"] >= 2:               return "give_up"      # ← 탈출구
    return "regenerate"

def give_up(state):
    return {"answer": "[안내] 정확한 답변을 생성하지 못했습니다. 상담원 연결을 도와드리겠습니다.",
            "status": "차단: 재생성 한도 초과 → 상담원 이관"}
```

### 7. 그래프 조립

```python
builder.add_edge(START, "validate_input")
builder.add_conditional_edges("validate_input", route_input, {"blocked": END, "ok": "generate_answer"})
builder.add_edge("generate_answer", "validate_grounding")
builder.add_edge("validate_grounding", "validate_output")
builder.add_conditional_edges("validate_output", route_output,
                              {"pass": END, "regenerate": "generate_answer", "give_up": "give_up"})
builder.add_edge("give_up", END)
```

## 주의사항

- 🚨 **`Day03_08` 노트북에 실제 OpenAI 키·LangSmith 키·네이버 Client ID/Secret이 하드코딩되어 있습니다.** 이 교안에서 노출이 가장 심한 파일입니다 → [[오류_환경설정_트러블슈팅]]
- **`InMemorySaver`는 프로세스가 죽으면 사라집니다.** 실서비스에서 사람 승인이 몇 분~몇 시간 걸린다면 `SqliteSaver`/`PostgresSaver` 같은 영속 체크포인터가 필요합니다.
- **`thread_id`가 겹치면 다른 사용자의 대화가 섞입니다.** 세션 ID를 반드시 사용자·대화 단위로 발급하세요 → [[개념_FastAPI_Agent_서비스화]]
- **재시도 상한을 반드시 두세요.** `retry >= 2` 같은 탈출구가 없으면 검증이 계속 불합격을 주는 무한 루프가 됩니다 → [[개념_Agent_통제력과_루프탈출]]
- **정규식 PII 탐지는 불완전합니다.** 이 3중 검문소는 최소 방어선이며, 전용 도구 병행이 실무 기준입니다 → [[개념_PII보호와_거버넌스]]
- **근거 검증은 LLM 호출을 추가합니다.** 답변 1건당 최소 2회(생성+검증), 재생성 시 4회 이상이 됩니다. 비용 설계를 함께 하세요 → [[개념_비용_라우팅_최적화]]
- **`input()` 기반 노드는 서버 환경에서 못 씁니다.** 원본은 try/except로 샘플 질문을 대신 쓰지만, 실제 앱에서는 `interrupt()` 로 대체해야 합니다.

## 관련 문서

- [[개념_조건부Edge와_ToolNode]] — 도구 실행 전 개입 지점
- [[개념_Agent_통제력과_루프탈출]] — 상한과 정직한 실패
- [[개념_PII보호와_거버넌스]] — 마스킹·인젝션 방어 심화
- [[개념_Fallback과_Rollback]] — 검증 불합격 시 되돌리기
- [[프로젝트_실전과제_가이드]] — 프로젝트 2번(카드민원)이 HITL 중심

## 출처

- Source: `Day3/Day03_08_조건분기_검증루프_Human_in_the_loop.ipynb`
- Source: `Day3/Day03_08_교안_조건분기_검증루프_Human_in_the_loop.pdf`
- Source: `바이브코딩실습/Day03_08_송금승인봇/app.py`, `Day03_07_자가평가답변봇/app.py`
