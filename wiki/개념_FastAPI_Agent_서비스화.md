# 개념: FastAPI로 Agent 서비스화

## 한눈에 보기

노트북 안의 Agent는 노트북 안에서만 삽니다. 모바일 앱·인터넷뱅킹·사내 포털이 쓰려면 **HTTP API**로 감싸야 합니다.
핵심 설계는 3가지 — 요청/응답을 Pydantic으로 검증, **`session_id` → LangGraph `thread_id` 매핑으로 대화 유지**, 예외를 HTTP 상태 코드로 변환.

## 핵심 개념

| 설계 요소 | 구현 | 왜 |
|---|---|---|
| **입력 검증** | `BaseModel` + `Field(min_length, max_length)` | 빈 문자열·초장문 요청을 Agent 앞에서 차단 (422 자동 반환) |
| **세션 유지** | `session_id` → `{"configurable": {"thread_id": session_id}}` | 고객 A와 B의 대화가 섞이지 않게 |
| **오류 처리** | `try/except` → `HTTPException(500, detail=...)` | 내부 스택 트레이스 노출 방지 |
| **헬스체크** | `GET /health` | 로드밸런서·모니터링이 살아있는지 확인 |
| **자동 문서** | `/docs` (Swagger UI) | FastAPI가 스키마로부터 자동 생성 |

**`create_react_agent` 를 쓰는 이유**: Day3에서는 `tools_condition` + `ToolNode`를 직접 조립했지만, 여기서는 서비스화가 주제이므로 Agent 자체는 한 줄로 만듭니다. 목적이 다르면 도구도 다릅니다. → [[개념_조건부Edge와_ToolNode]]

## 전체 동작 흐름

```
[클라이언트] POST /chat {"session_id": "customer-A", "message": "정기예금 추천해줘"}
      ↓ Pydantic 검증 (실패 시 422)
[FastAPI] config = {"configurable": {"thread_id": "customer-A"}}
      ↓
[Agent] create_react_agent(llm, tools, checkpointer=MemorySaver())
      ↓ 도구 선택 → 실행 → 답변  (checkpointer가 대화 이력 보관)
[FastAPI] ChatResponse(session_id, answer)   ← 예외 발생 시 500
      ↓
[클라이언트] {"session_id": "customer-A", "answer": "..."}
```

## 주요 코드 구성

| 파일 | 역할 |
|---|---|
| `Day4/Day04_03_FastAPI_서비스화_종합프로젝트.ipynb` | 통합 Agent → FastAPI → uvicorn → 시나리오 데모 |
| `Day4/개인실습/server.py` | 서버 코드 |
| `바이브코딩실습/Day04_03_상담봇API서비스/server.py` | POST /chat + 웹 채팅 화면 (**규칙 모드는 키 없이 동작**) |

## 실행 방법

```bash
pip install -qU langchain langgraph langchain-openai langchain-chroma fastapi uvicorn httpx nest_asyncio
$env:OPENAI_API_KEY = "sk-..."

# 이 폴더만 streamlit이 아니라 uvicorn입니다
cd 바이브코딩실습/Day04_03_상담봇API서비스 && uvicorn server:app --port 8000
# 브라우저에서 http://127.0.0.1:8000/docs  ← Swagger UI
```

## 핵심 코드

### 1. 통합 Agent 조립 — RAG + DB 조회 + 환율 도구

도구 3종을 `create_react_agent`로 묶고 `MemorySaver`로 세션 기억을 붙입니다.

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

@tool
def search_terms(query: str) -> str:
    """예금·적금 상품의 약관과 규정을 검색합니다. 금리, 중도해지, 예금자보호 등 규정 질문에 사용하세요."""
    return "\n".join(doc.page_content for doc in terms_db.similarity_search(query, k=2))

@tool
def lookup_product(product_name: str) -> str:
    """예금·적금 상품의 기본 정보(금리, 기간)를 조회합니다."""
    products = {"KB스타 정기예금": {"금리": "연 3.5%", "기간": "12~36개월"}}   # (Mock)
    info = products.get(product_name)
    return str(info) if info else "해당 상품을 찾을 수 없습니다."

@tool
def get_exchange_rate(currency: str) -> str:
    """통화 코드(USD, JPY 등)의 원화 환율을 조회합니다."""
    rates = {"USD": 1385.50, "JPY": 9.42, "EUR": 1490.20}          # (Mock)
    rate = rates.get(currency)
    return f"{currency} = {rate}원" if rate else "지원하지 않는 통화입니다."

agent = create_react_agent(
    llm,
    tools=[search_terms, lookup_product, get_exchange_rate],
    prompt="당신은 KB국민은행 상담 Agent입니다. 필요한 도구를 골라 정확하고 간결하게 답하세요.",
    checkpointer=MemorySaver(),          # ← 이게 있어야 thread_id별 대화가 유지됩니다
)
```

### 2. FastAPI 엔드포인트 ⭐ (재사용 템플릿)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="고객 세션 ID (대화방 번호)", min_length=1)
    message: str = Field(..., description="고객 질문", min_length=1, max_length=1000)

class ChatResponse(BaseModel):
    session_id: str
    answer: str

app = FastAPI(title="KB 상담 Agent API", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.session_id}}    # ← 세션 = 스레드
    try:
        result = agent.invoke({"messages": [("user", req.message)]}, config)
        answer = result["messages"][-1].content
    except Exception as e:
        # 내부 오류 메시지를 그대로 노출하지 않고 예외 타입만 전달
        raise HTTPException(status_code=500, detail=f"Agent 처리 중 오류: {type(e).__name__}")
    return ChatResponse(session_id=req.session_id, answer=answer)
```

> 💡 `min_length=1`, `max_length=1000` 만으로 빈 요청·초장문 요청이 **Agent에 도달하기 전에** 422로 거부됩니다. 비용 방어이자 보안 방어입니다.

### 3. TestClient로 서버 없이 검증

pytest 자동화 테스트에서도 그대로 쓰는 방법입니다.

```python
from fastapi.testclient import TestClient

client = TestClient(app)
print(client.get("/health").json())                      # {'status': 'ok'}

resp = client.post("/chat", json={"session_id": "customer-A", "message": "청년도약계좌 가입 조건은?"})
print(resp.status_code, resp.json()["answer"])

# 세션 분리 확인 — 같은 질문, 다른 결과
client.post("/chat", json={"session_id": "customer-A", "message": "월 얼마까지 넣을 수 있다고요?"})  # 문맥 이어짐
client.post("/chat", json={"session_id": "customer-B", "message": "월 얼마까지 넣을 수 있다고요?"})  # 새 세션

# 검증 실패 확인
resp = client.post("/chat", json={"session_id": "customer-C", "message": ""})
print(resp.status_code, resp.json()["detail"][0]["msg"])   # 422
```

### 4. 실서버 실행

프로덕션에서는 CLI로 띄웁니다.

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

노트북 안에서 띄우려면 별도 스레드 + `nest_asyncio`가 필요합니다(교육용).

```python
import threading, time, nest_asyncio, uvicorn
nest_asyncio.apply()
server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()
time.sleep(2)
```

### 5. 멀티턴 시나리오 호출

```python
import httpx

scenario = [
    "1천만원 정도 여유자금이 있는데 정기예금 상품 추천해주세요",
    "그 상품 중간에 해지하면 어떻게 되나요?",       # ← 앞 대화를 기억해야 답할 수 있음
    "참, 미국 달러 환율도 알려주세요",
]
for q in scenario:
    resp = httpx.post("http://127.0.0.1:8000/chat",
                      json={"session_id": "demo-고객", "message": q}, timeout=60)
    print(f"👤 {q}\n🤖 {resp.json()['answer']}\n" + "=" * 60)
```

## 실서비스 확장 체크리스트

교안 정리를 바탕으로 한 실무 항목입니다.

- [ ] **인증/인가** — API Key, OAuth, 사내 SSO 연동
- [ ] **`MemorySaver` → 영속 체크포인터** — 프로세스 재시작 시 대화 유실 방지 (`SqliteSaver`/`PostgresSaver`)
- [ ] **`workers > 1` 이면 인메모리 상태가 깨집니다** — 워커마다 별도 메모리를 가지므로 외부 저장소 필수
- [ ] **타임아웃과 재시도** — LLM 응답 지연이 그대로 API 지연이 됩니다 → [[개념_Fallback과_Rollback]]
- [ ] **감사 로그** — 요청마다 correlation_id·user_id 기록 → [[개념_트레이싱과_감사로그]]
- [ ] **PII 마스킹** — 요청/응답 양방향 → [[개념_PII보호와_거버넌스]]
- [ ] **레이트 리밋** — 고객당 요청 수 제한 (비용·남용 방어)
- [ ] **스트리밍 응답** — 긴 답변은 SSE로 흘려보내야 체감 속도가 개선됨

## 주의사항

- **`MemorySaver`는 프로세스 메모리입니다.** 재시작하면 모든 대화가 사라지고, 워커가 여러 개면 요청마다 다른 워커에 붙어 대화가 끊깁니다. **운영 전환 시 가장 먼저 바꿔야 할 부분입니다.**
- **`session_id`를 클라이언트가 자유롭게 정하면 남의 대화를 훔쳐볼 수 있습니다.** 인증된 사용자 ID에서 서버가 파생시키세요.
- **예외 detail에 원본 메시지를 넣지 마세요.** 위 코드가 `type(e).__name__` 만 넘기는 이유입니다. 상세 내용은 서버 로그에만 남깁니다.
- **`nest_asyncio` + 스레드 방식은 교육용입니다.** 프로덕션에서는 `uvicorn` CLI 또는 gunicorn+uvicorn worker를 쓰세요.
- **동기 `def chat`에서 `agent.invoke`(블로킹)를 호출**하므로 FastAPI가 스레드풀에서 실행합니다. 고동시성이 필요하면 `async def` + `agent.ainvoke`로 바꾸세요.
- **도구의 상품·환율 데이터는 전부 Mock입니다.** 실제 API 연결 시 실패 처리·타임아웃을 반드시 추가하세요.

## 관련 문서

- [[개념_MultiAgent_Supervisor]] — 서비스화할 Agent 만들기
- [[개념_Agent_통제력과_루프탈출]] — thread_id와 세션 설계
- [[개념_트레이싱과_감사로그]] — 운영 관측
- [[개념_Fallback과_Rollback]] — 장애 대비
- [[개념_PII보호와_거버넌스]] — API 앞단 방어

## 출처

- Source: `Day4/Day04_03_FastAPI_서비스화_종합프로젝트.ipynb`
- Source: `Day4/Day04_03_교안_FastAPI_서비스화_종합프로젝트.pdf`
- Source: `Day4/개인실습/server.py`, `바이브코딩실습/Day04_03_상담봇API서비스/server.py`
