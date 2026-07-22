# 개념: LangChain LCEL 기초

## 한눈에 보기

LangChain은 LLM 호출·프롬프트·출력 정돈·메모리·도구 연결을 표준 부품으로 제공하는 프레임워크이고, **LCEL(LangChain Expression Language)** 은 그 부품들을 파이프 연산자 `|` 로 이어 붙이는 문법입니다.
이 문서는 `ChatOpenAI` 호출부터 `prompt | llm | parser` 체인, 구조화 출력, 대화 메모리까지 — 이후 모든 RAG·Agent 코드의 밑바탕이 되는 최소 문법을 정리합니다.

## 핵심 개념

| 부품 | 하는 일 | 대표 클래스 |
|---|---|---|
| **모델 I/O** | LLM에 질문하고 답 받기 | `ChatOpenAI`, `init_chat_model` |
| **Prompt** | 질문 틀에 값 채우기 | `PromptTemplate`, `ChatPromptTemplate` |
| **Chain** | 여러 단계를 하나로 연결 | LCEL 파이프 `|` |
| **Output Parser** | 자유 텍스트 → 리스트/JSON | `StrOutputParser`, `JsonOutputParser` |
| **Memory** | 이전 대화 기억 | `InMemoryChatMessageHistory`, `RunnableWithMessageHistory` |
| **Tool/Agent** | 외부 기능 호출 | [[개념_Tool_Calling]] 참고 |

**LCEL이 왜 필요한가**: LCEL 없이 쓰면 `prompt.format()` → `client.chat.completions.create()` → `response.choices[0].message.content` 처럼 매 단계를 손으로 이어야 합니다. LCEL은 이 흐름을 `chain = prompt | model | parser` 한 줄로 만들고, 동시에 `.invoke()` / `.stream()` / `.batch()` 를 공짜로 얻게 해 줍니다.

> ⚠️ 원본 노트북은 `gpt-4.1`, `gpt-5`, `gpt-4o-mini`, `gemini-2.5-flash` 등을 섞어 씁니다. 모델명은 계정 권한에 따라 바꿔도 코드 구조는 동일합니다.

## 전체 동작 흐름

```
[입력 dict]
   ↓  PromptTemplate  — {변수}에 값을 채워 완성된 프롬프트 생성
[PromptValue]
   ↓  ChatOpenAI      — 모델 호출, AIMessage 반환
[AIMessage]
   ↓  OutputParser    — .content 추출 / 리스트·JSON으로 변환
[파이썬 값]
```

## 주요 코드 구성

| 대상 | 역할 |
|---|---|
| `Day01_01_LangChain_기초실습_최신버전.ipynb` | 위 6개 부품을 순서대로 실습하는 메인 노트북 |
| `Day1/개인실습/Day01_01_..._개인실습.ipynb` | 같은 내용의 빈칸 채우기 실습본 |
| `바이브코딩실습/Day01_01_금융용어봇/app.py` | LCEL 체인 하나로 만든 Streamlit 데모 |

## 실행 방법

```bash
pip install -U langchain langchain-openai langchain-community langchain-core openai pypdf beautifulsoup4

# Windows PowerShell — 키는 환경변수로
$env:OPENAI_API_KEY = "sk-..."      # 코드에 절대 적지 마세요

# 바이브코딩 데모
cd 바이브코딩실습/Day01_01_금융용어봇 && streamlit run app.py
```

## 핵심 코드

### 1. 키를 코드에 적지 않는 표준 패턴

교안 전체가 이 패턴을 씁니다. 환경변수가 있으면 그걸 쓰고, 없으면 실행 시점에 입력받습니다.

```python
import os
from getpass import getpass

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass("OpenAI API 키를 입력하세요: ")
```

Streamlit 앱에서는 `os.environ.get("OPENAI_API_KEY") or st.text_input("...", type="password")` 로 같은 원칙을 지킵니다.

### 2. 최소 LCEL 체인

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("{topic}에 대해 설명해주세요")
llm = ChatOpenAI(model="gpt-4.1", temperature=0.5)

chain = prompt | llm | StrOutputParser()   # 세 부품을 파이프로 연결
print(chain.invoke({"topic": "예금자보호제도"}))
```

`StrOutputParser()`가 없으면 결과는 `AIMessage` 객체이므로 `.content` 를 직접 꺼내야 합니다.

### 3. 체인을 체인의 입력으로 — 다단계 연결

앞 체인의 출력이 뒤 체인의 변수로 들어갑니다. 요약 → 번역처럼 단계를 이을 때 씁니다.

```python
chain1 = prompt1 | llm | StrOutputParser()          # 요약
# chain1의 결과를 prompt2의 {summary} 자리에 주입
chain2 = {"summary": chain1} | prompt2 | llm | StrOutputParser()   # 번역
chain2.invoke({"text": "인공지능은 ..."})
```

### 4. 구조화 출력 — 파서보다 안전한 방법

`OutputParser`는 "이런 형식으로 답해줘"라고 부탁하는 방식이라 형식이 깨질 수 있습니다.
`with_structured_output()` 은 모델의 Structured Outputs 기능을 써서 **스키마를 강제**합니다. 이후 Day3의 의도 분류, Day4의 Supervisor 라우팅이 모두 이 방식을 씁니다.

```python
from pydantic import BaseModel, Field

class Players(BaseModel):
    players: list[str] = Field(description="선수 이름 3명")

structured_llm = llm.with_structured_output(Players)
result = structured_llm.invoke("한국의 축구선수 3명을 추천해줘")
print(result.players)     # 파이썬 리스트로 바로 사용 가능
```

### 5. RunnablePassthrough — 입력을 그대로 흘려보내기

`chain.invoke({"num": 10})` 대신 `chain.invoke(10)` 처럼 스칼라를 바로 넣고 싶을 때 씁니다.
RAG 체인에서 "질문 문자열 하나를 받아 `{context, question}` 두 자리에 나눠 넣는" 용도로 계속 등장합니다. → [[개념_RAG_파이프라인]]

```python
from langchain_core.runnables import RunnablePassthrough

runnable_chain = {"num": RunnablePassthrough()} | prompt | llm
runnable_chain.invoke(10)      # dict로 감쌀 필요 없음
```

### 6. 대화 메모리 — `RunnableWithMessageHistory`

세션별 대화 기록을 자동 관리합니다. `session_id` 로 대화방을 구분하는 이 구조는 Day3의 LangGraph `thread_id`, Day4의 FastAPI 세션 설계와 정확히 같은 개념입니다.

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 도우미입니다."),
    MessagesPlaceholder(variable_name="history"),   # 과거 대화가 들어갈 자리
    ("user", "{input}"),
])

store = {}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversation = RunnableWithMessageHistory(
    prompt | llm, get_session_history,
    input_messages_key="input", history_messages_key="history",
)
config = {"configurable": {"session_id": "kb-demo"}}
conversation.invoke({"input": "진규는 사과 3개를 가지고 있습니다."}, config=config)
conversation.invoke({"input": "진규가 가진 과일 수는?"}, config=config)  # 앞 대화를 기억
```

### 7. 스트리밍

```python
for chunk in chain.stream({"topic": "축구"}):
    print(chunk, end="", flush=True)
```

## 주의사항

- **키 하드코딩 금지.** 원본 노트북 중 일부(Day3·Day4·Day5)는 실제 키가 박혀 있습니다 → [[오류_환경설정_트러블슈팅]]
- **레거시 문법 주의.** `llm.predict()`, `from langchain import PromptTemplate`, `LLMChain` 은 제거·비권장입니다. 최신 경로는 `langchain_core.prompts`, `langchain_core.output_parsers`, `langchain_openai` 입니다.
- **추론(reasoning) 모델은 `temperature`를 지원하지 않을 수 있습니다.** `gpt-5` 계열을 쓸 때 기존 인자를 그대로 넘기면 오류가 납니다.
- **Chat 모델 응답은 `AIMessage` 객체**입니다. `print(response)` 하면 메타데이터까지 나오므로 본문은 `.content` 로 꺼내거나 `StrOutputParser()` 를 체인에 붙이세요.
- Gemini 계열을 쓸 때 원본은 `InMemoryRateLimiter(requests_per_second=0.167)` 로 호출 속도를 제한합니다. 무료 티어 rate limit 회피용이며, 그만큼 실행이 느립니다.

## 관련 문서

- [[개념_RAG_파이프라인]] — 이 체인 문법 위에 검색을 얹습니다
- [[개념_Tool_Calling]] — 구조화 출력의 확장
- [[개념_LangGraph_State_Node_Edge]] — 체인으로 표현하기 어려운 분기·루프를 그래프로
- [[실습_바이브코딩_카탈로그]]

## 출처

- Source: `Day1/Day01_01_LangChain_기초실습_최신버전.ipynb`
- Source: `Day1/개인실습/Day01_01_LangChain_기초_개인실습.ipynb`
- Source: `바이브코딩실습/Day01_01_금융용어봇/app.py`, `프롬프트.txt`
