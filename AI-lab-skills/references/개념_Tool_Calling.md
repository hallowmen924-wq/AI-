# 개념: Tool Calling (Function Calling)

## 한눈에 보기

Tool Calling은 개발자가 함수 목록을 미리 정의해 두면 **LLM이 질문을 보고 적합한 함수와 인자를 스스로 골라 주는** 기능입니다.
LLM은 함수를 직접 실행하지 않습니다 — "이 함수를 이 인자로 부르세요"라고 알려줄 뿐이고, 실행은 우리 코드가 합니다. 이 구분이 Tool Calling 이해의 핵심입니다.

## 핵심 개념

**4단계 사이클**

```
① 도구 정의       — JSON Schema로 함수 이름·설명·파라미터를 기술
② 질문 + 도구목록  — 모델에 함께 전달 (tool_choice="auto")
③ 모델이 선택      — response.tool_calls 에 함수명과 인자(JSON)가 담겨 옴
④ 우리가 실행      — 함수 호출 → 결과를 role="tool" 메시지로 되돌려줌 → 모델이 최종 답변
```

| 개념 | 설명 |
|---|---|
| `description` | **모델이 도구를 고르는 유일한 근거.** 여기가 부실하면 엉뚱한 도구를 부릅니다 |
| `tool_calls` | 모델이 고른 함수와 인자. 없으면 도구 없이 바로 답한 것 |
| `tool_call_id` | 어떤 요청에 대한 결과인지 짝을 맞추는 ID. 반드시 함께 돌려줘야 합니다 |
| `tool_choice` | `"auto"`(모델 판단) / `"required"`(무조건 사용) / 특정 함수 강제 |

**Chat Completions API vs Responses API**: OpenAI는 Responses API를 후속으로 내놓았습니다. 원리는 같고 문법이 더 간결합니다(`client.responses.create`, `function_call_output`).
**LangChain에서는** 같은 개념을 `@tool` 데코레이터 + `llm.bind_tools([...])` 로 훨씬 짧게 씁니다. → [[개념_조건부Edge와_ToolNode]]

## 전체 동작 흐름

```
[사용자] "오늘 서울 날씨는?"
    ↓ messages + tools 전달
[LLM] tool_calls = [{name: "get_current_weather", arguments: '{"location":"서울"}'}]
    ↓ 우리 코드가 json.loads → 실제 함수 실행
[함수] {"location":"서울","temperature":"28","forecast":["sunny"]}
    ↓ role="tool" 메시지로 대화에 추가
[LLM] "서울은 현재 28도이고 맑습니다."
```

## 주요 코드 구성

| 파일 | 역할 |
|---|---|
| `Day2/Day02_03_Tool_Calling이해.ipynb` | OpenAI SDK 원형 문법 (날씨 → 다중 도구 → Wikipedia → Responses API) |
| `Day2/개인실습/Day02_02_Tool_Calling이해_개인실습.ipynb` | 빈칸 실습본 |
| `바이브코딩실습/Day02_04_금융계산도우미/app.py` | 이자·환율 계산 함수 자동 선택 데모 |

## 실행 방법

```bash
pip install --upgrade openai wikipedia
$env:OPENAI_API_KEY = "sk-..."

cd 바이브코딩실습/Day02_04_금융계산도우미 && streamlit run app.py
```

## 핵심 코드

### 1. 도구 정의 (Chat Completions 형식)

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "특정 지역의 현재 날씨를 알려줍니다.",   # ← 선택 기준이 되는 문장
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "지역 이름, 예: 서울"},
                "unit": {"type": "string", "enum": ["섭씨", "화씨"]},   # enum으로 값 제한
            },
            "required": ["location"],
        },
    },
}]
```

### 2. 실행 루프 — 다중 도구 대응 (재사용 템플릿) ⭐

```python
import json

available_functions = {
    "get_restaurant_recommendation": get_restaurant_recommendation,
    "get_book_recommendation": get_book_recommendation,
    "get_movie_recommendation": get_movie_recommendation,
}

response = client.chat.completions.create(
    model=MODEL, messages=messages, tools=tools, tool_choice="auto",
)
response_message = response.choices[0].message

if response_message.tool_calls:                 # 도구를 고르지 않았으면 None
    messages.append(response_message)           # ① 모델의 요청을 대화에 남기고
    for tool_call in response_message.tool_calls:
        function_args = json.loads(tool_call.function.arguments)     # 인자는 JSON 문자열
        function_response = available_functions[tool_call.function.name](**function_args)
        messages.append({                       # ② 결과를 tool 역할로 되돌림
            "role": "tool",
            "tool_call_id": tool_call.id,       # ← 짝 맞추기용, 빠뜨리면 오류
            "content": function_response,
        })
    second_response = client.chat.completions.create(model=MODEL, messages=messages)
    print(second_response.choices[0].message.content)   # ③ 최종 답변
```

> 💡 **두 번 호출한다**는 점이 핵심입니다. 1차는 "무슨 도구 쓸래?", 2차는 "이 결과로 답해줘".
> 비용도 두 배가 되므로 → [[개념_비용_라우팅_최적화]]

### 3. 진짜 외부 도구 연결 (Wikipedia)

Mock 함수와 달리 실제 외부 시스템을 부르는 예시입니다. 은행이라면 이 자리에 Core Banking 조회 API가 들어갑니다.

```python
import wikipedia

def search_wikipedia(query, lang="ko"):
    wikipedia.set_lang(lang)
    r = wikipedia.search(query, results=1)
    return wikipedia.page(r[0]).content[:4000]      # 토큰 폭발 방지를 위해 잘라서 반환
```

### 4. Responses API 버전 (최신 문법)

```python
tools_responses = [{
    "type": "function",                      # function 키 없이 평평한 구조
    "name": "get_current_weather",
    "description": "특정 지역의 현재 날씨를 알려줍니다.",
    "parameters": {...},
}]

response = client.responses.create(model=MODEL, input=input_list, tools=tools_responses)
input_list += response.output

for item in response.output:
    if item.type == "function_call":
        function_response = get_current_weather(**json.loads(item.arguments))
        input_list.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": function_response,
        })

second = client.responses.create(model=MODEL, input=input_list, tools=tools_responses)
print(second.output_text)                    # .choices[0].message.content 대신
```

## 주의사항

- **`description`이 곧 성능입니다.** "날씨 조회"보다 "특정 지역의 현재 기온과 날씨 상태를 조회합니다. 예보가 아닌 현재 시점입니다"처럼 **언제 써야 하는지**를 적으세요.
- **모델에 따라 도구 실행 능력이 크게 다릅니다.** 원본 노트북도 이를 명시합니다 — sLLM이나 소형 모델은 도구가 복잡하거나 컨텍스트가 길어지면 인자를 잘못 만들거나 도구를 건너뜁니다. 프롬프트에 사용 예시를 넣으면 개선됩니다.
- **`json.loads` 실패를 대비하세요.** 모델이 만든 arguments가 항상 유효한 JSON이라는 보장은 없습니다. try/except로 감싸고 실패 시 재요청하는 것이 안전합니다.
- **LLM이 고른 인자를 그대로 실행하면 위험합니다.** 송금·한도조정처럼 되돌릴 수 없는 작업은 실행 전에 사람 승인을 넣으세요 → [[개념_HITL과_검증루프]]
- **도구 실행 권한을 제한하세요.** 역할별로 쓸 수 있는 도구를 나누는 RBAC → [[개념_PII보호와_거버넌스]]
- 노트북에는 `MODEL = "gpt-5"` 로 되어 있습니다. 계정 권한에 따라 조정하세요.

## 관련 문서

- [[개념_조건부Edge와_ToolNode]] — LangGraph에서 이 루프를 자동화하는 방법
- [[개념_MCP_구조와_구현]] — 도구를 표준 규격으로 노출하기
- [[개념_Text_to_SQL]] — DB를 도구로 붙이는 특수 사례
- [[개념_MultiAgent_Supervisor]] — 도구가 많아질 때의 분업

## 출처

- Source: `Day2/Day02_03_Tool_Calling이해.ipynb`
- Source: `Day2/개인실습/Day02_02_Tool_Calling이해_개인실습.ipynb`
- Source: `바이브코딩실습/Day02_04_금융계산도우미/app.py`
