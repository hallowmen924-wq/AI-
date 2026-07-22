# 개념: MCP (Model Context Protocol) 구조와 구현

## 한눈에 보기

MCP는 **AI 모델과 외부 도구를 연결하는 표준 규격**입니다. 흔한 비유로 "AI 세계의 USB-C" — 한 번 MCP 서버로 만들어 두면 ChatGPT·Claude·Gemini 등 여러 AI가 같은 도구를 재사용할 수 있습니다.
이 문서는 `FastMCP`로 서버를 만들고, stdio/HTTP 두 방식으로 클라이언트를 붙이고, LangChain Agent에 그대로 꽂는 전 과정을 다룹니다.

## 핵심 개념

| 구분 | Function Calling | 플러그인 | **MCP** |
|---|---|---|---|
| 표준화 수준 | 벤더별 스키마 | 벤더 전용 | **업계 공통 프로토콜** |
| 재사용성 | 앱마다 재작성 | 해당 플랫폼만 | 여러 AI가 그대로 사용 |
| 배포 | 코드에 내장 | 스토어 등록 | 독립 서버 프로세스 |

**구조**

```
[AI Agent] ──(MCP Client)── 표준 프로토콜 ──(MCP Server)── [실제 시스템]
                                                            Core Banking / CRM / DW
```

| 용어 | 뜻 |
|---|---|
| **MCP Server** | 도구를 표준 규격으로 노출하는 프로세스. `@mcp.tool()` 데코레이터로 정의 |
| **MCP Client** | 서버에 연결해 `list_tools()` / `call_tool()` 을 호출하는 쪽 |
| **stdio 전송** | 클라이언트가 서버를 **하위 프로세스로 띄우고** 표준입출력으로 통신. 로컬 개발용 |
| **Streamable HTTP 전송** | 서버가 별도 장비에서 상시 운영되고 네트워크로 연결. **은행 실무 형태** |
| **langchain-mcp-adapters** | MCP 도구를 LangChain Tool 리스트로 자동 변환 |

**은행 시나리오**: Core Banking 조회, 환율 조회, 고객 등급 조회를 각각 MCP 서버로 감싸 두면, 상담봇·내부 포털·모바일 앱이 **같은 도구를 공유**합니다. 도구 로직을 한 곳에서만 관리하면 되고, 접근 통제도 서버 한 곳에 걸 수 있습니다.

## 전체 동작 흐름

```
① mcp_server.py 작성        @mcp.tool() 데코레이터로 함수 3개 노출
② 클라이언트 연결            stdio_client → ClientSession → initialize()
③ 도구 목록 조회            session.list_tools()
④ 도구 직접 호출            session.call_tool("get_deposit_rate", {"product": "KB정기예금"})
⑤ Agent에 연결              MultiServerMCPClient → get_tools() → create_agent(llm, tools)
⑥ HTTP 전송으로 전환         mcp.run(transport="streamable-http") + streamablehttp_client
```

## 주요 코드 구성

| 파일 | 역할 |
|---|---|
| `Day2/Day02_05_MCP구조이해_최신버전.ipynb` | 메인 노트북 (`%%writefile` 로 아래 .py 파일들을 생성) |
| `Day2/mcp_server.py` | stdio MCP 서버 (금리·환율·고객등급 3개 도구) |
| `Day2/mcp_list_demo.py` | 도구 목록 조회 클라이언트 |
| `Day2/mcp_call_demo.py` | 도구 직접 호출 클라이언트 |
| `Day2/mcp_agent_demo.py` | LLM이 도구를 선택하는 Agent 클라이언트 |
| `Day2/mcp_server_http.py` / `mcp_http_demo.py` | HTTP 전송 서버·클라이언트 쌍 |
| `Day2/개인실습/MCP은행도구체험판/` | Streamlit 체험판 |

## 실행 방법

```bash
pip install -q mcp==1.28.1 langchain-mcp-adapters==0.3.0 \
    langchain==1.3.11 langchain-openai==1.3.3 langgraph==1.2.7
$env:OPENAI_API_KEY = "sk-..."

cd Day2
python mcp_list_demo.py      # 도구 목록 조회 (서버는 자동으로 띄워집니다)
python mcp_call_demo.py      # 도구 직접 호출
python mcp_agent_demo.py     # LLM이 도구 선택

# HTTP 방식 — 서버를 먼저 별도 터미널에서 실행
python mcp_server_http.py    # 127.0.0.1:8000
python mcp_http_demo.py      # 다른 터미널에서
```

> ⚠️ **클라이언트를 Jupyter 셀 안에서 `await`로 직접 돌리지 마세요.** 서버를 하위 프로세스로 띄우는 비동기 코드라 노트북(특히 Windows)의 이벤트 루프 제약에 걸려 오류가 납니다. 원본 노트북도 `%%writefile`로 .py를 만든 뒤 `!{sys.executable} 파일명` 으로 실행합니다.

## 핵심 코드

### 1. MCP 서버 — 데코레이터 하나로 도구 노출

**docstring이 도구 설명이 됩니다.** LLM이 도구를 고르는 근거이므로 정성껏 쓰세요.

```python
# mcp_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("KB-Bank-Tools")

@mcp.tool()
def get_deposit_rate(product: str) -> str:
    """예금 상품명을 받아 연 금리를 조회한다. (Mock)"""
    rates = {"KB정기예금": "3.5%", "KB자유적금": "3.2%", "KB주택청약": "2.8%"}
    return f"{product}의 연 금리는 {rates.get(product, '정보 없음')} 입니다."

@mcp.tool()
def get_exchange_rate(currency: str) -> str:
    """통화 코드를 받아 원화 환율을 조회한다. (Mock)"""
    rates = {"USD": 1385.0, "JPY": 8.9, "EUR": 1495.0}
    return f"1 {currency} = {rates.get(currency, 0)}원 (Mock 기준)"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### 2. 클라이언트 — 목록 조회

```python
# mcp_list_demo.py
import sys, asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# sys.executable = 지금 실행 중인 파이썬. 가상환경이 달라지는 문제를 피합니다
server_params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()                 # 핸드셰이크 — 필수
            for t in (await session.list_tools()).tools:
                print(f"- {t.name}: {t.description}")

asyncio.run(main())
```

### 3. 도구 직접 호출

```python
result = await session.call_tool("get_deposit_rate", {"product": "KB정기예금"})
print(result.content[0].text)      # 결과는 content 리스트 안에 들어 있습니다
```

### 4. LangChain Agent에 MCP 도구 연결 ⭐

`MultiServerMCPClient`가 MCP 도구를 LangChain Tool로 자동 변환합니다. 이름이 "Multi"인 이유는 **여러 서버를 한 번에** 붙일 수 있기 때문입니다.

```python
# mcp_agent_demo.py
import sys, asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

async def main():
    client = MultiServerMCPClient({
        "bank": {"command": sys.executable, "args": ["mcp_server.py"], "transport": "stdio"},
        # "crm": {"url": "http://crm-server:8000/mcp", "transport": "streamable_http"},
    })
    tools = await client.get_tools()                  # → LangChain Tool 리스트
    agent = create_agent("openai:gpt-5-mini", tools)
    resp = await agent.ainvoke({"messages": [
        {"role": "user", "content": "미국 달러 환율이랑 KB정기예금 금리 알려줘"}
    ]})
    print(resp["messages"][-1].text)

asyncio.run(main())
```

### 5. HTTP 전송 — 네트워크 너머의 MCP 서버 (실무 형태)

```python
# 서버: 호스트/포트를 지정하고 transport만 바꾸면 끝
mcp = FastMCP("KB-Bank-Tools-HTTP", host="127.0.0.1", port=8000)
# ... @mcp.tool() 정의는 동일 ...
mcp.run(transport="streamable-http")

# 클라이언트
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client("http://127.0.0.1:8000/mcp") as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("get_exchange_rate", {"currency": "USD"})
```

## 주의사항

- **모든 도구가 Mock입니다.** 금리·환율·고객등급은 하드코딩 딕셔너리이며, docstring에 `(Mock)` 이 명시되어 있습니다. 실서비스 연결 시 이 자리를 실제 API 호출로 바꾸세요.
- **MCP 서버는 인증·인가가 기본 제공되지 않습니다.** HTTP로 노출하는 순간 네트워크에 접근 가능한 누구나 도구를 부를 수 있습니다. 사내 배포 시 네트워크 격리 + 인증 계층을 반드시 앞에 두세요.
- **버전 고정이 중요합니다.** MCP 생태계는 변화가 빨라 `mcp`/`langchain-mcp-adapters` 버전이 어긋나면 API가 달라집니다. 노트북에 명시된 `mcp==1.28.1`, `langchain-mcp-adapters==0.3.0` 조합이 검증된 것입니다.
- **HTTP 서버 프로세스 정리**: 노트북에서 `subprocess.Popen`으로 띄웠다면 `server_proc.terminate()` 로 반드시 종료하세요. 안 그러면 8000 포트가 계속 점유됩니다.
- **stdio 방식은 서버 파일 경로에 의존합니다.** `args=["mcp_server.py"]` 는 현재 작업 디렉터리 기준이라, 다른 폴더에서 실행하면 서버를 못 찾습니다. 절대 경로를 쓰거나 `cd Day2` 후 실행하세요.
- LLM이 부르는 도구도 결국 Tool Calling이므로, 실행 전 승인이 필요한 작업은 별도 장치가 필요합니다 → [[개념_HITL과_검증루프]]

## 관련 문서

- [[개념_Tool_Calling]] — MCP의 기반 개념
- [[개념_MultiAgent_Supervisor]] — 여러 도구·에이전트의 조율
- [[개념_LocalLLM_내부망_구축]] — 내부망에서 MCP 서버 운영
- [[실습_바이브코딩_카탈로그]]

## 출처

- Source: `Day2/Day02_05_MCP구조이해_최신버전.ipynb`
- Source: `Day2/{mcp_server.py, mcp_list_demo.py, mcp_call_demo.py, mcp_agent_demo.py, mcp_server_http.py, mcp_http_demo.py}`
- Source: `Day2/Day02_05_교안_MCP_기업_강의_20260709051423.pdf`, `Day2/이론_MCP.pdf`
- Source: `Day2/개인실습/MCP은행도구체험판/`, `바이브코딩실습/Day02_06_MCP은행도구/`
