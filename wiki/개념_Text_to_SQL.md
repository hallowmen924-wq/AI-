# 개념: Text-to-SQL (자연어로 DB에 질문하기)

## 한눈에 보기

Text-to-SQL은 자연어 질문을 LLM이 SQL로 번역하고, 그 SQL을 DB에서 실행해 실제 데이터를 답하는 구조입니다.
LangChain의 `create_sql_query_chain(llm, db)` 한 줄로 변환기를 만들 수 있지만, **금융권에서는 생성된 SQL을 그대로 실행하면 안 됩니다** — 이 문서는 그 가드레일까지 함께 다룹니다.

## 핵심 개념

| 단계 | 하는 일 |
|---|---|
| **스키마 전달** | `SQLDatabase.from_uri()` 가 테이블·컬럼 정보를 자동으로 읽어 프롬프트에 넣음 |
| **SQL 생성** | LLM이 질문 + 스키마를 보고 SELECT 문 작성 |
| **정제(clean)** | 코드펜스·`SQLQuery:` 접두어 등 군더더기 제거 |
| **검증(가드레일)** | SELECT만 허용, 테이블 화이트리스트, LIMIT 강제 |
| **실행** | `db.run(sql)` 또는 `sqlite3` 커서로 실행 |

**왜 스키마 전달이 자동인가**: `SQLDatabase`는 `CREATE TABLE` 문과 샘플 행을 읽어 프롬프트에 삽입합니다. 그래서 우리가 스키마를 설명하지 않아도 LLM이 JOIN 조건을 만들어 냅니다.

**Pandas Agent와의 차이**: DB(SQL) 대신 엑셀 같은 표 데이터에 질문하려면 `create_pandas_dataframe_agent`를 씁니다. 이쪽은 LLM이 **파이썬 코드를 생성해 실행**하므로 위험도가 훨씬 높습니다(아래 주의사항).

## 전체 동작 흐름

```
[질문] "Alice는 어떤 책을 빌려 갔나요?"
    ↓ create_sql_query_chain (스키마 자동 주입)
[SQL] SELECT b.title FROM books b JOIN rentals r ON b.id = r.book_id
      JOIN members m ON m.id = r.member_id WHERE m.name = 'Alice'
    ↓ clean_sql()  — 코드펜스/접두어 제거
    ↓ 가드레일 검사 — SELECT인가? 허용 테이블인가?
    ↓ db.run(sql)
[결과] [('The Catcher in the Rye',)]
    ↓ (선택) LLM이 자연어로 정리
[답변] "Alice님은 'The Catcher in the Rye'를 대여하셨습니다."
```

## 주요 코드 구성

| 파일 | 역할 |
|---|---|
| `Day2/Day02_04_SQLDB연동.ipynb` | DB 생성 → 체인 구성 → SQL 실행 → Pandas Agent |
| `Day2/library.db` | 실습용 SQLite (books / members / rentals 3테이블) |
| `Day2/개인실습/Day02_03_SQLDB연동_개인실습.ipynb` | 빈칸 실습본 |
| `바이브코딩실습/Day02_05_지점실적질문봇/app.py` | 자연어→SQL 데모 (**SELECT만 허용**) |

## 실행 방법

```bash
pip install -U langchain langchain-classic langchain-openai langchain-community \
    langchain-experimental pandas matplotlib tabulate
$env:OPENAI_API_KEY = "sk-..."

cd 바이브코딩실습/Day02_05_지점실적질문봇 && streamlit run app.py
```

## 핵심 코드

### 1. DB 연결과 스키마 확인

```python
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0)   # SQL 생성은 temperature=0
db = SQLDatabase.from_uri("sqlite:///library.db")
print(db.get_usable_table_names())    # ['books', 'members', 'rentals']
```

### 2. 질문 → SQL 변환

```python
from langchain_classic.chains import create_sql_query_chain   # langchain 1.x 경로

query_chain = create_sql_query_chain(llm, db)
response = query_chain.invoke({"question": "Alice는 어떤 책을 빌려 갔나요?"})
```

### 3. SQL 정제 — 실무에서 반드시 필요

모델은 종종 ` ```sql ` 코드펜스나 `SQLQuery:` 접두어, 뒤에 `SQLResult:` 설명까지 붙입니다.

```python
def clean_sql(text):
    sql = text.strip()
    if "sqlquery:" in sql.lower():                       # 접두어 제거
        idx = sql.lower().rfind("sqlquery:")
        sql = sql[idx + len("sqlquery:"):].strip()
    for stop in ("sqlresult:", "answer:"):               # 뒤따르는 설명 제거
        if stop in sql.lower():
            sql = sql[:sql.lower().find(stop)].strip()
    if sql.startswith("```"):                            # 코드펜스 제거
        sql = sql.strip("`").strip()
        if sql.lower().startswith("sql"):
            sql = sql[3:].strip()
    return sql
```

### 4. 가드레일 — 교안이 정한 규칙: **SELECT만 허용** ⭐

원본 노트북에는 이 검증 함수가 명시적으로 없습니다. 실습 앱과 실전프로젝트 가이드가 "SELECT만"을 요구하므로, **재사용할 때 반드시 아래 형태의 검증을 추가하세요.**

```python
import re

ALLOWED_TABLES = {"books", "members", "rentals"}

def validate_sql(sql: str) -> str:
    s = sql.strip().rstrip(";")
    low = s.lower()

    # ① SELECT로 시작하지 않으면 거부
    if not low.startswith("select"):
        raise ValueError("SELECT 문만 허용됩니다.")

    # ② 변경·삭제 키워드 차단
    for kw in ("insert", "update", "delete", "drop", "alter", "create",
               "truncate", "attach", "pragma"):
        if re.search(rf"\b{kw}\b", low):
            raise ValueError(f"허용되지 않은 키워드: {kw}")

    # ③ 세미콜론으로 여러 문장을 이어 붙이는 우회 차단
    if ";" in sql.strip().rstrip(";"):
        raise ValueError("다중 구문은 허용되지 않습니다.")

    # ④ 결과 폭주 방지
    if "limit" not in low:
        s += " LIMIT 100"
    return s

sql_query = validate_sql(clean_sql(response))
print(db.run(sql_query))
```

> 코드 레벨 방어에 더해 **DB 계정 자체를 읽기 전용으로** 만드는 것이 가장 확실합니다. 실전프로젝트 가이드도 "사내 DW 읽기 전용 계정"을 요구합니다.

### 5. Pandas DataFrame Agent (표 데이터 분석)

```python
from langchain_experimental.agents import create_pandas_dataframe_agent

agent = create_pandas_dataframe_agent(
    llm=ChatOpenAI(model='gpt-4.1-mini', temperature=0),
    df=data,
    verbose=True,
    agent_type='tool-calling',
    allow_dangerous_code=True,          # ⚠️ 아래 주의사항 필독
)
agent.invoke("year와 value의 분포를 산점도로 그려주세요")
```

## 주의사항

- 🚨 **`allow_dangerous_code=True` 는 임의 파이썬 코드 실행을 허용합니다.** LLM이 생성한 코드가 파일 삭제·네트워크 접근을 할 수 있습니다. 신뢰할 수 없는 입력이 들어오는 환경(고객 대면 서비스)에서는 **절대 쓰지 마세요.** 교육·내부 분석 용도로만, 격리된 환경에서 쓰세요.
- 🚨 **SQL 인젝션 관점**: 사용자가 "테이블을 지워줘"라고 하면 LLM이 순순히 `DROP TABLE`을 만들 수 있습니다. 위 `validate_sql` 같은 검증 없이 `db.run()`을 호출하지 마세요.
- **`create_sql_query_chain` import 경로가 langchain 1.x에서 `langchain_classic.chains`로 이동**했습니다. 구버전 예제를 그대로 쓰면 ImportError가 납니다.
- **테이블·컬럼이 많으면 스키마가 프롬프트를 다 차지합니다.** 실무 DW는 수백 테이블이므로, 사용할 테이블만 `include_tables=[...]` 로 좁혀야 합니다.
- **컬럼명이 한글이거나 의미가 불명확하면 정확도가 급락합니다.** 뷰(View)를 만들어 의미가 드러나는 이름으로 노출하는 것이 실무 요령입니다.
- **노트북 상단의 `!sudo apt-get install -y fonts-nanum` 은 Colab 전용**입니다. Windows 로컬에서는 실행되지 않으며, 대신 `plt.rc('font', family='Malgun Gothic')` 을 씁니다.
- `data.csv` 를 `encoding='cp949'` 로 읽습니다. 파일 인코딩이 다르면 `UnicodeDecodeError`가 납니다.

## 관련 문서

- [[개념_Tool_Calling]] — DB 조회를 도구 하나로 감싸는 방법
- [[개념_PII보호와_거버넌스]] — 조회 권한(RBAC)과 감사 로그
- [[프로젝트_실전과제_가이드]] — 프로젝트 3번(사내데이터 분석 Agent)이 이 주제
- [[개념_의도분류_기반_RAG_Agent]] — SQL과 RAG를 함께 쓰는 라우팅

## 출처

- Source: `Day2/Day02_04_SQLDB연동.ipynb`, `Day2/library.db`
- Source: `Day2/Day02_04_교안_Text-to-SQL_실습__자연어로_데이터베이스에_질문하기.pdf`
- Source: `바이브코딩실습/Day02_05_지점실적질문봇/app.py`
- Source: `실전프로젝트/03_사내데이터_분석Agent.md` (권한 가드레일 요구사항)
