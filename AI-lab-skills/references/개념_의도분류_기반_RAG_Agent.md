# 개념: 의도 분류 기반 RAG Agent (다중 DB)

## 한눈에 보기

실무 RAG는 하나의 벡터DB로 끝나지 않습니다. 사내 규정집·직원 정보·문서 양식처럼 **성격이 다른 DB를 각각 만들고**, 질문이 들어오면 어느 DB를 봐야 하는지 먼저 판단해야 합니다.
Part 1은 "분류 → 1회 검색 → 답변", Part 2는 "**분류 → 검색 → 초안 → 평가 → 추가 검색**" 반복 루프입니다. Part 2가 이 교안에서 가장 완성도 높은 RAG Agent입니다.

## 핵심 개념

| 개념 | 설명 |
|---|---|
| **의도 분류(Intent Classification)** | 질문이 어떤 종류인지 먼저 판단. 콜센터 상담원이 "대출 문의인지 계좌 문의인지"를 먼저 파악하는 것과 같음 |
| **retriever_map** | `{"policy": ..., "user": ..., "form": ...}` — 의도 → 리트리버 매핑 dict |
| **다중 검색 루프** | 한 번 검색으로 부족하면 평가 노드가 "추가 검색 필요"를 판정해 되돌아감 |
| **draft(초안)** | 검색할 때마다 답변을 덮어쓰지 않고 **보완**해 나감 |
| **retrieval_count** | `Annotated[int, operator.add]` 로 누적. 상한(3회)이 무한 루프를 막음 |

**Part 1 vs Part 2**

| | Part 1 | Part 2 |
|---|---|---|
| 검색 횟수 | 1회 | 최대 3회 |
| 분류 출력 | `intent` 만 | `retrieval_needed`, `target_db`, `query`(재작성된 검색어) |
| 답변 | 한 번에 생성 | 초안 → 평가 → 보완 반복 |
| 못 푸는 질문 | "휴가 신청서 작성해줘" (양식+직원정보 둘 다 필요) | 해결됨 |

**왜 Part 1이 실패하는가**: "야 나 김지훈인데, 휴가 신청서 작성해줘"는 **양식 DB(form)** 에서 서식을 가져오고 **직원 DB(user)** 에서 소속·직급을 가져와야 합니다. 한 번 검색으로는 불가능합니다.

## 전체 동작 흐름 (Part 2)

```
START → classify_intent
          │ retrieval_needed / target_db / query 를 구조화 출력으로 결정
          ↓
       retrieve_context  ← ─────────────┐   retrieval_count += 1
          │ 하이브리드 검색(BM25+Vector)  │
          ↓                              │
       generate_response                 │   draft를 "덮어쓰지 않고 보완"
          ↓                              │
       evaluate ──should_continue──┬─ 추가 검색 필요 ─┘
                                   ├─ retrieval_count >= 3 → END   (상한)
                                   ├─ error → handle_error → END
                                   └─ 충분함 → END
```

## 주요 코드 구성

| 파일 | 역할 |
|---|---|
| `Day3/Day03_06_통합Agent구현_의도분류기반RAG1.ipynb` | Part 1 — 분류 + 단일 검색 |
| `Day3/Day03_07_통합Agent구현_의도분류기반RAG2.ipynb` | **Part 2 — 반복 검색 루프 (핵심)** |
| `Day3/rag/company_policies.pdf` | 사내 규정집 (policy DB) |
| `Day3/rag/employee_data.csv` | 직원 정보 (user DB) |
| `Day3/rag/templates/*.md` | 문서 양식 6종 (form DB) |
| `Day3/chroma_db_*/` | 실행 시 생성되는 벡터DB (UUID 접미사) |
| `바이브코딩실습/Day03_06_의도분류상담봇/app.py` | Streamlit 데모 |

## 실행 방법

```bash
pip install -qU langchain langchain-core langchain-openai langchain-community langchain-classic \
    langchain-chroma chromadb langchain-text-splitters langgraph pymupdf python-dotenv kiwipiepy rank_bm25
$env:OPENAI_API_KEY = "sk-..."

# Colab에서는 설치 후 반드시 "런타임 → 세션 다시 시작"
cd 바이브코딩실습/Day03_06_의도분류상담봇 && streamlit run app.py
```

## 핵심 코드

### 1. 구조화 출력으로 분류 + 검색어 생성 ⭐

한 번의 LLM 호출로 **네 가지**를 동시에 결정합니다. 이 스키마가 Part 2의 심장입니다.

```python
from pydantic import BaseModel, Field
from typing import Literal

class IntentResult(BaseModel):
    explanation: str                                       # 판단 근거(디버깅용)
    retrieval_needed: bool = Field(description='DB 검색이 필요한지의 여부')
    target_db: Literal["policy", "user", "form"] = Field(
        description="""사용자 질의를 해결하기 위해, 추가로 검색해야 하는 DB의 종류:
policy: 사내 규정집
user: 직원 정보
form: 문서 양식 모음""")
    query: str = Field(description='벡터 데이터베이스에 검색할 쿼리')   # ← 질문을 검색어로 재작성
```

> 💡 `query` 필드가 중요합니다. 사용자의 구어체 질문("야 나 김지훈인데...")을 그대로 검색하면 결과가 나쁩니다. LLM이 검색에 적합한 형태로 다시 씁니다 (query rewriting).

### 2. State — 누적 카운터

```python
from typing import Annotated, List, Optional
import operator

class State(TypedDict):
    question: str
    target_db: str
    query: str
    retrieval_needed: bool
    context: List[str]
    draft: str
    error: Optional[str]
    retrieval_count: Annotated[int, operator.add]    # ← 노드가 1을 반환하면 누적
```

### 3. DB별 하이브리드 리트리버 구성 ⭐

**DB 성격에 따라 chunk 크기와 BM25 가중치를 다르게** 준 점이 핵심입니다.

```python
from kiwipiepy import Kiwi
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

kiwi = Kiwi()
def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text)]

# ① policy — 규정집 PDF: 긴 조문이라 chunk 3000, 균형 가중치
policy_retriever = EnsembleRetriever(
    retrievers=[
        BM25Retriever.from_documents(split_docs, preprocess_func=kiwi_tokenize),  # k=5
        policy_vectorstore.as_retriever(search_kwargs={"k": 5}),
    ],
    weights=[0.5, 0.5],
)

# ② user — 직원 CSV: 이름·스킬 등 고유명사 매칭이 중요 → BM25 비중 0.7
user_retriever = EnsembleRetriever(
    retrievers=[user_bm25_retriever, user_vector_retriever],
    weights=[0.7, 0.3],
)

# ③ form — 양식 md: 분할 없이 파일 하나 = 문서 하나
retriever_map = {"policy": policy_retriever, "user": user_retriever, "form": form_retriever}
```

CSV 인코딩 대응도 실무적으로 유용합니다:

```python
try:
    docs = CSVLoader(user_file, encoding='utf-8').load()
except (UnicodeDecodeError, RuntimeError):
    docs = CSVLoader(user_file, encoding='cp949').load()      # 한국 엑셀 대응
```

### 4. 검색 노드 — 카운터 증가

```python
def retrieve_context(state: State) -> dict:
    retriever = retriever_map.get(state['target_db'])
    if not retriever:
        return {"error": f"Unknown DB: {state['target_db']}"}
    results = retriever.invoke(state['query'])
    return {
        "context": [doc.page_content for doc in results],
        "retrieval_count": 1,                          # ← 리듀서가 누적
    }
```

### 5. 초안 "보완" 방식 생성 ⭐

덮어쓰지 않고 이전 초안 + 새 정보로 개선합니다. 이래야 여러 DB 결과가 하나의 답변에 합쳐집니다.

```python
def generate_response(state: State) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """사용자의 [질의]와 이에 대한 [중간 답변], 그리고 [추가 정보]가 주어집니다.
이를 활용하여, 중간 답변을 개선하세요.
추가 정보에 포함된 내용만을 사용해 개선하세요.
개선된 답변만 출력하고, 질의에 관련된 내용만 개선하세요."""),
        ("user", "[질의]: {question}\n---\n[중간 답변]: {draft}\n---\n[추가 정보]:{context}")
    ])
    draft = (prompt | llm).invoke({...})
    return {"draft": draft.content}
```

### 6. 평가 노드 — 추가 검색 판단

같은 `IntentResult` 스키마를 재사용해 "다음에 어느 DB를 뭘로 검색할지"까지 한 번에 받습니다.

```python
def evaluate(state: State) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 문서 작성 자동화를 돕는 에이전트입니다.
현재의 답변에서는 추가적인 DB 검색을 통해 채울 수 있는 내용이 들어있을 수 있습니다.
비어 있는 부분을 채우기 위해 다른 DB 검색을 통해 보완해야 하는지 판별하세요.
이미 충분한 정보가 주어진 경우에는, 추가로 검색할 필요가 없습니다."""),
        ("user", "질의: {question}\n[이전 검색 DB]: {target_db}\n[이전 검색 결과]: {context}\n현재 답변: {draft}")
    ])
    result = (prompt | llm.with_structured_output(IntentResult)).invoke({...})
    return {"retrieval_needed": result.retrieval_needed,
            "target_db": result.target_db, "query": result.query}
```

### 7. 라우터 — 상한이 최우선 ⭐

```python
def should_continue(state: State) -> Literal['handle_error', 'END', 'retrieve_context']:
    if state.get('error'):
        return 'handle_error'
    if state.get('retrieval_count', 0) >= 3:      # ← 무한 루프 차단이 먼저
        return 'END'
    if state.get('retrieval_needed'):
        return 'retrieve_context'
    return 'END'

builder.add_conditional_edges('evaluate', should_continue,
    {'END': END, 'handle_error': 'handle_error', 'retrieve_context': 'retrieve_context'})
```

### 8. 실행

```python
result = graph.invoke({'question': "야 나 김지훈인데, 25년 3월 22일부터 26일까지 휴가 신청서 작성해줘.",
                       'draft': '', 'retrieval_count': 0})   # 초기값을 반드시 명시
print(result['draft'])
```

## 주의사항

- **초기 State를 빠뜨리면 KeyError가 납니다.** `draft: ''`, `retrieval_count: 0` 을 반드시 넣으세요.
- **`chroma_db_*` 폴더가 실행할 때마다 UUID 접미사로 새로 생깁니다** (`"./chroma_db_policy_" + str(uuid.uuid4())[:5]`). 실습 중 충돌을 피하려는 의도지만, **디스크에 쓰레기가 쌓입니다.** 재사용 시 고정 경로 + 기존 컬렉션 재사용 로직으로 바꾸세요. 원본 저장소에 이미 6개 폴더가 남아 있습니다.
- ⚠️ **원본 평가 프롬프트에 "최소 2번이상 DB를 검색해서 빈 수치가 없도록 답해줘!" 라는 지시가 있습니다.** 불필요한 검색을 유도해 비용이 늘 수 있습니다. 실무 적용 시 재검토하세요.
- **`retrieval_count >= 3` 상한에 걸려 종료되면 답변이 미완성일 수 있습니다.** 그 상태를 사용자에게 알리는 처리를 추가하는 것이 정직합니다 → [[개념_Agent_통제력과_루프탈출]]
- **`langchain_classic.retrievers`** 경로를 씁니다. langchain 1.x 기준이며 구버전은 `langchain.retrievers` 입니다.
- **`kiwipiepy` 는 빌드가 필요할 수 있습니다.** Windows에서 설치가 실패하면 Visual C++ Build Tools를 확인하거나, BM25 토크나이저 없이 기본 공백 분리로 우회하세요.
- **`rag.zip` 압축 해제 코드**는 Colab 업로드 전제입니다. 로컬에서는 `Day3/rag/` 폴더가 이미 있으므로 건너뜁니다.

## 관련 문서

- [[개념_Graph_구조_패턴]] — Router 패턴의 기초
- [[개념_검색품질_개선]] — 하이브리드 검색과 Kiwi 토크나이저
- [[개념_Agent_통제력과_루프탈출]] — 반복 상한 설계
- [[개념_HITL과_검증루프]] — 답변 검증 강화
- [[프로젝트_실전과제_가이드]] — 프로젝트 1번(여신규정 상담봇)이 이 구조

## 출처

- Source: `Day3/Day03_06_통합Agent구현_의도분류기반RAG1.ipynb`
- Source: `Day3/Day03_07_통합Agent구현_의도분류기반RAG2.ipynb`
- Source: `Day3/rag/` (company_policies.pdf, employee_data.csv, templates/)
- Source: `바이브코딩실습/Day03_06_의도분류상담봇/app.py`, `Day03_05_통합상담Agent/app.py`
