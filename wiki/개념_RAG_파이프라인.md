# 개념: RAG 파이프라인 구축

## 한눈에 보기

RAG(Retrieval-Augmented Generation)는 LLM이 **미리 준비한 문서에서 근거를 찾아** 답하게 만드는 구조입니다.
파이프라인은 항상 같은 6단계 — 로드 → 분할 → 임베딩 → 벡터DB 저장 → 검색 → 프롬프트 결합 — 이며, 이 문서는 그 뼈대와 답변 품질 평가 방법까지 정리합니다.

## 핵심 개념

| 용어 | 뜻 | 왜 필요한가 |
|---|---|---|
| **Document Loader** | PDF·웹·CSV를 `Document` 객체로 변환 | LLM은 파일을 직접 못 읽음 |
| **Text Splitter** | 긴 문서를 조각(chunk)으로 자름 | 프롬프트 길이 제한 + 검색 정확도 |
| **Embedding** | 문장 → 숫자 벡터 | "의미가 가까움"을 계산 가능하게 |
| **Vector Store** | 벡터를 저장하고 최근접 검색 | 여기서는 Chroma |
| **Retriever** | 질문에 가까운 조각 k개 반환 | 체인에 꽂을 수 있는 표준 인터페이스 |

**핵심 직관**: RAG는 "모델을 똑똑하게 만드는 것"이 아니라 "모델에게 정답이 적힌 쪽지를 쥐여 주는 것"입니다. 그래서 답이 이상하면 원인은 대부분 LLM이 아니라 **검색**입니다. → [[개념_검색품질_개선]]

## 전체 동작 흐름

```
[PDF/웹/CSV]
   ↓ Loader                        PyPDFLoader / PyMuPDFLoader / WebBaseLoader / CSVLoader
[Document 리스트]
   ↓ RecursiveCharacterTextSplitter (chunk_size, chunk_overlap)
[조각 리스트]
   ↓ OpenAIEmbeddings              text-embedding-3-small / -large
[벡터]  →  Chroma에 저장

--- 질문이 들어오면 ---
[질문] → 같은 임베딩 모델로 벡터화 → Chroma 최근접 k개 검색
      → format_docs()로 텍스트 합치기 → 프롬프트의 {context}에 주입
      → LLM → 답변(+ 출처 페이지)
```

## 주요 코드 구성

| 파일 | 역할 |
|---|---|
| `Day1/Day01_02_RAG_아키텍처_설계_최신버전.ipynb` | 웹 문서 기반 RAG 최소 구현 + 평가 지표 |
| `Day1/Day01_02_RAG_아키텍처_설계_Ollama로컬버전.ipynb` | 같은 구조를 로컬 모델로 (→ [[개념_LocalLLM_내부망_구축]]) |
| `Day2/Day02_01_금융문서RAG구현_검색품질개선.ipynb` 1부 | PDF 기반 RAG + 페이지 출처 표기 + 요약 전략 |
| `바이브코딩실습/Day01_02_금융문서질문봇/app.py` | Streamlit 데모 |

## 실행 방법

```bash
pip install -U langchain langchain-openai langchain-community langchain-core \
    langchain-text-splitters langchain-chroma chromadb openai tiktoken pymupdf beautifulsoup4

$env:OPENAI_API_KEY = "sk-..."
# Chroma 텔레메트리 끄기(선택)
$env:ANONYMIZED_TELEMETRY = "False"
```

## 핵심 코드

### 1. 최소 RAG 파이프라인 (웹 문서)

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1) 로드 — bs_kwargs로 본문 영역만 파싱하면 광고·메뉴 노이즈가 줄어듭니다
loader = WebBaseLoader(
    web_paths=("https://example.com/article",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_="article_view")),
)
docs = loader.load()

# 2) 분할
splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

# 3~4) 임베딩 + 벡터DB
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="rag_lecture",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 5) 프롬프트 — "모르면 모른다고 하라"가 환각 방지의 최소 방어선
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=("다음 문맥만 사용하여 질문에 답하십시오. 답을 모르면 모른다고 말하십시오.\n"
              "질문: {question}\n문맥: {context}\n답변:"),
)

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

# 6) LCEL 체인 — 질문 문자열 하나가 {context}와 {question} 두 자리로 갈라집니다
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4.1", temperature=0.1)
    | StrOutputParser()
)
print(rag_chain.invoke("이 글에서 vector store를 만들 때 사용한 라이브러리는?"))
```

### 2. 페이지 출처를 함께 넘기기 (금융문서 필수)

금융 답변은 "어디에 그렇게 쓰여 있는가"를 반드시 제시해야 합니다.
분할 시 metadata에 페이지 번호를 심고, `format_docs`에서 그걸 텍스트에 함께 넣습니다.

```python
from langchain_core.documents import Document

split_docs = []
for i, page in enumerate(pages):                     # PyMuPDFLoader로 로드한 페이지들
    for j, chunk in enumerate(text_splitter.split_text(page.page_content)):
        split_docs.append(Document(
            page_content=chunk,
            metadata={"page_number": i + 1, "chunk_id": j + 1},
        ))

def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata.get('page_number', '?')}페이지] {doc.page_content}" for doc in docs
    )
```

시스템 프롬프트에 "3. 원본 출처 페이지수도 같이 출력하시오"를 명시하면 답변에 페이지가 표기됩니다.

### 3. 자가 검증 체인 — 초안을 한 번 더 검토

같은 근거 문서를 두 번 사용해서, 초안을 만들고 그 초안을 다시 검토하게 합니다.
LangGraph 없이 LCEL만으로 구현한 "검증 루프의 원시 버전"입니다. → [[개념_HITL과_검증루프]]

```python
verify_chain = (
    {
        "question": RunnablePassthrough(),
        "initial_answer": initial_chain,          # 초안 생성 체인을 그대로 입력으로
        "summaries": retriever | format_docs,
    }
    | verify_prompt      # "질문에 정확히 답했는가 / 문서와 일치하는가"를 점검하는 프롬프트
    | llm
    | StrOutputParser()
)
```

### 4. 검색 결과가 너무 길 때 — 3가지 요약 전략

| 전략 | 방식 | 언제 |
|---|---|---|
| **Stuff** | 전부 이어붙여 한 번에 요약 | 조각이 적고 짧을 때. 가장 빠름 |
| **Refine** | 첫 조각 요약 → 다음 조각으로 계속 보완 | 순서가 중요한 문서. 순차라 느림 |
| **Map-Reduce** | 조각별 요약(`chain.batch`) → 요약들을 다시 통합 | 조각이 많을 때. 병렬이라 빠름 |

```python
# Map-Reduce의 핵심은 batch() 한 줄 — 조각별 요약이 병렬로 실행됩니다
map_summaries = map_chain.batch([{"text": doc.page_content} for doc in relevant_docs])
reduce_summary = reduce_chain.invoke({"text": "\n---\n".join(map_summaries)})
```

### 5. 답변 품질 평가 — LLM-as-a-Judge

BLEU/ROUGE 같은 고전 지표는 표현이 달라지면 점수가 무너집니다. 그래서 LLM에게 채점을 시키고, 점수를 **구조화 출력**으로 받습니다.

```python
from pydantic import BaseModel, Field

class GradeResult(BaseModel):
    score: int = Field(description="0에서 100 사이의 평가 점수", ge=0, le=100)
    reason: str = Field(description="점수를 준 근거를 한 문장으로 설명")

grader = ChatOpenAI(model="gpt-4.1", temperature=0.1).with_structured_output(GradeResult)
```

4대 관점: **Document Relevance**(검색 문서가 질문과 관련 있나) / **Answer Faithfulness**(답변이 문서에 근거하나) / **Answer Helpfulness**(도움이 되나) / **Answer Correctness**(정답과 맞나).
자세한 평가 설계는 → [[개념_Agent_평가]]

## 주의사항

- **`chunk_size`/`chunk_overlap`은 검색 정확도의 출발점입니다.** 너무 크면 관련 없는 내용이 섞이고, 너무 작으면 문맥이 끊깁니다. overlap은 경계에 걸친 내용을 놓치지 않게 해 재현율을 올립니다. 실험 방법 → [[개념_검색품질_개선]]
- **적재 시와 검색 시 임베딩 모델이 같아야 합니다.** 모델을 바꾸면 벡터DB를 처음부터 다시 만들어야 합니다.
- **`text-embedding-3-small` vs `-large`**: small은 저렴·빠름, large는 정확도 우선. 교안은 상황에 따라 둘 다 씁니다.
- **문서 유형을 무시하고 글자 수로만 자르면 안 됩니다.** 약관 조항이 잘리고 FAQ의 질문·답변이 분리됩니다 → [[개념_문서유형별_전처리]]
- **PDF 경로가 Colab 기준(`/content/test.pdf`)** 인 노트북이 있습니다. 로컬 실행 시 상대 경로로 바꾸세요.
- **`WebBaseLoader` 사용 시 `USER_AGENT` 환경변수**를 설정하지 않으면 경고가 뜹니다: `os.environ["USER_AGENT"] = "kb-rag-lecture/1.0"`
- **PII가 섞인 문서를 그대로 적재하면 검색으로 누구에게나 노출됩니다.** 적재 전 마스킹은 선택이 아니라 필수입니다 → [[개념_PII보호와_거버넌스]]

## 관련 문서

- [[개념_LangChain_LCEL_기초]] — 이 문서의 체인 문법 전제
- [[개념_검색품질_개선]] — 답이 이상할 때 가장 먼저 볼 곳
- [[개념_문서유형별_전처리]] — 적재 전 단계
- [[개념_GraphRAG_Vector_융합]] — 관계가 걸린 질문에 대한 확장
- [[개념_의도분류_기반_RAG_Agent]] — 여러 DB를 쓰는 RAG

## 출처

- Source: `Day1/Day01_02_RAG_아키텍처_설계_최신버전.ipynb`
- Source: `Day2/Day02_01_금융문서RAG구현_검색품질개선.ipynb` (1부)
- Source: `바이브코딩실습/Day01_02_금융문서질문봇/app.py`
