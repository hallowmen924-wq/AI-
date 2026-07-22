# 개념: 로컬 LLM과 내부망(온프레미스) 구축

## 한눈에 보기

지금까지의 실습은 외부 API(OpenAI)를 호출했습니다. 하지만 은행 내부망에서는 **데이터가 밖으로 나갈 수 없고, 인터넷도 안 되는** 경우가 많습니다.
좋은 소식은 **코드가 거의 안 바뀐다**는 점입니다 — `ChatOpenAI → ChatOllama`, `OpenAIEmbeddings → OllamaEmbeddings`. 진짜 어려운 건 코드가 아니라 **패키지·모델 반입, 권한, 승인**입니다.

## 핵심 개념

| 운영 방식 | 설명 | 트레이드오프 |
|---|---|---|
| **외부 LLM API** | 승인된 OpenAI/Azure/Claude 사용 | 구축 간단 / **입력 데이터가 외부로 전송** |
| **내부망 오픈소스 LLM** | 사내 GPU 서버에 직접 설치 | 데이터 유출 없음 / GPU·운영 인력 필요 |
| **사내 LLM 플랫폼** | 기관이 이미 구축한 플랫폼 사용 | 가장 현실적 / 기능·모델 선택 제약 |

| 도구 | 역할 |
|---|---|
| **Hugging Face Hub** | "모델계의 GitHub". `snapshot_download`로 모델 파일 확보 |
| **`HF_HUB_OFFLINE=1`** | 캐시만으로 모델 로드. **오프라인 동작 실증용** |
| **`pipeline`** | 전처리 → 추론 → 후처리를 하나로 묶은 최상위 API |
| **`sentence-transformers`** | 로컬 임베딩 모델 (RAG의 심장) |
| **Ollama** | 로컬 LLM을 API 서버 형태로 제공. `langchain-ollama`로 연결 |
| **vLLM** | OpenAI 호환 API를 제공하는 고성능 서빙 — **코드 변경 없이 전환 가능** |

**실습 모델 3종 (전부 소형, CPU에서도 동작)**

| 모델 | 용도 |
|---|---|
| `snunlp/KR-FinBert-SC` | 한국어 금융 감성 분류 |
| `intfloat/multilingual-e5-small` | 다국어 임베딩 (RAG 검색) |
| `Qwen2.5-0.5B-Instruct` | 로컬 생성 LLM |

## 전체 동작 흐름

### 내부망 반입 절차 4단계
```
① 외부망에서 확보    pip download / snapshot_download
② 보안 검토·승인     라이선스·출처·버전·파일 크기 확인      ← 여기가 2~5일 걸립니다
③ 내부망 반입        wheelhouse / 모델 캐시 디렉터리 복사
④ 오프라인 실행 검증  HF_HUB_OFFLINE=1 로 실증
```

### 완전 로컬 RAG
```
[질문] → HuggingFaceEmbeddings(multilingual-e5-small)
      → InMemoryVectorStore 검색
      → apply_chat_template로 프롬프트 구성
      → HuggingFacePipeline(Qwen2.5-0.5B)
      → 답변            ← API 키 없음, 네트워크 없음
```

## 주요 코드 구성

| 파일 | 역할 |
|---|---|
| `LocalLLM_실습/HuggingFace실습/Day00_1_HuggingFace_기초_모델놀이터.ipynb` | pipeline, tokenizer, 모델 기초 |
| `LocalLLM_실습/HuggingFace실습/Day00_2_HuggingFace_심화_내부망RAG.ipynb` | **오프라인 모드 + 완전 로컬 RAG (핵심)** |
| `LocalLLM_실습/HuggingFace실습/Day00_3_Ollama_기초_로컬LLM놀이터.ipynb` | Ollama 사용법 |
| `LocalLLM_실습/HuggingFace실습/offline_test.py` | 오프라인 로드 실증 스크립트 |
| `Day1/Day01_02_RAG_아키텍처_설계_Ollama로컬버전.ipynb` | Day1 RAG의 Ollama 버전 |
| `내부망 환경에서 LLM 구축 시 고려사항.txt` | **10개 항목 실무 가이드** |
| `실전프로젝트/00_시작하기.md` | `llm_provider.py` 치환 패턴 + 치환표 |

## 실행 방법

```bash
pip install torch transformers huggingface_hub sentence-transformers accelerate
pip install langchain langchain-huggingface langchain-ollama

# Ollama 사용 시
ollama serve
ollama pull qwen2.5:7b      # LLM
ollama pull bge-m3          # 임베딩
```

## 핵심 코드

### 1. 백엔드 전환 레이어 ⭐⭐ (가장 재사용 가치 높음)

모든 LLM 호출이 여기를 거치게 하면 **환경변수 하나로** OpenAI ↔ Ollama를 바꿀 수 있습니다.

```python
# common/llm_provider.py
import os

BACKEND = os.environ.get("LLM_BACKEND", "openai")   # openai | ollama

def get_llm(temperature: float = 0.0):
    if BACKEND == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.environ.get("LLM_MODEL", "qwen2.5:7b"), temperature=temperature)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.environ.get("LLM_MODEL", "gpt-4.1"), temperature=temperature)

def get_embeddings():
    if BACKEND == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="bge-m3")
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")
```

> ⚠️ **임베딩을 바꾸면 벡터DB를 다시 만들어야 합니다.** 프로젝트 시작 시 하나로 정하고 끝까지 가세요.

### 2. 모델 다운로드와 캐시 위치

```python
from huggingface_hub import snapshot_download

local_path = snapshot_download(repo_id="snunlp/KR-FinBert-SC", revision="main")
print(local_path)
# 캐시 위치: HF_HOME 환경변수 (기본 ~/.cache/huggingface/hub)
#            models--snunlp--KR-FinBert-SC/ 형태로 저장됩니다
```

이 캐시 디렉터리를 통째로 내부망에 복사하면 반입이 끝납니다.

### 3. 오프라인 모드 실증 ⭐

내부망에서 진짜 동작하는지 **외부망에서 미리 확인**하는 방법입니다.

```python
# offline_test.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"          # ← 네트워크 접근 차단

from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBert-SC")
print("[성공] 오프라인 모드에서 캐시만으로 모델 로드 완료!")
```

경로를 직접 지정하는 방식도 있습니다:

```python
model = SentenceTransformer("./models/bge-m3", local_files_only=True)
```

### 4. 패키지 오프라인 반입

```bash
# 외부망에서
pip download -r requirements.txt -d wheelhouse

# 내부망에서
pip install --no-index --find-links=./wheelhouse -r requirements.txt
```

버전은 반드시 고정합니다:
```text
pandas==2.2.3
fastapi==0.115.0
langchain-core==0.3.15
```

Docker를 쓴다면:
```bash
docker pull python:3.11-slim
docker save python:3.11-slim -o python-image.tar     # 외부망
docker load -i python-image.tar                       # 내부망
```

### 5. `pipeline` — 가장 쉬운 사용법

```python
from transformers import pipeline

clf = pipeline("text-classification", model="snunlp/KR-FinBert-SC", device=device)
for news, result in zip(financial_news, clf(financial_news)):
    print(f"[{result['label']:>8s}] ({result['score']:.3f}) {news}")
```

`pipeline` = 토크나이저 + 모델 + 후처리의 포장입니다. 분해하면 `tokenizer(text) → model(**inputs).logits → softmax` 3단계입니다.

### 6. 로컬 임베딩과 의미 검색

e5 계열 모델은 **접두어 규칙**이 있습니다 — 문서는 `passage: `, 질문은 `query: `.

```python
from sentence_transformers import SentenceTransformer

emb_model = SentenceTransformer("intfloat/multilingual-e5-small", device=device)
doc_vecs = emb_model.encode([f"passage: {d}" for d in faq_docs], normalize_embeddings=True)
q_vec = emb_model.encode(f"query: {query}", normalize_embeddings=True)
scores = doc_vecs @ q_vec        # 정규화했으므로 내적 = 코사인 유사도
```

### 7. 로컬 LLM 생성 — 채팅 템플릿

```python
llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
llm_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype="auto").to(device)

messages = [
    {"role": "system", "content": "당신은 KB은행의 친절한 금융 상담사입니다. 두 문장 이내로 답하세요."},
    {"role": "user", "content": "예금자보호제도가 뭔가요?"},
]
prompt_text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

generated = llm_model.generate(**llm_tokenizer(prompt_text, return_tensors="pt").to(device),
                               max_new_tokens=120,
                               do_sample=False,                       # ← 결정적 생성
                               pad_token_id=llm_tokenizer.eos_token_id)
```

> ✅ **`do_sample=False`(탐욕적 생성)** 는 같은 입력 → 같은 출력입니다. **금융 상담·규정 안내는 일관성이 우선**이므로 결정적 생성이 기본입니다. 창의적 답변이 필요할 때만 `do_sample=True, temperature=0.7`.

### 8. LangChain 연동 — 완전 로컬 RAG ⭐

```python
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.vectorstores import InMemoryVectorStore
from transformers import pipeline as hf_pipeline

local_embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True, "prompt": "passage: "},
    query_encode_kwargs={"normalize_embeddings": True, "prompt": "query: "},
)
vectorstore = InMemoryVectorStore.from_texts(faq_docs, embedding=local_embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

gen_pipe = hf_pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer,
                       max_new_tokens=150, do_sample=False, return_full_text=False)
local_llm = HuggingFacePipeline(pipeline=gen_pipe)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | local_llm | StrOutputParser()
)
```

**LCEL 체인 구조가 그대로**입니다. 기존 Agent·RAG 코드는 LLM/임베딩 객체 선언부만 바꾸면 내부망용으로 전환됩니다. → [[개념_RAG_파이프라인]]

## 실무 치환표

| 오늘 쓴 것 | 사내망에서 | 난이도 |
|---|---|---|
| `ChatOpenAI` | `ChatOllama` / vLLM(OpenAI 호환 → **코드 그대로**) | 쉬움 |
| `OpenAIEmbeddings` | `OllamaEmbeddings("bge-m3")` | 쉬움 |
| 웹 URL 문서 | 사내 파일서버 (HWP·스캔 PDF) | **어려움** |
| SQLite | 사내 DW **읽기 전용 계정** | 보통 |
| LangSmith (외부 SaaS) | **Langfuse self-host** (사내 Docker) | 보통 |
| `pip install` | 사내 PyPI 미러 / 오프라인 wheel | **어려움** |
| 웹 검색 도구 | 사내 검색 또는 제거 | 보통 |

### 진짜 오래 걸리는 것 (코드가 아닙니다)

| | 강의실 | 사내망 |
|---|---|---|
| 패키지·모델 반입 | 30초 | **2~5일** (보안 검토·승인) |
| 문서 확보 | 즉시 | **1~2주** (소유 부서 협의) |
| DB 조회 권한 | 즉시 | **1주+** (정보보호부 승인) |

> 💡 실제 프로젝트라면 **첫날에 반입 신청부터 넣고**, 승인 대기 중에 개발하세요. 이 순서를 모르면 3주짜리가 6주가 됩니다.

## 내부망 구축 10개 항목 요약

1. 환경 확인 — 인터넷 가능 여부, OS/Python, GPU/CUDA, 외부 API 허용 여부, 반입 절차
2. 운영 방식 결정 — 외부 API / 내부 오픈소스 / 사내 플랫폼
3. 패키지 사전 준비 — `pip download` → wheelhouse, 버전 고정
4. 모델 파일 사전 반입 — 라이선스·출처·버전·크기 확인
5. (옵션) Docker 이미지 사전 저장
6. 데이터 보안·권한 설계 — `사용자 인증 → 문서 접근 권한 확인 → 허용된 문서만 검색 → 답변`
7. 작은 PoC부터 — 업로드 → 분할 → 검색 → 생성 → 출처 → 권한 순으로 단계 확장
8. RAG는 검색 품질이 핵심 → [[개념_검색품질_개선]]
9. Agent 실행 권한 제한 — `제안 → 입력검증 → 사용자 승인 → 허용 도구 실행 → 기록` → [[개념_HITL과_검증루프]]
10. 장애 대비 — GPU 장애→소형 모델, LLM 장애→검색 결과만 제공 → [[개념_Fallback과_Rollback]]

## 주의사항

- **소형 모델은 도구 호출과 긴 컨텍스트에 약합니다.** Qwen2.5-0.5B 같은 모델로 Multi-Agent를 돌리면 실패율이 높습니다. LangGraph 실습은 최소 7B급 이상을 권합니다 → [[개념_조건부Edge와_ToolNode]]
- **모델 라이선스를 반드시 확인하세요.** 상업적 이용 불가 모델이 있습니다. 교안도 "모델·패키지 라이선스 상업적 이용 가능(법무 확인)"을 배포 체크리스트에 넣고 있습니다.
- **`InMemoryVectorStore` 는 실습용**입니다. 프로세스가 끝나면 사라집니다. 내부망 운영에는 Chroma persist나 사내 벡터DB를 쓰세요.
- **GPU 없이 CPU로 돌리면 매우 느립니다.** 실습 모델이 전부 소형인 이유가 이것입니다. 실서비스 규모라면 GPU와 양자화(quantization) 검토가 필요합니다.
- **`HF_HUB_OFFLINE=1` 은 import 전에 설정해야 합니다.** transformers를 먼저 import하면 적용되지 않습니다.
- **Ollama는 별도 프로세스**입니다. `ollama serve` 가 떠 있지 않으면 연결 오류가 납니다.
- **Day1 Ollama 버전 노트북**은 같은 RAG를 로컬로 구현한 것이므로, 최신 버전과 나란히 비교하면 치환 지점을 정확히 볼 수 있습니다.

## 관련 문서

- [[개념_RAG_파이프라인]] — 치환 대상이 되는 기본 구조
- [[개념_트레이싱과_감사로그]] — Langfuse self-host
- [[개념_PII보호와_거버넌스]] — 데이터가 밖으로 안 나가는 이유
- [[개념_Fallback과_Rollback]] — 내부망 장애 대비
- [[프로젝트_실전과제_가이드]] — 실무 전환 체크리스트
- [[MOC_금융_AI]]

## 출처

- Source: `LocalLLM_실습/HuggingFace실습/Day00_1~3_*.ipynb`, `offline_test.py`, `README.md`
- Source: `내부망 환경에서 LLM 구축 시 고려사항.txt`
- Source: `실전프로젝트/00_시작하기.md` (llm_provider.py, 치환표)
- Source: `Day1/Day01_02_RAG_아키텍처_설계_Ollama로컬버전.ipynb`
