# 개념: GraphRAG + Vector 융합 검색

## 한눈에 보기

Vector DB는 "의미가 비슷한 문서"는 잘 찾지만 **문서 사이의 관계**는 모릅니다. "상품 → 약관 → 수수료 규정 → 면제 예외" 처럼 근거가 사슬로 이어진 질문에서는 반쪽짜리 답이 나옵니다.
지식그래프로 관계를 표현하고, Vector 검색 결과를 출발점으로 **그래프를 따라가며 근거를 보강**하는 것이 융합 검색입니다.

## 핵심 개념

| 구분 | Vector DB | Graph DB | 융합 |
|---|---|---|---|
| 잘하는 것 | 의미 유사도 | 관계 추적(N-hop) | 둘 다 |
| 못하는 것 | 문서 간 관계 | 표현이 다른 질문 매칭 | — |
| 대표 질문 | "노후 대비 상품 추천" | "이 규정의 예외는?" | "중도상환수수료와 면제 조건 전부" |

- **노드(Node)**: 문서 하나 (상품설명서, 약관, 규정, 예외조건)
- **엣지(Edge)**: 문서 사이 관계 (`근거 약관`, `상세 규정`, `예외 조건`)
- **N-hop 확장**: 출발 노드에서 N번 건너간 이웃까지 근거로 수집
- **트리플(Triple)**: `(주체, 관계, 대상)` — LLM이 문장에서 자동 추출해 그래프를 키움

**왜 중요한가 (금융 관점)**: "중도상환수수료 1.2% 부과"만 답하고 "다자녀·청년 면제"를 빠뜨리면 **불완전판매**가 됩니다. 예외 조건은 대부분 별도 문서에 있고, 그 연결은 벡터 유사도로 잡히지 않습니다.

## 전체 동작 흐름

```
[질문] "중간에 갚으면 수수료가 있나요? 면제되는 경우까지 전부"
   ↓
① Vector 검색 (k=1)  →  [KB든든전세자금대출]  ← 1개만 확보
   ↓
② 그래프 확장 (2-hop)
   KB든든전세자금대출 --[근거 약관]--> 전세자금대출약관
                     --[상세 규정]--> 중도상환수수료규정
                     --[예외 조건]--> 수수료면제예외조건      ← 자동 확보
   ↓
③ 문서 + 관계 경로를 함께 프롬프트에 주입
   ↓
[답변] "3년 이내 1.2% 부과. 단, 3자녀 이상·청년 우대형은 면제
        (근거: 중도상환수수료규정, 수수료면제예외조건)"
```

## 주요 코드 구성

| 파일 | 역할 |
|---|---|
| `Day2/Day02_02_GraphRAG_Vector융합_평가자동화.ipynb` | **설치 없이 실행되는 메인 실습** (networkx 사용) |
| `Day2/(심화)_RAG_아키텍처_설계_Neo4j버전.ipynb` | 실무용 Neo4j 버전 (**별도 인스턴스 필요**) |
| `바이브코딩실습/Day02_03_관계검색비교기/app.py` | Vector 단독 vs 융합 비교 데모 |

## 실행 방법

```bash
pip install -U langchain langchain-openai langchain-core langchain-chroma chromadb networkx
$env:OPENAI_API_KEY = "sk-..."
$env:ANONYMIZED_TELEMETRY = "False"
```

메인 노트북은 **`networkx`(파이썬 메모리 내 그래프)** 를 쓰므로 서버·계정 설치가 필요 없습니다.
실무에서 그래프가 커지면 Neo4j 같은 전용 그래프DB로 옮기고 Cypher로 질의합니다.

## 핵심 코드

### 1. 지식그래프 구축

```python
import networkx as nx

NODE_DOCS = {
    "KB든든전세자금대출": "무주택 세대주의 전세 보증금 마련 지원. 대출 기간 최대 2년.",
    "중도상환수수료규정": "대출 실행일로부터 3년 이내 상환 시 상환 금액의 1.2퍼센트 부과.",
    "수수료면제예외조건": "3자녀 이상 다자녀 가구 및 청년 우대형 가입자는 전액 면제.",
    # ...
}

G = nx.DiGraph()
for node_id in NODE_DOCS:
    G.add_node(node_id)
G.add_edge("KB든든전세자금대출", "전세자금대출약관", relation="근거 약관")
G.add_edge("전세자금대출약관", "중도상환수수료규정", relation="상세 규정")
G.add_edge("중도상환수수료규정", "수수료면제예외조건", relation="예외 조건")
```

### 2. N-hop 이웃 확장 (재사용 가치 높음) ⭐

방향 그래프이지만 **양방향 모두 탐색**합니다. "규정 → 예외"뿐 아니라 "예외 → 상위 규정"도 근거로 필요하기 때문입니다.

```python
def graph_neighbors(node_id):
    neighbors = []
    for _, dst, data in G.out_edges(node_id, data=True):      # 나가는 관계
        neighbors.append((dst, f"{node_id} --[{data['relation']}]--> {dst}"))
    for src, _, data in G.in_edges(node_id, data=True):       # 들어오는 관계
        neighbors.append((src, f"{src} --[{data['relation']}]--> {node_id}"))
    return neighbors

def expand_with_graph(start_nodes, hops=2):
    collected, visited, frontier = {}, set(start_nodes), list(start_nodes)
    for _ in range(hops):                       # BFS를 hops 횟수만큼
        next_frontier = []
        for node in frontier:
            for neighbor, path_desc in graph_neighbors(node):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                collected[neighbor] = path_desc  # 어떤 경로로 왔는지도 함께 저장
                next_frontier.append(neighbor)
        frontier = next_frontier
    return collected
```

### 3. 융합 검색 — Vector로 진입점을 찾고 Graph로 넓히기

```python
def hybrid_search(question, k=1, hops=2):
    hits = graph_store.similarity_search(question, k=k)          # ① 의미 검색
    start_nodes = [doc.metadata["node_id"] for doc in hits]
    expanded = expand_with_graph(start_nodes, hops=hops)         # ② 관계 확장

    context_docs = {n: NODE_DOCS.get(n, f"({n}) 본문 미등록") for n in start_nodes}
    for node_id in expanded:
        context_docs[node_id] = NODE_DOCS.get(node_id, f"({node_id}) 본문 미등록")
    return context_docs, list(expanded.values())                 # 문서 + 관계 경로
```

### 4. 관계 경로까지 프롬프트에 넣기

관계 경로를 함께 주면 LLM이 "어떤 규정을 거쳐 예외가 적용되는지"를 설명할 수 있습니다.

```python
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 KB국민은행 상담원입니다. 아래 근거 문서와 문서 사이의 관계 경로만 사용해 답하세요.\n"
     "1. 근거에 없는 내용은 지어내지 마세요.\n"
     "2. 예외 조건이 있으면 반드시 함께 안내하세요.\n"      # ← 불완전판매 방지
     "3. 답변 끝에 (근거: 사용한 문서 이름들) 형식으로 출처를 표기하세요.\n\n"
     "[근거 문서]\n{context}\n\n[문서 관계 경로]\n{paths}"),
    ("human", "{question}"),
])
```

### 5. LLM으로 그래프 자동 구축 — 트리플 추출

관계를 손으로 다 넣을 수는 없습니다. 구조화 출력으로 문장에서 `(주체, 관계, 대상)`을 뽑아 그래프를 키웁니다.

```python
from pydantic import BaseModel, Field
from typing import List

class Triple(BaseModel):
    head: str = Field(description="관계의 주체 (예: 상품명, 약관명)")
    relation: str = Field(description="관계 이름 (예: 근거 약관, 예외 조건)")
    tail: str = Field(description="관계의 대상")

class TripleList(BaseModel):
    triples: List[Triple]

extractor = llm.with_structured_output(TripleList)
result = extractor.invoke("다음 문장에서 (주체, 관계, 대상) 트리플을 모두 추출하세요:\n" + sentence)
for t in result.triples:
    G.add_edge(t.head, t.tail, relation=t.relation)
```

### 6. 평가 자동화로 개선 효과 증명하기

같은 질문 세트를 두 방식(Vector 단독 / 융합)으로 돌리고 LLM 채점관이 점수를 매깁니다. **배포 기준 점수(70점)** 를 두어 자동 게이트로 쓸 수 있습니다.

```python
DEPLOY_THRESHOLD = 70

def run_eval(answer_fn, name):
    scores = []
    for case in EVAL_SET:                       # {question, expected} 목록
        answer = answer_fn(case["question"])
        grade = grader.invoke(EVAL_PROMPT.format(
            question=case["question"], expected=case["expected"], answer=answer))
        scores.append(grade.score)
    avg = sum(scores) / len(scores)
    print(f"평균 {avg:.1f}점 → {'✅ 배포 가능' if avg >= DEPLOY_THRESHOLD else '🚨 배포 보류'}")
    return avg
```

## 주의사항

- **networkx는 인메모리**입니다. 프로세스가 끝나면 그래프가 사라집니다. 실무에서는 Neo4j 등에 영속화하세요.
- **Neo4j 심화 노트북은 별도 인스턴스가 필요합니다.** DB URL·계정이 준비되지 않으면 실행되지 않습니다.
- **hop 수를 늘리면 근거가 폭증합니다.** 2-hop이 실용적 상한이며, 3-hop 이상은 관련 없는 문서까지 끌고 와 오히려 정확도가 떨어지고 토큰 비용이 급증합니다.
- **트리플 자동 추출은 완벽하지 않습니다.** 같은 개체가 다른 이름으로 추출되면(`KB정기예금` vs `KB 정기예금`) 그래프가 파편화됩니다. 실무에서는 개체 정규화(entity resolution) 단계가 별도로 필요합니다.
- **평가 점수는 채점 LLM에 의존합니다.** 채점 모델을 바꾸면 점수 기준선이 달라지므로 A/B 비교는 반드시 같은 채점기로 하세요.
- 노트북 안 `run_eval` 비교 결과는 실행 시점 LLM 응답에 따라 달라집니다. 특정 점수 차이를 사실로 인용하지 마세요.

## 관련 문서

- [[개념_검색품질_개선]] — Vector 단독의 개선 한계
- [[개념_RAG_파이프라인]] — 기본 구조
- [[개념_Agent_평가]] — 평가 자동화 심화
- [[개념_금융_AI]] 관련 → [[MOC_금융_AI]]

## 출처

- Source: `Day2/Day02_02_GraphRAG_Vector융합_평가자동화.ipynb`
- Source: `Day2/(심화)_RAG_아키텍처_설계_Neo4j버전.ipynb`
- Source: `Day2/Day02_02_Graph_Vector_융합_RAG.pdf`
- Source: `바이브코딩실습/Day02_03_관계검색비교기/app.py`
