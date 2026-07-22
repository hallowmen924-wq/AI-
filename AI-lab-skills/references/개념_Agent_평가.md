# 개념: Agent 평가 (RAGAS · LLM-as-a-Judge · 회귀 테스트)

## 한눈에 보기

LLM 답변은 정답이 하나로 딱 떨어지지 않아 단순 문자열 비교로 채점할 수 없습니다. 그래서 **여러 각도의 지표**로 봅니다.
답변 품질은 RAGAS 3대 지표(Faithfulness·Answer Relevancy·Context Recall), 운영 품질은 성공률·응답 지연, 배포 안전은 pytest 회귀 테스트로 잡습니다.

## 핵심 개념

### RAGAS 3대 지표

| 지표 | 쉬운 뜻 | 무엇을 잡아내나 |
|---|---|---|
| **Faithfulness** | 답변이 검색된 근거에 충실한가 | **환각** — 근거에 없는 내용을 지어냄 |
| **Answer Relevancy** | 답변이 질문에 맞는 답인가 | 동문서답 |
| **Context Recall** | 정답에 필요한 근거를 다 가져왔는가 | **검색 실패** — 검색 단계 문제 |

평가 데이터 4종: `user_input`(질문), `response`(AI 답변), `retrieved_contexts`(검색된 근거), `reference`(정답).

> 💡 Faithfulness가 낮으면 → 생성 프롬프트 문제. Context Recall이 낮으면 → 검색 문제([[개념_검색품질_개선]]). 지표를 나눠 보는 이유가 이 **원인 분리**입니다.

### 운영 지표

| 지표 | 정의 |
|---|---|
| **Task 성공률** | 답변을 못 찾은 케이스를 제외한 비율 |
| **평균 응답 지연** | 요청부터 답변까지 시간 |

### 회귀 테스트

배포마다 품질이 떨어지지 않았는지 자동 검사. 예: "답변에는 반드시 출처가 포함돼야 한다"를 테스트로 고정.

## 전체 동작 흐름

```
[평가 데이터셋]  question / response / retrieved_contexts / reference
       ↓
 ┌─ RAGAS evaluate()      → Faithfulness, ResponseRelevancy, LLMContextRecall
 ├─ 운영 지표 측정         → 성공률, 평균 지연
 └─ pytest 회귀 테스트     → 출처 포함 여부 등 규칙 고정
       ↓
 배포 게이트: 기준 점수 미달이면 배포 차단
```

## 주요 코드 구성

| 파일 | 역할 |
|---|---|
| `Day5/Day05_04_Agent평가_RAGAS.ipynb` | RAGAS 평가 + 운영 지표 + pytest |
| `Day5/개인실습/Day05_03_Agent평가_RAGAS_개인실습.ipynb` | 빈칸 실습본 |
| `바이브코딩실습/Day05_04_답변채점기/app.py` | **RAGAS 대신 LLM 채점으로 3대 지표 구현 (합격선 0.7)** |
| `Day2/Day02_02_GraphRAG_..._평가자동화.ipynb` | LLM-as-a-Judge 평가 자동화 |
| `Day1/Day01_02_RAG_아키텍처_설계_최신버전.ipynb` | 고전 지표(BLEU/ROUGE) vs LLM 채점 비교 |

## ⚠️ RAGAS 라이브러리 버전 충돌 — 가장 중요한 주의사항

**`ragas` 는 langchain 1.x와 충돌합니다.** 교안 문서(`바이브코딩실습/README.md`, 강의 환경 노트)가 이를 명시하고 있습니다.

- `Day05_04_Agent평가_RAGAS.ipynb` 는 `ragas==0.2.15` + `langchain-openai==0.3.17` 을 설치합니다 — **강의 표준 환경(langchain 1.3.11)과 다른 환경**을 가정한 코드입니다.
- 그래서 **바이브코딩 실습 앱 `Day05_04_답변채점기` 는 ragas를 쓰지 않고 동일한 3대 지표를 LLM 채점(합격선 0.7)으로 구현**했습니다.
- **재사용 권장**: langchain 1.x 프로젝트에서는 ragas를 도입하지 말고, 아래 "LLM 채점 방식"을 쓰세요.

## 실행 방법

```bash
# ⚠️ 별도 가상환경을 권장합니다 (langchain 1.x와 충돌)
pip install -q ragas==0.2.15 langchain-openai==0.3.17
$env:OPENAI_API_KEY = "sk-..."

# 충돌 없는 대안 — 바이브코딩 앱
cd 바이브코딩실습/Day05_04_답변채점기 && streamlit run app.py
```

## 핵심 코드

### 1. 평가 데이터셋 구성

```python
samples = [
    {"user_input": "예금자 보호 한도는 얼마인가요?",
     "response": "1인당 최고 5천만원까지 보호됩니다.",
     "retrieved_contexts": ["예금자보호법상 1인당 원리금 5천만원까지 보호된다."],
     "reference": "1인당 5천만원까지 보호됩니다."},
    {"user_input": "정기예금 중도해지하면 이자는?",
     "response": "중도해지 시 약정금리보다 낮은 중도해지 이율이 적용됩니다.",
     "retrieved_contexts": ["정기예금 중도해지 시 중도해지 이율이 적용된다."],
     "reference": "중도해지 이율이 적용되어 약정금리보다 낮아집니다."},
]
```

### 2. RAGAS 자동 평가

```python
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

eval_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
eval_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

result = evaluate(
    dataset=EvaluationDataset.from_list(samples),
    metrics=[Faithfulness(), ResponseRelevancy(), LLMContextRecall()],
    llm=eval_llm, embeddings=eval_emb,
)
print(result)
```

### 3. LLM-as-a-Judge 방식 ⭐ (langchain 1.x 권장 대안)

`ragas` 없이 같은 개념을 구현합니다. 구조화 출력으로 점수를 받는 것이 핵심입니다.

```python
from pydantic import BaseModel, Field

class GradeResult(BaseModel):
    score: int = Field(ge=0, le=100, description="답변 품질 점수 (0~100)")
    reason: str = Field(description="채점 이유 한 문장")

grader = llm.with_structured_output(GradeResult)

EVAL_PROMPT = (
    "당신은 금융 상담 답변 채점관입니다.\n"
    "질문: {question}\n기대 포인트: {expected}\n실제 답변: {answer}\n\n"
    "실제 답변이 기대 포인트를 얼마나 정확하고 빠짐없이 담았는지 0~100점으로 채점하세요."
)

DEPLOY_THRESHOLD = 70          # 배포 게이트

def run_eval(answer_fn, name):
    scores = []
    for case in EVAL_SET:
        answer = answer_fn(case["question"])
        grade = grader.invoke(EVAL_PROMPT.format(
            question=case["question"], expected=case["expected"], answer=answer))
        scores.append(grade.score)
        print(f"  [{grade.score:3d}점] {case['question']} — {grade.reason}")
    avg = sum(scores) / len(scores)
    print(f"  평균 {avg:.1f}점 → {'✅ 배포 가능' if avg >= DEPLOY_THRESHOLD else '🚨 배포 보류'}")
    return avg
```

각 지표를 따로 채점하려면 프롬프트만 바꾸면 됩니다 — Faithfulness는 "근거 문서에 없는 내용이 있는가", Context Recall은 "정답에 필요한 근거가 검색되었는가".

### 4. 운영 지표 — 성공률과 지연

```python
import time

def is_success(answer):
    return "죄송" not in answer         # 실패 응답 패턴으로 판정

success, latencies = 0, []
for q in questions:
    t0 = time.time()
    ans = agent(q)
    latencies.append(time.time() - t0)
    if is_success(ans):
        success += 1

print(f"Task 성공률: {success}/{len(questions)} = {success/len(questions):.0%}")
print(f"평균 응답 지연: {sum(latencies)/len(latencies):.3f}s")
```

### 5. pytest 회귀 테스트 ⭐

배포마다 자동으로 돌려 규칙이 깨지지 않았는지 확인합니다.

```python
# test_agent.py
def test_answer_has_source():
    assert "출처" in agent_answer("보호 한도는?")      # 금융 답변은 반드시 출처 포함

def test_answer_not_empty():
    assert len(agent_answer("보호 한도는?")) > 0
```

```bash
python -m pytest -q test_agent.py
```

### 6. 고전 지표 (참고 — Day1)

| 지표 | 측정 | 한계 |
|---|---|---|
| BLEU | 단어 배열·어순 일치 | 표현이 다르면 정답이어도 0점 |
| ROUGE-L | 핵심 어구 포함률 | 위와 동일 |
| BERTScore | 의미적 유사도 | 모델 다운로드 필요, 느림 |
| UniEval(구현판) | 단어 집합 겹침 비율 | 매우 단순 |

→ 결론: 자유 형식 답변에는 **LLM 채점이 실용적**입니다.

## A/B 테스트와 자동화 파이프라인

- **A/B 테스트**: 프롬프트 A안과 B안을 같은 질문 세트로 평가해 점수가 높은 쪽 채택. 반드시 **같은 채점기**를 쓰세요.
- **배포 게이트**: 코드 배포 → 평가 + pytest 회귀 → 기준 점수 미달이면 배포 차단.

## 주의사항

- 🚨 **`Day05_04` 노트북에 실제 OpenAI 키가 하드코딩되어 있습니다** → [[오류_환경설정_트러블슈팅]]
- ⚠️ **ragas ↔ langchain 1.x 충돌.** 위 별도 절 참고. langchain 1.x 프로젝트에는 도입하지 마세요.
- **채점 LLM을 바꾸면 점수 기준선이 달라집니다.** 버전 간 비교를 하려면 채점 모델을 고정하세요.
- **평가 자체가 LLM 호출이라 비용이 듭니다.** 질문 20개 × 지표 3개 = 60회 호출. CI에서 매번 돌리면 비용이 쌓입니다. 핵심 세트만 매 배포, 전체 세트는 주기적으로가 현실적입니다.
- **평가 데이터셋을 실제 상담 로그로 만들면 개인정보가 섞입니다.** 마스킹 후 사용하세요 → [[개념_PII보호와_거버넌스]]
- **`is_success` 를 "죄송" 문자열로 판정하는 방식은 취약합니다.** 실서비스에서는 실패를 명시적 플래그로 표현하세요(예: State에 `status` 필드).
- **정답(`reference`)이 없는 지표도 있습니다.** Faithfulness는 근거만 있으면 되지만 Context Recall은 정답이 필요합니다. 데이터 준비 비용이 다릅니다.

## 관련 문서

- [[개념_검색품질_개선]] — Context Recall이 낮을 때 볼 곳
- [[개념_GraphRAG_Vector_융합]] — 평가 자동화 실전 예
- [[개념_트레이싱과_감사로그]] — Score를 대시보드에 축적
- [[개념_HITL과_검증루프]] — 런타임 검증(평가와 다름)
- [[개념_RAG_파이프라인]] — 4대 평가 관점

## 출처

- Source: `Day5/Day05_04_Agent평가_RAGAS.ipynb`
- Source: `Day5/Day05_04_교안_Agent평가_RAGAS.pdf`
- Source: `Day1/Day01_02_RAG_아키텍처_설계_최신버전.ipynb` (고전 지표·LLM 채점)
- Source: `Day2/Day02_02_GraphRAG_Vector융합_평가자동화.ipynb` (배포 게이트)
- Source: `바이브코딩실습/Day05_04_답변채점기/app.py`, `바이브코딩실습/README.md` (ragas 충돌 명시)
