# 에이전트 개발 전 체크리스트

이 문서는 공통 모듈을 활용해 에이전트를 개발하기 전에 확인해야 할 항목을 정리한 문서다.

## 1. 공통모듈 담당자에게 확인할 것

### 1) 패키지/설치 관련
- 사내 PyPI 사용 여부
- 설치 명령어
- 사용할 extra 종류
  - `llm`
  - `retrieval`
  - `lancedb`
  - `examples`
  - `dev`
- 패키지 버전 정책

### 2) LLM 연결 관련
- 내부 GenAI 서버의 base URL
- 인증 방식
  - API key
  - 토큰
  - mTLS
  - 사내 인증 헤더
- 사용 모델명
- endpoint 경로
  - 예: `/v1/chat/completions`
- timeout / retry / rate limit 정책

### 3) Retrieval/DB 관련
- PostgreSQL / pgvector 접속 정보
- DSN 형식
- 테이블 이름 / 스키마 이름
- 허용 필터 키
- 검색/리랭크 설정 기준

### 4) Embedding / Reranker 관련
- embedding endpoint 주소
- reranker endpoint 주소
- 모델명
- 호출 방식

### 5) 설정 파일 관련
- 어떤 YAML 파일을 써야 하는지
- `agent.local.yaml`, `models.local.yaml` 같은 로컬 설정 파일 규칙
- 시크릿/환경변수를 어디에 넣는지

### 6) 네트워크 접근 관련
- VPN 필요 여부
- 사내망 접근 가능 여부
- 프록시 / 게이트웨이 사용 여부
- 방화벽 / 인증서 / DNS 문제 여부

### 7) 로컬 개발 대안 관련
- 로컬 Ollama / vLLM 사용 가능 여부
- mock / stub 사용 가능 여부
- 샘플 데이터 제공 가능 여부

---

## 2. 환경 설정 전에 해야 할 것

### 1) Python 환경 준비
- Python 3.12 사용
- `uv` 설치

예시:

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) 프로젝트 의존성 설치

```bash
uv sync --extra dev --extra llm
```

### 3) 로컬 LLM 준비
- 로컬 Ollama 또는 vLLM이 가능하면 가장 쉽게 시작 가능
- 내부망 endpoint를 쓰는 경우 담당자가 제공한 정보 사용

예시:

```bash
ollama serve
ollama pull qwen2.5:3b
```

### 4) 설정 파일 준비

예제 설정 파일을 로컬용으로 복사한다.

```bash
cp examples/llm_chat/configs/agent.example.yaml examples/llm_chat/configs/agent.local.yaml
cp examples/llm_chat/configs/models.example.yaml examples/llm_chat/configs/models.local.yaml
```

예시 내용:

```yaml
agent:
  name: local-llm-example
  environment: local

llm:
  model_registry_path: models.local.yaml
  default_model: local-ollama
  api_key: null
  temperature: 0.2
  max_tokens: 1024
  timeout: 180
```

```yaml
models:
  local-ollama:
    provider: vllm
    base_url: http://localhost:11434
    endpoint_path: /v1/chat/completions
    timeout: 180
```

### 5) 환경변수 / 시크릿 준비
- LLM API key
- DB DSN
- S3 관련 값(필요 시)

예시:

```powershell
$env:KBCARD_POSTGRES_DSN="..."
```

### 6) 예제 실행 확인

```bash
uv run --extra llm python examples/llm_chat/run_sync.py
```

성공하면 다음 단계로 넘어간다.

---

## 3. 개발 시작 전 체크리스트

- [ ] Python 3.12 확인
- [ ] uv 설치
- [ ] 의존성 설치 완료
- [ ] 로컬 LLM 또는 내부 endpoint 확인
- [ ] YAML 설정 파일 준비 완료
- [ ] 환경변수 / 시크릿 설정 완료
- [ ] 예제 실행 성공
- [ ] 담당자에게 LLM/DB/인증 정보 확인 완료

---

## 4. 담당자에게 바로 보낼 메시지 예시

아래 문장을 그대로 보내면 된다.

> 에이전트 개발 전에 공통모듈 연동을 위해 LLM endpoint 주소, 인증 방식, DB DSN 형식, 설정 파일 템플릿, 로컬 테스트 대안(예: Ollama/vLLM)을 알려주실 수 있나요?
