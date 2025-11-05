## korsrt-fixer 개요

- OCR 기반으로 추출한 SRT 자막과 같이 후보정이 필요한 자막파일이나 
  개인이 작업한 SRT 자막을 OpenAI LLM을 이용하여 오인식 문자와 맞춤법을 교정한다.
- 자막 번호와 타임스탬프는 그대로 유지하고, 확신이 없는 줄은 `[확인필요]` 접두어로 표시한다.
- 한국어 기준 23자를 초과하는 줄은 자연스러운 문맥을 유지하며 두 줄로 변환한다.

## 설치

```bash
git clone https://github.com/your-username/korsrt-fixer.git
cd korsrt-fixer
pip install -e .
```

## 설정

OpenAI API 키를 환경 변수로 설정:

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY=sk-your-api-key-here
```

또는 작업 디렉토리에 `.env` 파일 생성:

```env
OPENAI_API_KEY=sk-your-api-key-here
```

## 실행

```bash
korsrt-fixer <입력.srt>
```

- 변환 결과는 `<입력>_fixed.srt` 파일로 저장된다.

## 개발 환경

uv를 사용한 개발 환경 구축:

```bash
uv sync
```

개발 중 실행:

```bash
uv run fixer.py <입력.srt>
```
