from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

MODEL_NAME = "gpt-4o-mini"
CHUNK_SIZE = 30
MAX_LINE_LENGTH = 23
FLAG_PREFIX = "[확인필요]"
FLAG_PREFIX_WITH_SPACE = "[확인필요] "
ENGLISH_PATTERN = re.compile(r"[A-Za-z]")

SYSTEM_PROMPT = (
    "You are a meticulous Korean subtitle editor. "
    "Correct Korean spelling errors, fix incorrect spacing, and confidently repair OCR mistakes while preserving meaning and tone. "
    "If a line cannot be corrected with high confidence, prefix the original line with '[확인필요]'. "
    "Treat any line containing Latin letters or mixed Korean-English noise as unreliable and mark it with '[확인필요]'. "
    "Do not introduce any punctuation that was not already present in the original text, and never add sentence-ending periods. "
    "Ensure each subtitle line is at most 23 characters (including spaces), using at most two lines per entry. "
    "Split lines at natural break points (before particles, conjunctions, etc.) to maintain readability. "
    "Respond only with valid JSON that follows the schema the user provides."
)

class CorrectionItem(BaseModel):
    id: str = Field(..., description="Subtitle entry identifier")
    lines: List[str] = Field(..., min_length=1, description="Corrected subtitle lines")


class CorrectionResponse(BaseModel):
    items: List[CorrectionItem] = Field(
        ..., description="Collection of corrected subtitle entries"
    )


def _to_serializable_dict(value: Any) -> Dict[str, Any] | None:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        candidate = model_dump()
        if isinstance(candidate, dict):
            return candidate
    return None


def extract_corrections_payload(result: Any) -> Sequence[Any] | None:
    data = _to_serializable_dict(result)
    if data is not None:
        items = data.get("items")
        if isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
            return items
        if len(data) == 1:
            sole_value = next(iter(data.values()))
            return extract_corrections_payload(sole_value)

    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        return result

    attr_items = getattr(result, "items", None)
    if isinstance(attr_items, Sequence) and not isinstance(attr_items, (str, bytes, bytearray)):
        return attr_items

    return None


def read_srt(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as file:
        lines = file.read().splitlines()

    entries: List[Dict[str, Any]] = []
    idx = 0
    total = len(lines)

    while idx < total:
        if not lines[idx].strip():
            idx += 1
            continue

        entry_id = lines[idx].strip()
        idx += 1

        if idx >= total:
            raise ValueError(f"Incomplete subtitle block after id '{entry_id}'.")

        timestamp = lines[idx].strip()
        idx += 1

        text_lines: List[str] = []
        while idx < total and lines[idx].strip() != "":
            text_lines.append(lines[idx])
            idx += 1

        entries.append(
            {"id": entry_id, "timestamp": timestamp, "lines": text_lines or [""]}
        )

        while idx < total and lines[idx].strip() == "":
            idx += 1

    return entries


def write_srt(entries: List[Dict[str, Any]], path: Path) -> None:
    output_lines: List[str] = []

    for entry in entries:
        output_lines.append(entry["id"])
        output_lines.append(entry["timestamp"])
        output_lines.extend(entry["lines"])
        output_lines.append("")

    text = "\n".join(output_lines).rstrip() + "\n"
    path.write_text(text, encoding="utf-8")


def chunk_entries(entries: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(entries), size):
        yield entries[start : start + size]


def init_llm(api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=4096,
        api_key=api_key,
    )


def build_prompt(chunk: List[Dict[str, Any]]) -> str:
    payload = [
        {"id": entry["id"], "timestamp": entry["timestamp"], "lines": entry["lines"]}
        for entry in chunk
    ]
    chunk_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        "다음은 SRT 자막 항목들입니다. "
        "각 항목의 `lines` 배열에 포함된 문장들의 맞춤법을 교정하고, 확실히 바로잡을 수 있는 오인식 문자를 수정하세요.\n\n"
        "**교정 규칙:**\n"
        "- 한국어 맞춤법(띄어쓰기 포함)을 정확히 교정하세요\n"
        "- OCR 오인식 문자나 오타를 자연스럽게 수정하세요\n"
        "- 문맥을 고려하여 올바른 표현으로 교정하세요\n\n"
        "**줄 나누기 규칙 (매우 중요):**\n"
        "- 각 줄은 공백 포함 23자를 절대 초과하지 마세요\n"
        "- 최대 2줄까지만 사용하세요\n"
        "- 문맥상 자연스러운 위치(조사, 접속사 앞 등)에서 줄을 나누세요\n"
        "- 두 줄로 나눌 때는 균형있게 나누되, 의미 단위를 유지하세요\n"
        "- 23자 이내라면 한 줄로 유지해도 됩니다\n\n"
        "**기타 규칙:**\n"
        "- 신뢰할 수 없는 교정이라면 해당 줄 맨 앞에 '[확인필요]'를 붙이고 원본을 그대로 남기세요\n"
        "- 한국어와 영어가 섞이거나 영문자로만 구성된 줄은 '[확인필요]' 처리하세요\n"
        "- 원본에 없던 문장부호를 절대 새로 추가하지 마세요 (특히 마침표)\n"
        "- 문장의 의미가 변하지 않도록 주의하세요\n\n"
        f"입력 데이터:\n{chunk_json}\n\n"
        "반드시 다음 JSON 형식으로만 응답하세요:\n"
        "{\"items\": [{\"id\": \"1\", \"lines\": [\"교정된 자막 줄1\", \"교정된 자막 줄2\"]}, ...]}"
    )


def enforce_line_rules(lines: Sequence[str]) -> List[str]:
    """LLM이 생성한 줄에 대한 최소한의 검증 및 플래그 처리"""
    processed: List[str] = []

    for raw in lines:
        text = "" if raw is None else str(raw).strip()
        if not text:
            processed.append("")
            continue

        # 영문자가 포함된 경우 [확인필요] 플래그 추가 (LLM이 놓친 경우를 대비)
        needs_flag = bool(ENGLISH_PATTERN.search(text))
        if needs_flag and not text.startswith(FLAG_PREFIX):
            text = f"{FLAG_PREFIX_WITH_SPACE}{text}"

        processed.append(text)

    return processed


def prepare_structured_llm(llm: ChatOpenAI) -> Runnable:
    return llm.with_structured_output(CorrectionResponse, method="json_mode", include_raw=True)


def normalize_lines(lines: Any) -> List[str]:
    if isinstance(lines, str):
        return [lines]
    if isinstance(lines, Sequence):
        normalized: List[str] = []
        for item in lines:
            normalized.append("" if item is None else str(item))
        return normalized or [""]
    return [str(lines)]


def apply_corrections(chunk: List[Dict[str, Any]], corrections: Sequence[Any]) -> None:
    mapped: Dict[str, List[str]] = {}

    for item in corrections:
        if isinstance(item, BaseModel):
            payload = item.model_dump()
        elif isinstance(item, dict):
            payload = item
        else:
            continue

        entry_id = str(payload.get("id", "")).strip()
        if not entry_id:
            continue
        mapped[entry_id] = normalize_lines(payload.get("lines"))

    for entry in chunk:
        corrected = mapped.get(entry["id"])
        if corrected:
            candidate_lines = normalize_lines(corrected)
        else:
            original_lines = normalize_lines(entry.get("lines", [""]))
            candidate_lines = [
                line if line.startswith(FLAG_PREFIX) else f"{FLAG_PREFIX_WITH_SPACE}{line}".rstrip()
                for line in original_lines
            ]

        entry["lines"] = enforce_line_rules(candidate_lines)


def process_chunk(
    structured_llm: Runnable,
    chunk: List[Dict[str, Any]],
    attempt_limit: int = 3,
) -> None:
    prompt = build_prompt(chunk)

    for attempt in range(1, attempt_limit + 1):
        try:
            result = structured_llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
            
            # include_raw=True이므로 result는 딕셔너리 형태
            if isinstance(result, dict):
                raw_response = result.get("raw")
                parsed_response = result.get("parsed")
                parsing_error = result.get("parsing_error")
                
                if parsing_error:
                    print(
                        f"\n[경고] 구조화된 출력 파싱 실패 (시도 {attempt}/{attempt_limit})",
                        file=sys.stderr,
                    )
                    print(f"[오류 메시지] {parsing_error}", file=sys.stderr)
                    if raw_response:
                        raw_content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
                        print(f"[LLM 응답 내용]\n{raw_content}\n", file=sys.stderr)
                    continue
                
                # 파싱 성공
                if parsed_response:
                    corrections = extract_corrections_payload(parsed_response)
                    if corrections:
                        apply_corrections(chunk, corrections)
                        return
                    else:
                        print(
                            f"[경고] LLM 응답에서 보정 결과를 찾지 못했습니다 (시도 {attempt}/{attempt_limit}). "
                            f"응답 타입: {type(parsed_response).__name__}",
                            file=sys.stderr,
                        )
                        print(f"[응답 내용] {parsed_response}", file=sys.stderr)
                        continue
            else:
                # 이전 방식과 호환성 유지
                corrections = extract_corrections_payload(result)
                if corrections:
                    apply_corrections(chunk, corrections)
                    return
                    
        except OutputParserException as error:
            print(
                f"\n[경고] 구조화된 출력 파싱 실패 (시도 {attempt}/{attempt_limit})",
                file=sys.stderr,
            )
            print(f"[오류 메시지] {error}", file=sys.stderr)
            
            # 다양한 방법으로 raw output 추출 시도
            raw_output = None
            if hasattr(error, "llm_output"):
                raw_output = error.llm_output
            elif hasattr(error, "observation"):
                raw_output = error.observation
            elif len(error.args) > 0:
                raw_output = str(error.args[0])
            
            if raw_output:
                print(f"[LLM 응답 내용]\n{raw_output}\n", file=sys.stderr)
            else:
                print("[LLM 응답 내용] 응답 내용을 추출할 수 없습니다.", file=sys.stderr)
            continue
        except Exception as error:
            print(
                f"[경고] LLM 호출 중 오류 발생 (시도 {attempt}/{attempt_limit}): {error}",
                file=sys.stderr,
            )
            continue

    raise RuntimeError("LLM 응답 파싱에 반복적으로 실패했습니다.")


def derive_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    suffix = input_path.suffix or ".srt"
    return input_path.with_name(f"{stem}_fixed{suffix}")


def main() -> None:
    # .env 파일에서 환경변수 로드
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="OpenAI LLM 기반 자막 맞춤법 교정 도구")
    parser.add_argument("subtitle_path", help="보정할 SRT 파일 경로")
    args = parser.parse_args()

    input_path = Path(args.subtitle_path).resolve()
    if not input_path.exists():
        print(f"[오류] 파일을 찾을 수 없습니다: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        entries = read_srt(input_path)
    except ValueError as error:
        print(f"[오류] SRT 파싱에 실패했습니다: {error}", file=sys.stderr)
        sys.exit(1)

    if not entries:
        print("[안내] 처리할 자막 항목이 없습니다.", file=sys.stderr)
        sys.exit(0)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[오류] OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.", file=sys.stderr)
        sys.exit(1)

    llm = init_llm(api_key)
    structured_llm = prepare_structured_llm(llm)
    print(f"[정보] 총 {len(entries)}개의 자막 항목을 {MODEL_NAME} 모델로 처리합니다.")

    chunks = list(chunk_entries(entries, CHUNK_SIZE))
    total_batches = len(chunks)

    for batch_index, chunk in enumerate(chunks, start=1):
        print(f"[정보] 배치 {batch_index}/{total_batches}: 항목 {len(chunk)}개 처리 중...")
        process_chunk(structured_llm, chunk)

    output_path = derive_output_path(input_path)
    write_srt(entries, output_path)
    print(f"[완료] 보정된 자막을 저장했습니다: {output_path}")
    print("반드시 변환된 srt 파일에서 [확인필요] 플래그가 붙은 줄을 확인하세요.")


if __name__ == "__main__":
    main()
