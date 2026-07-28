from __future__ import annotations

import argparse
import csv
import re
import urllib.request
from pathlib import Path


FORK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEAKER_FILE = FORK_ROOT / "data" / "youdao" / "text" / "speaker2"
DEFAULT_OUTPUT = FORK_ROOT / "creator_tools" / "data" / "speaker_metadata.csv"
WIKI_URL = (
    "https://raw.githubusercontent.com/wiki/netease-youdao/EmotiVoice/"
    "%F0%9F%98%8A-voice-wiki-page.md"
)
ROW_PATTERN = re.compile(
    r"^\|\s*(?P<id>\d+)\s*\|\s*(?P<name>[^|]+?)\s*\|"
    r"\s*(?P<gender>[MF])\s*\|\s*(?P<description>[^|]*?)\s*\|$"
)

DESCRIPTION_TERMS = {
    "clear": "清晰",
    "soothing": "舒缓",
    "expressive": "有表现力",
    "crisp": "清脆",
    "melodic": "悦耳",
    "captivating": "有感染力",
    "rich": "醇厚",
    "resonant": "有共鸣",
    "engaging": "亲和",
    "smooth": "顺滑",
    "mellow": "温和",
    "charismatic": "有魅力",
    "dynamic": "有活力",
    "lively": "活泼",
    "energetic": "有朝气",
}


def parse_wiki(markdown: str) -> dict[str, dict[str, str]]:
    speakers: dict[str, dict[str, str]] = {}
    for line in markdown.splitlines():
        match = ROW_PATTERN.match(line.strip())
        if not match:
            continue
        row = match.groupdict()
        speakers[row["id"]] = {
            "speaker_id": row["id"],
            "english_name": row["name"].strip(),
            "gender": row["gender"],
            "description_en": row["description"].strip().rstrip("."),
        }
    return speakers


def translate_description(description: str) -> str:
    terms = []
    lowered = description.lower()
    for english, chinese in DESCRIPTION_TERMS.items():
        if english in lowered and chinese not in terms:
            terms.append(chinese)
    return "、".join(terms)


def read_markdown(path: Path | None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    request = urllib.request.Request(
        WIKI_URL,
        headers={"User-Agent": "creator-emotivoice-metadata/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def build_rows(
    speaker_ids: list[str],
    wiki_speakers: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    missing = [speaker_id for speaker_id in speaker_ids if speaker_id not in wiki_speakers]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Voice wiki is missing {len(missing)} IDs: {preview}")

    counters = {"F": 0, "M": 0}
    rows = []
    for speaker_id in speaker_ids:
        row = wiki_speakers[speaker_id]
        gender = row["gender"]
        counters[gender] += 1
        gender_zh = "女声" if gender == "F" else "男声"
        chinese_name = f"{gender_zh}{counters[gender]:04d}"
        description_zh = translate_description(row["description_en"])
        display_name = (
            f"{chinese_name}｜{row['english_name']}｜ID {speaker_id}"
        )
        rows.append(
            {
                **row,
                "gender_zh": gender_zh,
                "chinese_name": chinese_name,
                "description_zh": description_zh,
                "display_name": display_name,
                "pitch_hz": "",
                "duration_s": "",
                "voice_profile": "",
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build EmotiVoice speaker metadata from the official voice wiki."
    )
    parser.add_argument("--wiki", type=Path)
    parser.add_argument("--speaker-file", type=Path, default=DEFAULT_SPEAKER_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    speaker_ids = [
        line.strip()
        for line in args.speaker_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wiki_speakers = parse_wiki(read_markdown(args.wiki))
    rows = build_rows(speaker_ids, wiki_speakers)
    write_csv(rows, args.output)
    print(
        f"Wrote {len(rows)} speakers to {args.output} "
        f"({sum(row['gender'] == 'F' for row in rows)} female, "
        f"{sum(row['gender'] == 'M' for row in rows)} male)."
    )


if __name__ == "__main__":
    main()
