from creator_tools.build_metadata import build_rows, parse_wiki


def test_parse_wiki_and_build_chinese_aliases():
    markdown = """
| ID | Voice Name | Gender | Description |
|----|-------|--------|-------------|
| 8051 | Maria Kasper | F | Clear, soothing, expressive |
| 9017 | John Van Stan | M | Rich, resonant, engaging |
"""

    rows = build_rows(["8051", "9017"], parse_wiki(markdown))

    assert rows[0]["chinese_name"] == "女声0001"
    assert rows[0]["description_zh"] == "清晰、舒缓、有表现力"
    assert rows[0]["display_name"].endswith("Maria Kasper｜ID 8051")
    assert rows[1]["chinese_name"] == "男声0001"
