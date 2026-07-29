from __future__ import annotations

import csv
import hashlib
from html import escape
import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st


FORK_ROOT = Path(__file__).resolve().parents[1]
CREATOR_TOOLS_ROOT = Path(__file__).resolve().parent
if str(CREATOR_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(CREATOR_TOOLS_ROOT))
DEFAULT_CREATOR_ROOT = (
    FORK_ROOT.parents[1]
    if FORK_ROOT.parent.name == "third_party"
    else FORK_ROOT
)
PROJECT_ROOT = Path(
    os.environ.get("CREATOR_ROOT", str(DEFAULT_CREATOR_ROOT))
).resolve()
TOOL_ROOT = PROJECT_ROOT / "output" / "data" / "tools" / "emotivoice"
VOICE_SEARCH_ROOT = TOOL_ROOT / "voice-search"
METADATA_PATH = FORK_ROOT / "creator_tools" / "data" / "speaker_metadata.csv"
VOICE_MATCHES_PATH = PROJECT_ROOT / "config" / "voice_matches.json"
SAMPLES_ROOT = VOICE_SEARCH_ROOT / "reference-samples"
MODEL_PATH = (
    VOICE_SEARCH_ROOT
    / "models"
    / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
)
INDEX_PATH = VOICE_SEARCH_ROOT / "index" / "emotivoice-speakers.npz"
PAGE_SIZE = 12


st.set_page_config(
    page_title="Creator 声音库",
    page_icon="🎙️",
    layout="wide",
)
st.markdown(
    """
    <style>
    .stApp { background: #f6f4ef; color: #1d2927; }
    .block-container { max-width: 1180px; padding-top: 2rem; }
    [data-testid="stHeader"] { background: transparent; }
    .voice-hero {
      border: 1px solid #d7d8cf;
      border-radius: 24px;
      padding: 28px;
      background: linear-gradient(135deg, #173d38 0%, #25594f 65%, #bd7b40 140%);
      color: #fff;
      margin-bottom: 18px;
    }
    .voice-hero h1 { margin: 0 0 8px; font-size: clamp(1.7rem, 4vw, 2.6rem); }
    .voice-hero p { margin: 0; max-width: 720px; color: #e4f0ec; line-height: 1.7; }
    .selected-strip {
      border-left: 4px solid #bd7b40;
      border-radius: 12px;
      padding: 12px 16px;
      background: #fffaf3;
      margin: 8px 0 18px;
    }
    .voice-title { font-weight: 700; font-size: 1.02rem; margin-bottom: 2px; }
    .voice-meta { color: #5b6865; font-size: .88rem; margin-bottom: 8px; }
    .match-empty {
      border: 1px solid #e2c9a9;
      border-radius: 16px;
      background: #fff8ed;
      padding: 16px 18px;
      margin-bottom: 18px;
    }
    .match-empty strong { color: #7a431d; }
    .stButton > button, .stDownloadButton > button {
      min-height: 44px;
      border-radius: 12px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
      background: rgba(255,255,255,.78);
      border-color: #d7d8cf;
      border-radius: 16px;
    }
    div[data-testid="stAudio"] { min-height: 48px; }
    @media (max-width: 700px) {
      .block-container { padding: 1rem .8rem 2rem; }
      .voice-hero { border-radius: 18px; padding: 20px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_catalog(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


@st.cache_data(show_spinner=False)
def load_match_status(path: str) -> dict[str, object] | None:
    match_path = Path(path)
    if not match_path.is_file():
        return None
    payload = json.loads(match_path.read_text(encoding="utf-8"))
    return payload.get("matches", [None])[0]


def sample_path(speaker_id: str) -> Path:
    return SAMPLES_ROOT / f"{speaker_id}.wav"


def select_voice(speaker_id: str) -> None:
    st.session_state.selected_speaker_id = speaker_id
    st.session_state.generated_audio = b""


def find_voice(catalog: list[dict[str, str]], speaker_id: str):
    return next(
        (row for row in catalog if row["speaker_id"] == speaker_id),
        None,
    )


def render_voice_card(
    row: dict[str, str],
    *,
    key_prefix: str,
    rank: int | None = None,
    similarity: float | None = None,
) -> None:
    speaker_id = row["speaker_id"]
    selected = st.session_state.selected_speaker_id == speaker_id
    heading = row["chinese_name"]
    if rank is not None:
        heading = f"{rank}. {heading}"
    st.markdown(
        f'<div class="voice-title">{escape(heading)}</div>',
        unsafe_allow_html=True,
    )
    details = (
        f"{row['gender_zh']} · {row['voice_profile']} · "
        f"{row['english_name']} · ID {speaker_id}"
    )
    if similarity is not None:
        details = f"相似度 {similarity:.4f} · {details}"
    st.markdown(
        f'<div class="voice-meta">{escape(details)}</div>',
        unsafe_allow_html=True,
    )
    reference = sample_path(speaker_id)
    if reference.is_file():
        st.audio(str(reference), format="audio/wav")
    else:
        st.caption("试听样本缺失")
    label = "已用于配音" if selected else "用于配音"
    st.button(
        label,
        key=f"{key_prefix}-{speaker_id}",
        type="primary" if selected else "secondary",
        use_container_width=True,
        disabled=selected,
        on_click=select_voice,
        args=(speaker_id,),
    )


def render_selected_voice(
    catalog: list[dict[str, str]],
    *,
    include_audio: bool = False,
) -> dict[str, str] | None:
    selected_id = st.session_state.selected_speaker_id
    row = find_voice(catalog, selected_id) if selected_id else None
    if not row:
        st.info("尚未选择音色。请先在“声音库”中试听，再点击“用于配音”。")
        return None
    st.markdown(
        (
            '<div class="selected-strip"><strong>当前配音音色</strong><br>'
            f"{row['chinese_name']} · {row['english_name']} · "
            f"Speaker ID {row['speaker_id']}</div>"
        ),
        unsafe_allow_html=True,
    )
    if include_audio:
        reference = sample_path(row["speaker_id"])
        if reference.is_file():
            st.audio(str(reference), format="audio/wav")
    return row


def filter_catalog(
    catalog: list[dict[str, str]],
    query: str,
    gender: str,
) -> list[dict[str, str]]:
    normalized = query.strip().casefold()
    rows = [
        row
        for row in catalog
        if gender == "全部" or row["gender_zh"] == gender
    ]
    if not normalized:
        return rows
    searchable_fields = (
        "speaker_id",
        "chinese_name",
        "english_name",
        "voice_profile",
        "description_zh",
        "description_en",
    )
    return [
        row
        for row in rows
        if any(normalized in row.get(field, "").casefold() for field in searchable_fields)
    ]


def search_similar_voice(uploaded_file) -> list[dict[str, object]]:
    suffix = Path(uploaded_file.name).suffix.lower() or ".wav"
    with tempfile.TemporaryDirectory(prefix="creator-voice-search-") as directory:
        input_path = Path(directory) / f"query{suffix}"
        input_path.write_bytes(uploaded_file.getbuffer())
        from voice_search import find_similar_voices

        return find_similar_voices(
            input_path,
            model_path=MODEL_PATH,
            index_path=INDEX_PATH,
            metadata_path=METADATA_PATH,
            top_k=8,
            num_threads=4,
        )


catalog = load_catalog(str(METADATA_PATH))
st.session_state.setdefault("selected_speaker_id", "")
st.session_state.setdefault("catalog_page", 0)
st.session_state.setdefault("search_results", [])
st.session_state.setdefault("search_upload_signature", None)
st.session_state.setdefault("generated_audio", b"")

st.markdown(
    """
    <section class="voice-hero">
      <h1>Creator 声音库</h1>
      <p>先试听并选定一个 Speaker ID，再进入配音。浏览和选择不会加载合成模型；
      模型只在点击生成时启动。</p>
    </section>
    """,
    unsafe_allow_html=True,
)

voice_tab, tts_tab, search_tab = st.tabs(
    ["声音库", "文字转语音", "相似音色搜索"]
)

with voice_tab:
    match = load_match_status(str(VOICE_MATCHES_PATH))
    if match and match.get("status") == "no_confident_match":
        st.markdown(
            (
                '<div class="match-empty"><strong>摆烂的🐟：库内暂无可信匹配</strong><br>'
                "两轮共试听 10 个候选均已否决。相似度最高约 0.53，"
                "不足以继续作为推荐；请从声音库独立选音，或后续补充新的参考音色。</div>"
            ),
            unsafe_allow_html=True,
        )

    render_selected_voice(catalog)
    filter_col, gender_col = st.columns([2, 1])
    with filter_col:
        query = st.text_input(
            "搜索音色",
            placeholder="输入中文音色名、英文名或 Speaker ID",
        )
    with gender_col:
        gender = st.radio(
            "声音类型",
            ["全部", "男声", "女声"],
            horizontal=True,
        )

    filtered = filter_catalog(catalog, query, gender)
    page_count = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)
    st.session_state.catalog_page = min(
        max(0, st.session_state.catalog_page),
        page_count - 1,
    )
    page = st.session_state.catalog_page
    start = page * PAGE_SIZE
    page_rows = filtered[start : start + PAGE_SIZE]

    status_col, previous_col, next_col = st.columns([4, 1, 1])
    status_col.caption(
        f"共 {len(filtered):,} 个音色 · 第 {page + 1}/{page_count} 页"
    )
    if previous_col.button(
        "上一页",
        use_container_width=True,
        disabled=page == 0,
    ):
        st.session_state.catalog_page -= 1
        st.rerun()
    if next_col.button(
        "下一页",
        use_container_width=True,
        disabled=page >= page_count - 1,
    ):
        st.session_state.catalog_page += 1
        st.rerun()

    if not page_rows:
        st.warning("没有符合条件的音色，请换一个关键词。")
    for offset in range(0, len(page_rows), 2):
        columns = st.columns(2)
        for column, row in zip(columns, page_rows[offset : offset + 2]):
            with column:
                with st.container(border=True):
                    render_voice_card(row, key_prefix="catalog")

with tts_tab:
    selected = render_selected_voice(catalog, include_audio=True)
    with st.form("tts-form"):
        text = st.text_area(
            "配音文字",
            height=150,
            placeholder="输入要合成的中文或英文内容",
        )
        prompt = st.text_input(
            "声音风格",
            value="语气自然、温和、克制，节奏清晰",
            help="描述情绪和语气，不要填写人物身份。",
        )
        submitted = st.form_submit_button(
            "生成语音",
            type="primary",
            use_container_width=True,
            disabled=selected is None,
        )

    if submitted and selected:
        st.session_state.generated_audio = b""
        try:
            with st.spinner("首次生成正在加载本地模型，后续生成会复用模型……"):
                from tts_runtime import synthesize

                st.session_state.generated_audio = synthesize(
                    text,
                    selected["speaker_id"],
                    prompt,
                )
            st.success("语音已生成")
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            st.error(str(error))

    if st.session_state.generated_audio:
        st.audio(st.session_state.generated_audio, format="audio/wav")
        st.download_button(
            "下载 WAV",
            st.session_state.generated_audio,
            file_name="creator-voice.wav",
            mime="audio/wav",
            use_container_width=True,
        )

with search_tab:
    st.write(
        "上传一段单人、少背景音乐的音频或视频。系统只负责返回检索候选，"
        "不再把低相似度结果标记为可信推荐。"
    )
    uploaded = st.file_uploader(
        "参考音视频",
        type=["wav", "mp3", "m4a", "flac", "ogg", "mp4", "mov"],
    )
    upload_signature = None
    if uploaded is not None:
        upload_id = getattr(uploaded, "file_id", None)
        if upload_id is None:
            upload_id = hashlib.sha256(uploaded.getbuffer()).hexdigest()
        upload_signature = (
            upload_id,
            uploaded.name,
            uploaded.size,
        )
    if upload_signature != st.session_state.search_upload_signature:
        st.session_state.search_upload_signature = upload_signature
        st.session_state.search_results = []
    if st.button(
        "开始搜索",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None,
    ):
        try:
            with st.spinner("正在解码一次音频并计算声纹……"):
                st.session_state.search_results = search_similar_voice(uploaded)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            st.error(str(error))

    results = st.session_state.search_results
    if results:
        st.caption("搜索结果仅供试听比较；分数低时应视为库内无匹配。")
        for offset in range(0, len(results), 2):
            columns = st.columns(2)
            for column, result in zip(columns, results[offset : offset + 2]):
                row = find_voice(catalog, str(result["speaker_id"]))
                if not row:
                    continue
                with column:
                    with st.container(border=True):
                        render_voice_card(
                            row,
                            key_prefix="search",
                            rank=int(result["rank"]),
                            similarity=float(result["similarity"]),
                        )
