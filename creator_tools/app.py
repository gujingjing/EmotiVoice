from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer
from yacs import config as CONFIG


FORK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CREATOR_ROOT = (
    FORK_ROOT.parents[1]
    if FORK_ROOT.parent.name == "third_party"
    else FORK_ROOT
)
PROJECT_ROOT = Path(
    os.environ.get("CREATOR_ROOT", str(DEFAULT_CREATOR_ROOT))
).resolve()
RUNTIME_ROOT = Path(
    os.environ.get("EMOTIVOICE_RUNTIME_ROOT", str(FORK_ROOT))
).resolve()
VOICE_SEARCH_ROOT = (
    PROJECT_ROOT / "output" / "data" / "tools" / "emotivoice" / "voice-search"
)
METADATA_PATH = Path(
    os.environ.get(
        "EMOTIVOICE_METADATA_PATH",
        str(FORK_ROOT / "creator_tools" / "data" / "speaker_metadata.csv"),
    )
).resolve()
REFERENCE_DIR = VOICE_SEARCH_ROOT / "reference-samples"
INDEX_PATH = VOICE_SEARCH_ROOT / "index" / "emotivoice-speakers.npz"
VOICE_SEARCH_PYTHON = Path(
    os.environ.get(
        "EMOTIVOICE_SEARCH_PYTHON",
        "/home/jingjinggu/.local/share/creator/voice-search-env/bin/python",
    )
)
VOICE_SEARCH_SCRIPT = FORK_ROOT / "creator_tools" / "voice_search.py"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_WAV_VALUE = 32768.0

os.chdir(FORK_ROOT)
sys.path.insert(0, str(FORK_ROOT))

from config.joint.config import Config  # noqa: E402
from frontend import ROOT_DIR, G2p, g2p_cn_en, read_lexicon  # noqa: E402
from models.prompt_tts_modified.jets import JETSGenerator  # noqa: E402
from models.prompt_tts_modified.simbert import StyleEncoder  # noqa: E402


config = Config()
config.output_directory = str(RUNTIME_ROOT / "outputs")
config.bert_path = str(RUNTIME_ROOT / "WangZeJun" / "simbert-base-chinese")


def scan_checkpoint(directory: str, prefix: str, digits: int = 8) -> str:
    pattern = os.path.join(directory, prefix + "?" * digits)
    checkpoints = sorted(glob.glob(pattern))
    if not checkpoints:
        raise FileNotFoundError(f"找不到模型文件：{pattern}")
    return checkpoints[-1]


@st.cache_data
def load_speaker_metadata() -> list[dict[str, str]]:
    if not METADATA_PATH.is_file():
        return [
            {
                "speaker_id": speaker_id,
                "english_name": "",
                "gender": "",
                "gender_zh": "未知",
                "chinese_name": f"音色{index + 1:04d}",
                "description_zh": "",
                "display_name": f"音色{index + 1:04d}｜ID {speaker_id}",
                "voice_profile": "",
            }
            for index, speaker_id in enumerate(config.speakers)
        ]
    with METADATA_PATH.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


@st.cache_resource
def load_tts_models():
    generator_checkpoint = scan_checkpoint(
        f"{config.output_directory}/prompt_tts_open_source_joint/ckpt",
        "g_",
    )
    style_checkpoint = scan_checkpoint(
        f"{config.output_directory}/style_encoder/ckpt",
        "checkpoint_",
        6,
    )
    with open(config.model_config_path, encoding="utf-8") as file:
        model_config = CONFIG.load_cfg(file)
    model_config.n_vocab = config.n_symbols
    model_config.n_speaker = config.speaker_n_labels

    style_encoder = StyleEncoder(config)
    checkpoint = torch.load(style_checkpoint, map_location="cpu")
    style_encoder.load_state_dict(
        {key[7:]: value for key, value in checkpoint["model"].items()},
        strict=False,
    )
    style_encoder.eval()

    generator = JETSGenerator(model_config).to(DEVICE)
    checkpoint = torch.load(generator_checkpoint, map_location=DEVICE)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    with open(config.token_list_path, encoding="utf-8") as file:
        token_to_id = {
            token.strip(): index for index, token in enumerate(file.readlines())
        }
    with open(config.speaker2id_path, encoding="utf-8") as file:
        speaker_to_id = {
            speaker.strip(): index
            for index, speaker in enumerate(file.readlines())
        }
    lexicon = read_lexicon(f"{ROOT_DIR}/lexicon/librispeech-lexicon.txt")
    return (
        style_encoder,
        generator,
        tokenizer,
        token_to_id,
        speaker_to_id,
        lexicon,
        G2p(),
    )


def get_style_embedding(text: str, tokenizer, style_encoder) -> np.ndarray:
    tokens = tokenizer([text], return_tensors="pt")
    with torch.no_grad():
        output = style_encoder(
            input_ids=tokens["input_ids"],
            token_type_ids=tokens["token_type_ids"],
            attention_mask=tokens["attention_mask"],
        )
    return output["pooled_output"].cpu().squeeze().numpy()


def synthesize(
    speaker_id: str,
    text: str,
    prompt: str,
    models,
) -> np.ndarray:
    (
        style_encoder,
        generator,
        tokenizer,
        token_to_id,
        speaker_to_id,
        lexicon,
        g2p,
    ) = models
    phonemes = g2p_cn_en(text, g2p, lexicon).split()
    text_ids = [token_to_id[phoneme] for phoneme in phonemes]

    sequence = torch.tensor(text_ids, device=DEVICE).long().unsqueeze(0)
    sequence_length = torch.tensor([len(text_ids)], device=DEVICE)
    style = torch.from_numpy(
        get_style_embedding(prompt, tokenizer, style_encoder)
    ).to(DEVICE).unsqueeze(0)
    content = torch.from_numpy(
        get_style_embedding(text, tokenizer, style_encoder)
    ).to(DEVICE).unsqueeze(0)
    speaker = torch.tensor([speaker_to_id[speaker_id]], device=DEVICE)

    with torch.inference_mode():
        output = generator(
            inputs_ling=sequence,
            input_lengths=sequence_length,
            inputs_style_embedding=style,
            inputs_content_embedding=content,
            inputs_speaker=speaker,
            alpha=1.0,
        )
    audio = output["wav_predictions"].squeeze() * MAX_WAV_VALUE
    return audio.cpu().numpy().astype("int16")


def filter_speakers(
    metadata: list[dict[str, str]],
    gender: str,
    query: str,
) -> list[str]:
    query = query.strip().casefold()
    rows = [
        row
        for row in metadata
        if gender == "全部" or row["gender_zh"] == gender
    ]
    if query:
        rows = [
            row
            for row in rows
            if query
            in " ".join(
                [
                    row["speaker_id"],
                    row["chinese_name"],
                    row["english_name"],
                    row["voice_profile"],
                ]
            ).casefold()
        ]
    return [row["speaker_id"] for row in rows]


def run_voice_search(uploaded_file, top_k: int) -> list[dict[str, object]]:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as file:
        file.write(uploaded_file.getbuffer())
        input_path = Path(file.name)
    environment = os.environ.copy()
    voice_env_bin = str(VOICE_SEARCH_PYTHON.parent)
    voice_env_lib = str(VOICE_SEARCH_PYTHON.parent.parent / "lib")
    environment["PATH"] = (
        f"{voice_env_bin}:{environment.get('PATH', '')}"
    )
    environment["LD_LIBRARY_PATH"] = voice_env_lib
    try:
        result = subprocess.run(
            [
                str(VOICE_SEARCH_PYTHON),
                str(VOICE_SEARCH_SCRIPT),
                "search",
                str(input_path),
                "--top-k",
                str(top_k),
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        return json.loads(result.stdout)
    finally:
        input_path.unlink(missing_ok=True)


def render_tts_tab(metadata: list[dict[str, str]]) -> None:
    metadata_by_id = {row["speaker_id"]: row for row in metadata}
    filters = st.columns([1, 2])
    with filters[0]:
        gender = st.selectbox("性别筛选", ["全部", "女声", "男声"])
    with filters[1]:
        query = st.text_input(
            "搜索音色",
            placeholder="输入中文音色名、官方英文名或 Speaker ID",
        )

    speaker_ids = filter_speakers(metadata, gender, query)
    if not speaker_ids:
        st.warning("没有符合条件的音色，请更换关键词。")
        return

    speaker_id = st.selectbox(
        f"选择音色（当前 {len(speaker_ids)} 个）",
        speaker_ids,
        format_func=lambda value: metadata_by_id[value]["display_name"],
    )
    selected = metadata_by_id[speaker_id]
    detail = " · ".join(
        value
        for value in [
            selected["gender_zh"],
            selected["voice_profile"],
            selected["description_zh"],
        ]
        if value
    )
    st.caption(detail or "该音色暂无额外描述。")

    prompt = st.text_input(
        "情绪提示",
        value="平静，温暖，自然",
        help="可以输入开心、悲伤、坚定、温柔，或一句自然语言描述。",
    )
    text = st.text_area(
        "合成文本",
        value="今天的故事，从这里开始，让我们慢慢说给你听。",
        height=120,
    )
    if st.button("生成语音", type="primary", use_container_width=True):
        if not text.strip():
            st.error("请输入需要合成的文本。")
            return
        with st.spinner("正在加载模型并生成语音..."):
            audio = synthesize(
                speaker_id,
                text.strip(),
                prompt.strip() or "自然",
                load_tts_models(),
            )
        st.audio(audio, sample_rate=config.sampling_rate)
        st.success(f"已使用 {selected['chinese_name']}（ID {speaker_id}）生成。")


def render_search_tab(metadata: list[dict[str, str]]) -> None:
    st.write("上传一段单人说话音频或视频，系统会返回最接近的 EmotiVoice 音色。")
    st.caption(
        "建议使用 5–30 秒、背景音乐较轻的片段。结果表示音色接近度，不用于身份确认。"
    )
    uploaded = st.file_uploader(
        "上传声音",
        type=["wav", "mp3", "m4a", "flac", "ogg", "mp4", "mov"],
    )
    top_k = st.slider("候选数量", min_value=3, max_value=12, value=8)
    if not uploaded:
        return
    upload_id = hashlib.sha256(uploaded.getbuffer()).hexdigest()
    if st.session_state.get("voice_search_upload_id") != upload_id:
        st.session_state["voice_search_upload_id"] = upload_id
        st.session_state.pop("voice_search_results", None)
    if not INDEX_PATH.is_file() or not VOICE_SEARCH_PYTHON.is_file():
        st.error("音色检索索引尚未完成，请先运行索引构建脚本。")
        return
    if st.button("搜索相近音色", type="primary", use_container_width=True):
        with st.spinner("正在提取声纹并搜索 2,014 个音色..."):
            try:
                results = run_voice_search(uploaded, top_k)
            except subprocess.CalledProcessError as error:
                message = error.stderr.strip() or error.stdout.strip()
                st.error(f"声音分析失败：{message}")
                return
            except json.JSONDecodeError as error:
                st.error(f"检索结果格式错误：{error}")
                return
        st.session_state["voice_search_results"] = results

    results = st.session_state.get("voice_search_results")
    if not results:
        return

    st.dataframe(
        [
            {
                "排名": row["rank"],
                "中文音色名": row["chinese_name"],
                "性别": row["gender"],
                "Speaker ID": row["speaker_id"],
                "官方英文名": row["english_name"],
                "声音特征": row["voice_profile"],
                "相似度": row["similarity"],
            }
            for row in results
        ],
        use_container_width=True,
        hide_index=True,
    )
    result_by_id = {row["speaker_id"]: row for row in results}
    preview_id = st.selectbox(
        "试听候选音色",
        list(result_by_id),
        format_func=lambda value: (
            f"{result_by_id[value]['rank']}. "
            f"{result_by_id[value]['chinese_name']}｜ID {value}"
        ),
    )
    sample_path = REFERENCE_DIR / f"{preview_id}.wav"
    if sample_path.is_file():
        st.audio(str(sample_path))


st.set_page_config(
    page_title="EmotiVoice 本地配音",
    layout="wide",
)
st.title("EmotiVoice 本地配音")
st.caption(
    "2,014 个离线音色 · 中文/英文混合朗读 · RTX 3050 本地生成 · "
    "CAM++ 相似音色搜索"
)

speaker_metadata = load_speaker_metadata()
tts_tab, search_tab = st.tabs(["文字转语音", "相似音色搜索"])
with tts_tab:
    render_tts_tab(speaker_metadata)
with search_tab:
    render_search_tab(speaker_metadata)
