from __future__ import annotations

import io
import glob
import os
import sys
from functools import lru_cache
from pathlib import Path


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
    os.environ.get(
        "EMOTIVOICE_RUNTIME_ROOT",
        PROJECT_ROOT / "output" / "data" / "tools" / "emotivoice" / "app",
    )
).resolve()
WINDOWS_RUNTIME_ROOT = (
    PROJECT_ROOT
    / "output"
    / "data"
    / "tools"
    / "emotivoice"
    / "windows-runtime"
)
MODEL_PATH = (
    RUNTIME_ROOT
    / "outputs"
    / "prompt_tts_open_source_joint"
    / "ckpt"
    / "g_00140000"
)
MAX_WAV_VALUE = 32768.0


def _require_runtime_files() -> None:
    required = [
        MODEL_PATH,
        RUNTIME_ROOT / "outputs" / "style_encoder" / "ckpt" / "checkpoint_163431",
        RUNTIME_ROOT / "WangZeJun" / "simbert-base-chinese",
        RUNTIME_ROOT / "data" / "youdao" / "text" / "speaker2",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "EmotiVoice 运行文件不完整：\n" + "\n".join(missing)
        )


@lru_cache(maxsize=1)
def load_runtime():
    _require_runtime_files()
    if str(FORK_ROOT) not in sys.path:
        sys.path.insert(0, str(FORK_ROOT))

    import torch
    from transformers import AutoTokenizer
    from yacs import config as CONFIG

    previous_directory = Path.cwd()
    os.chdir(RUNTIME_ROOT)
    try:
        from config.joint.config import Config
    finally:
        os.chdir(previous_directory)
    from frontend import G2p, read_lexicon
    import jieba
    from models.prompt_tts_modified.jets import JETSGenerator
    from models.prompt_tts_modified.simbert import StyleEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    config.output_directory = str(RUNTIME_ROOT / "outputs")
    config.bert_path = str(RUNTIME_ROOT / "WangZeJun" / "simbert-base-chinese")
    config.model_config_path = str(
        RUNTIME_ROOT / "config" / "joint" / "config.yaml"
    )

    with Path(config.model_config_path).open(encoding="utf-8") as file:
        model_config = CONFIG.load_cfg(file)
    model_config.n_vocab = config.n_symbols
    model_config.n_speaker = config.speaker_n_labels

    style_encoder = StyleEncoder(config).to(device)
    style_checkpoint_path = _scan_checkpoint(
        Path(config.output_directory) / "style_encoder" / "ckpt",
        "checkpoint_",
        6,
    )
    style_checkpoint = torch.load(style_checkpoint_path, map_location="cpu")
    style_encoder.load_state_dict(
        {
            key[7:]: value
            for key, value in style_checkpoint["model"].items()
        },
        strict=False,
    )
    style_encoder.eval()

    model = JETSGenerator(model_config).to(device)
    generator_checkpoint = _scan_checkpoint(
        Path(config.output_directory) / "prompt_tts_open_source_joint" / "ckpt",
        "g_",
        8,
    )
    checkpoint = torch.load(generator_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["generator"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    with Path(config.token_list_path).open(encoding="utf-8") as file:
        token_to_id = {
            token.strip(): index for index, token in enumerate(file)
        }
    with Path(config.speaker2id_path).open(encoding="utf-8") as file:
        speaker_to_id = {
            speaker.strip(): index for index, speaker in enumerate(file)
        }
    jieba_cache = WINDOWS_RUNTIME_ROOT / "cache" / "jieba"
    jieba_cache.mkdir(parents=True, exist_ok=True)
    jieba.dt.tmp_dir = str(jieba_cache)
    jieba.dt.cache_file = str(jieba_cache / "jieba.cache")
    lexicon = read_lexicon(str(FORK_ROOT / "lexicon" / "librispeech-lexicon.txt"))
    g2p = G2p()
    return {
        "torch": torch,
        "device": device,
        "config": config,
        "model": model,
        "tokenizer": tokenizer,
        "style_encoder": style_encoder,
        "token_to_id": token_to_id,
        "speaker_to_id": speaker_to_id,
        "lexicon": lexicon,
        "g2p": g2p,
    }


def _scan_checkpoint(directory: Path, prefix: str, digits: int) -> Path:
    pattern = str(directory / f"{prefix}{'?' * digits}")
    checkpoints = sorted(Path(path) for path in glob.glob(pattern))
    if not checkpoints:
        raise FileNotFoundError(f"找不到模型文件：{pattern}")
    return checkpoints[-1]


def _style_embedding(runtime, prompt: str):
    torch = runtime["torch"]
    tokenizer = runtime["tokenizer"]
    style_encoder = runtime["style_encoder"]
    device = runtime["device"]
    encoded = tokenizer(
        [prompt],
        return_tensors="pt",
    )
    with torch.no_grad():
        output = style_encoder(
            input_ids=encoded["input_ids"].to(device),
            token_type_ids=encoded["token_type_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )
    return output["pooled_output"].squeeze(0)


def _phoneme_ids(runtime, text: str):
    from frontend import g2p_cn_en

    torch = runtime["torch"]
    phonemes = g2p_cn_en(text, runtime["g2p"], runtime["lexicon"])
    tokens = [token for token in phonemes.split() if token]
    token_to_id = runtime["token_to_id"]
    unknown = [token for token in tokens if token not in token_to_id]
    if unknown:
        raise ValueError(f"无法转换这些发音单元：{'、'.join(sorted(set(unknown)))}")
    return torch.LongTensor(
        [[token_to_id[token] for token in tokens]]
    ).to(runtime["device"])


def synthesize(text: str, speaker_id: str, prompt: str) -> bytes:
    text = text.strip()
    prompt = prompt.strip()
    if not text:
        raise ValueError("请输入要合成的文字。")
    if not prompt:
        raise ValueError("请输入声音风格描述。")

    runtime = load_runtime()
    torch = runtime["torch"]
    config = runtime["config"]
    try:
        speaker_index = runtime["speaker_to_id"][str(speaker_id)]
    except KeyError as error:
        raise ValueError(f"未知 Speaker ID：{speaker_id}") from error

    inputs = _phoneme_ids(runtime, text)
    style_embedding = _style_embedding(runtime, prompt)
    content_embedding = _style_embedding(runtime, text)
    speaker = torch.LongTensor([speaker_index]).to(runtime["device"])
    lengths = torch.LongTensor([inputs.size(1)]).to(runtime["device"])

    with torch.no_grad():
        output = runtime["model"](
            inputs_ling=inputs,
            input_lengths=lengths,
            inputs_speaker=speaker,
            inputs_style_embedding=style_embedding.unsqueeze(0),
            inputs_content_embedding=content_embedding.unsqueeze(0),
            alpha=1.0,
        )
    waveform = (
        output["wav_predictions"].squeeze().detach().cpu() * MAX_WAV_VALUE
    ).numpy().astype("int16")

    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, waveform, config.sampling_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
