from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
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
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "output"
    / "data"
    / "tools"
    / "emotivoice"
    / "voice-search"
    / "reference-samples"
)
DEFAULT_TEXT = "今天的故事，从这里开始，让我们慢慢说给你听。"
DEFAULT_PROMPT = "平静，温暖，自然"
MAX_WAV_VALUE = 32768.0


def prepare_app_imports() -> None:
    os.chdir(FORK_ROOT)
    sys.path.insert(0, str(FORK_ROOT))


def scan_checkpoint(directory: Path, prefix: str) -> Path:
    checkpoints = sorted(directory.glob(f"{prefix}*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint matching {prefix} in {directory}")
    return checkpoints[-1]


def load_models():
    from config.joint.config import Config
    from models.prompt_tts_modified.jets import JETSGenerator
    from models.prompt_tts_modified.simbert import StyleEncoder

    config = Config()
    config.output_directory = str(RUNTIME_ROOT / "outputs")
    config.style_encoder_ckpt = str(
        scan_checkpoint(
            RUNTIME_ROOT / "outputs" / "style_encoder" / "ckpt",
            "checkpoint_",
        )
    )
    config.bert_path = str(
        RUNTIME_ROOT / "WangZeJun" / "simbert-base-chinese"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required to build 2,014 reference samples quickly.")

    with open(config.model_config_path, encoding="utf-8") as file:
        model_config = CONFIG.load_cfg(file)
    model_config.n_vocab = config.n_symbols
    model_config.n_speaker = config.speaker_n_labels

    style_encoder = StyleEncoder(config)
    style_checkpoint = torch.load(config.style_encoder_ckpt, map_location="cpu")
    style_state = {
        key[7:]: value for key, value in style_checkpoint["model"].items()
    }
    style_encoder.load_state_dict(style_state, strict=False)
    style_encoder.eval()

    generator = JETSGenerator(model_config).to(device)
    generator_path = scan_checkpoint(
        Path(config.output_directory) / "prompt_tts_open_source_joint" / "ckpt",
        "g_",
    )
    generator_checkpoint = torch.load(generator_path, map_location=device)
    generator.load_state_dict(generator_checkpoint["generator"])
    generator.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    with open(config.token_list_path, encoding="utf-8") as file:
        token_to_id = {
            token.strip(): index for index, token in enumerate(file.readlines())
        }
    return config, device, style_encoder, generator, tokenizer, token_to_id


def style_embedding(text: str, tokenizer, model) -> np.ndarray:
    tokens = tokenizer([text], return_tensors="pt")
    with torch.no_grad():
        output = model(
            input_ids=tokens["input_ids"],
            token_type_ids=tokens["token_type_ids"],
            attention_mask=tokens["attention_mask"],
        )
    return output["pooled_output"].cpu().squeeze().numpy()


def prepare_conditioning(
    text: str,
    prompt: str,
    tokenizer,
    style_encoder,
    token_to_id: dict[str, int],
    device: torch.device,
):
    from frontend import g2p_cn_en
    from frontend_en import G2p, ROOT_DIR, read_lexicon

    lexicon = read_lexicon(f"{ROOT_DIR}/lexicon/librispeech-lexicon.txt")
    phonemes = g2p_cn_en(text, G2p(), lexicon).split()
    unknown = [phoneme for phoneme in phonemes if phoneme not in token_to_id]
    if unknown:
        raise ValueError(f"Unknown phonemes: {unknown}")

    sequence = torch.tensor(
        [token_to_id[phoneme] for phoneme in phonemes],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    sequence_length = torch.tensor([sequence.shape[1]], device=device)
    prompt_embedding = torch.from_numpy(
        style_embedding(prompt, tokenizer, style_encoder)
    ).to(device)
    content_embedding = torch.from_numpy(
        style_embedding(text, tokenizer, style_encoder)
    ).to(device)
    return sequence, sequence_length, prompt_embedding, content_embedding


def generate_batch(
    speaker_indices: list[int],
    sequence: torch.Tensor,
    sequence_length: torch.Tensor,
    prompt_embedding: torch.Tensor,
    content_embedding: torch.Tensor,
    generator,
):
    batch_size = len(speaker_indices)
    with torch.inference_mode():
        output = generator(
            inputs_ling=sequence.repeat(batch_size, 1),
            input_lengths=sequence_length.repeat(batch_size),
            inputs_style_embedding=prompt_embedding.unsqueeze(0).repeat(
                batch_size, 1
            ),
            inputs_content_embedding=content_embedding.unsqueeze(0).repeat(
                batch_size, 1
            ),
            inputs_speaker=torch.tensor(
                speaker_indices,
                dtype=torch.long,
                device=sequence.device,
            ),
            alpha=1.0,
        )

    waveforms = output["wav_predictions"].squeeze(1).cpu().numpy()
    mel_lengths = (
        output["log_duration_predictions"].sum(dim=-1).cpu().numpy().astype(int)
    )
    sample_lengths = mel_lengths * generator.upsample_factor
    return waveforms, sample_lengths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one fixed Chinese reference sample for every EmotiVoice ID."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    prepare_app_imports()
    (
        config,
        device,
        style_encoder,
        generator,
        tokenizer,
        token_to_id,
    ) = load_models()
    conditioning = prepare_conditioning(
        args.text,
        args.prompt,
        tokenizer,
        style_encoder,
        token_to_id,
        device,
    )

    speakers = list(config.speakers)
    stop = len(speakers) if args.limit is None else args.start + args.limit
    selected = list(enumerate(speakers))[args.start:stop]
    args.output.mkdir(parents=True, exist_ok=True)

    pending = [
        (index, speaker_id)
        for index, speaker_id in selected
        if args.force or not (args.output / f"{speaker_id}.wav").is_file()
    ]
    started = time.monotonic()
    completed = len(selected) - len(pending)
    print(
        f"Generating {len(pending)} of {len(selected)} speakers on {device}; "
        f"{completed} already exist."
    )

    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        waveforms, sample_lengths = generate_batch(
            [index for index, _ in batch],
            *conditioning,
            generator,
        )
        for row, ((_, speaker_id), sample_length) in enumerate(
            zip(batch, sample_lengths)
        ):
            waveform = waveforms[row, :sample_length]
            sf.write(
                args.output / f"{speaker_id}.wav",
                np.clip(waveform, -1.0, 1.0),
                config.sampling_rate,
                subtype="PCM_16",
            )
        completed += len(batch)
        elapsed = time.monotonic() - started
        rate = completed / elapsed if elapsed else 0.0
        print(
            f"{completed}/{len(selected)} complete "
            f"({rate:.2f} voices/s, {elapsed / 60:.1f} min)",
            flush=True,
        )


if __name__ == "__main__":
    main()
