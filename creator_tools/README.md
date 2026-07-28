# Creator voice tools

This extension keeps the upstream EmotiVoice synthesis engine intact and adds:

- a searchable Streamlit UI with stable Chinese aliases, gender, official voice
  names, and speaker IDs;
- a local CAM++ speaker-embedding index for WAV, MP3, and MP4 similarity search;
- batch reference generation for all EmotiVoice speakers;
- batch matching of local creator videos to nearby EmotiVoice voices.

## Runtime layout

The Python source lives in this fork. Large checkpoints, generated reference
audio, the ONNX model, and local reports stay outside Git:

```text
<runtime-root>/
├── outputs/
└── WangZeJun/

<creator-root>/output/data/tools/emotivoice/voice-search/
├── index/
├── models/
└── reference-samples/
```

Set these environment variables when the repository is embedded in another
workspace:

| Variable | Purpose |
| --- | --- |
| `CREATOR_ROOT` | Workspace containing `output/data/tools/emotivoice` |
| `EMOTIVOICE_RUNTIME_ROOT` | Directory containing `outputs` and `WangZeJun` |
| `EMOTIVOICE_SEARCH_PYTHON` | Python executable with `sherpa-onnx` installed |

## Commands

Build the voice metadata:

```bash
python creator_tools/build_metadata.py
```

Generate fixed Chinese reference clips:

```bash
python creator_tools/generate_references.py
```

Install and query the voice index:

```bash
python -m pip install -r creator_tools/requirements.voice-search.txt
python creator_tools/voice_search.py index
python creator_tools/voice_search.py search sample.mp4 --top-k 8
```

Run the enhanced UI:

```bash
streamlit run creator_tools/app.py --server.address=127.0.0.1
```

Similarity scores rank acoustic proximity only. They do not identify a person;
music, reverberation, multiple speakers, and real-versus-synthetic domain
differences can change the ranking.
