import numpy as np

from creator_tools.voice_search import normalize, search_embedding


def test_search_embedding_ranks_exact_voice_first():
    speaker_ids = np.asarray(["a", "b"])
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    metadata = [
        {
            "speaker_id": "a",
            "chinese_name": "女声·清亮自然·0001",
            "english_name": "Alice",
            "gender_zh": "女声",
            "voice_profile": "清亮、自然",
        },
        {
            "speaker_id": "b",
            "chinese_name": "男声·低沉自然·0001",
            "english_name": "Bob",
            "gender_zh": "男声",
            "voice_profile": "低沉、自然",
        },
    ]

    results = search_embedding(
        normalize(np.asarray([1.0, 0.0])),
        speaker_ids,
        embeddings,
        metadata,
        top_k=2,
    )

    assert results[0]["speaker_id"] == "a"
    assert results[0]["similarity"] == 1.0
