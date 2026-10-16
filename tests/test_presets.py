import gc

import numpy as np
import pytest
import torch

from mini_llm.config import Cfg, get_preset, resolve_device
from mini_llm.model import LLM


def test_preset_param_counts_frozen():
    # 34M/70M presets must never change; 153M is the new ~152.6M target.
    for name, expected in [("34M", 34_087_424), ("70M", 69_975_680)]:
        m = LLM(get_preset(name))
        assert sum(p.numel() for p in m.parameters()) == expected
        del m
        gc.collect()
    m = LLM(get_preset("153M"))
    n = sum(p.numel() for p in m.parameters())
    assert m.c.vocab == 32768
    assert 152_000_000 < n < 153_500_000, n
    del m
    gc.collect()


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    if torch.cuda.is_available():
        assert resolve_device("cuda") == "cuda"
    else:
        with pytest.raises(SystemExit):
            resolve_device("cuda")
    with pytest.raises(SystemExit):
        resolve_device("tpu")
    assert resolve_device("auto") in ("cuda", "cpu")
    assert resolve_device(None) in ("cuda", "cpu")


def test_data_vocab_guard(tmp_path):
    import json

    from mini_llm.pretrain import check_data_matches, check_ids_fit

    meta = tmp_path / "meta.json"
    tok = tmp_path / "tok.json"
    tok.write_bytes(b"fake-tokenizer")
    cfg = Cfg(vocab=32768, dim=32, layers=1, heads=2, ffn=32, ctx=16)

    meta.write_text(json.dumps({"vocab": 16384, "tokenizer_sha256": "x"}))
    with pytest.raises(SystemExit):
        check_data_matches(meta, cfg, tok)

    import hashlib

    h = hashlib.sha256(b"fake-tokenizer").hexdigest()
    meta.write_text(json.dumps({"vocab": 32768, "tokenizer_sha256": "dead"}))
    with pytest.raises(SystemExit):
        check_data_matches(meta, cfg, tok)

    meta.write_text(json.dumps({"vocab": 32768, "tokenizer_sha256": h}))
    check_data_matches(meta, cfg, tok)  # must not raise

    train = tmp_path / "train.bin"
    val = tmp_path / "val.bin"
    np.asarray([1, 2, 3], dtype=np.uint16).tofile(train)
    np.asarray([4, 5], dtype=np.uint16).tofile(val)
    mm = lambda p: np.memmap(str(p), dtype=np.uint16, mode="r")
    check_ids_fit(mm(train), mm(val), cfg)  # fits, no raise
    with pytest.raises(SystemExit):
        check_ids_fit(mm(train), mm(val), Cfg(vocab=4, dim=32, layers=1, heads=2, ffn=32, ctx=16))


def test_prefetch_propagates_exception():
    from mini_llm.prep import prefetch

    def boom():
        yield 1
        raise ValueError("x")

    with pytest.raises(ValueError):
        list(prefetch(boom()))
