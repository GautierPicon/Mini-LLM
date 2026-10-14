import torch

from mini_llm.config import Cfg
from mini_llm.configuration_mini_llm import MiniLlmConfig
from mini_llm.model import LLM
from mini_llm.modeling_mini_llm import MiniLlmForCausalLM


def tiny_cfg():
    return Cfg(vocab=256, dim=32, layers=2, heads=4, ffn=64, ctx=32)


def test_config_round_trip():
    cfg = tiny_cfg()
    hf = MiniLlmConfig.from_cfg(cfg)
    assert (hf.vocab_size, hf.hidden_size, hf.num_hidden_layers) == (256, 32, 2)
    assert (hf.num_attention_heads, hf.intermediate_size, hf.max_position_embeddings) == (4, 64, 32)
    assert hf.tie_word_embeddings is True


def test_logits_match_native():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    native = LLM(cfg).eval()
    hf = MiniLlmForCausalLM(MiniLlmConfig.from_cfg(cfg)).eval()
    missing, unexpected = hf.load_state_dict(native.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) <= {"head.weight", "cos", "sin"}
    hf.tie_weights()
    x = torch.randint(0, cfg.vocab, (2, 16))
    with torch.no_grad():
        ref, _ = native(x)
        got = hf(input_ids=x).logits
    assert torch.isfinite(got).all()
    assert (ref.float() - got.float()).abs().max().item() < 1e-5


def test_save_reload_round_trip(tmp_path):
    torch.manual_seed(1)
    cfg = tiny_cfg()
    native = LLM(cfg).eval()
    hf = MiniLlmForCausalLM(MiniLlmConfig.from_cfg(cfg)).eval()
    hf.load_state_dict(native.state_dict(), strict=False)
    hf.tie_weights()
    hf.save_pretrained(str(tmp_path), safe_serialization=True)
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "config.json").exists()
    from transformers import AutoModelForCausalLM

    rt = AutoModelForCausalLM.from_pretrained(str(tmp_path), dtype=torch.float32).eval()
    assert torch.isfinite(rt.cos).all() and torch.isfinite(rt.sin).all()
    x = torch.randint(0, cfg.vocab, (1, 24))
    with torch.no_grad():
        a = hf(input_ids=x).logits.float()
        b = rt(input_ids=x).logits.float()
    assert (a - b).abs().max().item() < 1e-5


def test_labels_match_native_loss():
    torch.manual_seed(2)
    cfg = tiny_cfg()
    native = LLM(cfg).eval()
    hf = MiniLlmForCausalLM(MiniLlmConfig.from_cfg(cfg)).eval()
    hf.load_state_dict(native.state_dict(), strict=False)
    hf.tie_weights()
    x = torch.randint(0, cfg.vocab, (2, 12))
    y = torch.randint(0, cfg.vocab, (2, 12))
    with torch.no_grad():
        _, ref_loss = native(x, y)
        got_loss = hf(input_ids=x, labels=y).loss
    assert abs(ref_loss.item() - got_loss.item()) < 1e-5
