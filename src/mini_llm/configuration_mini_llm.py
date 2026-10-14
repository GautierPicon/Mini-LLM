from transformers import PretrainedConfig


class MiniLlmConfig(PretrainedConfig):
    model_type = "mini_llm"

    def __init__(
        self,
        vocab_size=16384,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=1408,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            eos_token_id=kwargs.pop("eos_token_id", 0),
            pad_token_id=kwargs.pop("pad_token_id", 0),
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta

    @classmethod
    def from_cfg(cls, cfg, **kwargs):
        return cls(
            vocab_size=cfg.vocab,
            hidden_size=cfg.dim,
            num_hidden_layers=cfg.layers,
            num_attention_heads=cfg.heads,
            intermediate_size=cfg.ffn,
            max_position_embeddings=cfg.ctx,
            **kwargs,
        )
