from .kimi_k3 import KimiK3, KimiK3Block, KimiK3Config, KimiK3ForCausalLM
from .mtp import MTPBlock
from .transformer import Transformer, TransformerBlock, TransformerConfig, TransformerForCausalLM

ARCHITECTURES = {
    "transformer": (TransformerConfig, TransformerForCausalLM),
    "kimi_k3": (KimiK3Config, KimiK3ForCausalLM),
}
