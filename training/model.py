"""Trainable embedding model: decoder backbone + LoRA + pooling head.

Two backbones are supported, selected by `NVEmbedBuildConfig.base_kind`:
  * "qwen3":      a Qwen3 decoder, causal attention, last-token pooling (`EOSPoolHead`).
  * "nvembed_v2": the Mistral backbone of NV-Embed-v2 with bidirectional attention, pooled by a
                  freshly initialised latent-attention head (`LatentAttentionModel`).
LoRA is applied to the attention and MLP projections of the backbone; the backbone itself stays
frozen in its load dtype, and the LoRA weights are kept in fp32.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

from transformers import MistralModel, MistralConfig
from transformers.masking_utils import create_bidirectional_mask
from nvembed_layers import LatentAttentionModel, LatentAttentionConfig, BidirectionalMistralConfig

# LoRA targets: every attention and MLP projection. Mistral and Qwen3 use the same module names.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# base_kind -> checkpoint `model_type` values the loader can consume.
_ARCH_BY_KIND = {
    "qwen3": ("qwen3",),
}


def _checkpoint_model_type(name: str) -> Optional[str]:
    """`model_type` from a checkpoint's config.json, without instantiating anything. Returns None
    when it cannot be read; callers then skip the check."""
    import json as _json, os as _os
    try:
        if _os.path.isdir(name):
            p = _os.path.join(name, "config.json")
            if not _os.path.isfile(p):
                return None
            cfg = _json.load(open(p))
        else:
            from huggingface_hub import hf_hub_download
            cfg = _json.load(open(hf_hub_download(name, "config.json")))
    except Exception:
        return None
    mt = cfg.get("model_type")
    if mt == "nvembed" or "text_config" in cfg:
        return "nvembed"
    return mt


def assert_base_kind(name: str, kind: str) -> None:
    """Refuse a base_model whose architecture the selected loader cannot represent. A Mistral-shaped
    checkpoint can load into a Qwen3 class (or the reverse) with only warnings for dropped tensors."""
    mt = _checkpoint_model_type(name)
    allowed = _ARCH_BY_KIND.get(kind)
    if mt is None or allowed is None or mt in allowed:
        return
    raise RuntimeError(
        f"base_kind={kind!r} selects a loader for {allowed}, but {name} is model_type={mt!r}. "
        f"Loading it anyway can succeed silently with dropped tensors. "
        f"Set base_kind to the matching value (qwen3 for Qwen3 checkpoints).")


def assert_clean_load(info: dict, name: str, allow_missing=("inv_freq",)) -> None:
    """Turn from_pretrained's missing/unexpected/mismatched key warnings into an error. Rotary
    `inv_freq` buffers are recomputed at init and are legitimately absent from checkpoints."""
    missing = [k for k in (info.get("missing_keys") or [])
               if not any(k.endswith(s) for s in allow_missing)]
    unexpected = list(info.get("unexpected_keys") or [])
    mismatched = list(info.get("mismatched_keys") or [])
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"{name}: checkpoint/class mismatch — missing={len(missing)} {missing[:4]} | "
            f"unexpected={len(unexpected)} {unexpected[:4]} | mismatched={len(mismatched)} "
            f"{mismatched[:2]}. Refusing to train on a partially-loaded backbone.")


def expected_lora_params(hf_config, r: int, targets=LORA_TARGETS) -> int:
    """Analytic trainable-parameter count for a LoRA over `targets` on a Llama-family decoder.
      Mistral-7B (NV-Embed-v2), r=64, all 7 targets -> 167,772,160
      Qwen3-8B,                 r=64, all 7 targets -> 174,587,904
    """
    h = hf_config.hidden_size
    heads = hf_config.num_attention_heads
    hd = getattr(hf_config, "head_dim", None) or (h // heads)
    kv = getattr(hf_config, "num_key_value_heads", heads)
    inter = hf_config.intermediate_size
    shapes = {"q_proj": (h, heads * hd), "k_proj": (h, kv * hd), "v_proj": (h, kv * hd),
              "o_proj": (heads * hd, h), "gate_proj": (h, inter), "up_proj": (h, inter),
              "down_proj": (inter, h)}
    per_layer = sum(r * shapes[t][0] + shapes[t][1] * r for t in targets)
    return per_layer * hf_config.num_hidden_layers


def assert_trainable_params(model, expected: int, label: str = "") -> int:
    """Hard gate on the trainable-parameter count (PEFT attaching to fewer modules than intended
    does not raise). `expected<=0` disables it."""
    got = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if expected and got != expected:
        raise RuntimeError(
            f"{label}trainable params {got:,} != expected {expected:,}. PEFT attaches to fewer "
            f"modules than intended WITHOUT raising; a mismatch here means the adapter is not on "
            f"the network you think it is.")
    return got


# --------------------------------------------------------------------------- bidirectional Mistral
class BidirectionalMistralModel(MistralModel):
    """Mistral with full (non-causal) self-attention over non-pad tokens, as in NV-Embed.

    The 4D bidirectional mask is built up front with `masking_utils.create_bidirectional_mask`;
    everything else (rotary embeddings, gradient checkpointing, decoder loop, final norm) is the
    stock `MistralModel.forward`, which passes an already-4D mask straight to the layers.

    Mask semantics: key-side padding mask only. No causal triangle and no sliding window
    (`config.sliding_window` is ignored). Under eager the mask holds 0.0 / finfo.min; under sdpa
    it is boolean (True = attend), which gives the same softmax weights because every query row
    has at least one visible key.
    """
    config_class = BidirectionalMistralConfig

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        for layer in self.layers:
            layer.self_attn.is_causal = False   # governs the SDPA/FA2 fast paths

    def _bidirectional_mask(self, inputs_embeds, attention_mask, past_key_values):
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask                       # caller already prepared one
        mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        impl = self.config._attn_implementation
        if mask is None and impl in ("eager", "sdpa"):
            # create_bidirectional_mask returns None when no bias is needed (a batch with no
            # padding). Passing None on would make the stock forward build a causal mask, so build
            # the all-visible 4D mask explicitly.
            b, q = inputs_embeds.shape[0], inputs_embeds.shape[1]
            kv = attention_mask.shape[-1] if attention_mask is not None else q
            if impl == "eager":
                mask = torch.zeros((b, 1, q, kv), dtype=inputs_embeds.dtype,
                                   device=inputs_embeds.device)
            else:
                mask = torch.ones((b, 1, q, kv), dtype=torch.bool, device=inputs_embeds.device)
        return mask

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            # Embed here (not in super) because the mask factory needs dtype/device/shape. PEFT's
            # enable_input_require_grads hook fires on this call.
            inputs_embeds = self.embed_tokens(input_ids)
        bidi_mask = self._bidirectional_mask(inputs_embeds, attention_mask, past_key_values)
        return super().forward(
            input_ids=None,
            attention_mask=bidi_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )


@dataclass
class NVEmbedBuildConfig:
    base_model: str = "mistralai/Mistral-7B-v0.1"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    # latent-attention head (NV-Embed-v2 defaults)
    num_latents: int = 512
    num_cross_heads: int = 8
    cross_dim_head: int = 4096
    latent_dim: int = 4096
    hidden_dim: int = 4096
    output_normalize: bool = True
    # "last" / "eos": last-token pooling (qwen3). "latent": latent-attention head (nvembed_v2).
    pooling: str = "latent"
    # "causal" (qwen3) | "bidirectional" (nvembed_v2)
    attention: str = "bidirectional"
    gradient_checkpointing: bool = True
    dtype: str = "bfloat16"
    # Checkpoint layout of `base_model`:
    #   "nvembed_v2" the released NV-Embed-v2 -> _load_nvembed_v2_backbone. Its backbone tensors
    #                are namespaced under `embedding_model.` and its Mistral config is nested under
    #                config["text_config"], so a plain from_pretrained would match no keys.
    #   "qwen3"      a Qwen3 decoder -> _load_qwen3_backbone (causal, last-token pooling).
    base_kind: str = "nvembed_v2"
    # Attention kernel for the Qwen3 backbone; eager / sdpa / flash_attention_2 differ numerically.
    # Not used by the nvembed_v2 loader.
    attn_implementation: str = "sdpa"
    # Hard gate on the LoRA trainable-parameter count; 0 = use the analytic count.
    expected_trainable_params: int = 0


def _latent_config(cfg: NVEmbedBuildConfig) -> LatentAttentionConfig:
    return LatentAttentionConfig(
        num_latents_value=cfg.num_latents,
        num_cross_heads=cfg.num_cross_heads,
        cross_dim_head=cfg.cross_dim_head,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        output_normalize=cfg.output_normalize,
    )


def _load_nvembed_v2_backbone(name: str, dtype: torch.dtype) -> BidirectionalMistralModel:
    """Load only the bidirectional-Mistral backbone of the released NV-Embed-v2.

    The checkpoint stores the backbone under `embedding_model.` and the pooling head under
    `latent_attention_model.`; config["text_config"] is a full Mistral config. The backbone is
    rebuilt from text_config and loaded from the prefix-stripped tensors. The released pooling
    head is not loaded.
    """
    import glob, json as _json, os as _os
    from safetensors.torch import load_file

    path = name
    if not _os.path.isdir(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(name, allow_patterns=["*.json", "*.safetensors", "*.model"])
    cfg = _json.load(open(_os.path.join(path, "config.json")))
    tcfg = BidirectionalMistralConfig(**cfg["text_config"])
    tcfg.torch_dtype = dtype

    shards = sorted(glob.glob(_os.path.join(path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards under {path}")
    prefix = "embedding_model."
    sd = {}
    for shard in shards:
        for k, v in load_file(shard).items():
            if k.startswith(prefix):
                sd[k[len(prefix):]] = v.to(dtype)
    if not sd:
        raise RuntimeError(f"no '{prefix}*' tensors in {path} — checkpoint layout changed")

    # Build on the meta device (no 7B random-init allocation), then assign the loaded tensors.
    from accelerate import init_empty_weights
    with init_empty_weights():
        emb = BidirectionalMistralModel(tcfg)
    missing, unexpected = emb.load_state_dict(sd, strict=False, assign=True)
    # Rotary inv_freq buffers are recomputed at init and absent from the checkpoint; any other
    # missing tensor means the backbone would train on uninitialised weights.
    bad = [k for k in missing if not k.endswith("inv_freq")]
    if bad:
        raise RuntimeError(f"NV-Embed-v2 backbone missing {len(bad)} weights, e.g. {bad[:5]}")
    if unexpected:
        raise RuntimeError(f"NV-Embed-v2 backbone unexpected {len(unexpected)} keys, e.g. {unexpected[:5]}")
    for m in emb.modules():                  # materialize any meta buffers left by init_empty_weights
        for bname, buf in list(m.named_buffers(recurse=False)):
            if buf is not None and buf.is_meta:
                m.register_buffer(bname, torch.zeros_like(buf, device="cpu"), persistent=False)
    return emb.to(dtype)


# --------------------------------------------------------------------------- Qwen3
def _load_qwen3_backbone(name: str, dtype: torch.dtype, attn_implementation: str = "sdpa"):
    """Load a Qwen3 decoder backbone: causal attention, strict key matching.

    Qwen3 applies per-head RMSNorm to q and k (`q_norm` / `k_norm`), so it must be loaded as
    `Qwen3Model`. The attention kernel is passed explicitly so numerics do not depend on which
    kernels are installed. Every key must match (`assert_clean_load`).
    """
    from transformers import Qwen3Model
    assert_base_kind(name, "qwen3")
    kwargs = dict(dtype=dtype, attn_implementation=attn_implementation,
                  output_loading_info=True)
    model, info = Qwen3Model.from_pretrained(name, **kwargs)
    assert_clean_load(info, name)
    if getattr(model.config, "_attn_implementation", None) != attn_implementation:
        raise RuntimeError(f"{name}: asked for attn_implementation={attn_implementation!r}, got "
                           f"{getattr(model.config, '_attn_implementation', None)!r}")
    return model


class EOSPoolHead(nn.Module):
    """Last-token pooling: the hidden state of the last non-pad token, L2-normalised. No parameters.

    Requires right padding, so the last real token is at index sum(mask)-1. Under causal attention
    that token has attended to the whole sequence, instruction included, so the instruction is not
    masked out of the pool. Interface matches the latent head: forward(hiddens, mask).
    """
    def __init__(self, output_normalize: bool = True):
        super().__init__()
        self.output_normalize = output_normalize

    def forward(self, hiddens, attention_mask=None):
        if attention_mask is None:
            emb = hiddens[:, -1]
        else:
            # A 0 in column 0 means left padding or a zeroed instruction span; either moves
            # sum(mask)-1 off the final token while still returning a valid-looking vector.
            if attention_mask.numel() and not bool(attention_mask[:, 0].all()):
                raise RuntimeError(
                    "EOSPoolHead got a mask with a 0 in column 0 — either LEFT padding, or an "
                    "instruction span zeroed in the pool mask. Both move `sum(mask)-1` off the "
                    "final token while still returning a valid-looking vector. Set "
                    "tokenizer.padding_side='right' and mask_instruction=False for last-token "
                    "pooling.")
            last = attention_mask.long().sum(dim=1) - 1            # (B,)
            last = last.clamp(min=0)
            emb = hiddens[torch.arange(hiddens.size(0), device=hiddens.device), last]
        if self.output_normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb


class TrainableNVEmbed(nn.Module):
    def __init__(self, embedding_model: nn.Module, latent_model: nn.Module,
                 checkpoint_latent: bool = False):
        super().__init__()
        self.embedding_model = embedding_model
        self.latent_attention_model = latent_model    # the pooling head (latent or last-token)
        self.checkpoint_latent = checkpoint_latent    # gradient-checkpoint the pooling head

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                pool_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pool_mask is None:
            pool_mask = attention_mask
        # On CUDA the whole forward (backbone + head) runs under bf16 autocast to bound activation
        # memory; trainable weights are stored in fp32 and the returned embedding is cast to fp32.
        if input_ids.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                if self.checkpoint_latent and self.training:
                    from torch.utils.checkpoint import checkpoint
                    emb = checkpoint(self.latent_attention_model, hidden, pool_mask, use_reentrant=False)
                else:
                    emb = self.latent_attention_model(hidden, pool_mask)
            return emb.float()
        out = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return self.latent_attention_model(hidden, pool_mask)

    def trainable_parameter_count(self) -> dict:
        tot = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        latent = sum(p.numel() for p in self.latent_attention_model.parameters())
        return {"total": tot, "trainable": train, "latent_head": latent,
                "trainable_pct": round(100 * train / tot, 3)}


def build_trainable_nvembed(cfg: NVEmbedBuildConfig) -> TrainableNVEmbed:
    """Build the trainable model: backbone, LoRA adapter, pooling head.

    base_kind="nvembed_v2": the NV-Embed-v2 backbone with bidirectional attention, pooled by a
    latent-attention head that is freshly initialised here (not loaded from the released
    checkpoint) and trained in full, in fp32, alongside the LoRA adapter. The head is built after
    LoRA is attached, so its random initialisation follows the LoRA initialisation in RNG order.

    base_kind="qwen3": a causal Qwen3 backbone with last-token pooling; the head has no parameters.
    """
    dtype = getattr(torch, cfg.dtype)

    kind = cfg.base_kind
    targets = LORA_TARGETS
    if kind == "qwen3":
        # Last-token pooling and causal attention are coupled: the last token is the only one that
        # has seen the whole sequence.
        if cfg.pooling not in ("eos", "last"):
            raise ValueError(
                f"base_kind='qwen3' requires pooling='last' (or 'eos'); got {cfg.pooling!r}.")
        if cfg.attention == "bidirectional":
            raise ValueError("base_kind='qwen3' requires attention='causal'.")
        emb = _load_qwen3_backbone(cfg.base_model, dtype,
                                   attn_implementation=cfg.attn_implementation)
    elif kind == "nvembed_v2":
        if cfg.attention == "causal":
            raise ValueError("base_kind='nvembed_v2' requires attention='bidirectional'.")
        if cfg.pooling != "latent":
            raise ValueError(f"base_kind='nvembed_v2' requires pooling='latent'; got {cfg.pooling!r}.")
        emb = _load_nvembed_v2_backbone(cfg.base_model, dtype)
    else:
        raise ValueError(f"unsupported base_kind {kind!r}; supported: 'qwen3', 'nvembed_v2'.")

    from peft import LoraConfig, get_peft_model

    # LoRA on the backbone projections; the latent head (when present) is trained in full.
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      bias="none", target_modules=targets, task_type=None)
    emb = get_peft_model(emb, lora)
    # PEFT attaching to fewer modules than intended does not raise, so check the count.
    _exp = (cfg.expected_trainable_params
            or expected_lora_params(emb.config, cfg.lora_r, targets))
    assert_trainable_params(emb, _exp, label=f"{cfg.base_model} LoRA r={cfg.lora_r}: ")
    # LoRA adapters as fp32 master weights (the frozen base stays in `dtype`); autocast casts in forward.
    for n, p in emb.named_parameters():
        if "lora_" in n and p.requires_grad:
            p.data = p.data.float()

    if cfg.gradient_checkpointing:
        emb.enable_input_require_grads()
        emb.gradient_checkpointing_enable()

    if cfg.pooling in ("eos", "last"):
        head = EOSPoolHead(output_normalize=cfg.output_normalize)
        ckpt_head = False
    else:
        lat_cfg = _latent_config(cfg)
        lat_cfg.hidden_dim = cfg.hidden_dim
        lat_cfg.latent_dim = cfg.hidden_dim
        head = LatentAttentionModel(lat_cfg).float()  # freshly initialised head: fp32 master weights
        for p in head.parameters():
            p.requires_grad_(True)
        ckpt_head = cfg.gradient_checkpointing

    return TrainableNVEmbed(emb, head, checkpoint_latent=ckpt_head)
