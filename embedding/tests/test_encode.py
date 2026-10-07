"""CPU unit tests for encode.py. Run: python -m pytest tests/ -q

NVEMBED_AUTHORS_CODE may point at modeling_nvembed.py inside the NV-Embed-v2 download (next to
configuration_nvembed.py); the latent-head test
compares our pooling head with the authors' class, and is skipped when the file is absent.
"""
import importlib
import importlib.machinery
import importlib.util
import os
import sys

import pytest
import torch
from transformers import AutoTokenizer, MistralConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import encode  # noqa: E402

AUTHORS = os.environ.get("NVEMBED_AUTHORS_CODE", "")
BERT = os.environ.get("TEST_BERT_TOKENIZER", "")      # e.g. a MedCPT tower directory
DECODER = os.environ.get("TEST_DECODER_TOKENIZER", "")  # e.g. a BMRetriever-7B snapshot


def tiny_mistral(impl):
    cfg = MistralConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, vocab_size=100, max_position_embeddings=64, sliding_window=4)
    cfg._attn_implementation = impl
    torch.manual_seed(0)
    return encode.BidirectionalMistral(cfg).eval()


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_bidirectional_sees_later_tokens(impl):
    m = tiny_mistral(impl)
    a = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]])
    b = a.clone()
    b[0, -1] = 42
    ones = torch.ones_like(a)
    ha = m(input_ids=a, attention_mask=ones).last_hidden_state
    hb = m(input_ids=b, attention_mask=ones).last_hidden_state
    assert not torch.allclose(ha[0, 0], hb[0, 0]), "first token must see the last one"


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_right_padding_does_not_leak(impl):
    m = tiny_mistral(impl)
    x = torch.tensor([[5, 6, 7, 8, 9, 10]])
    padded = torch.tensor([[5, 6, 7, 8, 9, 10, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    h1 = m(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state
    h2 = m(input_ids=padded, attention_mask=mask).last_hidden_state[:, :6]
    assert torch.allclose(h1, h2, atol=1e-5)


def test_eager_equals_sdpa():
    x = torch.tensor([[5, 6, 7, 8, 9, 10, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8]])
    mask = torch.tensor([[1] * 6 + [0, 0], [1] * 8])
    he = tiny_mistral("eager")(input_ids=x, attention_mask=mask).last_hidden_state
    hs = tiny_mistral("sdpa")(input_ids=x, attention_mask=mask).last_hidden_state
    keep = mask.bool()
    assert torch.allclose(he[keep], hs[keep], atol=1e-5)


@pytest.mark.skipif(not os.path.isfile(AUTHORS), reason="set NVEMBED_AUTHORS_CODE")
def test_latent_head_matches_authors():
    pkg = os.path.dirname(AUTHORS)
    spec = importlib.util.spec_from_file_location("nvembed_authors", os.path.join(pkg, "__init__.py"),
                                                  submodule_search_locations=[pkg])
    if not os.path.exists(os.path.join(pkg, "__init__.py")):
        spec = importlib.machinery.ModuleSpec("nvembed_authors", None, is_package=True)
        spec.submodule_search_locations = [pkg]
    sys.modules["nvembed_authors"] = importlib.util.module_from_spec(spec)
    mod = importlib.import_module("nvembed_authors.modeling_nvembed")
    cfg = mod.LatentAttentionConfig(num_latents_value=8, num_cross_heads=2, cross_dim_head=16, latent_dim=32,
                                    hidden_dim=32, output_normalize=True)
    torch.manual_seed(0)
    ref = mod.LatentAttentionModel(cfg).eval()
    ours = encode.LatentAttentionPool(dim=32, heads=2, dim_head=16, latents=8).eval()
    ours.load_state_dict(ref.state_dict(), strict=True)
    hidden = torch.randn(3, 7, 32)
    mask = torch.tensor([[1] * 7, [0, 0, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0]])
    want = ref(hidden, mask)
    got = torch.nn.functional.normalize(ours(hidden, mask), dim=-1)
    assert torch.allclose(got, want, atol=1e-6)


def test_last_token_pooling_picks_last_real_token():
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    am = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    v = encode.pool(hidden, am, am, "last", None)
    assert torch.equal(v[0], hidden[0, 1]) and torch.equal(v[1], hidden[1, 3])


def test_mean_pooling_ignores_masked_tokens():
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 3.0], [9.0, 9.0]]])
    pm = torch.tensor([[1, 1, 0]])
    v = encode.pool(hidden, pm, pm, "mean", None)
    assert torch.allclose(v, torch.tensor([[0.5, 1.5]]))


@pytest.mark.skipif(not BERT, reason="set TEST_BERT_TOKENIZER")
def test_pair_with_empty_title_matches_r2med():
    tok = AutoTokenizer.from_pretrained(BERT)
    enc = encode.tokenize(tok, [("", "some clinical text")], 512, "none", pair=True)
    ref = tok([""], ["some clinical text"], max_length=512, truncation="longest_first")
    assert enc["input_ids"][0].tolist() == ref["input_ids"][0]
    assert enc["input_ids"][0][1].item() == tok.sep_token_id


@pytest.mark.skipif(not DECODER, reason="set TEST_DECODER_TOKENIZER")
def test_eos_modes():
    tok = AutoTokenizer.from_pretrained(DECODER)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    long_text = "word " * 100
    after = encode.tokenize(tok, [long_text, "short"], 16, "after_truncation")
    assert after["input_ids"].shape[1] == 16
    assert after["input_ids"][0, -1].item() == tok.eos_token_id          # clipped text still ends in EOS
    n = after["attention_mask"][1].sum().item()
    assert after["input_ids"][1, n - 1].item() == tok.eos_token_id
    appended = encode.tokenize(tok, [long_text], 16, "append")
    assert appended["input_ids"][0, -1].item() != tok.eos_token_id       # clipping eats the appended EOS
    lengths = encode.true_lengths(tok, [long_text, "short"], "none", False)
    assert lengths[0] > 16


def test_plan_batches_respects_token_budget_and_count():
    lengths = [900, 500, 100, 100, 100, 100, 50]
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    plan = encode.plan_batches(order, __import__("numpy").array(lengths), 512, 4, 1024)
    assert [len(b) for b in plan] == [2, 4, 1]          # 2 x 512 fits; then 4 x 100 hits the count cap
    assert sorted(i for b in plan for i in b) == list(range(len(lengths)))
    assert encode.plan_batches(order, __import__("numpy").array(lengths), 512, 3, 0) == [order[:3], order[3:6], order[6:]]
