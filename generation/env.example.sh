# Copy to env.sh, fill in, and `source env.sh` before running a stage.
# LLM access: one OpenAI-compatible endpoint. Empty LLM_BASE_URL uses the OpenRouter API.
export LLM_BASE_URL=""
# API key; the variable name is set by `models.api_key_env` in the config.
export OPENROUTER_API_KEY=""
# Input data root (entities, taxonomy, import rows) and output root for run directories.
export GEN_RESOURCES=""
export GEN_RUN_ROOT=""
# Prompt directory; default is ./prompts.
export GEN_PROMPTS=""
# Optional file of KEY=value secrets read by the stages.
export GEN_SECRETS=""
# Corpus metadata table (parquet or JSONL: id, title, abstract, journal_name,
# cited_by_count, doc_type, year), used by stages 02, 04, 05 and 06.
export GEN_METADATA_TABLE=""
# Corpus search for stages 02 and 04 (see `corpus_search` in the config): for each dense
# retriever, a model config for corpus.encode_queries and an index directory for
# corpus.exact_search, both built by you over your corpus; plus a BM25 index directory.
export GEN_SEARCH_DEVICE="cpu"
export GEN_OPENAI_EMBED_MODEL_CONFIG=""
export GEN_OPENAI_EMBED_INDEX=""
export GEN_BMRETRIEVER_MODEL_CONFIG=""
export GEN_BMRETRIEVER_INDEX=""
export GEN_MEDCPT_MODEL_CONFIG=""
export GEN_MEDCPT_INDEX=""
export GEN_SPECTER2_MODEL_CONFIG=""
export GEN_SPECTER2_INDEX=""
export GEN_BM25_INDEX=""
# Stage 03 (frozen base-encoder scan). GEN_BASE_INDEX: the base-encoder corpus index
# directory as downloaded from the Hub (index_meta.json, vectors/, ids/).
export GEN_BASE_INDEX=""
# Encoder model config; default ../embedding/configs/models/reasonembed.yaml (relative to
# this directory). Weights are found through ENCODE_WEIGHTS_ROOT (see embedding/).
export GEN_BASE_MODEL_CONFIG=""
# Benchmark hold-out work ids dropped from every scan ranking; default
# resources/benchmark_holdout_ids.txt (11,059 ids, the list the run applied).
export GEN_HOLDOUT_IDS=""
# cpu or cuda for the stage-03 scan.
export GEN_SCAN_DEVICE="cpu"
