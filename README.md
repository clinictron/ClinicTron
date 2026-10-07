# ClinicTron

ClinicTron is a set of retrieval models for clinical medicine: three LoRA adapters,
ClinicTron-BGE, ClinicTron-Qwen and ClinicTron-NV, that improve how an embedding model finds
biomedical and clinical documents. This repository holds the code to encode and search with them,
to train them from the released data, to produce that training data, and to run the agent
evaluations. Models, training data and the 58.5M-article embeddings are on Hugging Face at
https://huggingface.co/clinictronanon.

The OpenAlex corpus text (titles, abstracts, metadata) is not part of the release. Stages that
read it take a table you supply.

| Folder | What it does |
|---|---|
| `embedding/` | Encode queries and documents with any of the nine models, and score a benchmark. |
| `search/` | Exact search over the published 58.5M-paper embeddings. |
| `training/` | Fine-tune ClinicTron-BGE, ClinicTron-Qwen and ClinicTron-NV from the released training data. |
| `generation/` | The pipeline that produced the training data: write questions, retrieve candidates, grade them, build batches. |
| `evaluation/` | Multi-round agent search over the corpus, with a judged comparison on physician questions. |

## Installation

One GPU with CUDA 12.8. Tested with Python 3.13.

```bash
python -m venv ct && . ct/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Quick start

### Encode

`embedding/encode.py` takes a model config and a benchmark config. `configs/models/` holds one
file per model, with the base revision, the adapter, the pooling and the prompt template.
`configs/benchmarks/` holds the instruction and maximum length for every task. The adapters are
fetched from Hugging Face automatically.

Put your documents in `<dir>/corpus.parquet` with columns `_id`, `title`, `text`, add a
`queries.parquet` with one row, then:

```bash
cd embedding
python encode.py --model configs/models/clinictron_bge.yaml --benchmark configs/benchmarks/corpus_encode.yaml \
    --task corpus --data-dir <dir> --batch-size 16 --max-batch-tokens 0 --store-dtype float16 --out shard.npz
```

This is how the published corpus embeddings were made. Vectors agree with the published ones to a
cosine similarity above 0.9997; they are not bit-identical.

Two things decide whether the vectors are right:

- Load the base model and the adapter separately, as `encode.py` does. Merging the adapter in bf16 loses most of the LoRA update.
- Use `corpus_encode.yaml` for a corpus and a benchmark config for a benchmark. The two handle the end of sequence differently.

### Search

Download one `clinictronanon/<encoder>-openalex-58m` repository, encode your queries with
`encode.py`, then:

```bash
python search/search.py --index <index_dir> --queries queries.npz --top-k 100 --out run.tsv
```

A full scan of 58.5M vectors takes about 11 minutes on one GPU and is limited by disk speed.

## Evaluation

### R2MED, BEIR, BRIGHT and MTEB

`prepare_data.py` builds R2MED, the biomedical BEIR tasks, BRIGHT and the MTEB retrieval tasks
from public datasets at pinned revisions. Encode a task with its benchmark config, then score:

```bash
cd embedding
python prepare_data.py r2med --task IIYi-Clinical --out data/
python encode.py --model configs/models/clinictron_bge.yaml --benchmark configs/benchmarks/r2med.yaml \
    --task r2med_iiyi_clinical --batch-size 8 --out iiyi.npz
python score.py --bank iiyi.npz > score.json
```

`score.json` holds nDCG@10 and the other metrics. The published query and document vectors for
every model and task are in `clinictronanon/clinictron-benchmark-bank`; `score.py --bank` scores
those directly. `python embedding/r2med_groups.py scores.json` prints the R2MED grouped means.

- The instruction strings and lengths are set per task in `configs/benchmarks/`; changing them changes the scores.
- ClinicTron-NV is scored with mean pooling and the adapter, as in `configs/models/clinictron_nv.yaml`, not with the latent-attention head. It was trained with a new head that was not saved.

### PMC-Patients

The ClinicTron-NV index, the encoded queries and the labels are in
`clinictronanon/clinictron-benchmark-pmc-patients`:

```bash
hf download clinictronanon/clinictron-benchmark-pmc-patients --repo-type dataset --local-dir pmc \
    --include "clinictron_nv/*" --include "queries/clinictron_nv.npz" --include "qrels_test.tsv"
python search/search.py --index pmc/clinictron_nv --queries pmc/queries/clinictron_nv.npz --top-k 1000 --device cuda --out run.tsv
python embedding/score.py --run run.tsv --qrels pmc/qrels_test.tsv > score.json
```

ClinicTron-BGE, ReasonEmbed and NV-Embed-v2 store most PMC-Patients articles in their 58.5M-paper
index. Build one index first (about 93 GB), then run the same three commands on it:

```bash
python search/assemble_pmc_index.py --selection pmc/bridge/selection_clinictron_bge.parquet \
    --corpus clinictron-bge-openalex-58m --gap pmc/clinictron_bge_gap --out pmc_bge
```

### Agent evaluation

`evaluation/` runs a multi-round search agent on PMC-Patients and on physician questions, where a
judge compares the agent's citations with those of a comparator system. Both use the same stages
with a different config. The search tools read a corpus database and embedding indexes that you
supply; see `evaluation/env.example.sh`.

```bash
cd evaluation
python 01_sample_questions.py --config configs/pmc_patients.yaml --run-dir runs/pmc200
python 02_run_agent_search.py --config configs/pmc_patients.yaml --run-dir runs/pmc200 --arm clinictron_bge --agent opus
python 03_score_rankings.py --run-dir runs/pmc200
```

## Training

Download `batches.jsonl` from `clinictronanon/clinictron-train-p7-kl14-mix` into
`training/data/p7_kl14_mix/`, then:

```bash
cd training
python train.py --config configs/clinictron_bge.json                       # one GPU
torchrun --nproc_per_node 8 train.py --config configs/clinictron_bge.json  # several GPUs
```

The three configs are the settings that trained the three models. One step takes about 12 minutes
on one 96 GB GPU, and a run is 204 steps.

### Ablation

The training sets for the three ablation variants are in the `ablation/` folder of
`clinictronanon/clinictron-train-p7-kl14-mix`. Each is 74 training steps:

```bash
cd training
python train.py --config configs/ablation_papers_only.json
```

Score the step-74 adapter on R2MED as above. `generation/configs/ablation_*.yaml` and
`generation/prompts/ablation/` hold the writer settings that differ from the full recipe.
`analysis/unretrieved_positive_rate.py` computes the unretrieved-positive rate from the outputs of
generation stages 03, 05 and 07.

## Generating training data

The stages in `generation/` run in order, `00` to `08`. Copy `env.example.sh` to `env.sh`, fill it
in, and `source` it. You need an API key for the writer and grader models, a metadata table for
your corpus, and an embedding index for each retriever. Stage 08 writes the same `batches.jsonl`
that is on Hugging Face, so training does not require running any of this.

This code differs from the run that produced the released data in three ways:

- Candidate search is exact. The original run used approximate nearest-neighbour and BM25 services; you supply the BM25 index.
- Stage 00 keeps 4,238 imported questions; the original run kept 4,241.
- Stage 05 shows the grader more article metadata than the original run did.

## Models and data

| For | Repository | Used by |
|---|---|---|
| Fine-tuned adapters | `clinictronanon/ClinicTron-BGE`, `ClinicTron-Qwen`, `ClinicTron-NV` | `embedding/` (fetched automatically) |
| Training data, main and ablation | `clinictronanon/clinictron-train-p7-kl14-mix` | `training/` |
| PMC-Patients vectors, queries and labels | `clinictronanon/clinictron-benchmark-pmc-patients` | `search/`, `embedding/score.py` |
| Benchmark query and document vectors | `clinictronanon/clinictron-benchmark-bank` | `embedding/score.py --bank` |
| 58.5M-article indexes, one per encoder | `clinictronanon/<encoder>-openalex-58m` for `clinictron-bge`, `reasonembed`, `nvembed-v2`, `bmretriever-2b`, `medcpt`, `openai-text-embedding-3-small` | `search/`, `generation/` stage 03, `evaluation/` |
