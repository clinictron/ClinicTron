# Copy, fill in, and `source` before running a stage. Every path is a placeholder.
export PAPERS_DB_PSQL="psql -d papers_db"          # psql command reaching the corpus database (table papers_index)
export OPENROUTER_API_KEY=""                       # judge and OpenRouter agents; OpenAI embedding queries
export NCBI_API_KEY=""                             # PubMed E-utilities (PubMed arm, 02b)
export PMC_PATIENTS_RAW="/path/to/PMC-Patients"    # ReCDS-PAR test queries and qrels (01, PMC-Patients)
export AGENT_EVAL_QUESTIONS="/path/to/questions_100.json"            # questions + the comparator system's citations
export AGENT_EVAL_CITATION_MATCHES="/path/to/citation_matches_100.csv"  # comparator citation -> corpus work_id
export CLINICTRON_BGE_BASE="/path/to/base_model"   # ClinicTron-BGE = base model + LoRA adapter
export CLINICTRON_BGE_ADAPTER="/path/to/adapter"
export CLINICTRON_BGE_INDEX_DIR="/path/to/clinictron_bge_index"
export AGENT_EVAL_INDEX_REASONEMBED="/path/to/reasonembed_index"
export AGENT_EVAL_INDEX_NVEMBED="/path/to/nvembed_index"
export AGENT_EVAL_DOCMAT_DIR="/path/to/docmats"      # BMRetriever-2B, MedCPT, OpenAI-3-small matrices
export AGENT_EVAL_NVEMBED_PYTHON="/path/to/python"   # environment with transformers 4.42 for NV-Embed-v2
export AGENT_EVAL_NVEMBED_CODE_DIR="/path/to/nvembed_encoder_module"
export MULTI_TURN_CLAUDE_CWD="/tmp/multi_turn_claude_cwd"  # empty directory for the headless Claude CLI
export MULTI_TURN_MIN_FREE_RAM_GB=45
