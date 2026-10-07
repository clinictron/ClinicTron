"""Shared by the stage scripts: machine paths from the environment, secrets, the sample, psql."""
import csv
import io
import json
import os
import shlex
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def env(name):
    """A machine path from the environment (see env.example.sh)."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set (see env.example.sh)")
    return value


def secret(name):
    """One API key from the environment (OPENROUTER_API_KEY, NCBI_API_KEY)."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set: export it to call this service")
    return value


def load_sample(run_dir, limit=None):
    """The sampled queries, in sample order: [{qid, text, gold_pmids}]."""
    with open(f"{run_dir}/sample.jsonl") as fh:
        rows = [json.loads(line) for line in fh]
    return rows[:limit] if limit else rows


def sql_list(values):
    """Quoted SQL list for ids we produced ourselves; refuses anything with a quote."""
    values = list(values)
    if any("'" in v for v in values):
        raise ValueError("identifier contains a quote")
    return ",".join(f"'{v}'" for v in values)


def psql_rows(select_sql):
    """Run a read-only SELECT against the paper corpus database and return its rows as lists of strings.
    PAPERS_DB_PSQL is the psql command that reaches it, e.g. "psql -d papers_db". The database holds
    table papers_index (the 58.5M-article corpus) with a ParadeDB BM25 index on title and abstract."""
    command = os.environ.get("PAPERS_DB_PSQL")
    if not command:
        raise SystemExit("PAPERS_DB_PSQL is not set: the corpus database is not configured")
    sql = f"COPY ({select_sql}) TO STDOUT WITH (FORMAT csv)"
    r = subprocess.run(shlex.split(command) + ["-c", sql],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"psql failed: {r.stderr.strip()[-300:]}")
    return list(csv.reader(io.StringIO(r.stdout)))
