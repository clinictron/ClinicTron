#!/usr/bin/env python
"""Stage 2b (physician-questions benchmark, optional): PubMed abstracts for the sources our corpus has no abstract for.

Every source the judge would show without an abstract, on either side, is looked up in PubMed by its title:
the comparator's citations as the comparator wrote them, and every source of every arm after every hop. A PubMed
record is accepted when its title is within 3 character edits of the cited title (after lower-casing and
removing punctuation), its year is within one of the cited year, and, when the citation names authors, its
first author's surname is among the first cited author's names. FDA labels are not looked up.

A source "has no abstract" when it has no corpus record or its corpus abstract is shorter than 50 characters
("No abstract", "Peer Reviewed").

  usage: 02b_pubmed_abstracts.py --run-dir RUN

Writes RUN/pubmed_abstracts.json: {key: {title, year, first_author, pubmed: best record or null}} with one
entry per source looked up; key is "cmp:<rank>:<n>" for a comparator citation and the citation id for ours.
03_judge.py shows the PubMed abstract of an accepted record whenever this file is in the run folder. Refuses
a run that has already been judged: its judgments were made on the lists without these abstracts. To re-judge
a finished run, copy its sample.jsonl and arms/*/citations.json into a new run folder first.
"""
import argparse
import csv
import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import tools
from common import env
from pubmed_search import PubMed, _text

MIN_ABSTRACT = 50  # characters; shorter corpus "abstracts" are placeholders
STOP = {"a", "an", "the", "of", "in", "on", "for", "and", "or", "to", "with", "at", "by", "from", "as", "is", "are",
        "vs", "versus"}


def has_abstract(text):
    return len((text or "").strip()) >= MIN_ABSTRACT


def accepted(title, year, first_author, record):
    """The acceptance rule. `first_author` is the first cited author as written ("" when none is named)."""
    year_ok = bool(year) and record["year"].isdigit() and abs(int(record["year"]) - int(year)) <= 1
    author_ok = not first_author or (bool(record["first_author"]) and record["first_author"].lower() in first_author.lower())
    return tools.title_edits(title, record["title"]) <= 3 and year_ok and author_ok


def fetch(pm, pmids):
    root = pm._get("efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"},
                   ET.fromstring)
    out = {}
    for art in root.iter("PubmedArticle"):
        date = art.find(".//JournalIssue/PubDate")
        year = _text(date.find("Year")) if date is not None else ""
        if not year and date is not None:  # e.g. MedlineDate "1998 Dec-1999 Jan"
            year = (re.search(r"\d{4}", _text(date)) or [""])[0]
        out[art.findtext(".//PMID")] = {
            "title": _text(art.find(".//ArticleTitle")), "journal": _text(art.find(".//Journal/Title")), "year": year,
            "first_author": _text(art.find(".//AuthorList/Author/LastName")),
            "abstract": " ".join(_text(a) for a in art.findall(".//Abstract/AbstractText")).strip()}
    return out


def lookup(pm, cited):
    """cited {title, year, first_author} -> the best PubMed record (accepted first, then closest title) or None."""
    title = cited["title"].strip().rstrip(".")
    words = [w for w in re.findall(r"[a-z0-9]+", title.lower()) if w not in STOP]  # PubMed cannot match a long quoted phrase
    ids = pm.search(" AND ".join(f"{w}[Title]" for w in words) or f'"{title}"[Title]', 10)
    if not ids:  # fall back to a proximity phrase search on the title
        ids = pm.search('"' + title.replace('"', " ") + '"[Title:~3]', 10)
    best = None
    for pmid, r in (fetch(pm, ids) if ids else {}).items():
        cand = {"pmid": pmid, **r, "edits": tools.title_edits(title, r["title"]),
                "accepted": accepted(title, cited["year"], cited["first_author"], r)}
        if best is None or (cand["accepted"], -cand["edits"]) > (best["accepted"], -best["edits"]):
            best = cand
    return {**cited, "pubmed": best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    if glob.glob(f"{a.run_dir}/arms/*/judge"):
        raise SystemExit(f"{a.run_dir} has judge outputs; copy sample.jsonl and arms/*/citations.json to a new run folder")

    cited = {}  # key -> {title, year, first_author}, or a work_id until the corpus record is read
    arms = sorted(glob.glob(f"{a.run_dir}/arms/*/citations.json"))
    ranks = set()
    for path in arms:
        with open(path) as fh:
            for q in json.load(fh).values():
                ranks.add(q["rank"])
                for c in (c for snapshot in q["snapshots"].values() for c in snapshot):
                    if not c["id"].startswith("FDA:"):
                        cited[c["id"]] = c.get("work_id") or {"title": c["title"], "year": c["year"], "first_author": ""}
    with open(env("AGENT_EVAL_CITATION_MATCHES")) as fh:
        match = {(int(r["rank"]), int(r["n"])): r["work_id"] for r in csv.DictReader(fh) if r["in_corpus"] == "True"}
    with open(env("AGENT_EVAL_QUESTIONS")) as fh:
        cmp = {(q["rank"], c["n"]): c for q in json.load(fh)["questions"] if q["rank"] in ranks
              for c in q["comparator_citations"]}
    meta = tools.papers({w for w in cited.values() if isinstance(w, str)} | {match[k] for k in cmp if k in match})
    for key, w in list(cited.items()):
        if isinstance(w, str):
            if has_abstract(meta[w]["abstract"]):
                del cited[key]
            else:
                cited[key] = {"title": meta[w]["title"], "year": meta[w]["year"], "first_author": ""}
    for (rank, n), c in cmp.items():
        if (rank, n) not in match or not has_abstract(meta[match[rank, n]]["abstract"]):
            cited[f"cmp:{rank}:{n}"] = {"title": c["title"], "year": c["year"] or "",
                                       "first_author": (c.get("authors") or "").split(",")[0].strip()}

    pm = PubMed(lambda m: print(m, flush=True))
    with ThreadPoolExecutor(3) as ex:
        out = dict(zip(cited, ex.map(lambda c: lookup(pm, c), cited.values())))
    with open(f"{a.run_dir}/pubmed_abstracts.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    got = lambda o: bool(o["pubmed"] and o["pubmed"]["accepted"] and has_abstract(o["pubmed"]["abstract"]))
    for side, keys in (("comparator", [k for k in out if k.startswith("cmp:")]), ("ours", [k for k in out if not k.startswith("cmp:")])):
        print(f"{side}: {len(keys)} sources without an abstract looked up, {sum(got(out[k]) for k in keys)} PubMed abstracts")
    print(f"-> {a.run_dir}/pubmed_abstracts.json ({len(arms)} arms)")


if __name__ == "__main__":
    main()
