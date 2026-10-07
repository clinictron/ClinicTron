"""The search helpers that need no GPU: keyword search over the corpus, paper records, and FDA drug labels.

  keyword_search(query, k)  BM25 over title + abstract of the OpenAlex corpus (papers_db), best first (the BM25 arm)
  papers(work_ids)          corpus fields for OpenAlex identifiers (the dense-search hits)
  papers_by_pmid(pmids)     the same fields for PubMed identifiers (the PubMed hits that are in the corpus)
  resolve(title)            the corpus record with this title, or None (the zero-shot arm's existence check)
  fda_label(drug, section)  one FDA drug label from openFDA, looked up by brand or generic name

The dense search itself is dense_clinictron_bge_search.py / dense_search.py; PubMed is pubmed_search.py.

  usage: tools.py keyword "droperidol antiemetic" [k]
         tools.py fda "teriparatide" [section]
"""
import functools
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from common import psql_rows, sql_list

PAPER_COLUMNS = ("work_id, coalesce(pmid,''), coalesce(nullif(type,''),'article'), coalesce(title,''), "
                 "coalesce(journal_name,''), coalesce(publication_year::text,''), coalesce(cited_by_count::text,''), "
                 "coalesce(nullif(abstract,''),'No abstract')")
PAPER_KEYS = ("work_id", "pmid", "type", "title", "journal", "year", "cited_by", "abstract")

OPENFDA = "https://api.fda.gov/drug/label.json"
TITLE_MAX_EDITS = 3  # a listed title counts as a corpus title when, after normalising, they differ by at most this many
                     # inserted or deleted characters: punctuation, capitalisation and a spelling variant pass
                     # (leukaemia/leukemia 1, randomised/randomized 2, lymphocyte/lymphocytic 3); a missing or
                     # different word does not ("Fourth " 7, "in adults " 10). Titles that
                     # match, with only punctuation or spelling slack; a ratio (95 or 98) could not give both on
                     # short and long titles at once.


def keyword_search(query, k=25):
    """Top-k papers by BM25 score of `query` against title and abstract, best first."""
    q = "$kw$" + query.replace("$", " ").replace("\x00", " ") + "$kw$"  # no "$" inside the SQL quoting tag
    rows = psql_rows(
        f"SELECT {PAPER_COLUMNS} FROM papers_index WHERE work_id @@@ paradedb.boolean(should => ARRAY["
        f"paradedb.match('title', {q}), paradedb.match('abstract', {q})]) "
        f"ORDER BY paradedb.score(work_id) DESC LIMIT {int(k)}")
    return [dict(zip(PAPER_KEYS, r)) for r in rows]


def _select(where, values):
    out, ids = {}, sorted(set(values))
    for s in range(0, len(ids), 2000):
        for r in psql_rows(f"SELECT {PAPER_COLUMNS} FROM papers_index WHERE {where} IN ({sql_list(ids[s:s + 2000])})"):
            yield dict(zip(PAPER_KEYS, r))


def papers(work_ids):
    """work_id -> paper fields."""
    return {p["work_id"]: p for p in _select("work_id", work_ids)}


def papers_by_pmid(pmids):
    """pmid -> paper fields, for the PubMed hits that have a corpus record (one pass over the table; the
    corpus has no index on pmid, about a minute for a few thousand ids)."""
    return {p["pmid"]: p for p in _select("pmid", pmids)}


def normalise_title(title):
    """Lower case, letters and digits only, single spaces: what two titles are compared on."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", title.lower())).strip()


def title_candidates(title, k=20):
    """The k best BM25 matches of `title` on corpus titles."""
    q = "$kw$" + title.replace("$", " ").replace("\x00", " ") + "$kw$"
    rows = psql_rows(f"SELECT {PAPER_COLUMNS} FROM papers_index WHERE work_id @@@ paradedb.match('title', {q}) "
                     f"ORDER BY paradedb.score(work_id) DESC LIMIT {int(k)}")
    return [dict(zip(PAPER_KEYS, r)) for r in rows]


def title_edits(a, b):
    """Characters inserted or deleted to turn one normalised title into the other (rapidfuzz Indel distance)."""
    from rapidfuzz.distance import Indel
    a, b = normalise_title(a), normalise_title(b)
    return Indel.distance(a, b) if a and b else 10 ** 6


def resolve(title):
    """The corpus record whose title is `title`, or None: the existence check of the zero-shot arm, as in
    OpenScholar (Asai et al., 2024) with our corpus as the reference set. The closest title wins; the corpus
    holds duplicates of some papers, so among equally close titles the record with a PMID, then the most cited."""
    same = [(title_edits(title, rec["title"]), rec) for rec in title_candidates(title)]
    same = [(d, rec) for d, rec in same if d <= TITLE_MAX_EDITS]
    return min(same, key=lambda x: (x[0], not x[1]["pmid"], -int(x[1]["cited_by"] or 0)))[1] if same else None


def _openfda(params):
    """The results of one openFDA request; [] when nothing matches. A rate limit, server error or
    timeout is retried."""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(OPENFDA + "?" + urllib.parse.urlencode(params), timeout=30) as r:
                return json.load(r)["results"]
        except urllib.error.HTTPError as e:
            if e.code == 404:  # openFDA answers "no match" with 404
                return []
            if (e.code != 429 and e.code < 500) or attempt == 4:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 4:
                raise
        time.sleep(15 * (attempt + 1))


def _matches(name):
    """openFDA search expressions for a drug name, most specific first: the brand name, then the generic
    name, EQUAL to `name`; then the brand name, then the generic name, CONTAINING `name` (so that
    "metoprolol" still finds "metoprolol succinate", but "hydrochlorothiazide" does not pick a
    combination product over hydrochlorothiazide itself)."""
    fields = ("openfda.brand_name", "openfda.generic_name")
    for field in fields:
        same = [t["term"] for t in _openfda({"search": f'{field}:"{name}"', "count": f"{field}.exact", "limit": 1000})
                if t["term"].lower() == name.lower()]
        if same:
            yield " ".join(f'{field}.exact:"{s}"' for s in same)  # a space is OR in openFDA
    for field in fields:
        yield f'{field}:"{name}"'


@functools.lru_cache(maxsize=None)
def _label(name):
    """The openFDA record of the label chosen for `name` (rule in fda_label), or None. Kept in memory:
    every section of a label costs one lookup."""
    for match in _matches(name):
        applications = [a["term"] for a in _openfda({"search": match, "count": "openfda.application_number.exact",
                                                     "limit": 1000})]
        if applications:
            break
    else:
        return None
    original = sorted((a for a in applications if re.match(r"(NDA|BLA)\d+$", a)),
                      key=lambda a: int(re.sub(r"\D", "", a)))
    if original:  # every label under that application is the same drug product; take the newest
        match = f'openfda.application_number:"{original[0]}"'
    return _openfda({"search": match, "limit": 1, "sort": "effective_time:desc"})[0]


def fda_label(drug, section=None):
    """One label for `drug`, or None if openFDA has none.

    A drug has one label per manufacturer. We take the original manufacturer's label: among the
    labels matching `drug` (brand name first, then generic name; see _matches), the one filed
    under the lowest-numbered original application (NDA or BLA, not a generic's ANDA), in its most
    recently updated version; if there is no original application, the most recently updated label.
    Without `section`: the label's identity and the sections it has. With `section`: also its text,
    or `missing_section` if the label has no such section.
    """
    label = _label(drug.replace('"', " ").replace("/", " ").strip())
    if label is None:
        return None
    info = {"drug_asked": drug,
            "brand_name": (label["openfda"].get("brand_name") or [""])[0],
            "generic_name": (label["openfda"].get("generic_name") or [""])[0],
            "manufacturer": (label["openfda"].get("manufacturer_name") or [""])[0],
            "application": (label["openfda"].get("application_number") or [""])[0],
            "label_id": label["set_id"], "updated": label["effective_time"],
            "sections": [s for s in FDA_SECTIONS if s in label]}
    if section in info["sections"]:
        info["text"] = "\n".join(label[section])
    elif section:  # the agent is told it then receives the list of sections the label has
        info["missing_section"] = section
    return info


if __name__ == "__main__":
    if sys.argv[1] == "keyword":
        for p in keyword_search(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 25):
            print(p["work_id"], p["type"], p["year"], f"cites:{p['cited_by']}", p["title"][:100])
    else:
        print(json.dumps(fda_label(*sys.argv[2:4]), indent=1)[:3000])
