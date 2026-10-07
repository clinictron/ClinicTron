"""PubMed E-utilities client: Best Match search and title/abstract/journal/year fetch, rate-limited."""
import json
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import requests

from common import secret

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def _text(el):
    return ET.tostring(el, encoding="unicode", method="text").strip() if el is not None else ""


class PubMed:
    def __init__(self, log):
        self.key = secret("NCBI_API_KEY")
        self.log = log
        self.meta = {}  # pmid -> {title, abstract, journal, year}
        self.lock = threading.Lock()
        self.next_slot = 0.0

    def _get(self, path, params, parse):
        """GET with retries -> parse(body). An unparseable body counts as a failed try."""
        for attempt in range(4):
            with self.lock:  # one request per 0.13 s stays under NCBI's 10 per second
                now = time.time()
                self.next_slot = max(now, self.next_slot + 0.13)
                wait = self.next_slot - now
            time.sleep(wait)
            try:
                r = requests.get(f"{EUTILS}/{path}", params=dict(params, api_key=self.key),
                                 timeout=60)
                if r.status_code != 200:
                    raise RuntimeError(f"http {r.status_code}")
                return parse(r.text)
            except Exception as e:  # message withheld: it can carry the request URL and key
                self.log(f"  pubmed error {path} try{attempt}: {type(e).__name__}")
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"pubmed failed after retries: {path}")

    def search(self, term, retmax):
        """Best Match search -> PMIDs. Zero hits is a real result and returns []."""
        res = self._get("esearch.fcgi",
                        {"db": "pubmed", "term": term, "retmax": retmax,
                         "sort": "relevance", "retmode": "json"},
                        lambda body: json.loads(body)["esearchresult"])
        if "ERROR" in res:
            self.log(f"  pubmed rejected a query, counted as zero hits: {term[:120]!r}")
        return res.get("idlist", [])

    def fetch_meta(self, pmids):
        todo = [p for p in pmids if p not in self.meta]
        if todo:
            root = self._get("efetch.fcgi",
                             {"db": "pubmed", "id": ",".join(todo),
                              "rettype": "abstract", "retmode": "xml"},
                             ET.fromstring)
            for art in root.iter("PubmedArticle"):
                abstract = " ".join(_text(a) for a in art.findall(".//Abstract/AbstractText"))
                date = art.find(".//JournalIssue/PubDate")
                year = _text(date.find("Year")) if date is not None else ""
                if not year and date is not None:  # e.g. MedlineDate "1998 Dec-1999 Jan"
                    year = (re.search(r"\d{4}", _text(date)) or [""])[0]
                self.meta[art.findtext(".//PMID") or ""] = {
                    "title": _text(art.find(".//ArticleTitle")), "abstract": abstract.strip(),
                    "journal": _text(art.find(".//Journal/Title")), "year": year}
        missing = [p for p in todo if p not in self.meta]
        if missing:
            self.log(f"  pubmed returned no record for {len(missing)} ids: {missing[:5]}")
            for p in missing:
                self.meta[p] = {"title": "(title unavailable)", "abstract": "", "journal": "", "year": ""}


def pubmed_tool(pm, queries, retmax):
    """{qid: query text} -> {qid: [up to retmax docs]} in Best Match order."""
    def one(item):
        qid, term = item
        ids = pm.search(term, retmax)
        pm.fetch_meta(ids)
        return qid, [{"pmid": p, **pm.meta[p]} for p in ids]

    with ThreadPoolExecutor(3) as ex:
        return dict(ex.map(one, queries.items()))
