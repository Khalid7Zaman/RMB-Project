"""Export the cited references (in citation order) to RIS, BibTeX and EndNote XML,
and format the IEEE-style bibliography used in the manuscript.

    python refs/export.py <cited_keys.json> refs/
"""
import html
import json
import re
import sys

DB = json.load(open("refs/references.json"))


def ieee_name(a):
    fam, giv = [x.strip() for x in a.split(",", 1)] if "," in a else (a, "")
    if not giv:
        return fam
    ini = " ".join(p[0] + "." for p in re.split(r"[\s]+", giv.replace("-", " -")) if p and p[0].isalpha())
    if "-" in giv:  # e.g. Dong-Hyun -> D.-H.
        ini = "-".join(x[0] + "." for x in giv.split("-"))
        ini = " ".join(w[0] + "." if "-" not in w else "-".join(t[0] + "." for t in w.split("-")) for w in giv.split())
    return f"{ini} {fam}"


def ieee_authors(auth):
    names = [ieee_name(a) for a in auth]
    if len(names) > 6:
        return f"{names[0]} *et al.*"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + ", and " + names[-1]


def ieee_entry(k):
    r = DB[k]
    a = ieee_authors(r["authors"])
    if r["type"] == "CONF":
        s = f"{a}, “{r['title']},” in *{r['journal']}*, {r['year']}"
        if r.get("pages"):
            s += f", pp. {r['pages'].replace('-', '–')}"
    else:
        s = f"{a}, “{r['title']},” *{r['journal']}*"
        if r.get("volume"):
            s += f", vol. {r['volume']}"
        if r.get("issue"):
            s += f", no. {r['issue']}"
        pg = r.get("pages", "")
        if pg:
            s += f", pp. {pg.replace('-', '–')}" if "-" in pg else f", Art. no. {pg}"
        s += f", {r['year']}"
    s += f", doi: {r['doi']}." if r.get("doi") else (f". [Online]. Available: {r['url']}" if r.get("url") else ".")
    return s


def ris(keys):
    out = []
    for k in keys:
        r = DB[k]
        out.append("TY  - " + {"CONF": "CPAPER", "BOOK": "BOOK", "PREPRINT": "UNPB"}.get(r["type"], "JOUR"))
        for a in r["authors"]:
            out.append(f"AU  - {a}")
        out.append(f"TI  - {r['title']}")
        if r["type"] == "BOOK":
            out.append(f"PB  - {r['journal']}")
        elif r["type"] in ("CONF", "PREPRINT"):
            out.append(f"T2  - {r['journal']}")
        else:
            out += [f"JO  - {r['journal']}", f"T2  - {r['journal']}"]
        out.append(f"PY  - {r['year']}")
        for tag, f in (("VL", "volume"), ("IS", "issue")):
            if r.get(f):
                out.append(f"{tag}  - {r[f]}")
        pg = r.get("pages", "")
        if pg:
            if "-" in pg:
                sp, ep = pg.split("-", 1); out += [f"SP  - {sp}", f"EP  - {ep}"]
            else:
                out.append(f"SP  - {pg}")
        if r.get("doi"):
            out += [f"DO  - {r['doi']}", f"UR  - https://doi.org/{r['doi']}"]
        elif r.get("url"):
            out.append(f"UR  - {r['url']}")
        out.append(f"ID  - {k}")
        out.append("ER  - ")
        out.append("")
    return "\n".join(out)


def bib(keys):
    out = []
    for k in keys:
        r = DB[k]
        typ = {"CONF": "inproceedings", "BOOK": "book", "PREPRINT": "misc"}.get(r["type"], "article")
        venue_key = {"inproceedings": "booktitle", "book": "publisher", "misc": "howpublished"}.get(typ, "journal")
        f = {"author": " and ".join(r["authors"]), "title": "{" + r["title"] + "}",
             venue_key: r["journal"], "year": str(r["year"])}
        for x in ("volume", "issue", "pages", "doi", "url"):
            if r.get(x):
                f["number" if x == "issue" else x] = r[x].replace("-", "--") if x == "pages" else r[x]
        body = ",\n".join(f"  {a} = {{{b}}}" for a, b in f.items())
        out.append(f"@{typ}{{{k},\n{body}\n}}\n")
    return "\n".join(out)


def endnote_xml(keys):
    e = html.escape
    recs = []
    for i, k in enumerate(keys, 1):
        r = DB[k]
        rt = {"CONF": ("47", "Conference Paper"), "BOOK": ("6", "Book"), "PREPRINT": ("34", "Unpublished Work")}.get(
            r["type"], ("17", "Journal Article"))
        au = "".join(f"<author>{e(a)}</author>" for a in r["authors"])
        x = (f"<record><rec-number>{i}</rec-number><ref-type name=\"{rt[1]}\">{rt[0]}</ref-type>"
             f"<contributors><authors>{au}</authors></contributors><titles><title>{e(r['title'])}</title>"
             f"<secondary-title>{e(r['journal'])}</secondary-title></titles>"
             f"<periodical><full-title>{e(r['journal'])}</full-title></periodical>")
        if r.get("pages"): x += f"<pages>{e(r['pages'])}</pages>"
        if r.get("volume"): x += f"<volume>{e(r['volume'])}</volume>"
        if r.get("issue"): x += f"<number>{e(r['issue'])}</number>"
        x += f"<dates><year>{r['year']}</year></dates>"
        if r.get("doi"): x += f"<electronic-resource-num>{e(r['doi'])}</electronic-resource-num>"
        url = f"https://doi.org/{r['doi']}" if r.get("doi") else r.get("url", "")
        if url: x += f"<urls><related-urls><url>{e(url)}</url></related-urls></urls>"
        x += f"<label>{k}</label></record>"
        recs.append(x)
    return '<?xml version="1.0" encoding="UTF-8"?><xml><records>' + "".join(recs) + "</records></xml>"


if __name__ == "__main__":
    keys = json.load(open(sys.argv[1]))
    od = sys.argv[2]
    missing = [k for k in keys if k not in DB]
    assert not missing, missing
    open(f"{od}/RMB-FF_references.ris", "w").write(ris(keys))
    open(f"{od}/RMB-FF_references.bib", "w").write(bib(keys))
    open(f"{od}/RMB-FF_references_endnote.xml", "w").write(endnote_xml(keys))
    unused = sorted(set(DB) - set(keys) - {"_note"})
    print(f"exported {len(keys)} references; uncited in DB (not exported): {unused}")
