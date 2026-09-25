import re
import unicodedata

_COMBINING = re.compile(r"[̀-ͯ]")
_APOS = re.compile(r"['‘’`´]")
_DOMAIN = re.compile(r"\.(com|net|org|info|biz|co|in|fr|us)\b")
_PUNCT = re.compile(r"[!\"#$%&()*+,\-./:;<=>?@\[\\\]^_{|}~‐-―“-‟ ·«»•]")
_ORD_SUFFIX = re.compile(r"^(\d+)(st|nd|rd|th|er|e|eme)$")

NAME_ABBR = {
    "pvt": "private", "ltd": "limited", "corp": "corporation", "inc": "incorporated",
    "co": "company", "intl": "international", "assn": "association", "svc": "service",
    "svcs": "services", "mfg": "manufacturing", "bros": "brothers", "dept": "department",
    "univ": "university", "mgmt": "management", "amp": "and", "&": "and",
}

ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "avnue": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "hwy": "highway",
    "pkwy": "parkway", "pl": "place", "cir": "circle", "ter": "terrace", "sq": "square",
    "apt": "apartment", "ste": "suite", "bldg": "building", "flr": "floor", "nr": "near",
    "opp": "opposite", "rte": "route", "n": "north", "s": "south", "e": "east", "w": "west",
}

ORDINALS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}

IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "ts", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl",
    "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py", "ladakh": "la",
}

_STATE_MAP = {**IN_STATES, **US_STATES}
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _STATE_MAP), key=len, reverse=True)) + r")\b")


def _base(s):
    if s is None or s != s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = _COMBINING.sub("", s).lower()
    return s


def norm_name(s):
    s = _base(s)
    s = _DOMAIN.sub(" ", s)
    s = s.replace("&", " and ")
    s = _APOS.sub("", s)
    s = s.replace(".", "")
    s = _PUNCT.sub(" ", s)
    return " ".join(NAME_ABBR.get(t, t) for t in s.split())


def norm_addr(s):
    s = _base(s)
    s = _APOS.sub("", s)
    s = s.replace(".", "")
    s = _PUNCT.sub(" ", s)
    s = _STATE_RE.sub(lambda m: _STATE_MAP[m.group(1)], s)
    out = []
    for t in s.split():
        t = ORDINALS.get(t, t)
        m = _ORD_SUFFIX.match(t)
        if m:
            t = m.group(1)
        t = ADDR_ABBR.get(t, t)
        if t in ("no", "number", "nr", "num"):
            continue
        out.append(t)
    return " ".join(out)


def learn_lexicon(names_ref, names_other, min_j=0.25):
    """Learn non-ASCII token -> Latin token map from matched (reference, other) normalized name pairs."""
    from collections import Counter, defaultdict
    cn, cl, co = Counter(), Counter(), defaultdict(lambda: defaultdict(float))
    for a, b in zip(names_ref, names_other):
        lt = a.split()
        bt = b.split()
        nt = [(i, t) for i, t in enumerate(bt) if not t.isascii()]
        if not nt:
            continue
        cl.update(set(lt))
        cn.update({t for _, t in nt})
        for i, t in nt:
            for j, l in enumerate(lt):
                co[t][l] += 1.0 / (1 + abs(i - j))
    lex = {}
    for t, c in co.items():
        l, w = max(c.items(), key=lambda kv: kv[1] / (cn[t] + cl[kv[0]] - kv[1]))
        if w / (cn[t] + cl[l] - w) >= min_j:
            lex[t] = l
    return lex


def apply_lexicon(name_n, lex):
    return " ".join(lex.get(t, t) for t in name_n.split())


def normalize_frame(df, lexicon=None):
    """Adds name_n, addr_n (normalized) and name_nonlatin (raw name contains non-ASCII characters)."""
    raw = df.business_name.fillna("")
    out = df.copy()
    out["name_nonlatin"] = ~raw.map(str.isascii).to_numpy()
    names = [norm_name(x) for x in df.business_name.values]
    if lexicon:
        names = [apply_lexicon(x, lexicon) if not x.isascii() else x for x in names]
    out["name_n"] = names
    out["addr_n"] = [norm_addr(x) for x in df.business_address.values]
    return out.drop(columns=["business_name", "business_address"])
