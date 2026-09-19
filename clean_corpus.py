"""Turn `pdftotext` output of the papers into prose suitable for language modeling.

PDF extraction of a journal article interleaves body text with material that is
not prose at all: masthead and submission dates, author and affiliation blocks,
running headers repeated on every page, figure axis labels, statistics strings
and bare equation fragments. Feeding those to an n-gram model produces sentences
made of institute names and axis units, so they are removed here rather than
worked around downstream.

The pass runs in three stages:

  1. line level   - cut front matter, drop repeated running headers, captions,
                    unit labels and symbol-heavy lines
  2. text level   - rejoin hyphenated and wrapped lines, strip inline citations
                    and parenthetical statistics
  3. sentence level - keep only sentences that read as prose

`clean(path)` returns the cleaned text for one file; running the module writes
the concatenation of every file in `text/` to `corpus.txt`.
"""
import re, glob, os
from collections import Counter

# ---------------------------------------------------------------- line level

# Everything from one of these headings onward is back matter, not prose.
REF_HEAD = re.compile(r'^\s*(references|bibliography|literature cited|acknowledge?ments?|'
                      r'author contributions|competing interests|funding|data availability|'
                      r'declaration of|conflict of interest|supplementary (material|information))'
                      r'\s*[:.]?\s*$', re.I)

# The body starts here; anything before it is masthead, authors and affiliations.
BODY_START = re.compile(r'^\s*(abstract|significance|summary|introduction|'
                        r'\d*\.?\s*introduction)\s*[:.]?\s*$', re.I)

CAPTION = re.compile(r'^\s*(fig(ure)?\.?\s*\d|table\s*\d|supplementary|extended data|'
                     r'panel\b|scale bar)', re.I)

# Caption bodies keyed by panel letter: "D) Frequency spectra of both test sounds",
# or run together so the next panel's letter trails the sentence ("... cluster. e.").
PANEL = re.compile(r'(^|\s)\(?[A-H]\)\s')
PANEL_TAIL = re.compile(r'\s[a-h]\.\s*$')

# Section headings and numbered TOC entries sitting alone between blank lines.
HEADING = re.compile(r'^(?:[IVX]+\.|\d+(?:\.\d+)*\.?)?\s*[A-Z][^.!?:]{0,60}$')

JUNK = re.compile(r'^(.{0,3}|\d+|.*(doi:|https?://|©|arXiv:|bioRxiv|preprint|'
                  r'article info|keywords:|all rights reserved|creative commons|'
                  r'corresponding author|e-?mail:|received:|accepted:|published online:|'
                  r'check for updates|author/funder|peer review|this version posted).*)$', re.I)

# Axis and colourbar labels: "Freq [kHz]", "speed [cm/s]", "response size [deg]".
UNIT_LABEL = re.compile(r'\[\s*(k?hz|m?s(ec)?|deg|cm/s|mm|µ?m|a\.?u\.?|spikes/s|%)\s*\]', re.I)

# Affiliation lines: "3 Institute of Science and Technology Austria, ..."
AFFILIATION = re.compile(r'^\s*\d*\s*(institute|department|faculty|center|centre|school|'
                         r'laboratory|university|max[- ]planck|bernstein)\b', re.I)

def nonword_ratio(line):
    letters = sum(c.isalpha() or c.isspace() for c in line)
    return 1 - letters / max(len(line), 1)

def digit_ratio(line):
    return sum(c.isdigit() for c in line) / max(len(line), 1)

def drop_headings(lines):
    """Remove standalone heading lines, which otherwise glue onto the next sentence."""
    out = []
    for i, line in enumerate(lines):
        s = line.strip()
        alone = (i == 0 or not lines[i-1].strip()) and \
                (i + 1 >= len(lines) or not lines[i+1].strip())
        if s and alone and len(s.split()) <= 8 and HEADING.match(s):
            continue
        out.append(line)
    return out


def running_headers(lines):
    """Lines repeated across pages - journal name, article type, paper title."""
    counts = Counter(l.strip() for l in lines if 0 < len(l.strip()) < 90)
    return {l for l, n in counts.items() if n >= 3}

def keep_line(line, headers):
    s = line.strip()
    if not s:
        return True                       # blank lines carry paragraph structure
    if s in headers or CAPTION.match(s) or JUNK.match(s) or AFFILIATION.match(s):
        return False
    if UNIT_LABEL.search(s):
        return False
    if nonword_ratio(s) > 0.30 or digit_ratio(s) > 0.15:
        return False
    return True

def trim_front_matter(lines):
    """Drop the masthead/author block ahead of the abstract or first real prose."""
    for i, l in enumerate(lines[:80]):
        if BODY_START.match(l):
            return lines[i+1:]
    for i, l in enumerate(lines[:80]):    # no heading: first line of real prose
        if len(l.split()) >= 15 and l.rstrip().endswith(('.', ':')):
            return lines[i:]
    return lines

# ---------------------------------------------------------------- text level

# "(Doi and Lewicki, 2014; Tkacik et al., 2010)" and "[12,15-18]".
CITE_PAREN = re.compile(r'\s*\((?:[^()]{0,300}?(?:et al\.?|19\d\d|20\d\d)[^()]{0,60}?)\)')
CITE_BRACK = re.compile(r'\s*\[[\d,\s–-]+\]')
CITE_SUP   = re.compile(r'(?<=[a-z])\d{1,3}(?=[\s.,;])')

# "(P = 0.016)", "(n = 24 cells)", "(r = 0.71, P < 0.001)", "(Fig. 2f)".
STATS_PAREN = re.compile(r'\s*\((?:[^()]*?[=<>≈±][^()]*?|\s*[a-z]\s*|\s*fig[^()]{0,20})\)', re.I)

# Bare statistics and figure references left outside parentheses.
STATS_BARE = re.compile(r'\b[a-zA-Z]{1,3}\s*[=<>]\s*[-−]?\d*\.?\d+(?:\s*[×x]\s*10[-−]?\d+)?')
FIG_REF    = re.compile(r'\b(?:see\s+)?[Ff]igs?\.?\s*\d+\s*[a-h]?\b')
NARR_CITE  = re.compile(r'\b[A-Z][A-Za-z\u2019-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z\u2019-]+)?'
                        r'\s+et al\.?,?\s*\(?(?:19|20)\d\d\)?')

# Ligatures survive extraction; some PDFs also decode to mojibake ("pðytþ1 jzt").
LIGATURES = str.maketrans({'\ufb00': 'ff', '\ufb01': 'fi', '\ufb02': 'fl',
                           '\ufb03': 'ffi', '\ufb04': 'ffl', '\u2019': "'",
                           '\u201c': '"', '\u201d': '"'})
MOJIBAKE = re.compile(r'[\u00f0\u00fe\u00de\u00bc\ufffd\u02dc]')

def strip_inline(text):
    text = text.translate(LIGATURES)
    text = CITE_PAREN.sub('', text)
    text = CITE_BRACK.sub('', text)
    text = NARR_CITE.sub('', text)
    text = STATS_PAREN.sub('', text)
    text = FIG_REF.sub('', text)
    text = STATS_BARE.sub('', text)
    text = CITE_SUP.sub('', text)
    return re.sub(r'\s+([,.;:])', r'\1', text)

# ------------------------------------------------------------ sentence level

# A sentence of real prose almost always contains one of these.
FUNCTION_WORDS = set('''the a an of in on at by for with from to into this that these those
is are was were be been being has have had we our it its they their which who when where
as and or but if than then thus because while although however can could may might will
would should do does did not more most such each between among during before after'''.split())

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z(])')

# Equation prose ("where wt is the gain at time t, and C 0 is ...") reads as words
# but is really notation; these markers catch it.
MATH = re.compile(r'[=<>≈±∝∑∫√^_·×]|[\u0370-\u03ff]')

# A single symbol is normal in methods prose ("we used both periods to estimate
# beta"); notation only dominates once symbols or bare variable letters pile up.
def math_heavy(sentence):
    toks = sentence.split()
    symbols = len(MATH.findall(sentence))
    # "a" and "I" are words, not variables, so they must not count as notation.
    singles = sum(1 for t in toks if t.strip('.,;:()') not in ('a', 'A', 'I')
                  and len(t.strip('.,;:()')) == 1 and t.strip('.,;:()').isalpha())
    return symbols >= 2 or singles >= 2 or (symbols and len(toks) < 8)

def is_prose(sentence):
    toks = sentence.split()
    if len(toks) < 6:
        return False
    if PANEL.search(sentence) or PANEL_TAIL.search(sentence) or math_heavy(sentence):
        return False
    if MOJIBAKE.search(sentence):          # mis-decoded equation text
        return False
    if sentence.rstrip().endswith(':') and len(toks) < 15:
        return False                       # lead-in to a display equation
    words = [t for t in toks if re.match(r"^[A-Za-z][A-Za-z'’-]*$", t.strip('.,;:()'))]
    if len(words) / len(toks) < 0.72:
        return False
    return bool(FUNCTION_WORDS & {w.lower().strip('.,;:()') for w in words})

def prose_only(text):
    out = []
    for para in re.split(r'\n\s*\n', text):
        kept = [s for s in SENT_SPLIT.split(re.sub(r'\s+', ' ', para).strip()) if is_prose(s)]
        if kept:
            out.append(' '.join(kept))
    return '\n\n'.join(out)

# ------------------------------------------------------------------- driver

def clean(path):
    lines = open(path, errors='ignore').read().split('\n')
    lines = trim_front_matter(lines)
    lines = drop_headings(lines)
    headers = running_headers(lines)

    body = []
    for line in lines:
        if REF_HEAD.match(line):
            break
        if keep_line(line, headers):
            body.append(line.rstrip())

    text = '\n'.join(body)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)          # rejoin hyphenated breaks
    text = re.sub(r'(?<![.!?:;])\n(?=[a-z])', ' ', text)  # unwrap mid-sentence breaks
    text = strip_inline(text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return prose_only(text)

if __name__ == '__main__':
    chunks = []
    for f in sorted(glob.glob('text/*.txt')):
        c = clean(f)
        chunks.append(c)
        print(f'{os.path.basename(f):75s} {len(c.split()):6d} words')
    corpus = '\n\n'.join(chunks)
    open('corpus.txt', 'w').write(corpus)
    print('total', len(corpus.split()), 'words ->', 'corpus.txt')
