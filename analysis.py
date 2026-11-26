"""Scoring and pre-specified statistics (PREREGISTRATION.md + AMENDMENT_1.md).
Outputs: results/analysis_main.json, results/word_level.csv (no audio)."""
import os, json, math, csv
import numpy as np
from scipy.stats import binomtest
from common import load_rows, RUNS, RESULTS, read_jsonl, align
from common import norm_words as _nw
SCORING = os.environ.get('SCORING', 'prereg')   # prereg | digits (post-hoc: digit strings spelled out)
WFILTER = os.environ.get('WFILTER', 'none')     # none | common10k (post-hoc: drop target words outside 10k frequent words)
DIG = ['ZERO','ONE','TWO','THREE','FOUR','FIVE','SIX','SEVEN','EIGHT','NINE']
def norm_words(t):
    ws = _nw(t)
    if SCORING == 'digits':
        out = []
        for w in ws:
            out += [DIG[int(ch)] for ch in w] if w.isdigit() else [w]
        return out
    return ws
COMMON = set(l.strip().upper() for l in open(os.path.join(os.path.dirname(RUNS), 'data', 'google-10000-english-usa.txt'), encoding='utf-8')) if WFILTER == 'common10k' else None
SUFFIX = '' if (SCORING == 'prereg' and WFILTER == 'none') else f'_{SCORING}_{WFILTER}'

rows = load_rows(with_audio=False)
subs = json.load(open(os.path.join(RUNS, 'subsets.json')))
S1000, S500 = set(subs['subset1000']), set(subs['subset500'])
ALL = {r['uid'] for r in rows}
spk = {r['uid']: r['speaker'] for r in rows}
age = {r['uid']: r['age'] for r in rows}

# word table
W = []  # (uid, idx, speaker, age, primary, le5, le3)
ref_tokens = {}
for r in rows:
    toks, owner = [], []
    for k, w in enumerate(r['words']):
        t = norm_words(w['text'])
        toks += t; owner += [k] * len(t)
        acc = w['accuracy']; mis = bool(w['mispronunciations'])
        keepw = COMMON is None or all(t in COMMON for t in _nw(w['text']))
        W.append((r['uid'], k, r['speaker'], r['age'], (1 if mis else (0 if acc == 10 else -1)) if keepw else -1,
                  (1 if acc <= 5 else (0 if acc == 10 else -1)) if keepw else -1, (1 if acc <= 3 else (0 if acc == 10 else -1)) if keepw else -1))
    ref_tokens[r['uid']] = (toks, owner, len(r['words']))
W = np.array(W, dtype=object)
UID = W[:, 0]; SPK = W[:, 2]; AGE = W[:, 3].astype(int)
LAB = {'primary': W[:, 4].astype(int), 'le5': W[:, 5].astype(int), 'le3': W[:, 6].astype(int)}
word_index = {(u, k): i for i, (u, k) in enumerate(zip(W[:, 0], W[:, 1]))}


def score_run(name):
    recs = {}
    for x in read_jsonl(os.path.join(RUNS, name + '.jsonl')):
        if x.get('hyp') is not None:
            recs[x['uid']] = x
    acc = np.full(len(W), -1, dtype=int)  # -1 missing
    errs = 0; nref = 0; fail = 0
    for u, x in recs.items():
        toks, owner, nw = ref_tokens[u]
        hyp = norm_words(x['hyp'])
        a, d = align(toks, hyp)
        errs += d; nref += len(toks)
        per = [True] * nw
        for t_ok, k in zip(a, owner):
            per[k] = per[k] and t_ok
        if sum(a) < 0.2 * len(a):
            fail += 1
        for k in range(nw):
            acc[word_index[(u, k)]] = int(per[k])
    return acc, {'n_utts': len(recs), 'wer': errs / max(nref, 1), 'gross_failures': fail}


def metrics(acc, mask, lab='primary'):
    L = LAB[lab]
    pos = mask & (L == 1) & (acc >= 0); neg = mask & (L == 0) & (acc >= 0)
    return {'n_pos': int(pos.sum()), 'n_neg': int(neg.sum()),
            'FAR': float(acc[pos].mean()) if pos.any() else None,
            'FRR': float(1 - acc[neg].mean()) if neg.any() else None}


rng_global = np.random.default_rng(2027)
SPEAKERS = sorted(set(SPK))
sp_idx = {s: i for i, s in enumerate(SPEAKERS)}
SPI = np.array([sp_idx[s] for s in SPK])


def boot_delta(a, b, sel, B=2000, seed=2027, frr=False):
    """speaker-cluster bootstrap of rate(b) - rate(a) over words in sel. For FRR, rate = 1-acc."""
    va, vb = a[sel].astype(float), b[sel].astype(float)
    if frr:
        va, vb = 1 - va, 1 - vb
    s = SPI[sel]
    n = np.bincount(s, minlength=len(SPEAKERS)); sa = np.bincount(s, va, len(SPEAKERS)); sb = np.bincount(s, vb, len(SPEAKERS))
    rng = np.random.default_rng(seed)
    present = np.where(n > 0)[0]
    draws = rng.choice(present, size=(B, len(present)), replace=True)
    N = n[draws].sum(1); A = sa[draws].sum(1); Bv = sb[draws].sum(1)
    d = (Bv - A) / N
    est = (vb.sum() - va.sum()) / len(va)
    return float(est), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def mcnemar(a, b, sel):
    x, y = a[sel], b[sel]
    b10 = int(((x == 1) & (y == 0)).sum()); b01 = int(((x == 0) & (y == 1)).sum())
    p = binomtest(min(b10, b01), b10 + b01, 0.5).pvalue if b10 + b01 > 0 else 1.0
    return b10, b01, float(p)


def contrast(A, B, base_mask, lab='primary', extra_mask=None):
    a, b = runs[A][0], runs[B][0]
    m = base_mask & (a >= 0) & (b >= 0)
    if extra_mask is not None:
        m &= extra_mask
    L = LAB[lab]
    pos = m & (L == 1); neg = m & (L == 0)
    far = boot_delta(a, b, pos); frr = boot_delta(a, b, neg, frr=True)
    mc = mcnemar(a, b, pos); mcf = mcnemar(1 - a, 1 - b, neg)
    return {'A': A, 'B': B, 'label': lab, 'n_pos': int(pos.sum()), 'n_neg': int(neg.sum()),
            'FAR_A': float(a[pos].mean()), 'FAR_B': float(b[pos].mean()),
            'dFAR': far[0], 'dFAR_CI': [far[1], far[2]], 'mcnemar_pos': {'A1B0': mc[0], 'A0B1': mc[1], 'p': mc[2]},
            'FRR_A': float(1 - a[neg].mean()), 'FRR_B': float(1 - b[neg].mean()),
            'dFRR': frr[0], 'dFRR_CI': [frr[1], frr[2]], 'mcnemar_neg_p': mcf[2]}


RUNLIST = [f[:-6] for f in os.listdir(RUNS) if f.endswith('.jsonl')]
runs = {n: score_run(n) for n in RUNLIST}
mask_all = np.ones(len(W), bool)
mask_1000 = np.array([u in S1000 for u in UID]); mask_500 = np.array([u in S500 for u in UID])
child = AGE <= 15

out = {'runs': {}, 'confirmatory': [], 'sensitivity': [], 'age': [], 'robustness': {}, 'efficiency': {}}
for n, (acc, info) in sorted(runs.items()):
    out['runs'][n] = dict(info, full=metrics(acc, mask_all), subset1000=metrics(acc, mask_1000), subset500=metrics(acc, mask_500))

CONF = [('H1', 'whisper-tiny_R0', 'whisper-tiny_R1', 'all'), ('H1', 'whisper-base_R0', 'whisper-base_R1', 'all'),
        ('H1', 'whisper-small_R0', 'whisper-small_R1', 'all'),
        ('H1', 'gpt-4o-mini-transcribe_R0_n1000', 'gpt-4o-mini-transcribe_R1_n1000', '1000'),
        ('H2a', 'gemini-2.5-flash_R0V0_n1000', 'gemini-2.5-flash_R1V0_n1000', '1000'),
        ('H2a', 'gpt-audio-mini_R0V0_n1000', 'gpt-audio-mini_R1V0_n1000', '1000'),
        ('H2b', 'gemini-2.5-flash_R0V0_n1000', 'gemini-2.5-flash_R0V1_n1000', '1000'),
        ('H2b', 'gpt-audio-mini_R0V0_n1000', 'gpt-audio-mini_R0V1_n1000', '1000'),
        ('H3', 'whisper-tiny_R0', 'whisper-small_R0', 'all')]
MASK = {'all': mask_all, '1000': mask_1000, '500': mask_500}
res = []
for h, A, B, m in CONF:
    if A in runs and B in runs:
        c = contrast(A, B, MASK[m]); c['hyp'] = h; c['sample'] = m; res.append(c)
# Holm
ps = sorted([(c['mcnemar_pos']['p'], i) for i, c in enumerate(res)])
k = len(ps); prev = 0
for rank, (p, i) in enumerate(ps):
    adj = min(1.0, max(prev, (k - rank) * p)); prev = adj; res[i]['p_holm'] = adj
for c in res:
    pred_up = c['hyp'] in ('H1', 'H2a', 'H3')
    ci = c['dFAR_CI']
    c['supported'] = bool(c['p_holm'] < 0.05 and ((ci[0] > 0) if pred_up else (ci[1] < 0)))
out['confirmatory'] = res
for h, A, B, m in CONF:
    if A in runs and B in runs:
        for lab in ('le5', 'le3'):
            c = contrast(A, B, MASK[m], lab); c['hyp'] = h; out['sensitivity'].append(c)
        for g, gm in (('child', child), ('adult', ~child)):
            c = contrast(A, B, MASK[m], extra_mask=gm); c['hyp'] = h; c['group'] = g; out['age'].append(c)
# whisper models on the 1000 subset (cross-family descriptive)
for n in ('whisper-tiny_R0', 'whisper-tiny_R1', 'whisper-base_R0', 'whisper-base_R1', 'whisper-small_R0', 'whisper-small_R1'):
    if n in runs:
        out['runs'][n]['subset1000_wer'] = None
# robustness
rb = {}
if 'whisper-base_R0_beam5_n500' in runs and 'whisper-base_R1_beam5_n500' in runs:
    rb['beam5'] = contrast('whisper-base_R0_beam5_n500', 'whisper-base_R1_beam5_n500', mask_500)
    if 'whisper-base_R0' in runs and 'whisper-base_R1' in runs:
        rb['greedy_same500'] = contrast('whisper-base_R0', 'whisper-base_R1', mask_500)
for alt in ('R0V1', 'R1V1'):
    A, B = f'gpt-audio-mini_{alt}_n1000', f'gpt-audio-mini_{alt}_alt_n500'
    if A in runs and B in runs:
        rb[f'alt_wording_{alt}'] = contrast(A, B, mask_500)
    if f'gpt-audio-mini_R0V0_n1000' in runs and B in runs:
        rb[f'alt_vs_R0V0_{alt}'] = contrast('gpt-audio-mini_R0V0_n1000', B, mask_500)
for c0 in ('R0V0', 'R1V0'):
    A, B = f'gpt-audio-mini_{c0}_n1000', f'gpt-audio-mini_{c0}_rep_n500'
    if A in runs and B in runs:
        a, b = runs[A][0], runs[B][0]; m = mask_500 & (a >= 0) & (b >= 0)
        ra = {x['uid']: norm_words(x['hyp']) for x in read_jsonl(os.path.join(RUNS, A + '.jsonl'))}
        rbb = {x['uid']: norm_words(x['hyp']) for x in read_jsonl(os.path.join(RUNS, B + '.jsonl'))}
        common_u = [u for u in rbb if u in ra]
        rb[f'repeat_{c0}'] = {'word_accept_agreement': float((a[m] == b[m]).mean()), 'n_words': int(m.sum()),
                              'identical_transcripts': float(np.mean([ra[u] == rbb[u] for u in common_u])), 'n_utts': len(common_u),
                              'contrast': contrast(A, B, mask_500)}
out['robustness'] = rb
# efficiency
eff = {}
for n in runs:
    recs = read_jsonl(os.path.join(RUNS, n + '.jsonl'))
    if n.startswith('whisper') and recs and 'sec' in recs[0]:
        eff[n] = {'rtf': sum(x['sec'] for x in recs) / sum(x['dur'] for x in recs), 'n': len(recs),
                  'mean_sec': float(np.mean([x['sec'] for x in recs]))}
    elif recs and 'latency' in recs[0]:
        eff[n] = {'median_latency_s': float(np.median([x['latency'] for x in recs if x.get('latency') is not None])), 'n': len(recs)}
out['efficiency'] = eff
# interaction descriptive
inter = {}
for sname in ('gemini-2.5-flash', 'gpt-audio-mini'):
    inter[sname] = {c: out['runs'].get(f'{sname}_{c}_n1000', {}).get('subset1000') for c in ('R0V0', 'R1V0', 'R0V1', 'R1V1')}
out['interaction'] = inter
out['counts'] = {'utts': len(rows), 'speakers': len(SPEAKERS), 'words': int(len(W)),
                 'pos_primary': int((LAB['primary'] == 1).sum()), 'neg': int((LAB['primary'] == 0).sum()),
                 'pos_le5': int((LAB['le5'] == 1).sum()), 'pos_le3': int((LAB['le3'] == 1).sum()),
                 'pos_primary_1000': int(((LAB['primary'] == 1) & mask_1000).sum()), 'neg_1000': int(((LAB['primary'] == 0) & mask_1000).sum()),
                 'child_utts': int(sum(1 for r in rows if r['age'] <= 15))}
json.dump(out, open(os.path.join(RESULTS, f'analysis_main{SUFFIX}.json'), 'w', encoding='utf-8'), indent=1)
with open(os.path.join(RESULTS, f'word_level{SUFFIX}.csv'), 'w', newline='', encoding='utf-8') as f:
    wr = csv.writer(f); names = sorted(runs)
    wr.writerow(['uid', 'word_idx', 'speaker', 'age', 'label_primary', 'label_le5', 'label_le3'] + names)
    for i in range(len(W)):
        wr.writerow(list(W[i]) + [int(runs[n][0][i]) for n in names])
print(json.dumps({'counts': out['counts'], 'confirmatory': [{k: c[k] for k in ('hyp', 'A', 'B', 'n_pos', 'FAR_A', 'FAR_B', 'dFAR', 'dFAR_CI', 'p_holm', 'supported', 'FRR_A', 'FRR_B')} for c in res]}, indent=1))


