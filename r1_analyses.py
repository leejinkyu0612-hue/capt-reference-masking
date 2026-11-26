"""Analyses added after review round R1 (exploratory/post-hoc unless noted; no new model calls).
1) speaker-cluster sign-flip permutation tests (10,000) for the nine confirmatory contrasts + Holm; 10,000-resample bootstrap CIs
2) Youden's J = 1 - FAR - FRR per system and condition, with Delta-J bootstrap CI
3) masking/unmasking transitions of mispronounced words (R0 -> R1)
4) matched-sample comparison (811 utterances common to all six systems; 1,000 for Whisper + audio LLMs)
5) R x V interaction (difference in differences) for audio LLMs, bootstrap CI
6) characteristics of the utterances missing from the gpt-4o-mini-transcribe R1 run
Output: results/r1_analyses.json"""
import os, json
import numpy as np
import analysis as A
from common import RESULTS, RUNS, read_jsonl

B = 10000
rng = np.random.default_rng(2027)
NS = len(A.SPEAKERS)
draws = None


def spk_sums(v, sel):
    return np.bincount(A.SPI[sel], v[sel].astype(float), NS), np.bincount(A.SPI[sel], minlength=NS).astype(float)


def boot(fn_stat, sel_list, Bn=B, seed=2027):
    """generic speaker-cluster bootstrap. fn_stat receives speaker weights (counts per speaker)."""
    r = np.random.default_rng(seed)
    present = np.unique(np.concatenate([A.SPI[s] for s in sel_list]))
    out = np.empty(Bn)
    for b in range(Bn):
        idx = r.choice(present, size=len(present), replace=True)
        w = np.bincount(idx, minlength=NS).astype(float)
        out[b] = fn_stat(w)
    return np.percentile(out, [2.5, 97.5]).tolist()


def rate(acc, sel, w=None, invert=False):
    s, n = spk_sums(1 - acc if invert else acc, sel)
    if w is None:
        return s.sum() / n.sum()
    return (w * s).sum() / (w * n).sum()


def signflip_p(a, b, sel, Bn=B, seed=2027):
    sa, n = spk_sums(a, sel); sb, _ = spk_sums(b, sel)
    d = sb - sa; obs = d.sum() / n.sum()
    r = np.random.default_rng(seed)
    signs = r.choice([-1.0, 1.0], size=(Bn, NS))
    perm = (signs * d).sum(1) / n.sum()
    return float(obs), float((np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (Bn + 1))


R = A.runs
out = {'confirmatory_cluster': [], 'youden': [], 'transitions': [], 'matched': {}, 'interaction': {}, 'missing_transcribe': {}}
MASK = {'all': A.mask_all, '1000': A.mask_1000}
pos = A.LAB['primary'] == 1; neg = A.LAB['primary'] == 0
for h, a0, b0, m in A.CONF:
    a, b = R[a0][0], R[b0][0]
    base = MASK[m] & (a >= 0) & (b >= 0)
    sp, sn = base & pos, base & neg
    d, p = signflip_p(a, b, sp)
    ci = boot(lambda w: rate(b, sp, w) - rate(a, sp, w), [sp])
    out['confirmatory_cluster'].append({'hyp': h, 'A': a0, 'B': b0, 'dFAR': d, 'CI10k': ci, 'p_signflip': p, 'n_pos': int(sp.sum())})
ps = sorted([(c['p_signflip'], i) for i, c in enumerate(out['confirmatory_cluster'])]); prev = 0; k = len(ps)
for rank, (p, i) in enumerate(ps):
    adj = min(1.0, max(prev, (k - rank) * p)); prev = adj; out['confirmatory_cluster'][i]['p_holm'] = adj
SYS = [('Whisper tiny', 'whisper-tiny_R0', 'whisper-tiny_R1', 'all'), ('Whisper base', 'whisper-base_R0', 'whisper-base_R1', 'all'),
       ('Whisper small', 'whisper-small_R0', 'whisper-small_R1', 'all'), ('gpt-4o-mini-transcribe', 'gpt-4o-mini-transcribe_R0_n1000', 'gpt-4o-mini-transcribe_R1_n1000', '1000'),
       ('gemini-2.5-flash', 'gemini-2.5-flash_R0V0_n1000', 'gemini-2.5-flash_R1V0_n1000', '1000'), ('gpt-audio-mini', 'gpt-audio-mini_R0V0_n1000', 'gpt-audio-mini_R1V0_n1000', '1000')]
for name, a0, b0, m in SYS:
    a, b = R[a0][0], R[b0][0]
    base = MASK[m] & (a >= 0) & (b >= 0); sp, sn = base & pos, base & neg
    J = lambda acc, w=None: 1 - rate(acc, sp, w) - rate(acc, sn, w, invert=True)
    jci = boot(lambda w: J(b, w) - J(a, w), [sp, sn], Bn=B)
    out['youden'].append({'system': name, 'J_R0': J(a), 'J_R1': J(b), 'dJ': J(b) - J(a), 'dJ_CI': jci})
    x, y = a[sp], b[sp]
    out['transitions'].append({'system': name, 'n_pos': int(sp.sum()), 'masked_R0rej_R1acc': float(((x == 0) & (y == 1)).mean()),
                               'unmasked_R0acc_R1rej': float(((x == 1) & (y == 0)).mean()),
                               'share_of_R0_detected_that_became_accepted': float(((x == 0) & (y == 1)).sum() / max((x == 0).sum(), 1))})
# matched samples
common = A.mask_1000.copy()
for name, a0, b0, m in SYS:
    common &= (R[a0][0] >= 0) & (R[b0][0] >= 0)
for name, a0, b0, m in SYS:
    a, b = R[a0][0], R[b0][0]; sp = common & pos
    out['matched'].setdefault('all6', []).append({'system': name, 'n_pos': int(sp.sum()), 'FAR_R0': rate(a, sp), 'FAR_R1': rate(b, sp), 'dFAR': rate(b, sp) - rate(a, sp)})
out['matched']['n_utts_all6'] = int(len(set(A.UID[common])))
c5 = A.mask_1000.copy()
for name, a0, b0, m in [s for s in SYS if 'transcribe' not in s[1]]:
    c5 &= (R[a0][0] >= 0) & (R[b0][0] >= 0)
for name, a0, b0, m in [s for s in SYS if 'transcribe' not in s[1]]:
    a, b = R[a0][0], R[b0][0]; sp = c5 & pos
    out['matched'].setdefault('five_1000', []).append({'system': name, 'n_pos': int(sp.sum()), 'FAR_R0': rate(a, sp), 'FAR_R1': rate(b, sp), 'dFAR': rate(b, sp) - rate(a, sp)})
# interaction
for s in ('gemini-2.5-flash', 'gpt-audio-mini'):
    v = {c: R[f'{s}_{c}_n1000'][0] for c in ('R0V0', 'R1V0', 'R0V1', 'R1V1')}
    base = A.mask_1000.copy()
    for c in v: base &= v[c] >= 0
    sp = base & pos
    did = lambda w=None: (rate(v['R1V1'], sp, w) - rate(v['R0V1'], sp, w)) - (rate(v['R1V0'], sp, w) - rate(v['R0V0'], sp, w))
    out['interaction'][s] = {'FAR': {c: rate(v[c], sp) for c in v}, 'R_effect_V0': rate(v['R1V0'], sp) - rate(v['R0V0'], sp),
                             'R_effect_V1': rate(v['R1V1'], sp) - rate(v['R0V1'], sp), 'DiD': did(), 'DiD_CI': boot(lambda w: did(w), [sp])}
# refusal-excluded contrasts (10,000 resamples)
import re as _re
REF = _re.compile(r"\b(sorry|can't|cannot|can not|unable|i'm not able|i am not able)\b", _re.I)
out['refusal_excluded'] = []
for s in ('gemini-2.5-flash', 'gpt-audio-mini'):
    for Bc in ('R1V0', 'R0V1'):
        a0, b0 = f'{s}_R0V0_n1000', f'{s}_{Bc}_n1000'
        bad = set()
        for n_ in (a0, b0):
            bad |= {x['uid'] for x in read_jsonl(os.path.join(RUNS, n_ + '.jsonl')) if x.get('hyp') and REF.search(x['hyp'])}
        keep = np.array([u not in bad for u in A.UID])
        a, b = R[a0][0], R[b0][0]; sp = A.mask_1000 & keep & (a >= 0) & (b >= 0) & pos
        out['refusal_excluded'].append({'A': a0, 'B': b0, 'excluded_utts': len(bad), 'dFAR': rate(b, sp) - rate(a, sp), 'CI10k': boot(lambda w: rate(b, sp, w) - rate(a, sp, w), [sp])})
# missing utterances in transcribe R1
r1 = {x['uid'] for x in read_jsonl(os.path.join(RUNS, 'gpt-4o-mini-transcribe_R1_n1000.jsonl'))}
s1000 = set(json.load(open(os.path.join(RUNS, 'subsets.json')))['subset1000'])
miss = s1000 - r1
mm = np.array([u in miss for u in A.UID]); inc = A.mask_1000 & ~mm
out['missing_transcribe'] = {'n_missing_utts': len(miss), 'pos_words_missing': int((mm & pos).sum()), 'pos_words_included': int((inc & pos).sum()),
                             'child_share_missing_words': float((A.AGE[mm] <= 15).mean()), 'child_share_included_words': float((A.AGE[inc] <= 15).mean()),
                             'pos_rate_missing': float((mm & pos).sum() / max(mm.sum(), 1)), 'pos_rate_included': float((inc & pos).sum() / max(inc.sum(), 1))}
json.dump(out, open(os.path.join(RESULTS, 'r1_analyses.json'), 'w', encoding='utf-8'), indent=1)
for c in out['confirmatory_cluster']:
    print(c['hyp'], c['A'], '->', c['B'].split('_', 1)[1], 'd=%.1f CI[%.1f,%.1f] p=%.4g holm=%.4g' % (100 * c['dFAR'], 100 * c['CI10k'][0], 100 * c['CI10k'][1], c['p_signflip'], c['p_holm']))
for y in out['youden']:
    print('J', y['system'], '%.1f -> %.1f d=%.1f [%.1f,%.1f]' % (100 * y['J_R0'], 100 * y['J_R1'], 100 * y['dJ'], 100 * y['dJ_CI'][0], 100 * y['dJ_CI'][1]))
for t in out['transitions']:
    print('T', t)
print('matched', out['matched'])
print('inter', out['interaction'])
print('missing', out['missing_transcribe'])
print('refex', out['refusal_excluded'])

