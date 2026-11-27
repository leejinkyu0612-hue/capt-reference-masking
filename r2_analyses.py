"""Analyses added after review round R2 (exploratory/post-hoc; no new API calls).
1) control experiment summary (silence / mismatched audio with target prompt; unrelated prompt with real audio)
2) gpt-4o-mini-transcribe missing-data bounds and R0 comparison of included vs missing utterances
3) audio-LLM analysis on utterances with a valid (non-refusal) response in all four conditions
4) refusal regex audit: refusals whose target sentence itself contains a matched word
5) label checks (overlap of definitions) and an FRR sensitivity using all non-mispronounced words as negatives
6) example words masked by reference context (for the paper)
Output: results/r2_analyses.json"""
import os, re, json
import numpy as np
import analysis as A
from common import RUNS, RESULTS, read_jsonl, norm_words, align, load_rows

out = {}
pos = A.LAB['primary'] == 1; neg = A.LAB['primary'] == 0
REF = re.compile(r"\b(sorry|can't|cannot|can not|unable|i'm not able|i am not able)\b", re.I)
rows = {r['uid']: r for r in load_rows(with_audio=False)}


def accept_vec(recs):
    """per-utterance list of per-word acceptance for given records"""
    res = {}
    for x in recs:
        r = rows[x['uid']]
        toks, owner, nw = A.ref_tokens[x['uid']]
        a, _ = align(toks, norm_words(x['hyp']))
        per = [True] * nw
        for ok, k in zip(a, owner):
            per[k] = per[k] and ok
        res[x['uid']] = per
    return res


# 1) control
ctrl = {}
for c in ('C', 'D', 'E'):
    recs = read_jsonl(os.path.join(RUNS, f'control_whisper-base_{c}.jsonl'))
    acc = accept_vec(recs)
    allw = [v for per in acc.values() for v in per]
    exact = np.mean([norm_words(x['hyp']) == norm_words(x['ref']) for x in recs])
    lab = {u: [1 if w['mispronunciations'] else (0 if w['accuracy'] == 10 else -1) for w in rows[u]['words']] for u in acc}
    p_ = [a for u in acc for a, l in zip(acc[u], lab[u]) if l == 1]; n_ = [a for u in acc for a, l in zip(acc[u], lab[u]) if l == 0]
    ctrl[c] = {'n_utts': len(recs), 'target_word_acceptance': float(np.mean(allw)), 'exact_target_output': float(exact),
               'FAR': float(np.mean(p_)) if p_ else None, 'FRR': float(1 - np.mean(n_)) if n_ else None, 'n_pos': len(p_)}
# same 300 utterances under the main R0 / R1 runs
uids300 = [x['uid'] for x in read_jsonl(os.path.join(RUNS, 'control_whisper-base_C.jsonl'))]
m300 = np.array([u in set(uids300) for u in A.UID])
for nm, run in (('A_real_R0', 'whisper-base_R0'), ('B_real_R1', 'whisper-base_R1')):
    acc = A.runs[run][0]
    ctrl[nm] = {'target_word_acceptance': float(acc[m300].mean()), 'FAR': float(acc[m300 & pos].mean()),
                'FRR': float(1 - acc[m300 & neg].mean()), 'n_pos': int((m300 & pos).sum()),
                'exact_target_output': float(np.mean([norm_words(x['hyp']) == norm_words(x['ref']) for x in read_jsonl(os.path.join(RUNS, run + '.jsonl')) if x['uid'] in set(uids300)]))}
out['control'] = ctrl

# 2) missing-data bounds for gpt-4o-mini-transcribe
a = A.runs['gpt-4o-mini-transcribe_R0_n1000'][0]; b = A.runs['gpt-4o-mini-transcribe_R1_n1000'][0]
sel = A.mask_1000 & (a >= 0) & pos
miss = sel & (b < 0); inc = sel & (b >= 0)
far0 = a[sel].mean()
b_lo = b.copy(); b_lo[miss] = 0; b_hi = b.copy(); b_hi[miss] = 1
out['transcribe_missing'] = {'n_pos_with_R0': int(sel.sum()), 'n_pos_missing_R1': int(miss.sum()), 'FAR_R0_included': float(a[inc].mean()), 'FAR_R0_missing': float(a[miss].mean()),
                             'dFAR_bound_low': float(b_lo[sel].mean() - far0), 'dFAR_bound_high': float(b_hi[sel].mean() - far0),
                             'FRR_R0_included': float(1 - a[A.mask_1000 & (a >= 0) & neg & (b >= 0)].mean()), 'FRR_R0_missing': float(1 - a[A.mask_1000 & (a >= 0) & neg & (b < 0)].mean())}

# 3) audio LLMs: valid responses in all four conditions
four = {}
for s in ('gemini-2.5-flash', 'gpt-audio-mini'):
    bad = set()
    for c in ('R0V0', 'R1V0', 'R0V1', 'R1V1'):
        bad |= {x['uid'] for x in read_jsonl(os.path.join(RUNS, f'{s}_{c}_n1000.jsonl')) if x.get('hyp') and REF.search(x['hyp'])}
    keep = np.array([u not in bad for u in A.UID])
    v = {c: A.runs[f'{s}_{c}_n1000'][0] for c in ('R0V0', 'R1V0', 'R0V1', 'R1V1')}
    base = A.mask_1000 & keep
    for c in v: base &= v[c] >= 0
    sp, sn = base & pos, base & neg
    F = {c: float(v[c][sp].mean()) for c in v}; R_ = {c: float(1 - v[c][sn].mean()) for c in v}
    four[s] = {'excluded_utts': len(bad), 'n_pos': int(sp.sum()), 'FAR': F, 'FRR': R_, 'R_effect_V0': F['R1V0'] - F['R0V0'], 'V_effect_R0': F['R0V1'] - F['R0V0'],
               'DiD': (F['R1V1'] - F['R0V1']) - (F['R1V0'] - F['R0V0']), 'coverage': {c: 1 - len({x['uid'] for x in read_jsonl(os.path.join(RUNS, f'{s}_{c}_n1000.jsonl')) if x.get('hyp') and REF.search(x['hyp'])}) / 1000 for c in v}}
out['four_condition_valid'] = four

# 4) refusal audit
aud = []
for s in ('gemini-2.5-flash', 'gpt-audio-mini'):
    for c in ('R0V0', 'R1V0', 'R0V1', 'R1V1'):
        for x in read_jsonl(os.path.join(RUNS, f'{s}_{c}_n1000.jsonl')):
            if x.get('hyp') and REF.search(x['hyp']):
                in_ref = bool(REF.search(x['ref'].replace("CAN'T", "can't")))
                looks_refusal = bool(re.search(r"\b(sorry|i can't|i cannot|unable to|i'm not able)\b", x['hyp'], re.I)) and len(norm_words(x['hyp'])) > len(norm_words(x['ref'])) + 3
                aud.append({'run': f'{s}_{c}', 'uid': x['uid'], 'ref_contains_match': in_ref, 'looks_like_refusal': looks_refusal, 'hyp': x['hyp'][:160]})
out['refusal_audit'] = {'n_matched': len(aud), 'n_ref_contains_match': sum(a_['ref_contains_match'] for a_ in aud),
                        'n_not_refusal_like': sum(not a_['looks_like_refusal'] for a_ in aud), 'examples_not_refusal_like': [a_ for a_ in aud if not a_['looks_like_refusal']][:10]}

# 5) label checks
words = [(u, w) for u in rows for w in rows[u]['words']]
out['labels'] = {'mispron_and_acc10': sum(1 for _, w in words if w['mispronunciations'] and w['accuracy'] == 10),
                 'mispron_total': sum(1 for _, w in words if w['mispronunciations']), 'acc10_total': sum(1 for _, w in words if w['accuracy'] == 10),
                 'excluded_middle': sum(1 for _, w in words if not w['mispronunciations'] and w['accuracy'] != 10)}
frr_all = {}
nonpos = A.LAB['primary'] != 1
for k, (a0, b0, m) in {'Whisper tiny': ('whisper-tiny_R0', 'whisper-tiny_R1', 'all'), 'Whisper base': ('whisper-base_R0', 'whisper-base_R1', 'all'),
                       'Whisper small': ('whisper-small_R0', 'whisper-small_R1', 'all'), 'gemini-2.5-flash': ('gemini-2.5-flash_R0V0_n1000', 'gemini-2.5-flash_R1V0_n1000', '1000'),
                       'gpt-audio-mini': ('gpt-audio-mini_R0V0_n1000', 'gpt-audio-mini_R1V0_n1000', '1000')}.items():
    aa, bb = A.runs[a0][0], A.runs[b0][0]
    mm = (A.mask_all if m == 'all' else A.mask_1000) & (aa >= 0) & (bb >= 0) & nonpos
    frr_all[k] = {'FRR_R0': float(1 - aa[mm].mean()), 'FRR_R1': float(1 - bb[mm].mean()), 'n': int(mm.sum())}
out['frr_all_nonpositive'] = frr_all

# 6) examples
ex = []
r0 = {x['uid']: x['hyp'] for x in read_jsonl(os.path.join(RUNS, 'whisper-small_R0.jsonl'))}
r1 = {x['uid']: x['hyp'] for x in read_jsonl(os.path.join(RUNS, 'whisper-small_R1.jsonl'))}
a0v, a1v = A.runs['whisper-small_R0'][0], A.runs['whisper-small_R1'][0]
idx = np.where(pos & (a0v == 0) & (a1v == 1))[0]
for i in idx[:40]:
    u, k = A.UID[i], A.W[i, 1]
    w = rows[u]['words'][k]
    ex.append({'uid': u, 'word': w['text'], 'mispron': w['mispronunciations'], 'ref': rows[u]['text'], 'R0': r0[u], 'R1': r1[u]})
out['examples_whisper_small'] = ex
json.dump(out, open(os.path.join(RESULTS, 'r2_analyses.json'), 'w', encoding='utf-8'), indent=1, default=float)
print(json.dumps({k: out[k] for k in ('control', 'transcribe_missing', 'four_condition_valid', 'labels', 'frr_all_nonpositive')}, indent=1, default=float))
print({k: out['refusal_audit'][k] for k in ('n_matched', 'n_ref_contains_match', 'n_not_refusal_like')})
for e in ex[:12]: print(e['word'], e['mispron'], '|', e['ref'], '|', e['R0'], '|', e['R1'])
