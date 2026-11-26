"""Exploratory/post-hoc analyses (added after inspecting outputs; not part of the pre-specified plan).
1) refusal / non-transcription responses per run (regex), 2) audio-LLM contrasts excluding utterances with a refusal
in either condition, 3) digit-string outputs per run. Output: results/posthoc.json"""
import os, re, json
import numpy as np
import analysis as A   # reuses scoring (respects SCORING / WFILTER env)
from common import RUNS, RESULTS, read_jsonl

REF = re.compile(r"\b(sorry|can't|cannot|can not|unable|i'm not able|i am not able)\b", re.I)
out = {'refusals': {}, 'digit_outputs': {}, 'contrasts_excl_refusals': []}
ref_utts = {}
for n in sorted(A.runs):
    recs = read_jsonl(os.path.join(RUNS, n + '.jsonl'))
    rf = {x['uid'] for x in recs if x.get('hyp') and REF.search(x['hyp'])}
    ref_utts[n] = rf
    out['refusals'][n] = {'n_refusal': len(rf), 'n_utts': len(recs)}
    out['digit_outputs'][n] = sum(1 for x in recs if x.get('hyp') and re.search(r'\d', x['hyp']))
for sysn in ('gemini-2.5-flash', 'gpt-audio-mini'):
    for B in ('R1V0', 'R0V1'):
        a, b = f'{sysn}_R0V0_n1000', f'{sysn}_{B}_n1000'
        bad = ref_utts[a] | ref_utts[b]
        keep = np.array([u not in bad for u in A.UID])
        c = A.contrast(a, b, A.mask_1000, extra_mask=keep)
        c['excluded_utts'] = len(bad)
        out['contrasts_excl_refusals'].append(c)
json.dump(out, open(os.path.join(RESULTS, 'posthoc.json'), 'w', encoding='utf-8'), indent=1)
for c in out['contrasts_excl_refusals']:
    print(c['A'], '->', c['B'], 'excl', c['excluded_utts'], 'FAR %.1f -> %.1f d=%+.1f [%+.1f, %+.1f] FRR %.1f -> %.1f' % (
        100 * c['FAR_A'], 100 * c['FAR_B'], 100 * c['dFAR'], 100 * c['dFAR_CI'][0], 100 * c['dFAR_CI'][1], 100 * c['FRR_A'], 100 * c['FRR_B']))
print({k: v['n_refusal'] for k, v in out['refusals'].items() if v['n_refusal']})
