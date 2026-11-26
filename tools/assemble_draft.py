# -*- coding: utf-8 -*-
"""Fill the S06 draft placeholders from logged analysis outputs, renumber citations in order of first
appearance (IEEE), and write the manuscript Markdown.  Usage: python assemble_draft.py front.md back.md out.md"""
import sys, os, re, json
EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(EXP, 'results'); RUNS = os.path.join(EXP, 'runs')
A = json.load(open(os.path.join(RES, 'analysis_main.json'), encoding='utf-8'))
AD = json.load(open(os.path.join(RES, 'analysis_main_digits_none.json'), encoding='utf-8'))
AF = json.load(open(os.path.join(RES, 'analysis_main_digits_common10k.json'), encoding='utf-8'))
PH = json.load(open(os.path.join(RES, 'posthoc.json'), encoding='utf-8'))
pc = lambda x: f"{100 * x:.1f}"
sd = lambda x: f"{100 * x:+.1f}"


def conf(a, A_, B_, src=None):
    for c in (src or a)['confirmatory']:
        if c['A'] == A_ and c['B'] == B_:
            return c
    raise KeyError((A_, B_))


def find(lst, A_, B_, **kw):
    for c in lst:
        if c['A'] == A_ and c['B'] == B_ and all(c.get(k) == v for k, v in kw.items()):
            return c
    raise KeyError((A_, B_, kw))


P = {}
SYS = {'T': ('whisper-tiny_R0', 'whisper-tiny_R1'), 'B': ('whisper-base_R0', 'whisper-base_R1'), 'S': ('whisper-small_R0', 'whisper-small_R1'),
       'G': ('gpt-4o-mini-transcribe_R0_n1000', 'gpt-4o-mini-transcribe_R1_n1000'),
       'GF': ('gemini-2.5-flash_R0V0_n1000', 'gemini-2.5-flash_R1V0_n1000'), 'GA': ('gpt-audio-mini_R0V0_n1000', 'gpt-audio-mini_R1V0_n1000')}
ds = []
for k, (a0, a1) in SYS.items():
    c = conf(A, a0, a1)
    P[f'{k}_FAR0'], P[f'{k}_FAR1'] = pc(c['FAR_A']), pc(c['FAR_B'])
    P[f'{k}_FRR0'], P[f'{k}_FRR1'] = pc(c['FRR_A']), pc(c['FRR_B'])
    P[f'{k}_D'] = f"{sd(c['dFAR'])} [{pc(c['dFAR_CI'][0])}, {pc(c['dFAR_CI'][1])}]"
    P[f'{k}_WER'] = f"{pc(A['runs'][a0]['wer'])}/{pc(A['runs'][a1]['wer'])}"
    ds.append(100 * c['dFAR'])
sup = [c for c in A['confirmatory'] if c['supported']]
P['N_SUP'] = {7: 'seven', 8: 'eight', 9: 'all'}.get(len(sup), str(len(sup)))
assert max(c['p_holm'] for c in A['confirmatory'] if c['hyp'] in ('H1', 'H2a')) < 0.001, 'table note p<0.001 not valid'
assert len(A['confirmatory']) == 9
P['MIN_D'], P['MAX_D'] = f"{min(ds):.1f}", f"{max(ds):.1f}"
cga = conf(A, 'gpt-audio-mini_R0V0_n1000', 'gpt-audio-mini_R0V1_n1000'); cgf = conf(A, 'gemini-2.5-flash_R0V0_n1000', 'gemini-2.5-flash_R0V1_n1000')
P['GA_V_FAR'], P['GA_V_D'], P['GA_V_FRR'] = pc(cga['FAR_B']), f"{100 * cga['dFAR']:.1f}", pc(cga['FRR_B'])
P['GF_V_D'], P['GF_V_P'] = f"{100 * cgf['dFAR']:.1f}", f"{cgf['p_holm']:.2f}"
P['GA_REFUSE'] = str(PH['refusals']['gpt-audio-mini_R0V0_n1000']['n_refusal']); P['GA_V_REFUSE'] = str(PH['refusals']['gpt-audio-mini_R0V1_n1000']['n_refusal'])
ex = find(PH['contrasts_excl_refusals'], 'gpt-audio-mini_R0V0_n1000', 'gpt-audio-mini_R0V1_n1000')
P['GA_V_EX_D'] = f"{100 * ex['dFAR']:.1f} [{pc(ex['dFAR_CI'][0])}, {pc(ex['dFAR_CI'][1])}]"
P['GF_R1V1'] = pc(A['runs']['gemini-2.5-flash_R1V1_n1000']['subset1000']['FAR']); P['GA_R1V1'] = pc(A['runs']['gpt-audio-mini_R1V1_n1000']['subset1000']['FAR'])
h3 = conf(A, 'whisper-tiny_R0', 'whisper-small_R0'); P['H3_D'] = f"{sd(h3['dFAR'])} [{pc(h3['dFAR_CI'][0])}, {pc(h3['dFAR_CI'][1])}]"
E = A['efficiency']
P['RTF_T'] = f"{(E['whisper-tiny_R0']['rtf'] + E['whisper-tiny_R1']['rtf']) / 2:.2f}"
P['RTF_B'] = f"{(E['whisper-base_R0']['rtf'] + E['whisper-base_R1']['rtf']) / 2:.2f}"
P['RTF_S'] = f"{(E['whisper-small_R0']['rtf'] + E['whisper-small_R1']['rtf']) / 2:.2f}"
P['LAT_G'] = f"{E['gpt-4o-mini-transcribe_R0_n1000']['median_latency_s']:.1f}"; P['LAT_GF'] = f"{E['gemini-2.5-flash_R0V0_n1000']['median_latency_s']:.1f}"
P['LAT_GA'] = f"{E['gpt-audio-mini_R0V0_n1000']['median_latency_s']:.1f}"
COL = {'T': SYS['T'], 'B': SYS['B'], 'S': SYS['S'], 'GF': SYS['GF'], 'GA': SYS['GA']}
for k, (a0, a1) in COL.items():
    P[f'RB_P_{k}'] = sd(conf(A, a0, a1)['dFAR'])
    P[f'RB_5_{k}'] = sd(find(A['sensitivity'], a0, a1, label='le5')['dFAR'])
    P[f'RB_3_{k}'] = sd(find(A['sensitivity'], a0, a1, label='le3')['dFAR'])
    P[f'RB_C_{k}'] = sd(find(A['age'], a0, a1, group='child')['dFAR'])
    P[f'RB_A_{k}'] = sd(find(A['age'], a0, a1, group='adult')['dFAR'])
    P[f'RB_D_{k}'] = sd(conf(AD, a0, a1)['dFAR'])
    P[f'RB_F_{k}'] = sd(conf(AF, a0, a1)['dFAR'])
rb = A['robustness']
P['BEAM_D'] = f"{sd(rb['beam5']['dFAR'])} pp, compared with {sd(rb['greedy_same500']['dFAR'])}"
P['ALT_R0'], P['ALT_R0_ORIG'] = pc(rb['alt_wording_R0V1']['FAR_B']), pc(rb['alt_wording_R0V1']['FAR_A'])
P['ALT_R1'], P['ALT_R1_ORIG'] = pc(rb['alt_wording_R1V1']['FAR_B']), pc(rb['alt_wording_R1V1']['FAR_A'])
P['REP_R0'], P['REP_R1'] = pc(rb['repeat_R0V0']['word_accept_agreement']), pc(rb['repeat_R1V0']['word_accept_agreement'])
C = A['counts']
P['N_CHILD'] = f"{C['child_utts']:,} utterances"; P['N_POS'] = f"{C['pos_primary']:,}"; P['N_NEG'] = f"{C['neg']:,}"
P['TRANSFORMERS'] = json.load(open(os.path.join(RUNS, 'whisper-tiny_R0.meta.json')))['transformers']
P['REPO'] = 'https://github.com/leejinkyu0612-hue/capt-reference-masking'
P['ABSTRACT'] = (
    "Read-aloud practice on e-learning platforms often turns automatic speech recognition (ASR) transcripts into word-level "
    "pronunciation feedback. Because the application knows the target sentence, developers can pass it to the recognizer as "
    "context, but whether this hides learners' pronunciation errors has not been measured against expert labels. This paper "
    "evaluates this design choice on the speechocean762 test set (2,500 read utterances by 125 Mandarin-L1 learners of English, "
    "scored by five experts). Three Whisper models run on a CPU, a commercial transcription API and two audio large language "
    "models (LLMs) transcribed each utterance with and without the target sentence as context, and the audio LLMs were also given "
    "a verbatim instruction. A target word was accepted when the aligned transcript contained it, and false acceptance (FAR) and "
    "false rejection (FRR) rates were computed against the expert labels with speaker-cluster bootstrap intervals. Reference-text "
    f"context increased the FAR of mispronounced words for all six systems, by {P['MIN_D']} to {P['MAX_D']} percentage points, "
    "while it reduced FRR. The verbatim instruction lowered FAR for one of the two audio LLMs only, and larger Whisper models "
    "accepted more mispronounced words. The results held under stricter and looser error labels and for child and adult speakers. "
    "For computer-assisted pronunciation training, the target sentence and model size should therefore be treated as assessment "
    "parameters and evaluated with FAR and FRR rather than with word error rate.")

front, back, out = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(front, encoding='utf-8').read().rstrip() + '\n\n' + open(back, encoding='utf-8').read()
text = text.replace('between 26 and 27 November 2026', 'on 27 November 2026 (KST)')
miss = set(re.findall(r'\{\{([A-Z0-9_]+)\}\}', text)) - set(P)
if miss:
    raise SystemExit(f'missing placeholders: {sorted(miss)}')
for k, v in P.items():
    text = text.replace('{{' + k + '}}', v)
# renumber citations by first appearance
body, refs = text.split('## References', 1)
ref_lines = [l for l in refs.strip().split('\n') if l.strip()]
order = []
for m in re.finditer(r'\[(\d+)\]', body):
    n = int(m.group(1))
    if n not in order:
        order.append(n)
unused = [i + 1 for i in range(len(ref_lines)) if i + 1 not in order]
if unused:
    raise SystemExit(f'uncited references: {unused}')
mp = {old: new + 1 for new, old in enumerate(order)}
body = re.sub(r'\[(\d+)\]', lambda m: f'[{mp[int(m.group(1))]}]', body)
body = re.sub(r'\[\d+\](?:, \[\d+\])+', lambda m: ', '.join(f'[{x}]' for x in sorted(int(y) for y in re.findall(r'\d+', m.group(0)))), body)
new_refs = [ref_lines[old - 1] for old in order]
text = body + '## References\n\n' + '\n'.join(new_refs) + '\n'
open(out, 'w', encoding='utf-8').write(text)
json.dump(P, open(os.path.splitext(out)[0] + '_values.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('written', out, 'words', len(re.findall(r'[A-Za-z0-9]+', body)))



