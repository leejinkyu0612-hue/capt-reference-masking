import os, json
from common import RUNS, read_jsonl
keep=set(json.load(open(os.path.join(RUNS,'subsets.json')))['subset1000'])
for src,dst in [('gemini-2.5-flash_R1V0','gemini-2.5-flash_R1V0_n1000'),('gpt-audio-mini_R1V0','gpt-audio-mini_R1V0_n1000'),('gpt-4o-mini-transcribe_R1','gpt-4o-mini-transcribe_R1_n1000')]:
    p=os.path.join(RUNS,dst+'.jsonl')
    if os.path.exists(p): print('exists',dst); continue
    seen=set(); out=[]
    for x in read_jsonl(os.path.join(RUNS,src+'.jsonl')):
        if x['uid'] in keep and x.get('hyp') is not None and x['uid'] not in seen: seen.add(x['uid']); out.append(x)
    open(p,'w',encoding='utf-8').write(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out)); print(dst,len(out))
