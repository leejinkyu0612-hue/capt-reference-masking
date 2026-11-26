import random, os, json
from common import load_rows, RUNS, read_jsonl
ids=sorted(r['uid'] for r in load_rows(with_audio=False))
perm=random.Random(2027).sample(ids,len(ids)); keep=set(perm[:1000])
json.dump({'seed':2027,'subset1000':perm[:1000],'subset500':perm[:500]},open(os.path.join(RUNS,'subsets.json'),'w'))
for src,dst in [('gemini-2.5-flash_R0V0','gemini-2.5-flash_R0V0_n1000'),('gpt-audio-mini_R0V0','gpt-audio-mini_R0V0_n1000'),('gpt-4o-mini-transcribe_R0','gpt-4o-mini-transcribe_R0_n1000')]:
    recs=[x for x in read_jsonl(os.path.join(RUNS,src+'.jsonl')) if x['uid'] in keep and x.get('hyp') is not None]
    seen=set(); out=[]
    for x in recs:
        if x['uid'] not in seen: seen.add(x['uid']); out.append(x)
    with open(os.path.join(RUNS,dst+'.jsonl'),'w',encoding='utf-8') as f:
        for x in out: f.write(json.dumps(x,ensure_ascii=False)+'\n')
    print(dst,len(out))
