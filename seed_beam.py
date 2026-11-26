import os, json
from common import RUNS, read_jsonl
s500=set(json.load(open(os.path.join(RUNS,'subsets.json')))['subset500'])
src=os.path.join(RUNS,'whisper-base_R1_beam5_n1000.jsonl'); dst=os.path.join(RUNS,'whisper-base_R1_beam5_n500.jsonl')
recs=[x for x in read_jsonl(src) if x['uid'] in s500]
open(dst,'w',encoding='utf-8').write(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in recs)); print('seeded',len(recs))
import shutil; shutil.copy(os.path.join(RUNS,'whisper-base_R1_beam5_n1000.meta.json'), os.path.join(RUNS,'whisper-base_R1_beam5_n500.meta.json'))
