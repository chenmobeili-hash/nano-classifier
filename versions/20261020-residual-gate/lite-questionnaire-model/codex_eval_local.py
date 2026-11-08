"""Run frozen local-attention experiment; no fitted preprocessing or epoch selection."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,time,hashlib
from pathlib import Path
import numpy as np
import codex_local_attn as A
from classifier import load_references,DEFAULT_REFERENCES,read_answer,inventory
TARGET=.968421052631579
VARIANTS={'local':True,'uniform':False}
STAGE=Path(r'C:\Users\betraysilence\Music\nano-classfier\不确定新增不新增的模型')
CASES=[(f'{prefix}.txt','GLM') for prefix in ('ofm-glm53flash-06','ofm-glm53flash-07','ofm-glm53flash-08','ofm-glm53flash-09','cmdc-glm53flash-01','cmdc-glm53flash-03','zcode-glm53flash-01')]+[('probe-01.txt','Claude'),('codex-sonnet55-01.txt','Claude'),('ofm-deepseek41flash-01.txt','DeepSeek'),('qwen-free-01.txt','Qwen')]

def summary(rows,v,threshold=None):
    margins=np.array([r[v]['margin'] for r in rows]); threshold=float(np.quantile(margins,1-TARGET)) if threshold is None else threshold
    accepted=[r for r in rows if r[v]['margin']>=threshold]
    hits=sum(r[v]['top']==r['expected'] for r in rows)
    ah=sum(r[v]['top']==r['expected'] for r in accepted)
    return dict(variant=v,tested=len(rows),forced_hits=hits,forced_accuracy=hits/len(rows),threshold=threshold,accepted=len(accepted),accepted_hits=ah,coverage=len(accepted)/len(rows),accepted_accuracy=ah/len(accepted) if accepted else None)

def prediction(data,p,flag,classes):
    prob=A.forward(data,p,flag)[0][0];order=np.argsort(-prob)
    return dict(top=classes[int(order[0])],margin=float(prob[order[0]]-prob[order[1]]),probabilities=prob.tolist())

def main():
    records,skipped=load_references(DEFAULT_REFERENCES);assert len(records)==95
    assert not {r['source'] for r in records}&{n for n,_ in CASES}
    classes=sorted({r['family'] for r in records});data=A.encode(records)
    result=dict(engine='codex-local-A',seed=42,target_coverage=TARGET,inventory=inventory(records,skipped),classes=classes,
        config=dict(d=A.D,centers=A.CENTERS,window=1,epochs=A.EPOCHS,lr=A.LR,l2=A.L2,parameters_local=493,parameters_uniform=485,selection='fixed before held-out results'),gradient_check=A.gradcheck(),runs={})
    for mode in ('sample','model'):
        cache={};rows=[];times=[];started=time.perf_counter()
        for i,r in enumerate(records):
            key=i if mode=='sample' else r['model']
            if key not in cache:
                train=[j for j,s in enumerate(records) if (j!=i if mode=='sample' else s['model']!=r['model'])]
                td=tuple(x[train] for x in data);subset=[records[j] for j in train]; cache[key]={}
                for v,flag in VARIANTS.items():
                    tick=time.perf_counter();cache[key][v]=A.train(td,subset,classes,flag);times.append(dict(key=str(key),variant=v,seconds=time.perf_counter()-tick))
            row=dict(source=r['source'],model=r['model'],expected=r['family'])
            for v,flag in VARIANTS.items():row[v]=prediction(tuple(x[i:i+1] for x in data),cache[key][v],flag,classes)
            rows.append(row)
            if (i+1)%10==0:print(mode,i+1,flush=True)
        result['runs'][mode]=dict(rows=rows,summary=[summary(rows,v) for v in VARIANTS],seconds=time.perf_counter()-started,training_times=times)
        print(mode,result['runs'][mode]['summary'],flush=True)
    full={v:A.train(data,records,classes,flag) for v,flag in VARIANTS.items()}
    external=[]
    for name,truth in CASES:
        path=STAGE/name
        if not path.exists():raise FileNotFoundError(path)
        d=A.encode([read_answer(path)]);row=dict(source=name,expected=truth)
        for v,flag in VARIANTS.items():row[v]=prediction(d,full[v],flag,classes)
        external.append(row)
    result['external']=external
    result['external_summary']=[summary(external,v,result['runs']['sample']['summary'][i]['threshold']) for i,v in enumerate(VARIANTS)]
    result['training_diagnostics']={}
    for v,flag in VARIANTS.items():
        prob,(a,z,denom)=A.forward(data,full[v],flag)
        result['training_diagnostics'][v]=dict(accuracy=float(np.mean(prob.argmax(axis=1)==[classes.index(r['family']) for r in records])),attention_parameters=full[v]['a'].tolist(),mean_entropy=float(np.mean(-np.sum(a*np.log(np.maximum(a,1e-30)),axis=-1))))
    result['training_prediction_agreement']=float(np.mean(A.forward(data,full['local'],True)[0].argmax(1)==A.forward(data,full['uniform'],False)[0].argmax(1)))
    result['baseline_snapshots']={}
    for filename in ('evaluation-fusion.json','evaluation-fusion-external.json','evaluation-attn.json'):
        raw=Path(filename).read_bytes();obj=json.loads(raw.decode('utf-8-sig'))
        result['baseline_snapshots'][filename]=dict(sha256=hashlib.sha256(raw).hexdigest(),data=obj)
    Path('evaluation-attn-local-codex.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('external',result['external_summary'],flush=True)
if __name__=='__main__':main()
