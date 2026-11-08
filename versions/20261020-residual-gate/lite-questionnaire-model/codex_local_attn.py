"""Frozen experiment: within-question local attention, numpy/manual gradients only."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import hashlib
import numpy as np
from feature_profiles import normalized_answers
from nlp_core import sample_weights
D, CENTERS, EPOCHS, LR, L2 = 8, 16, 120, .02, .01

def vector(token):
    raw = hashlib.blake2b(token.encode('utf-8'), digest_size=D).digest()
    return (np.frombuffer(raw, dtype=np.uint8).astype(float) / 127.5 - 1) * np.sqrt(3)

def encode(records):
    values = np.zeros((len(records), 12, CENTERS, 3, D))
    mask = np.zeros(values.shape[:-1], dtype=bool)
    for n, record in enumerate(records):
        answers = normalized_answers(record)
        for q in range(12):
            text = answers.get('B%02d' % (q+1), '')
            tokens = [text[i:i+2] for i in range(max(0,len(text)-1))] or ([text] if text else [])
            if not tokens: continue
            centers = np.unique(np.linspace(0,len(tokens)-1,min(CENTERS,len(tokens))).astype(int))
            for t,i in enumerate(centers):
                for k,offset in enumerate((-1,0,1)):
                    j=i+offset
                    if 0<=j<len(tokens):
                        values[n,q,t,k]=vector(tokens[j]); mask[n,q,t,k]=True
    pair = values * values[:,:,:,1:2,:] / np.sqrt(D)
    return values, pair, mask

def init(seed=42):
    rng=np.random.default_rng(seed)
    return {'a':rng.normal(0,.05,D), 'W':rng.normal(0,.02,(12*D,5)), 'b':np.zeros(5)}

def forward(data,p,attention=True):
    v,f,mask=data
    scores=np.einsum('nqtkd,d->nqtk',f,p['a']) if attention else np.zeros(mask.shape)
    scores=np.where(mask,scores,-1e9)
    scores-=scores.max(axis=-1,keepdims=True)
    a=np.exp(scores)*mask
    a/=np.maximum(a.sum(axis=-1,keepdims=True),1e-30)
    denom=np.maximum(mask.any(axis=-1).sum(axis=-1),1)
    z=np.einsum('nqtk,nqtkd->nqd',a,v)/denom[:,:,None]
    z=z.reshape(len(v),-1)
    logits=z@p['W']+p['b']; logits-=logits.max(axis=1,keepdims=True)
    prob=np.exp(logits);prob/=prob.sum(axis=1,keepdims=True)
    return prob,(a,z,denom)

def loss_grad(data,y,weights,p,attention=True,l2=L2):
    prob,(a,z,denom)=forward(data,p,attention)
    loss=-np.sum(weights*np.log(np.maximum(prob[np.arange(len(y)),y],1e-300)))
    loss+=.5*l2*sum(np.sum(p[k]**2) for k in ('W','a') if attention or k!='a')
    g=prob.copy();g[np.arange(len(y)),y]-=1;g*=weights[:,None]
    grads={'W':z.T@g+l2*p['W'],'b':g.sum(axis=0),'a':np.zeros(D)}
    if attention:
        dz=(g@p['W'].T).reshape(-1,12,D)/denom[:,:,None]
        da=np.einsum('nqd,nqtkd->nqtk',dz,data[0])
        ds=a*(da-(a*da).sum(axis=-1,keepdims=True))
        grads['a']=np.einsum('nqtk,nqtkd->d',ds,data[1])+l2*p['a']
    return float(loss),grads

def train(data,records,classes,attention=True):
    p=init();y=np.array([classes.index(r['family']) for r in records]); weights=sample_weights(records,classes)
    m={k:np.zeros_like(v) for k,v in p.items()};s={k:v.copy() for k,v in m.items()}
    for t in range(1,EPOCHS+1):
        _,grads=loss_grad(data,y,weights,p,attention)
        for k in p:
            if k=='a' and not attention:continue
            m[k]=.9*m[k]+.1*grads[k];s[k]=.999*s[k]+.001*grads[k]**2
            p[k]-=LR*(m[k]/(1-.9**t))/(np.sqrt(s[k]/(1-.999**t))+1e-8)
    return p

def gradcheck():
    rng=np.random.default_rng(123)
    v=rng.normal(size=(3,12,2,3,D));mask=rng.random(v.shape[:-1])>.25
    mask[:,:,:,1]=True; mask[0,0]=False
    data=(v,v*v[:,:,:,1:2]/np.sqrt(D),mask)
    y=np.array([0,2,4]);weights=np.array([.2,.3,.5]); out={}
    for attention in (True,False):
        p=init(17);_,g=loss_grad(data,y,weights,p,attention);errors=[];absolute=[]
        for k in p:
            for idx in np.ndindex(p[k].shape):
                old=p[k][idx];p[k][idx]=old+1e-5;plus=loss_grad(data,y,weights,p,attention)[0]
                p[k][idx]=old-1e-5;minus=loss_grad(data,y,weights,p,attention)[0];p[k][idx]=old
                numeric=(plus-minus)/2e-5;analytic=g[k][idx]
                errors.append(abs(numeric-analytic)/max(1e-8,abs(numeric)+abs(analytic)))
                absolute.append(abs(numeric-analytic))
        out[str(attention)]={'max_relative_error':max(errors),'max_absolute_error':max(absolute),'coordinates':len(errors)}
    return out

if __name__=='__main__':
    import json
    print(json.dumps(gradcheck(),indent=2))
