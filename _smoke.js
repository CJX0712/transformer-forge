const fs=require('fs'),vm=require('vm'),path=require('path');
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const m=html.match(/<script id="engine">([\s\S]*?)<\/script>/);
const ctx={console,Math,Object,Array,Float64Array,Infinity,NaN,isFinite,globalThis:{}};
ctx.globalThis=ctx; vm.createContext(ctx); vm.runInContext(m[1],ctx,{filename:'engine.js'});
const TRF=ctx.TRF;

let pass=0, fail=0; const fails=[];
function ok(n,c,info){ if(c)pass++; else { fail++; fails.push(n+(info?' ('+info+')':'')); } console.log((c?'PASS':'FAIL')+' '+n+(info?'  '+info:'')); }

// 1. softmax 行和=1 + 因果掩码
{
  const model=TRF.createModel({vocab:4,dModel:12,nHeads:3,dFF:24,nLayers:1,maxLen:16,seed:11});
  const f=TRF.forward(model,[0,1,2,3,1,0]); const T=6; let smOK=true, maskOK=true, finite=true;
  for(let i=0;i<T;i++){ let s=0; for(let j=0;j<T;j++){ const a=f.lc[0].att.Aall[i*T+j]; if(!isFinite(a))finite=false; s+=a; if(j>i && a>1e-9)maskOK=false; } if(Math.abs(s-1)>1e-9)smOK=false; }
  ok("softmax-rows-sum-1", smOK);
  ok("causal-mask-upper-tri-zero", maskOK);
  ok("attention-finite", finite);
}

// 2. 梯度检验（LN + MHA + FFN + 交叉熵，数值 vs 解析）
{
  const gcm=TRF.createModel({vocab:3,dModel:6,nHeads:2,dFF:8,nLayers:1,maxLen:8,seed:5});
  const toks=[0,1,2,0], tgt=[0,0,1,2];
  const lossOf=model=>{ const f=TRF.forward(model,toks); return TRF.lossFromLogits(f.logits,4,3,tgt); };
  const G=TRF.zeroGrads(gcm); const f0=TRF.forward(gcm,toks); TRF.backward(gcm,f0,tgt,G,toks);
  const params=[], names=[], gps=[];
  const collect=(arr,nm,gp)=>{ params.push(arr); names.push(nm); gps.push(gp); };
  collect(gcm.tokenEmb,"tok",G.tokenEmb); collect(gcm.posEmb,"pos",G.posEmb); collect(gcm.Wlm,"Wlm",G.Wlm);
  const L=gcm.layers[0], GL=G.layers[0];
  collect(L.Wq,"Wq",GL.Wq); collect(L.Wk,"Wk",GL.Wk); collect(L.Wv,"Wv",GL.Wv); collect(L.Wo,"Wo",GL.Wo);
  collect(L.g1,"g1",GL.g1); collect(L.b1,"b1",GL.b1); collect(L.g2,"g2",GL.g2); collect(L.b2,"b2",GL.b2);
  collect(L.W1,"W1",GL.W1); collect(L.c1,"c1",GL.c1); collect(L.W2,"W2",GL.W2); collect(L.c2,"c2",GL.c2);
  const eps=1e-5; let maxRel=0, worst="";
  for(let p=0;p<params.length;p++){ const P=params[p], dP=gps[p];
    for(let idx=0; idx<P.length; idx+=3){
      const orig=P[idx]; P[idx]=orig+eps; const lp=lossOf(gcm); P[idx]=orig-eps; const lm=lossOf(gcm); P[idx]=orig;
      const num=(lp-lm)/(2*eps), ana=dP[idx];
      const rel=Math.abs(num-ana)/(Math.abs(num)+Math.abs(ana)+1e-9);
      if(rel>maxRel){ maxRel=rel; worst=names[p]+"["+idx+"]"; }
    }
  }
  ok("grad-check-backprop", maxRel<1e-5, "maxRel="+maxRel.toExponential(2)+" @ "+worst);
}

// 3. 前向确定性 + 反向有限（引擎可复现、无 NaN）
{
  const m2=TRF.createModel({vocab:4,dModel:8,nHeads:2,dFF:16,nLayers:1,maxLen:16,seed:3});
  const a=TRF.forward(m2,[0,1,2,3]), b=TRF.forward(m2,[0,1,2,3]);
  let detOK=true; for(let i=0;i<a.logits.length;i++) if(a.logits[i]!==b.logits[i]) detOK=false;
  const G=TRF.zeroGrads(m2); TRF.backward(m2,a,[0,0,1,2],G,[0,1,2,3]);
  let fin=true; for(const k of ['tokenEmb','posEmb','Wlm']) for(const v of G[k]) if(!isFinite(v)) fin=false;
  for(const l of G.layers) for(const k in l) for(const v of l[k]) if(!isFinite(v)) fin=false;
  ok("forward-deterministic", detOK);
  ok("backward-finite", fin);
}

// 4. 端到端：训练后 shift 精度
{
  const tm=TRF.train({vocab:4,T:8,dModel:20,nHeads:4,dFF:40,nLayers:1,steps:1500,batch:32,lr:0.01,seed:42,task:'shift'});
  const L=tm.lossCurve;
  ok("train-loss-decreases", L[L.length-1] < L[0]*0.8, "loss "+L[0].toFixed(3)+"->"+L[L.length-1].toFixed(3));
  ok("shift-accuracy>=0.95", tm.accuracy>=0.95, "acc="+(tm.accuracy*100).toFixed(1)+"%");
}

// 5. copy 基线可学
{
  const tm=TRF.train({vocab:4,T:8,dModel:16,nHeads:2,dFF:32,nLayers:1,steps:800,batch:32,lr:0.01,seed:7,task:'copy'});
  ok("copy-accuracy>=0.98", tm.accuracy>=0.98, "acc="+(tm.accuracy*100).toFixed(1)+"%");
}

fs.writeFileSync(path.join(__dirname,'_smoke.log'), `PASS ${pass} / ${pass+fail}\n`+(fail?'FAIL '+fails.join(', '):'ALL GREEN')+'\n');
console.log(`\n=== ${pass} / ${pass+fail} `+(fail?'FAIL':'ALL GREEN'));
process.exit(fail?1:0);
