// _probe.js — 注意力热力图探针（Node 无头可读）
// 训练一个 shift 模型并打印因果注意力权重 ASCII 热力图。
// shift 任务要求模型在位置 i 预测 token i-1，因此学成的注意力应集中在下对角带 (i == j-1)。
const fs=require('fs'),vm=require('vm'),path=require('path');
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const m=html.match(/<script id="engine">([\s\S]*?)<\/script>/);
const ctx={console,Math,Object,Array,Float64Array,Infinity,NaN,isFinite};
ctx.globalThis=ctx; vm.createContext(ctx); vm.runInContext(m[1],ctx,{filename:'engine.js'});
const TRF=ctx.TRF;

function avgDiag(A,T){ let s=0,n=0; for(let i=1;i<T;i++){ s+=A[i*T+(i-1)]; n++; } return s/n; }
function offDiag(A,T){ let s=0,n=0; for(let i=0;i<T;i++){ let r=0; for(let j=0;j<T;j++) if(Math.abs(j-(i-1))>0) r+=A[i*T+j]; n++; s+=r/T; } return s/n; }

function probe(task){
  const tm=TRF.train({vocab:4,T:8,dModel:20,nHeads:4,dFF:40,nLayers:1,steps:1000,batch:32,lr:0.01,seed:42,task});
  const ds=tm.testDS, sample=ds.X[0];
  const fwd=TRF.forward(tm.model,sample); const att=fwd.lc[0].att; const T=sample.length;
  const A=new Float64Array(T*T); for(let i=0;i<T;i++) for(let j=0;j<T;j++) A[i*T+j]=att.Aall[i*T+j];
  const G=9; const bar=v=>v>0.9?'#':String(Math.min(G,Math.round(v*G)));
  let out="\n=== task="+task+"  测试精度="+(tm.accuracy*100).toFixed(1)+"% ===\n    "+[...Array(T)].map((_,j)=>"k"+j).join(" ")+"  <- key j";
  for(let i=0;i<T;i++){ let row="q"+i+" "; for(let j=0;j<T;j++) row+=" "+bar(A[i*T+j]); out+="\n"+row; }
  out+="\n    下对角带(i=j-1)均值="+avgDiag(A,T).toFixed(3)+"   其余位置均值="+offDiag(A,T).toFixed(3);
  out+="\n    样本输入=["+sample.join(",")+"]  目标=["+ds.Y[0].join(",")+"]";
  console.log(out);
}
probe('shift');
probe('copy');
