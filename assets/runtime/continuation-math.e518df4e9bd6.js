import {softmax,dot,matmul,fmt} from './math.e518df4e9bd6.js';

// Exact, deliberately small teaching models. These are not learned checkpoints.
export const bpeCorpus=[{word:'low',count:5},{word:'lower',count:2},{word:'newest',count:6},{word:'widest',count:3}];
export const splitWord=word=>[...word,'</w>'];
export function pairCounts(rows){
 const counts=new Map();
 for(const {symbols,count} of rows)for(let i=0;i<symbols.length-1;i++){
  const key=JSON.stringify(symbols.slice(i,i+2));counts.set(key,(counts.get(key)||0)+count);
 }
 return [...counts].map(([key,count])=>({pair:JSON.parse(key),count})).sort((a,b)=>b.count-a.count||JSON.stringify(a.pair).localeCompare(JSON.stringify(b.pair),'en'));
}
export function mergePair(symbols,pair){
 const out=[];for(let i=0;i<symbols.length;i++){
  if(symbols[i]===pair[0]&&symbols[i+1]===pair[1]){out.push(pair.join(''));i++;}
  else out.push(symbols[i]);
 }return out;
}
export function trainBPE(n=8){
 let rows=bpeCorpus.map(({word,count})=>({word,count,symbols:splitWord(word)}));
 const history=[];
 for(let step=0;step<n;step++){
  const ranked=pairCounts(rows);if(!ranked.length)break;
  const winner=ranked[0];history.push({pair:winner.pair,count:winner.count});
  rows=rows.map(r=>({...r,symbols:mergePair(r.symbols,winner.pair)}));
 }return{rows,history,ranked:pairCounts(rows)};
}
export function applyBPE(word,n=8){
 const {history}=trainBPE(n);let symbols=splitWord(word);
 const stages=[symbols];for(const {pair} of history){symbols=mergePair(symbols,pair);stages.push(symbols);}
 return{symbols,stages};
}
export function contextualToy(context='river',causal=false){
 const bank=[1,1],cue=context==='river'?[2,0]:[0,2];
 const visible=causal&&context==='account'?[bank]:(context==='river'?[cue,bank]:[bank,cue]);
 const scores=visible.map(v=>dot(bank,v)/Math.sqrt(2)),weights=softmax(scores);
 const result=[0,1].map(j=>weights.reduce((s,w,i)=>s+w*visible[i][j],0));
 return{bank,cue,visible,scores,weights,result};
}
export function characterConv(reverse=false){
 const chars=['b','a','n','k'],x=[.2,.7,-.3,.9],kernel=reverse?[-1,1]:[1,-1];
 const raw=x.slice(0,-1).map((_,i)=>kernel[0]*x[i]+kernel[1]*x[i+1]);
 const relu=raw.map(v=>Math.max(0,v));return{chars,x,kernel,raw,relu,pooled:Math.max(...relu)};
}
export function elmoMix(logit=0){
 const layers=[[1,0],[.8,.4],[.2,1]],weights=softmax([0,0,Number(logit)]);
 return{layers,weights,result:[0,1].map(j=>weights.reduce((s,w,i)=>s+w*layers[i][j],0))};
}
export const lmLoss=p=>-Math.log(Number(p));
export const poolToy=()=>{
 const states=[[.2,.8],[.6,.4],[1,.1]];
 return{states,last:states.at(-1),max:[1,.8],mean:[.6,1.3/3],concat:[1,.1,1,.8,.6,1.3/3]};
};
export const adaptationResults=[
 {task:'CHEMPROT',metric:'micro-F1',domain:'Biomedical',scores:[81.9,84.2,82.6,84.4],sd:[1,.2,.4,.4]},
 {task:'RCT',metric:'micro-F1',domain:'Biomedical',scores:[87.2,87.6,87.7,87.8],sd:[.1,.1,.1,.1]},
 {task:'ACL-ARC',metric:'macro-F1',domain:'Computer science',scores:[63,75.4,67.4,75.6],sd:[5.8,2.5,1.8,3.8]},
 {task:'SCIERC',metric:'macro-F1',domain:'Computer science',scores:[77.3,80.8,79.3,81.3],sd:[1.9,1.5,1.5,1.8]},
 {task:'HYPERPARTISAN',metric:'macro-F1',domain:'News',scores:[86.6,88.2,90.4,90],sd:[.9,5.9,5.2,6.6]},
 {task:'AGNEWS',metric:'macro-F1',domain:'News',scores:[93.9,93.9,94.5,94.6],sd:[.2,.2,.1,.1]},
 {task:'HELPFULNESS',metric:'macro-F1',domain:'Reviews',scores:[65.1,66.5,68.5,68.7],sd:[3.4,1.4,1.9,1.8]},
 {task:'IMDB',metric:'macro-F1',domain:'Reviews',scores:[95,95.4,95.5,95.6],sd:[.2,.1,.1,.1]}
]; // Gururangan et al. (2020), Table 5, five-seed test means and standard deviations.
export const maftResults=[
 {name:'XLM-R-base',score:79.2,params:270,copies:1},
 {name:'+ MAFT',score:81.8,params:270,copies:1},
 {name:'+ MAFT, 70k vocab',score:80.7,params:140,copies:1},
 {name:'+ LAFT per language',score:82.3,params:270,copies:13}
]; // Alabi et al. (2022), Table 3, mean NER test F1 over 13 languages, five runs.
export const socialResults=[
 {task:'AfriSenti',base:51.39,adapted:58.21},
 {task:'AfriEmo',base:44.25,adapted:51.45},
 {task:'AfriHate',base:65.17,adapted:69.83}
]; // Belay et al. (2025), final paper Table 3, reported average macro-F1.
export const lughaResults=[
 {task:'AfriMMLU',scores:[30,33.2,34.3,33]},
 {task:'AfriMGSM',scores:[4.5,5.1,5,5.9]},
 {task:'AfriXNLI',scores:[34.6,38,37.9,38.1]}
]; // Buzaaba et al. (2025), Table 1, Avg excludes English and French.
export const lughaNames=['Llama-3.1-8B','Lugha: WURA','Lugha: + education','Lugha: + maths'];
export function tokenBudget(percent=40,total=10){const education=total*Number(percent)/100;return{education,wura:total-education,total};}
export function loraCounts(rank=8,width=4096){const full=width*width,trainable=2*width*Number(rank);return{full,trainable,percent:100*trainable/full};}
export function loraToy(rank=1){
 const W=[[1,0],[0,1]],x=[[2],[1]];
 const A=Number(rank)===1?[[1,-1]]:[[1,0],[0,1]];
 const B=Number(rank)===1?[[.2],[.5]]:[[.2,0],[0,.5]];
 const Ax=matmul(A,x),update=matmul(B,Ax),base=matmul(W,x),out=base.map((r,i)=>[r[0]+update[i][0]]);
 return{W,x,A,B,Ax,update,base,out,delta:matmul(B,A)};
}
export function adapterToy(){
 const x=[1,.5],down=[.2,.4],u=dot(down,x),activation=Math.max(0,u),up=[.5,-.25],delta=up.map(w=>w*activation);
 return{x,down,u,activation,up,delta,out:x.map((v,i)=>v+delta[i])};
}
export function learningRates(top=.0001,factor=2.6){return[top/(factor*factor),top/factor,top];}
export const mixedLoss=lambda=>Number(lambda)*.2+(1-Number(lambda))*.8;
