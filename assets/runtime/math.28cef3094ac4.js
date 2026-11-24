export const softmax=a=>{const m=Math.max(...a),z=a.map(x=>Math.exp(x-m)),s=z.reduce((a,b)=>a+b,0);return z.map(x=>x/s);};
export const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
export const matmul=(a,b)=>a.map(row=>b[0].map((_,j)=>dot(row,b.map(r=>r[j]))));
export const transpose=a=>a[0].map((_,i)=>a.map(r=>r[i]));
export const sigmoid=x=>1/(1+Math.exp(-x));
export const corpus=[['I','like','NLP'],['I','like','deep','learning'],['I','study','NLP']];
export const vocabulary=['I','like','NLP','deep','learning','study','</s>'];
export function bigram(context,alpha=0){const counts=Object.fromEntries(vocabulary.map(w=>[w,0]));for(const s of corpus){const words=['<s>',...s,'</s>'];for(let i=1;i<words.length;i++)if(words[i-1]===context)counts[words[i]]++;}const total=Object.values(counts).reduce((a,b)=>a+b,0);return vocabulary.map(word=>({word,count:counts[word],p:(counts[word]+alpha)/(total+alpha*vocabulary.length)}));}
export function rnn(xs=[1,.5,-1]){let h=0;return xs.map(x=>{const prev=h,pre=.6*x+.8*h;h=Math.tanh(pre);return {x,prev,pre,h};});}
export function lstm(f=.9){const prev=.8,i=.2,g=-.5,o=.7,keep=f*prev,write=i*g,c=keep+write,h=o*Math.tanh(c);return {prev,f,i,g,o,keep,write,c,h};}
// This deck defines z as the NEW-state fraction; some papers reverse its meaning.
export function gru(z=.25,r=.3){const prev=.8,x=.4,reset=r*prev,candidate=Math.tanh(.6*x+.8*reset),keep=(1-z)*prev,write=z*candidate,h=keep+write;return {prev,x,z,r,reset,candidate,keep,write,h};}
export const attentionStates=[[1,0],[0,1],[1,1]];
export function crossAttention(q=[1,2]){const scores=attentionStates.map(h=>dot(q,h)),weights=softmax(scores),context=[0,1].map(j=>weights.reduce((s,w,i)=>s+w*attentionStates[i][j],0));return {q,scores,weights,context};}
export const attentionX=[[1,0],[0,1],[1,1]],WQ=[[1,0],[0,1]],WK=[[1,1],[0,1]],WV=[[1,0],[0,2]];
export function selfAttention(mask=false){const Q=matmul(attentionX,WQ),K=matmul(attentionX,WK),V=matmul(attentionX,WV),scores=matmul(Q,transpose(K)).map(row=>row.map(x=>x/Math.sqrt(2))),weights=scores.map((row,i)=>softmax(row.map((x,j)=>mask&&j>i?-Infinity:x))),out=matmul(weights,V);return {Q,K,V,scores,weights,out};}
export const fmt=(x,d=3)=>Number(x).toFixed(d);
