import assert from 'node:assert/strict';import fs from 'node:fs';
import {softmax,bigram,rnn,lstm,gru,crossAttention,selfAttention,matmul,WV,attentionX} from './math.js';
import {COURSE} from './content.js';import {renderScene} from './scenes.js';
const near=(a,b,eps=1e-10)=>assert.ok(Math.abs(a-b)<eps,`${a} != ${b}`);
near(softmax([1000,1000])[0],.5);near(softmax([-Infinity,1,2]).reduce((a,b)=>a+b,0),1);
const rows=bigram('I');near(rows.find(x=>x.word==='like').p,2/3);near(rows.find(x=>x.word==='study').p,1/3);
for(const a of [0,.1,1])near(bigram('like',a).reduce((s,r)=>s+r.p,0),1);
near(bigram('like',1).find(x=>x.word==='study').p,1/9);
near(rnn()[0].h,Math.tanh(.6));near(rnn()[2].h,Math.tanh(-.6+.8*Math.tanh(.3+.8*Math.tanh(.6))));
near(lstm().c,.62);near(lstm().h,.7*Math.tanh(.62));near(gru(0).h,.8);near(gru(1).h,gru(1).candidate);
near(gru(.25,0).candidate,Math.tanh(.24));near(crossAttention().weights.reduce((s,x)=>s+x,0),1);
const masked=selfAttention(true),full=selfAttention(false);assert.deepEqual(masked.weights[0],[1,0,0]);assert.deepEqual(masked.out[0],[1,0]);
for(const a of [masked,full])for(const row of a.weights)near(row.reduce((s,x)=>s+x,0),1);
assert.notDeepEqual(masked.out,full.out);assert.deepEqual(matmul(attentionX,WV),[[1,0],[0,2],[1,2]]);
let scenes=0;for(const d of COURSE){assert.ok(d.slides.length>=15);for(const s of d.slides){assert.ok(s.title&&s.notes);if(s.scene){const r=renderScene(s.scene);assert.ok(r.caps.length>1);assert.ok(!/NaN|undefined/.test(r.svg));const steps=[...r.svg.matchAll(/data-at="(\d+)"/g)].map(x=>Number(x[1]));assert.ok(Math.max(...steps)<r.caps.length,s.scene);scenes++;}}assert.ok(fs.existsSync(`part-${d.id}/index.html`));}
for(const s of ['lstm','gru','attention-math','smoothing']){const r=renderScene(s,s==='attention-math'?{mask:'causal',row:'2'}:s==='lstm'?{f:'.05'}:s==='gru'?{z:'1',r:'0'}:{alpha:'0'});assert.ok(!/NaN|undefined/.test(r.svg));}
console.log(`Passed numeric, masking, gate-boundary and content checks: ${COURSE.reduce((s,d)=>s+d.slides.length,0)} slides, ${scenes} animated scenes.`);
