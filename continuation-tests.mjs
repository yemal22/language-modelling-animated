import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {matmul} from './math.js';
import{trainBPE,applyBPE,mergePair,contextualToy,characterConv,elmoMix,lmLoss,poolToy,loraCounts,loraToy,adapterToy,learningRates,mixedLoss,tokenBudget,adaptationResults,socialResults,lughaResults}from'./continuation-math.js';
import{MORE_COURSE}from'./continuation-content.js';
import{renderScene}from'./scenes.js';
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,a+' != '+b);
assert.deepEqual(trainBPE(1).history[0],{pair:['e','s'],count:9});
assert.deepEqual(mergePair(['a','a','a'],['a','a']),['aa','a']);
assert.deepEqual(applyBPE('lowest',8).symbols,['low','est</w>']);
for(let n=0;n<=8;n++)for(const word of ['low','lower','newest','widest','lowest'])assert.equal(applyBPE(word,n).symbols.join('').replace('</w>',''),word);
assert.deepEqual(contextualToy('account',true).result,[1,1]);assert.deepEqual(contextualToy('account',false).result,[.5,1.5]);assert.deepEqual(contextualToy('river',true).result,[1.5,.5]);
near(characterConv().pooled,1);near(characterConv(true).pooled,1.2);
for(const logit of [-2,0,2]){const q=elmoMix(logit);near(q.weights.reduce((a,b)=>a+b,0),1);for(const v of q.result)assert.ok(v>=0&&v<=1);}
assert.ok(elmoMix(2).weights[2]>elmoMix(0).weights[2]);
near(lmLoss(1),0);assert.ok(lmLoss(.95)<lmLoss(.05));near(poolToy().mean[1],1.3/3);
const a=adapterToy();near(a.out[0],1.2);near(a.out[1],.4);
for(const rank of [1,2]){const q=loraToy(rank);assert.deepEqual(matmul(q.delta,q.x),q.update);q.out.forEach((r,i)=>near(r[0],q.base[i][0]+q.update[i][0]));}
near(loraToy(1).out[0][0],2.2);near(loraToy(1).out[1][0],1.5);
assert.deepEqual(loraCounts(8),{full:16777216,trainable:65536,percent:.390625});
near(learningRates()[2]/learningRates()[1],2.6);near(mixedLoss(0),.8);near(mixedLoss(1),.2);near(tokenBudget(40).education,4);
assert.equal(adaptationResults.length,8);
assert.ok(adaptationResults.find(r=>r.task==='HYPERPARTISAN').scores[2]>adaptationResults.find(r=>r.task==='HYPERPARTISAN').scores[3]);
near(socialResults[0].adapted-socialResults[0].base,6.82);assert.equal(lughaResults[0].scores[2],34.3);
let scenes=0;
for(const d of MORE_COURSE)for(const s of d.slides){
 if(!s.scene)continue;scenes++;
 const r=renderScene(s.scene);
 for(const c of r.controls||[]){
  const values=c.type==='range'?[c.min,c.max]:c.options.map(o=>Array.isArray(o)?o[0]:o);
  for(const value of values){
   const v=renderScene(s.scene,{[c.key]:String(value)});
   assert.ok(!/NaN|undefined/.test(v.svg),s.scene);
   assert.ok(!v.svg.includes(String.fromCharCode(92)+'n'),s.scene+' contains a literal line-break escape');
   assert.equal(v.caps.length,r.caps.length,s.scene+' must preserve build count after control changes');
  }
 }
}
for(const d of MORE_COURSE){
 const file='part-'+d.id+'/index.html',html=fs.readFileSync(file,'utf8');
 for(const match of html.matchAll(/(?:href|src)="([^"]+)"/g)){
  const target=match[1];if(target.startsWith('http')||target==='../')continue;
  const local=path.resolve(path.dirname(file),target);assert.ok(fs.existsSync(local),local);
 }
 const config=html.match(/<script>(window.MathJax=.*?)<\/script>/)[1],sandbox={window:{}};vm.runInNewContext(config,sandbox);
 assert.equal(sandbox.window.MathJax.tex.inlineMath[0][0],'\\(');
}
const html=fs.readFileSync('part-8/index.html','utf8'),runtime=html.match(/src="([^"]*engine[^"]*)"/)[1];
const runtimePath=path.resolve('part-8',runtime),folder=path.dirname(runtimePath);
for(const f of fs.readdirSync(folder).filter(f=>f.endsWith('.js')&&f.includes(path.basename(runtimePath).split('.')[1]))){
 const code=fs.readFileSync(path.join(folder,f),'utf8');
 for(const m of code.matchAll(/from '\.\/([^']+)'/g))assert.ok(fs.existsSync(path.join(folder,m[1])),f+' missing '+m[1]);
}
console.log('Passed BPE, context visibility, ELMo, losses, adapter/LoRA, source-value and asset-graph checks for '+scenes+' new scenes.');
