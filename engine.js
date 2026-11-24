import {COURSE} from './content.js';
import {renderScene} from './scenes.js';
const part=Number(document.body.dataset.part),deck=COURSE.find(d=>d.id===part);
if(!deck)throw Error('Unknown course part');
const canvas=document.getElementById('canvas'),root=document.getElementById('root'),sections=document.getElementById('sections'),hud=document.getElementById('hud'),notes=document.getElementById('notes'),overview=document.getElementById('overview');
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
let cur=0,beat=0,blank=false,max=0,notesOn=false,active=null,sceneData=null,pendingMath=Promise.resolve();const state=deck.slides.map(()=>({}));const seq=[...new Set(deck.slides.map(s=>s.sec))];
document.title=`Part ${part} · ${deck.title}`;
function resize(){const scale=Math.min(innerWidth/1600,innerHeight/900);canvas.style.transform=`scale(${scale})`;}
addEventListener('resize',resize);resize();
function content(s){if(s.kind==='cover')return `<div class="center"><div class="kicker">Language modelling · Part ${part} of 4</div><h1>${s.title}</h1><div class="subtitle">${s.subtitle}</div><div class="credit">A study companion to Shamsuddeen Muhammad’s lecture<br>Prepared for Morel KPAVODE · 24 November 2026</div></div>`;
 if(s.kind==='scene'){sceneData=renderScene(s.scene,state[cur]);return `<h1>${s.title}</h1><div class="scene">${sceneData.svg}</div><div class="controls" aria-label="Teaching example controls"></div><div class="caption" aria-live="polite"></div>`;}
 return `<h1>${s.title}</h1><div class="body">${s.html.replaceAll('</s>','&lt;/s&gt;')}</div>`;}
function makeControls(){const host=active.querySelector('.controls');if(!host)return;const controls=sceneData.controls||[];host.hidden=!controls.length;
 controls.forEach(c=>{const label=document.createElement('label');label.htmlFor='ctrl-'+c.key;label.textContent=c.label;host.append(label);let el=document.createElement(c.type==='select'?'select':'input');el.id='ctrl-'+c.key;el.dataset.control='true';
 if(c.type==='select'){for(const option of c.options){const [value,name]=Array.isArray(option)?option:[option,option];const o=document.createElement('option');o.value=value;o.textContent=name;el.append(o);}el.value=state[cur][c.key]??(Array.isArray(c.options[0])?c.options[0][0]:c.options[0]);}
 else{el.type='range';el.min=c.min;el.max=c.max;el.step=c.step;el.value=state[cur][c.key]??c.value;}
 const out=document.createElement('output');if(c.type!=='select'){out.textContent=Number(el.value).toFixed(2);el.setAttribute('aria-valuetext',out.textContent);}host.append(el);if(c.type!=='select')host.append(out);
 el.addEventListener('input',()=>{state[cur][c.key]=el.value;const data=renderScene(deck.slides[cur].scene,state[cur]);sceneData=data;active.querySelector('.scene').innerHTML=data.svg;out.textContent=Number(el.value).toFixed(2);el.setAttribute('aria-valuetext',el.value);apply(false);});
 });
}
function typeset(){if(window.MathJax?.typesetPromise)pendingMath=window.MathJax.typesetPromise([active]).catch(()=>{});}
addEventListener('load',typeset);
function render(){const s=deck.slides[cur];sceneData=null;root.innerHTML=`<section class="slide ${s.kind==='cover'?'cover':''}" aria-label="Slide ${cur+1}">${content(s)}<div class="source">${s.source||'Lecture adaptation · Original worked diagrams and computations'}</div></section>`;active=root.firstElementChild;
 max=s.kind==='scene'?sceneData.caps.length-1:Math.max(0,...[...active.querySelectorAll('[data-at]')].map(x=>Number(x.dataset.at)));
 beat=Math.max(0,Math.min(beat===Infinity?max:beat,max));makeControls();apply(false);typeset();
}
function apply(animate=true){const s=deck.slides[cur];canvas.classList.toggle('preblank',blank);active.querySelectorAll('[data-at]').forEach(el=>{const at=Number(el.dataset.at),until=Number(el.dataset.until??99);el.classList.toggle('on',at<=beat&&beat<=until);el.classList.toggle('hot',at===beat&&el.classList.contains('atom'));});
 const caption=active.querySelector('.caption');if(caption)caption.textContent=sceneData.caps[beat]||'';
 hud.textContent=`${cur+1} / ${deck.slides.length} · ${beat} / ${max}`;
 sections.innerHTML=seq.map(sec=>{const positions=deck.slides.map((x,i)=>x.sec===sec?i:-1).filter(i=>i>=0),i=seq.indexOf(sec),ci=seq.indexOf(s.sec);return `<button class="${i===ci?'active':i<ci?'done':''}" data-jump="${positions[0]}" style="--progress:${100*(positions.indexOf(cur)+1)/positions.length}%">${esc(sec)}</button>`;}).join('');
 const base=s.notes||'Use the visible example to explain the mechanism. All toy values are chosen for teaching.';notes.innerHTML=`<strong>${esc(s.title)}</strong><p>${esc(base)}</p>${sceneData?`<p>${esc(sceneData.caps[beat])}</p>`:''}${s.source?`<p>${s.source}</p>`:''}`;notes.classList.toggle('open',notesOn);
 document.getElementById('prev').disabled=cur===0&&beat===0&&!blank;document.getElementById('next').disabled=cur===deck.slides.length-1&&beat===max;
 const hash=`#${cur+1}/${beat}${blank?'/blank':''}`;history.replaceState(null,'',location.pathname+location.search+hash);
 window.deckState=Object.freeze({part,slide:cur+1,title:s.title,beat,max,blank,scene:s.scene||null});
}
function jump(i,b=0,isBlank=false){cur=Math.max(0,Math.min(deck.slides.length-1,i));beat=b;blank=isBlank;render();}
function next(){canvas.classList.remove('backward');if(blank){blank=false;return apply();}if(beat<max){beat++;return apply();}if(cur<deck.slides.length-1)jump(cur+1,0,new URLSearchParams(location.search).get('blank')!=='0');}
function prev(){canvas.classList.add('backward');if(blank){if(cur>0)jump(cur-1,Infinity,false);return;}if(beat>0){beat--;return apply();}if(cur>0)jump(cur-1,Infinity,false);}
function openOverview(){overview.classList.toggle('open');if(overview.classList.contains('open'))overview.querySelector('button').focus();}
overview.innerHTML=`<button class="close" aria-label="Close slide overview">Close</button><h2>Part ${part} · ${deck.title}</h2><div class="grid">${deck.slides.map((s,i)=>`<button data-jump="${i}">${i+1}. ${esc(s.title)}</button>`).join('')}</div>`;
overview.querySelector('.close').addEventListener('click',openOverview);
document.addEventListener('click',e=>{const j=e.target.closest('[data-jump]');if(j){overview.classList.remove('open');jump(Number(j.dataset.jump));return;}if(e.target.closest('button,a,input,select,option,summary,#notes,#overview'))return;if(!notesOn)next();});
document.getElementById('prev').addEventListener('click',prev);document.getElementById('next').addEventListener('click',next);document.getElementById('overview-button').addEventListener('click',openOverview);
document.addEventListener('keydown',e=>{if(e.ctrlKey||e.metaKey||e.altKey||e.target.closest('input,select,textarea'))return;const k=e.key.toLowerCase();if(overview.classList.contains('open')){if(k==='escape'||k==='o')openOverview();return;}
 if(!['arrowright','arrowleft',' ','enter','pagedown','pageup','home','end','n','o','f','escape'].includes(k))return;e.preventDefault();
 if(k==='arrowright'||k===' '||k==='enter')next();else if(k==='arrowleft')prev();else if(k==='pagedown')jump(cur+1);else if(k==='pageup')jump(cur-1);else if(k==='home')jump(0);else if(k==='end')jump(deck.slides.length-1);else if(k==='n'){notesOn=!notesOn;apply();}else if(k==='o')openOverview();else if(k==='escape'){notesOn=false;apply();}else if(k==='f'){if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen?.();}
});
function readHash(){const m=location.hash.match(/^#(\d+)(?:\/(\d+))?(\/blank)?$/);if(m)jump(Number(m[1])-1,m[2]===undefined?Infinity:Number(m[2]),!!m[3]);else jump(0);}
addEventListener('hashchange',readHash);readHash();
