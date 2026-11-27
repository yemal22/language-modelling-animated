import {fmt,softmax} from './math.js';
import {trainBPE,applyBPE,mergePair,contextualToy,characterConv,elmoMix,lmLoss,poolToy,adaptationResults,maftResults,socialResults,lughaResults,lughaNames,tokenBudget,loraCounts,loraToy,adapterToy,learningRates,mixedLoss} from './continuation-math.js';
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const A=(at,html,until=99)=>'<g class="atom" data-at="'+at+'" data-until="'+until+'">'+html+'</g>';
function text(x,y,s,cls='',anchor='middle'){
 const lines=String(s).split('\n'),dy=cls.includes('small')?27:34;
 return '<text x="'+x+'" y="'+(y-(lines.length-1)*dy/2)+'" text-anchor="'+anchor+'" class="svgtxt '+cls+'">'+lines.map((l,i)=>'<tspan x="'+x+'" dy="'+(i?dy:0)+'">'+esc(l)+'</tspan>').join('')+'</text>';
}
const box=(x,y,w,h,s,k=0,style='')=>A(k,'<rect class="node '+style+'" x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="7"/>'+text(x+w/2,y+h/2+8,s,(style.includes('dark')?'white ':'')+(String(s).split('\n').some(l=>l.length*14>w-24)?'small':'')));
const line=(x1,y1,x2,y2,k=0,cls='')=>A(k,'<path class="edge '+cls+'" d="M'+x1+','+y1+' L'+x2+','+y2+'"/>');
const path=(d,k=0,cls='')=>A(k,'<path class="edge '+cls+'" d="'+d+'"/>');
const op=(x,y,s,k=0)=>A(k,'<circle class="op" cx="'+x+'" cy="'+y+'" r="25"/>'+text(x,y+9,s));
const wrap=s=>'<svg viewBox="0 0 1380 520" role="img" aria-label="Animated architecture diagram"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10z" fill="#64748b"/></marker></defs>'+s+'</svg>';
function matrix(x,y,m,label,at=0,cell=62){
 let s=text(x+m[0].length*cell/2,y-22,label,'small bold');
 for(let i=0;i<m.length;i++)for(let j=0;j<m[i].length;j++)s+='<rect class="matrix-cell" x="'+(x+j*cell)+'" y="'+(y+i*cell)+'" width="'+cell+'" height="'+cell+'" fill="#edf2f7"/>'+text(x+(j+.5)*cell,y+(i+.5)*cell+7,typeof m[i][j]==='number'?fmt(m[i][j],2):m[i][j],'small');
 return A(at,s);
}
function symbols(symbols,x,y,at=0){let s='';for(const token of symbols){const w=Math.max(48,token.length*12+24);s+=box(x,y,w,50,token,at,'data');x+=w+8;}return s;}
const choice=(key,label,options)=>({type:'select',key,label,options});
const range=(key,label,min,max,step,value)=>({type:'range',key,label,min,max,step,value});
function axis(x,y,w){let s='<path d="M'+x+','+y+' H'+(x+w)+'" stroke="#94a3b8"/>';for(const v of [0,25,50,75,100])s+=text(x+w*v/100,y+30,String(v),'small');return A(0,s);}
function scoreRows(names,scores,sd=null){
 let s=axis(420,440,650);
 names.forEach((name,i)=>{const y=85+i*85;s+=A(i+1,text(25,y+24,name,'small','start')+'<rect x="420" y="'+y+'" width="'+(650*scores[i]/100)+'" height="35" fill="'+(i===0?'#94a3b8':'#4f6d9a')+'"/>'+text(1100,y+25,fmt(scores[i],1)+(sd?' ± '+fmt(sd[i],1):''),'small bold','start'));});
 return s;
}
export const MORE_SCENES={
 'contextual-toy':state=>{
  const context=state.context||'river',causal=state.visibility==='causal',q=contextualToy(context,causal);
  let s=box(20,80,300,100,'bank lookup\n[1.00, 1.00]',0,'data')+box(20,320,300,100,(context==='river'?'river before bank':'account after bank')+'\n['+q.cue.join(', ')+']',1,'data')+box(470,170,420,155,'Identity Q, K, V\nvisible weights: '+q.weights.map(x=>fmt(x,2)).join(', '),2)+path('M320,130 H390 V210 H470',2)+(causal&&context==='account'?A(2,text(390,370,'×','bold')):path('M320,370 H390 V285 H470',2))+box(1060,190,300,115,'bank state\n['+q.result.map(x=>fmt(x,2)).join(', ')+']',3,'dark')+line(890,245,1060,245,3)+A(0,text(690,35,context==='river'?'river  bank':'bank  account','bold'))+A(3,text(690,490,causal&&context==='account'?'Causal bank cannot read the future word “account”.':'Same input vector, context-dependent output. Chosen vectors, exact attention arithmetic.','small'));
  return{svg:wrap(s),controls:[choice('context','Phrase',[['river','river bank'],['account','bank account']]),choice('visibility','Visibility',[['bidirectional','Bidirectional'],['causal','Causal']])],caps:['The embedding lookup gives bank the same starting vector in both phrases.','The neighbouring token supplies another vector. Its position determines whether a causal model can see it.','Attention mixes only permitted inputs. This toy uses chosen vectors and identity projections.','The resulting token state depends on available context. It is still a vector for this token occurrence.']};
 },
 'character-cnn':state=>{
  const q=characterConv(state.filter==='reverse');
  const s=box(15,185,275,120,'b  a  n  k\n'+q.x.join('  '),0,'data')+box(365,185,270,120,'Width-2 convolution\n['+q.kernel.join(', ')+']',1)+line(290,245,365,245,1)+box(710,185,275,120,'ReLU at each window\n'+q.relu.map(x=>fmt(x,1)).join('  '),2)+line(635,245,710,245,2)+box(1060,185,300,120,'Max over positions\n'+fmt(q.pooled,2),3,'data')+line(985,245,1060,245,3)+box(1030,405,330,80,'Highway + projection\nword input vector',4)+line(1210,305,1210,405,4)+A(1,text(690,100,'Window scores: '+q.raw.map(x=>fmt(x,2)).join('   '),'bold'))+A(0,text(690,35,'One scalar character channel and one chosen filter','small'))+A(4,text(420,455,'A real character CNN combines many filters and channels.','small'));
  return{svg:wrap(s),controls:[choice('filter','Filter',[['normal','[1, −1]'],['reverse','[−1, 1]']])],caps:['Each character has an embedding. Four chosen scalar embeddings make this example transparent.','Slide a width-2 filter over adjacent characters and compute each dot product.','ReLU sets negative responses to zero.','Max-pooling keeps the strongest response across the word.','ELMo combines many such features, then applies highway layers and a projection before its LSTMs.']};
 },
 'elmo-architecture':state=>{
  let s='';
  ['the','river','bank'].forEach((w,i)=>{
   const x=20+i*310;
   s+=box(x,450,250,55,w,0,'data')+box(x,355,250,65,'Character CNN input',1,'data')+line(x+125,450,x+125,420,1);
   for(const [y,layer,at]of[[220,1,2],[65,2,3]]){
    s+=box(x,y,250,110,'Forward LSTM '+layer+'\nBackward LSTM '+layer,at);
    const from=layer===1?355:220;
    s+=line(x+70,from,x+70,y+110,at)+line(x+185,from,x+185,y+110,at);
    if(i<2)s+=line(x+250,y+28,x+310,y+28,at)+line(x+310,y+83,x+250,y+83,at);
   }
  });
  s+=box(1040,175,320,150,'Task-specific\nweighted mixture',4)+path('M890,387 H970 V285 H1040',4)+line(890,275,1040,250,4)+path('M890,120 H970 V215 H1040',4)+A(4,text(1180,135,'h₀, h₁, h₂ for bank','small'))+A(3,text(690,35,'Forward and backward channels remain separate through the stack','small'));
  return{svg:wrap(s),caps:['A word starts as characters. The selected occurrence is bank.','The character CNN produces one context-independent input per word.','Layer 1 reads left-to-right and right-to-left with separate recurrent states.','Layer 2 stacks within each direction. Contextual representations concatenate the two directional states.','A downstream task learns how to mix the input representation and both contextual layers.']};
 },
 'elmo-mixture':state=>{
  const q=elmoMix(Number(state.logit??0));let s='';
  q.layers.forEach((h,i)=>{const y=60+i*135;s+=box(30,y,300,85,'h'+i+' = ['+h.join(', ')+']',0,'data')+box(450,y,320,85,'s'+i+' = '+fmt(q.weights[i]),1)+line(330,y+42,450,y+42,1)+path('M770,'+(y+42)+' H940 V245 H1030',2);});
  s+=box(1030,190,330,110,'Mixed vector\n['+q.result.map(x=>fmt(x)).join(', ')+']',2,'dark')+A(3,text(690,495,'γ = 1 · weights sum to '+fmt(q.weights.reduce((a,b)=>a+b,0),3),'bold'));
  return{svg:wrap(s),controls:[range('logit','Top-layer logit',-2,2,.25,0)],caps:['Three chosen vectors stand for a token’s input, lower-layer and upper-layer states.','Softmax across layer logits produces task-specific mixture weights. Move the slider.','Multiply each layer vector by its scalar weight, then sum the vectors.','The mixture stays the same width. These are layer weights, distinct from attention over token positions.']};
 },
 'bpe-training':state=>{
  const n=Number(state.merges??0),q=trainBPE(n),winner=q.ranked[0],sample=q.rows.find(r=>r.symbols.some((s,i)=>s===winner?.pair[0]&&r.symbols[i+1]===winner?.pair[1]));
  let s=A(0,text(690,35,'Character BPE · end marker </w> · weighted counts','small'));
  q.rows.forEach((r,i)=>s+=A(0,text(25,105+i*75,r.word+' ×'+r.count,'small','start'))+symbols(r.symbols,210,73+i*75,0));
  s+=A(1,text(1150,75,'Top next pairs','small bold'));
  q.ranked.slice(0,3).forEach((p,i)=>s+=A(1,text(1150,130+i*65,p.pair.join(' + ')+' : '+p.count,'small')));
  if(winner)s+=box(980,305,380,80,'Choose '+winner.pair.join(' + ')+'\nweighted count '+winner.count,2);
  if(sample)s+=A(3,text(25,425,'After next merge in '+sample.word+':','small','start'))+symbols(mergePair(sample.symbols,winner.pair),360,397,3);
  s+=A(3,text(690,505,'Completed merges: '+n+' · pair counts are recomputed after every merge','small'));
  return{svg:wrap(s),controls:[range('merges','Completed merges',0,8,1,0)],caps:['The complete training corpus is shown with word frequencies. The slider chooses how many merges have already happened.','Count adjacent pairs, weighting each occurrence by the word’s frequency. Ties use a declared alphabetical rule.','Choose the current most frequent pair: '+(winner?winner.pair.join(' + ')+' with count '+winner.count:'none remain')+'.','Replace the chosen adjacent pair everywhere. Then recount before the following merge.']};
 },
 'bpe-encoding':state=>{
  const n=Number(state.rules??8),q=applyBPE('lowest',n),rules=trainBPE(8).history;
  let s=box(20,195,260,95,'Unseen word\nlowest',0,'data')+line(280,242,380,242,1)+box(380,30,510,380,'',1);
  rules.forEach((r,i)=>s+=A(1,text(405+(i>=4?250:0),100+(i%4)*80,(i+1)+'. '+r.pair.join(' + '),'small','start')));
  s+=line(890,242,950,242,2)+symbols(q.symbols,950,215,2)+A(3,text(690,490,'Apply learned rules in order. Do not learn new merges from this test word.','small bold'));
  return{svg:wrap(s),controls:[range('rules','Learned rules applied',0,8,1,8)],caps:['Lowest is absent from the four-word training corpus. Its letters are in the initial alphabet.','Use the ordered merge rules learned from that corpus.','After '+n+' rules: '+q.symbols.join(' + ')+'. The slider replays how the segmentation changes.','Training learns the vocabulary and rule order. Encoding reuses them on new text.']};
 },
 'transfer-body':state=>{
  let s='';
  [['Embeddings only',80,false],['Whole-model pretraining',790,true]].forEach(([title,x,whole])=>{
   s+=A(0,text(x+250,40,title,'bold'))+box(x,390,500,80,'Word embeddings · pretrained',0,'data')+box(x,235,500,85,whole?'Network layers · pretrained':'Network layers · random start',1,whole?'data':'train')+line(x+250,390,x+250,320,1)+box(x,80,500,85,'Task head · newly initialised',2,'train')+line(x+250,235,x+250,165,2);
  });
  s+=A(3,text(690,510,'Full fine-tuning can update inherited weights as well as the new head.','small'));
  return{svg:wrap(s),caps:['Both systems may begin with pretrained token vectors.','Whole-model pretraining also supplies the body’s weights. The task can start from reusable representations.','A task-specific head learns the required outputs.','Choose independently whether the inherited body stays frozen or updates during task fine-tuning.']};
 },
 'masked-objective':state=>{
  const target=state.target==='movie'?'movie':'great',p=Number(state.p??.6),tokens=['the','movie','was','great'];
  let s=symbols(tokens.map(w=>w===target?'[MASK]':w),50,55,0)+box(230,190,520,110,'Bidirectional encoder\ncorrupted input',1)+path('M350,105 V190',1)+box(940,190,400,110,'Selected-token prediction\np('+target+') = '+fmt(p,2),2)+line(750,245,940,245,2)+box(440,385,500,90,'Loss = −ln p = '+fmt(lmLoss(p)),3,'data')+path('M1140,300 V430 H940',3)+A(1,text(1130,70,'Saved target: '+target,'bold'))+A(3,text(690,510,'A chosen target probability, not an actual pretrained model prediction','small'));
  return{svg:wrap(s),controls:[choice('target','Prediction target',[['great','great'],['movie','movie']]),range('p','Correct-token probability',.05,.95,.05,.6)],caps:['Select a token to hide and retain its original token ID as the training target.','The encoder reads the corrupted sequence with bidirectional attention.','At the selected position, a vocabulary head assigns a probability to the original token.','The selected target’s contribution is −log p. A higher correct-token probability gives a smaller loss.']};
 },
 'ulmfit-stages':state=>{
  const names=['General LM','In-domain LM','Task classifier'],data=['Wikipedia text','Review text, labels ignored','Reviews + sentiment labels'];
  let s='';
  names.forEach((name,i)=>{const x=20+i*465;s+=box(x,40,420,80,data[i],0,'data')+box(x,240,420,130,'3-layer AWD-LSTM\n'+name,i+1)+line(x+210,120,x+210,240,i+1);if(i)s+=line(x-45,305,x,305,i+1);s+=A(i+1,text(x+210,440,i<2?'Next-token loss':'Supervised classification loss','small bold'));});
  s+=A(4,text(690,510,'Inherit the body weights. Replace the vocabulary head for classification.','small'));
  return{svg:wrap(s),caps:['ULMFiT separates three stages and the data available in each.','Pretrain the recurrent LM on a broad text corpus.','Continue next-token prediction on the target domain’s text.','Add the classifier and learn from labelled task examples.','The body transfers between stages, while the prediction head changes for the new task.']};
 },
 'ulmfit-pooling':state=>{
  const q=poolToy();let s=matrix(30,170,q.states,'Three token states H',0,75);
  [['Last',q.last,1],['Max',q.max,2],['Mean',q.mean,3]].forEach(([name,v,at],i)=>{const y=30+i*150;s+=box(420,y,390,95,name+': ['+v.map(x=>fmt(x,2)).join(', ')+']',at,'data')+path('M180,282 H310 V'+(y+47)+' H420',at);});
  s+=box(1020,160,340,160,'Concatenate\n6 features\nthen task classifier',4)+path('M810,77 H900 V210 H1020',4)+line(810,227,1020,240,4)+path('M810,377 H900 V270 H1020',4)+A(4,text(690,510,'Three pooling operations over H, not three different language models','small'));
  return{svg:wrap(s),caps:['The recurrent body gives a hidden vector at every token position. These three vectors are chosen.','Keep the final hidden state.','Take the maximum separately in each feature coordinate.','Average each feature over positions.','Concatenate last, max and mean. The supervised head uses the resulting 3d-dimensional vector.']};
 },
 'adaptation-data':state=>{
  const method=state.method||'dapt',data={dapt:['Domain text','Biomedical papers','Reusable across domain tasks'],tapt:['Task training inputs','Reviews, labels ignored','Close to one task distribution'],lapt:['Target-language text','Swahili text','Reusable across language tasks']}[method];
  let s=box(20,195,300,110,'Pretrained model',0,'data')+box(465,30,470,110,data[0]+'\n'+data[1],1,'data')+box(465,225,470,110,'Continue the LM objective\n'+method.toUpperCase(),2)+line(700,140,700,225,2)+line(320,250,465,280,2)+box(1070,225,290,110,'Adapted checkpoint',3,'dark')+line(935,280,1070,280,3)+A(3,text(690,465,data[2],'bold'))+A(3,text(690,510,'The adaptation name describes the corpus. Parameter-update choices are separate.','small'));
  return{svg:wrap(s),controls:[choice('method','Adaptation corpus',[['dapt','DAPT · domain'],['tapt','TAPT · task inputs'],['lapt','LAPT · language']])],caps:['Start from an existing checkpoint.','Select the relevant text. This stage uses token targets constructed from that text.','Continue the same family of language-modelling objective on the selected corpus.','Save the adapted checkpoint, then fine-tune and evaluate the downstream task.']};
 },
 'adaptation-stack':state=>{
  const labels=['General\npretraining','Language\nadaptation','Domain\nadaptation','Task\nadaptation','Labelled task\nfine-tuning'];
  let s='';
  labels.forEach((name,i)=>{s+=box(10+i*280,205,240,120,name,i,i>0&&i<4?'data':'')+(i?line(250+(i-1)*280,265,290+(i-1)*280,265,i):'');});
  s+=A(0,text(690,60,'A possible sequence, with optional middle stages','bold'))+A(4,text(690,450,'Each added stage needs a baseline and validation evidence.','small'));
  return{svg:wrap(s),caps:['General pretraining supplies the starting model.','Optional LAPT gives it more examples in the target language.','Optional DAPT changes the domain distribution.','Optional TAPT focuses on the task’s own training inputs.','Labelled fine-tuning optimises the required task. This ordering is a design option, not a universal recipe.']};
 },
 'dapt-results':state=>{
  const r=adaptationResults[Number(state.task||0)],names=['RoBERTa baseline','+ DAPT','+ TAPT','+ DAPT + TAPT'];
  return{svg:wrap(A(0,text(690,35,r.task+' · '+r.domain+' · test '+r.metric,'bold'))+scoreRows(names,r.scores,r.sd)+A(4,text(690,510,'Means ± standard deviations across five seeds · source: published Table 5','small'))),controls:[choice('task','Task',adaptationResults.map((r,i)=>[String(i),r.task]))],caps:['All four conditions start from RoBERTa and receive the same downstream task fine-tuning.','Baseline test mean: '+fmt(r.scores[0],1)+'.','DAPT adds domain-text pretraining: '+fmt(r.scores[1],1)+'.','TAPT adds task-input pretraining: '+fmt(r.scores[2],1)+'.','DAPT then TAPT: '+fmt(r.scores[3],1)+'. Select another task. The highest mean and gain depend on the task.']};
 },
 'adaptation-split':state=>{
  let s='';
  [['Training inputs + labels',30],['Validation inputs + labels',510],['Test inputs + labels',990]].forEach(([name,x])=>s+=box(x,30,350,90,name,0,'data'));
  s+=box(30,230,350,110,'TAPT: hide train labels\nthen task training',1)+line(205,120,205,230,1)+box(510,230,350,110,'Choose method\nand stopping point',2)+line(685,120,685,230,2)+box(990,230,350,110,'Final evaluation\nof selected method',3)+line(1165,120,1165,230,3)+line(380,285,510,285,2)+line(860,285,990,285,3)+A(4,text(690,460,'Deduplicate added pretraining text against the held-out evaluation data.','bold'))+A(4,text(690,510,'Inductive setup: model fitting uses training data, including its unlabelled-input view.','small'));
  return{svg:wrap(s),caps:['Declare the split before adaptation.','Use only the training inputs for ordinary inductive TAPT, then train the task loss with training labels.','Validation selects the adaptation method, hyperparameters and training duration.','The held-out test estimates the selected system’s performance.','No-label pretraining can still expose test content. Document overlap checks for extra corpora.']};
 },
 'maft-architecture':state=>{
  const shared=state.mode!=='laft';let s=A(0,text(690,35,shared?'MAFT: shared adaptation':'LAFT: per-language adaptation','bold'));
  ['Swahili text','Hausa text','Yoruba text'].forEach((name,i)=>{const x=50+i*455;s+=box(x,90,370,90,name,0,'data');if(shared)s+=path('M'+(x+185)+',180 V225 H690 V270',1);else s+=box(x,320,370,120,'Separate adapted\nbackbone '+(i+1),2)+line(x+185,180,x+185,320,1);});
  if(shared)s+=box(435,270,510,85,'Mixed-language MLM batches',1)+box(435,415,510,75,'One adapted backbone',2,'dark')+line(690,355,690,415,2);
  s+=A(3,text(690,shared?515:510,'Three example corpora are drawn. AfroXLMR MAFT used 20 languages.','small'));
  return{svg:wrap(s),controls:[choice('mode','Storage strategy',[['maft','MAFT · one shared backbone'],['laft','LAFT · per-language backbones']])],caps:['Begin with one multilingual starting checkpoint and language-specific text corpora.','MAFT mixes languages within one continued-pretraining process. LAFT trains separate adaptations.','The chosen strategy determines whether one or several adapted backbones must be stored.','Then fine-tune on labelled tasks. Shared storage, language coverage and per-language scores all matter.']};
 },
 'maft-results':state=>{
  let s=A(0,text(690,35,'NER test F1 · average over 13 languages · five runs','bold'))+axis(400,430,530);
  maftResults.forEach((r,i)=>{const y=90+i*85;s+=A(i+1,text(20,y+24,r.name,'small','start')+'<rect x="400" y="'+y+'" height="35" width="'+(530*r.score/100)+'" fill="#4f6d9a"/>'+text(950,y+24,fmt(r.score,1),'small bold','start')+text(1080,y+13,r.copies+' × '+r.params+'M','small','start')+text(1080,y+43,'stored backbone params','tiny','start'));});
  s+=A(4,text(690,510,'The LAFT comparison uses 13 separately adapted backbones. Counts exclude task heads.','small'));
  return{svg:wrap(s),caps:['This chart transcribes Table 3 of the AfroXLMR paper.','XLM-R-base: 79.2 average NER F1, one 270M backbone.','MAFT: 81.8 with one 270M backbone.','Pruned-vocabulary MAFT: 80.7 with one 140M backbone.','Per-language LAFT: 82.3, storing 13 backbones. The shared model trades a little mean quality for simpler storage.']};
 },
 'social-pipeline':state=>{
  let s='';
  const names=['AfroXLMR\nlanguage-adapted encoder','DAPT\nAfriSocial text','TAPT\noptional task inputs','Supervised\nclassification task'];
  names.forEach((name,i)=>{const x=10+i*350;s+=box(x,240,310,125,name,i,i===0?'data':'')+(i?line(x-40,302,x,302,i):'');});
  s+=box(360,35,500,90,'Relevant social and news text\nconstructs LM targets',1,'data')+line(515,125,515,240,1)+A(4,text(690,480,'Language adaptation and domain adaptation address different distribution gaps.','bold'));
  return{svg:wrap(s),caps:['AfroXLMR already adapted a multilingual encoder to African languages.','Continue masked-LM training on relevant social/news domain text.','A task-input pretraining stage can focus the distribution further.','Fine-tune the classifier using its labels.','Keep the base checkpoint and task evaluation fixed when measuring the additional benefit.']};
 },
 'social-results':state=>{
  let s=A(0,text(690,30,'Reported average macro-F1 · AfroXLMR vs +DAPT','bold'))+axis(360,420,720);
  socialResults.forEach((r,i)=>{const y=75+i*125;s+=A(i+1,text(20,y+36,r.task,'bold','start')+'<rect x="360" y="'+y+'" height="29" width="'+(720*r.base/100)+'" fill="#94a3b8"/>'+'<rect x="360" y="'+(y+38)+'" height="29" width="'+(720*r.adapted/100)+'" fill="#4f6d9a"/>'+text(1110,y+23,fmt(r.base,2),'small','start')+text(1110,y+61,fmt(r.adapted,2),'small bold','start')+text(1300,y+36,'+'+fmt(r.adapted-r.base,2),'small bold'));});
  s+=A(4,text(690,500,'Different datasets cover different language sets. These are F1-point differences.','small'));
  return{svg:wrap(s),caps:['Grey is baseline AfroXLMR. Blue adds domain-adaptive pretraining.','AfriSenti: 51.39 to 58.21 average macro-F1.','AfriEmo: 44.25 to 51.45.','AfriHate: 65.17 to 69.83.','The final paper reports gains in these averages. Individual language results can improve by different amounts or decline.']};
 },
 'language-mixture':state=>{
  const p=Number(state.education??40),q=tokenBudget(p);let s=box(20,130,300,100,'Llama-3.1-8B\nstarting checkpoint',0,'data')+box(485,40,380,90,'WURA\n'+fmt(q.wura,1)+'B tokens',1,'data')+box(485,320,380,90,'English educational text\n'+fmt(q.education,1)+'B tokens',1,'data')+path('M865,85 H950 V215 H1050',2)+path('M865,365 H950 V265 H1050',2)+box(1050,180,310,120,'Continue next-token\nprediction\n10B-token budget',2)+line(320,180,1050,240,2)+A(3,'<rect x="55" y="455" height="30" width="'+(1260*(1-p/100))+'" fill="#4f6d9a"/><rect x="'+(55+1260*(1-p/100))+'" y="455" height="30" width="'+(1260*p/100)+'" fill="#8b614a"/>')+A(3,text(690,510,'Published education recipe: 60% WURA + 40% FineWeb-Edu. Other slider ratios are hypothetical.','small'));
  return{svg:wrap(s),controls:[range('education','Education mix (%)',0,100,10,40)],caps:['Start from a pretrained causal decoder.','Allocate a fixed 10B-token budget between two text sources.','Continue the same next-token objective on the mixture. The slider changes only the budget calculation.','At '+p+'% education, use '+fmt(q.education,1)+'B education tokens and '+fmt(q.wura,1)+'B WURA tokens. Performance requires a separate measured experiment.']};
 },
 'lugha-results':state=>{
  const r=lughaResults[Number(state.metric||0)];
  return{svg:wrap(A(0,text(690,35,r.task+' · IrokoBench average excluding English and French','bold'))+scoreRows(lughaNames,r.scores)+A(4,text(690,510,'Same 8B starting family · reported paper Table 1 · score scale 0–100','small'))),controls:[choice('metric','Benchmark',lughaResults.map((r,i)=>[String(i),r.task]))],caps:['Select one task. The bars compare the same metric and language averaging rule.','Llama-3.1-8B base: '+fmt(r.scores[0],1)+'.','WURA-only continuation: '+fmt(r.scores[1],1)+'.','Education mixture: '+fmt(r.scores[2],1)+'.','Maths mixture: '+fmt(r.scores[3],1)+'. '+(r.task==='AfriMMLU'?'Education leads this average.':r.task==='AfriMGSM'?'Maths data leads this average.':'Small differences need cautious interpretation.')+' No single mixture wins every task.']};
 },
 'inkuba-architecture':state=>{
  let s=box(30,440,610,65,'Token IDs · vocabulary 61,788',0,'data')+box(30,335,610,65,'Token embeddings · width 2048',1,'data')+line(335,440,335,400,1)+box(30,120,610,165,'Causal self-attention\nPosition-wise MLP · width 5632',2)+line(335,335,335,285,2)+box(30,15,610,65,'Vocabulary head · next-token distribution',3,'dark')+line(335,120,335,80,3);
  const details=['8 decoder layers','32 attention heads','Hidden width 2048','FFN width 5632','Context: 2048 tokens','About 0.422B parameters'];
  details.forEach((v,i)=>s+=A(2,text(810,110+i*58,v,'bold','start')));
  s+=A(2,text(335,310,'repeat decoder block ×8','small'))+A(4,text(1000,490,'Architecture facts from\nInkubaLM Table 4','small'));
  return{svg:wrap(s),caps:['A multilingual BPE tokenizer maps text to IDs.','An embedding table maps those IDs to 2048-dimensional vectors.','Eight causal decoder layers contextualise the known prefix with attention and MLPs. Norms and positional operations sit inside these layers.','The vocabulary head predicts the next token.','This small decoder uses the same broad autoregressive task as larger language models. Size alone does not settle task quality.']};
 },
 'fine-tuning-updates':state=>{
  const full=state.mode==='full';let s='';
  for(let i=0;i<3;i++){const y=380-i*120;s+=box(140,y,560,85,'Pretrained block '+(i+1)+' · '+(full?'TRAIN':'FROZEN'),i+1,full?'train':'frozen')+(i?line(420,y+120,420,y+85,i+1):'');}
  s+=box(140,15,560,85,'New task head · TRAIN',4,'train')+line(420,140,420,100,4)+A(0,text(1060,150,'Task input','bold'))+path('M1060,185 V422 H700',1)+A(4,text(1060,305,full?'Body + head learn':'Head learns;\nbody features stay fixed','bold'))+A(4,text(690,510,'TRAIN marks optimiser-updated parameters. A frozen backbone still performs computation.','small'));
  return{svg:wrap(s),controls:[choice('mode','Parameter updates',[['head','Feature extraction · head only'],['full','Full fine-tuning']])],caps:['Use one pretrained backbone and a newly initialised task head.','The input passes through the first block.','Higher blocks transform its representation.','The last block supplies task features.','Choose whether the task loss updates the head alone or both the head and body.']};
 },
 'layer-learning-rates':state=>{
  const rates=learningRates(Number(state.rate||.0001));let s='';
  rates.forEach((lr,i)=>{const y=370-i*145;s+=box(60,y,570,95,'Body layer '+(i+1),i,'data')+box(880,y,440,95,'η'+(i+1)+' = '+lr.toExponential(2),i+1,'train')+line(630,y+47,880,y+47,i+1);});
  s+=A(3,text(690,510,'Example ratio: each lower layer uses the next layer’s rate divided by 2.6.','small'));
  return{svg:wrap(s),controls:[choice('rate','Top-body rate',[['0.0001','0.0001'],['0.001','0.001']])],caps:['Lower layers can retain more of the starting representation.','Assign the lowest layer a smaller learning rate.','Use a larger rate in the middle layer.','The top-body rate is largest in this example. Training still determines the actual gradients and updates.']};
 },
 'gradual-unfreeze':state=>{
  const phase=Number(state.phase||0);let s='';
  for(let i=0;i<3;i++){const y=370-i*120,train=phase>=3-i;s+=box(120,y,750,85,'Body layer '+(i+1)+' · '+(train?'TRAIN':'FROZEN'),i,train?'train':'frozen');}
  s+=box(120,10,750,85,'Task head · TRAIN',3,'train')+A(3,text(1120,200,'Phase '+phase,'bold'))+A(3,text(1120,290,phase===0?'Head only':phase===3?'All layers train':phase+' top body\nlayer'+(phase>1?'s':'')+' train','small'))+A(3,text(690,510,'A schedule changes the trainable set over time. Validate the schedule on the actual task.','small'));
  return{svg:wrap(s),controls:[choice('phase','Unfreezing phase',[['0','0 · head only'],['1','1 · top layer'],['2','2 · top two'],['3','3 · all layers']])],caps:['The lower body begins frozen.','A schedule may expose the middle layer later.','The upper body can adapt before lower layers do.','The task head trains throughout. Select a phase to inspect the trainable parameters.']};
 },
 'multitask-loss':state=>{
  const l=Number(state.lambda??.5);
  const s=box(450,365,480,100,'Shared pretrained body',0,'data')+box(30,160,420,100,'Target task head\nL₁ = 0.2',1,'train')+box(930,160,420,100,'Auxiliary head / LM\nL₂ = 0.8',2,'train')+path('M570,365 V310 H240 V260',1)+path('M810,365 V310 H1140 V260',2)+box(450,15,480,90,'Weighted loss\n'+fmt(l,2)+' L₁ + '+fmt(1-l,2)+' L₂ = '+fmt(mixedLoss(l)),3,'data')+path('M240,160 V60 H450',3)+path('M1140,160 V60 H930',3)+A(3,text(690,510,'The weighted scalar objective is computed. Accuracy and gradient compatibility require experiments.','small'));
  return{svg:wrap(s),controls:[range('lambda','Target-task weight λ',0,1,.1,.5)],caps:['Two objectives share body parameters. They may use different task heads.','The target examples contribute their own loss.','An auxiliary task or LM objective contributes another loss.','The optimiser uses a weighted combination. A related task can help, but task gradients may also conflict.']};
 },
 'adapter-architecture':state=>{
  const q=adapterToy();
  let s=box(20,200,230,95,'x = [1, 0.5]',0,'data')+box(350,350,220,100,'Down: d → r\\nWdown x = '+fmt(q.u,2),1,'train')+path('M250,247 H285 V400 H350',1)+box(650,350,220,100,'ReLU\\n'+fmt(q.activation,2),2)+line(570,400,650,400,2)+box(950,350,220,100,'Up: r → d\\nΔ = [0.20, −0.10]',3,'train')+line(870,400,950,400,3)+op(1240,247,'+',4)+line(250,247,1215,247,4,'thick')+path('M1170,400 H1240 V272',4)+box(1110,50,260,95,'y = [1.20, 0.40]',4,'dark')+line(1240,222,1240,145,4)+A(4,text(740,210,'Unchanged residual input','small'))+A(4,text(690,510,'Chosen 2 → 1 → 2 adapter · exact arithmetic','small'));
  return{svg:wrap(s),caps:['The frozen body supplies a hidden vector x.','A trainable down projection maps width d into a small bottleneck r.','Apply a nonlinearity inside the added module.','A trainable up projection returns to width d.','Add the module’s output to x. This nonlinear adapter differs from LoRA’s linear weight-update branch.']};
 },
 'lora-architecture':state=>{
  const s=box(20,195,190,100,'Input x',0,'data')+box(350,170,470,140,'W₀ · FROZEN\ndout × din',1,'frozen')+line(210,245,350,245,1)+box(350,390,220,85,'A · TRAIN\nr × din',2,'train')+path('M210,245 H265 V432 H350',2)+box(720,390,220,85,'B · TRAIN\ndout × r',3,'train')+line(570,432,720,432,3)+A(4,text(1050,415,'× α/r','small bold'))+op(1120,245,'+',5)+line(820,245,1095,245,5)+path('M940,432 H1120 V270',5)+box(1210,195,160,100,'Output y',5,'dark')+line(1145,245,1210,245,5)+A(1,text(690,70,'Learn a change to this projection, while keeping its base weights','bold'));
  return{svg:wrap(s),caps:['LoRA adapts a selected linear projection.','The pretrained W₀ branch stays frozen.','Train A to project the input into r coordinates.','Train B to project back to the output width. Standard LoRA adds no activation between A and B.','Scale the added branch by α/r.','Add the low-rank update to the frozen branch. Gradients update A and B, not W₀.']};
 },
 'lora-numeric':state=>{
  const q=loraToy(Number(state.rank||1));
  const s=matrix(25,80,q.W,'W₀ frozen',0)+matrix(250,80,q.x,'x',0)+matrix(500,80,q.base,'W₀x',1)+matrix(25,330,q.A,'A trainable',2)+matrix(285,330,q.Ax,'Ax',2)+matrix(500,330,q.B,'B trainable',3)+matrix(780,330,q.update,'BAx',3)+path('M624,142 H1000 V220',4)+path('M842,392 H1000 V270',4)+op(1000,245,'+',4)+matrix(1190,210,q.out,'Output y',4)+line(1025,245,1190,245,4)+A(4,text(690,505,'Chosen matrices · α/r = 1 · rank changes the stated example, not a measured training outcome','small'));
  return{svg:wrap(s),controls:[choice('rank','Toy update rank',[['1','r = 1'],['2','r = 2']])],caps:['The base matrix is identity and x = [2, 1].','The frozen branch returns W₀x = [2, 1].','Compute Ax in the small rank space.','Multiply by B to obtain the update in output space.','Add the branches coordinate by coordinate: ['+q.out.map(r=>fmt(r[0],2)).join(', ')+']. The selected matrices define this teaching example.']};
 },
 'lora-parameters':state=>{
  const r=Number(state.rank??8),q=loraCounts(r),num=x=>x.toLocaleString('en-US');
  const s=box(20,80,620,130,'Full update\n4096 × 4096\n'+num(q.full)+' trainable parameters',0,'data')+box(750,80,610,130,'LoRA update · rank '+r+'\n2 × 4096 × '+r+'\n'+num(q.trainable)+' trainable parameters',1,'train')+A(2,text(690,310,fmt(q.percent,3)+'% of this matrix’s full-update count','bold'))+A(2,'<rect x="50" y="360" width="1260" height="35" fill="#edf2f7"/><rect x="50" y="360" width="'+(1260*q.percent/100)+'" height="35" fill="#8b614a"/>')+A(3,text(690,480,'The original matrix remains stored. Other layers, activations and optimiser state also cost memory.','small'));
  return{svg:wrap(s),controls:[range('rank','Rank r',1,128,1,8)],caps:['A full update would train all 16,777,216 entries of this square matrix.','LoRA trains r(din + dout) entries in A and B.','At rank '+r+', the count is '+num(q.trainable)+', or '+fmt(q.percent,3)+'% for this selected projection.','This percentage describes trainable additions for one matrix. It is not the full model’s size or a complete GPU-memory estimate.']};
 }
};
export function renderMoreScene(name,state={}){return MORE_SCENES[name]?.(state);}
