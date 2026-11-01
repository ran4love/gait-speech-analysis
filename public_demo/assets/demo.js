/* Standalone public demo. Never call a server, accept files, or run neural models. */
(() => {
  const $=id=>document.getElementById(id), fixture=window.PUBLIC_FIXTURE;
  let language='en', responses=[], revision=0, exported=null, opened=false, time=0, playing=false, lastTick=null, message=null;
  try {language=localStorage.getItem('adualnet_public_demo_language')==='zh'?'zh':'en';} catch (_) {}
  const labels={
    idle:['Example ready','示例已准备'],review:['Human review needed','待人工核查'],done:['Demo outputs generated','演示结果已生成'],
    window:['Choose a finite task window within 0–12 s, with end later than start.','请在 0–12 秒内设置有限的任务时窗，结束必须晚于开始。'],
    metadata:['Enter a fictional reviewer code and record the scoring rule.','请填写虚构核查者代码并记录评分规则。'],
    confirm:['Complete both review confirmations before generating outputs.','请先完成两项核查确认再生成结果。'],
    onset:['An included response lies outside the task [start, end) interval or has an invalid onset.','有计入的回答不在任务 [开始,结束) 时窗内，或其起始时间无效。'],
    text:['Every included response needs reviewed text.','每条计入的回答都需要核对后的文本。'],
    correctness:['Explicitly mark every included response as correct or incorrect.','请逐条标记每个计入回答的正确性。'],
    duplicate:['Duplicate included response and onset; review the response rows.','出现重复的回答与起始时间，请核查回答行。'],
    pending:['Not reviewed','待确认'],correct:['Correct','正确'],incorrect:['Incorrect','错误'],seek:['Seek','定位'],remove:['Remove','删除'],
    included:['Include response','计入回答'],onsetLabel:['Response onset','回答起始秒'],textLabel:['Response text','回答文本'],gradeLabel:['Response correctness','回答正确性'],
    generated:['Outputs generated in this browser. Downloads reflect this confirmed revision.','已在本浏览器生成结果，下载内容对应本次确认版本。'],
    dirty:['Review changed. Previous download controls are hidden until both sections are confirmed again.','核查内容已更改。重新确认两部分前，旧版下载入口暂时隐藏。'],
    limit:['This demonstration allows up to 100 response rows.','本演示最多允许 100 条回答。'],
    total:['Included responses','计入回答数'],ncorrect:['Correct responses','正确回答数'],accuracy:['Reviewed accuracy','人工核查正确率'],
    reference:['Precomputed reference','预计算参考'],live:['Live browser calculation','浏览器即时计算'],missing:['Missing','缺失'],
    frame:['Frame','帧'],draft:['review_draft','核查草稿']
  };
  const T=key=>labels[key]?.[language==='en'?0:1]??key;
  function setLanguage(next) {
    language=next==='zh'?'zh':'en';document.documentElement.lang=language==='zh'?'zh-CN':'en';
    try {localStorage.setItem('adualnet_public_demo_language',language);} catch (_) {}
    for(const node of document.querySelectorAll('[data-en][data-zh]'))node.textContent=node.dataset[language];
    $('lang-en').setAttribute('aria-pressed',String(language==='en'));$('lang-zh').setAttribute('aria-pressed',String(language==='zh'));
    $('review-state').textContent=T(exported?'done':opened?'review':'idle');
    if(opened)renderRows();if(exported)renderResults();if(message)showMessage(message.key,message.kind);draw();
  }
  function showMessage(key,kind='error') {message={key,kind};$('review-message').textContent=T(key);$('review-message').className=kind;}
  function invalidate() {
    $('gait-confirm').checked=false;$('speech-confirm').checked=false;exported=null;
    $('result-content').hidden=true;$('results-empty').hidden=false;$('review-state').textContent=T('review');
    if(opened)showMessage('dirty','');
  }
  function openExample() {
    if(opened){$('review').scrollIntoView({block:'start'});return;}
    opened=true;responses=JSON.parse(JSON.stringify(fixture.responses));$('transcript').value=fixture.raw_transcript;
    $('review-placeholder').hidden=true;$('workspace').hidden=false;$('review-state').textContent=T('review');
    renderRows();draw();$('review').scrollIntoView({block:'start'});
  }
  const number=value=>String(value).trim()===''?NaN:Number(value);
  function readReview() {
    return {reviewer:$('reviewer').value,rule:$('scoring-rule').value,notes:$('notes').value,corrected_transcript:$('transcript').value,
      start:number($('task-start').value),end:number($('task-end').value),gait_window:{start:0,end:12,fixed:true},
      gaitConfirmed:$('gait-confirm').checked,speechConfirmed:$('speech-confirm').checked,
      responses:JSON.parse(JSON.stringify(responses))};
  }
  function renderRows() {
    const tbody=$('responses');tbody.replaceChildren();
    responses.forEach((row,index)=>{
      const tr=document.createElement('tr');
      const cell=node=>{const td=document.createElement('td');td.append(node);tr.append(td);};
      const include=document.createElement('input');include.type='checkbox';include.checked=row.include;include.setAttribute('aria-label',T('included')+' '+(index+1));
      include.onchange=()=>{row.include=include.checked;invalidate();};cell(include);
      const onset=document.createElement('input');onset.type='number';onset.step='.01';onset.min=0;onset.max=12;onset.value=Number.isFinite(row.onset)?row.onset:'';onset.setAttribute('aria-label',T('onsetLabel')+' '+(index+1));
      onset.oninput=()=>{row.onset=number(onset.value);invalidate();};cell(onset);
      const text=document.createElement('input');text.type='text';text.maxLength=200;text.value=row.text;text.setAttribute('aria-label',T('textLabel')+' '+(index+1));
      text.oninput=()=>{row.text=text.value;invalidate();};cell(text);
      const grade=document.createElement('select');grade.setAttribute('aria-label',T('gradeLabel')+' '+(index+1));
      for(const [value,key] of [['','pending'],['1','correct'],['0','incorrect']]){const option=document.createElement('option');option.value=value;option.textContent=T(key);grade.append(option);}
      grade.value=row.correct==null?'':String(row.correct);grade.onchange=()=>{row.correct=grade.value===''?null:Number(grade.value);invalidate();};cell(grade);
      const actions=document.createElement('div');actions.className='row-actions';
      const seek=document.createElement('button');seek.type='button';seek.className='secondary';seek.textContent=T('seek');seek.onclick=()=>{if(Number.isFinite(row.onset)){playing=false;time=Math.min(359/30,Math.max(0,row.onset));draw();}};
      const remove=document.createElement('button');remove.type='button';remove.className='secondary';remove.textContent=T('remove');remove.onclick=()=>{responses.splice(index,1);invalidate();renderRows();};
      actions.append(seek,remove);cell(actions);tbody.append(tr);
    });
  }
  function draw() {
    const canvas=$('pose-view'),ctx=canvas.getContext('2d'),frame=Math.max(0,Math.min(359,Math.round(time*30)));
    const points=fixture.frames[frame].instances[0].keypoints;
    ctx.fillStyle='#102038';ctx.fillRect(0,0,640,360);ctx.strokeStyle='#233a57';ctx.lineWidth=1;
    for(let x=0;x<640;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,360);ctx.stroke();}
    for(let y=0;y<360;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(640,y);ctx.stroke();}
    ctx.strokeStyle='#385473';ctx.beginPath();ctx.moveTo(0,321);ctx.lineTo(640,321);ctx.stroke();
    const bones=[[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
    ctx.lineCap='round';ctx.strokeStyle='#2f5278';ctx.lineWidth=13;
    for(const [a,b]of bones){ctx.beginPath();ctx.moveTo(...points[a]);ctx.lineTo(...points[b]);ctx.stroke();}
    ctx.fillStyle='#2f5278';ctx.beginPath();ctx.arc(points[0][0]+10,points[0][1],17,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle='#75b8ff';ctx.lineWidth=2;
    for(const [a,b]of bones){ctx.beginPath();ctx.moveTo(...points[a]);ctx.lineTo(...points[b]);ctx.stroke();}
    points.forEach((p,i)=>{ctx.fillStyle=i>=15?'#ffca77':'#a8d2ff';ctx.beginPath();ctx.arc(p[0],p[1],i>=15?5:3,0,Math.PI*2);ctx.fill();});
    ctx.fillStyle='#bdcee4';ctx.font='12px monospace';ctx.fillText('SYNTHETIC / COCO-17 / NO INFERENCE',20,26);
    ctx.fillText('t = '+time.toFixed(2)+' s',20,343);$('clock').textContent=time.toFixed(2)+' / 12.00 s · '+T('frame')+' '+frame;
    $('timeline').value=time;
  }
  function tick(now) {
    if(playing){if(lastTick!==null)time=Math.min(359/30,time+(now-lastTick)/1000);if(time>=359/30)playing=false;draw();}
    lastTick=now;requestAnimationFrame(tick);
  }
  function renderResults() {
    const live=Object.fromEntries(exported.features.filter(r=>r.domain==='speech').map(r=>[r.feature,r.value]));
    $('metric-summary').replaceChildren();
    for(const [label,value]of [['total',live.Num_DT],['ncorrect',live.Correct_Num_DT],['accuracy',live.ACC_DT==null?'—':(100*live.ACC_DT).toFixed(1)+'%']]){
      const div=document.createElement('div'),small=document.createElement('small'),strong=document.createElement('strong');small.textContent=T(label);strong.textContent=value;div.append(small,strong);$('metric-summary').append(div);
    }
    const body=$('feature-rows');body.replaceChildren();
    for(const row of exported.features){const tr=document.createElement('tr');for(const value of [row.domain,row.feature,row.value==null?T('missing'):Number(row.value.toFixed(6)),row.unit]){const td=document.createElement('td');td.textContent=value;if(row.value==null)td.className='missing';tr.append(td);}
      const td=document.createElement('td'),strong=document.createElement('strong'),note=document.createElement('span');strong.textContent=T(row.domain==='gait'?'reference':'live');note.className='source-note';note.textContent=row.status;td.append(strong,note);tr.append(td);body.append(tr);}
    $('qc-json').textContent=JSON.stringify({gait_qc:exported.gait_qc,provenance:exported.provenance},null,2);
  }
  function generate() {
    try {const result=DemoCore.result(fixture,readReview(),revision+1);revision++;exported=result;
      $('results-empty').hidden=true;$('result-content').hidden=false;$('review-state').textContent=T('done');renderResults();showMessage('generated','success');$('results').scrollIntoView({block:'start'});
    } catch(error){showMessage(Object.hasOwn(labels,error.message)?error.message:'confirm');}
  }
  function download(name,content,type) {
    const blob=new Blob([content],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  const featureRows=()=>[['domain','feature','value','unit','source','status'],...exported.features.map(r=>[r.domain,r.feature,r.value,r.unit,r.source,r.status])];
  function downloadXlsx() {
    if(!exported)return;
    const wide=[['ID','synthetic','review_revision',...exported.features.map(r=>r.feature)],[fixture.id,true,exported.revision,...exported.features.map(r=>r.value)]];
    const annotations=[['onset_s','response_text','include','is_correct'],...exported.review.responses.map(r=>[r.onset,r.text,r.include,r.correct])];
    const audit=[['field','value'],...Object.entries(exported.provenance).map(([k,v])=>[k,typeof v==='object'?JSON.stringify(v):v]),['reviewer',exported.review.reviewer],['notes',exported.review.notes],['corrected_transcript',exported.review.corrected_transcript]];
    const qc=[['field','value'],...Object.entries(exported.gait_qc).map(([k,v])=>[k,typeof v==='object'&&v!==null?JSON.stringify(v):v])];
    download('ADualNet_synthetic_review_r'+exported.revision+'.xlsx',DemoXlsx.workbook([{name:'Features',rows:wide},{name:'Definitions_and_sources',rows:featureRows()},{name:'Reviewed_responses',rows:annotations},{name:'QC',rows:qc},{name:'Provenance',rows:audit}]),'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
  }
  $('lang-en').onclick=()=>setLanguage('en');$('lang-zh').onclick=()=>setLanguage('zh');$('open-demo').onclick=openExample;
  $('play').onclick=()=>{playing=!playing;if(time>=359/30)time=0;lastTick=null;};
  $('back').onclick=()=>{playing=false;time=Math.max(0,(Math.round(time*30)-1)/30);draw();};
  $('next').onclick=()=>{playing=false;time=Math.min(359/30,(Math.round(time*30)+1)/30);draw();};
  $('timeline').oninput=()=>{playing=false;time=Number($('timeline').value);draw();};
  $('add-response').onclick=()=>{if(responses.length>=100){showMessage('limit');return;}responses.push({onset:Math.round(time*100)/100,text:'',include:true,correct:null});invalidate();renderRows();};
  for(const id of ['reviewer','notes','scoring-rule','transcript','task-start','task-end'])$(id).addEventListener('input',invalidate);
  $('generate').onclick=generate;$('draft').onclick=()=>download('ADualNet_synthetic_review_draft.json',JSON.stringify({fixture_id:fixture.id,synthetic:true,confirmed:false,review:readReview()},null,2),'application/json');
  $('download-xlsx').onclick=downloadXlsx;
  $('download-csv').onclick=()=>{if(exported)download('ADualNet_synthetic_features_r'+exported.revision+'.csv',DemoCore.csv(featureRows()),'text/csv;charset=utf-8');};
  $('download-json').onclick=()=>{if(exported)download('ADualNet_synthetic_provenance_r'+exported.revision+'.json',JSON.stringify(exported,null,2),'application/json');};
  setLanguage(language);requestAnimationFrame(tick);
})();
