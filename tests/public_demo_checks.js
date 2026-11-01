/* Run with Node in CI; also callable by the local JavaScript test runtime. */
((root)=>{
  function run(core,xlsx,fixture) {
    let count=0;
    const assert=(value,message)=>{if(!value)throw Error(message);count++;};
    const rejects=(fn,key)=>{try{fn();throw Error('expected '+key);}catch(error){assert(error.message===key,'Wrong rejection: '+error.message);}};
    const review={reviewer:'DEMO_TEST',rule:'Synthetic sequential subtraction rule',start:0,end:12,
      gaitConfirmed:true,speechConfirmed:true,notes:'Synthetic only',corrected_transcript:'93 86 78 71',
      responses:fixture.responses.map((r,i)=>({...r,correct:i===2?0:1}))};
    const original=JSON.stringify(fixture);
    const result=core.result(fixture,review,1), metrics=Object.fromEntries(result.features.map(r=>[r.feature,r.value]));
    assert(metrics.Num_DT===4,'Total count');assert(metrics.Correct_Num_DT===3,'Correct count');assert(metrics.ACC_DT===.75,'Accuracy');
    assert(metrics.Timecount_DT_1stNum===2,'First response latency');assert(metrics.Correct_persec_DT===.25,'Rate');
    assert(review.responses[2].text==='78'&&review.responses[2].correct===0,'Preserve actual response error');
    assert(metrics.stance_times===null&&metrics.swing_times===null&&metrics.double_support_times===null,'Unavailable contact phases');
    assert(metrics.walking_speed===null,'No guessed metric distance');
    assert(JSON.stringify(fixture)===original,'Raw fixture not mutated');
    rejects(()=>core.result(fixture,{...review,gaitConfirmed:false},2),'confirm');
    rejects(()=>core.result(fixture,{...review,rule:''},2),'metadata');
    rejects(()=>core.result(fixture,{...review,start:4},2),'onset');
    rejects(()=>core.result(fixture,{...review,end:13},2),'window');
    rejects(()=>core.result(fixture,{...review,responses:review.responses.map((r,i)=>i===0?{...r,correct:null}:r)},2),'correctness');
    rejects(()=>core.result(fixture,{...review,responses:[...review.responses,review.responses[0]]},2),'duplicate');
    const zero=core.speech({...review,responses:[]});assert(zero.Num_DT===0&&zero.ACC_DT===null&&zero.Timecount_DT_1stNum===null,'Zero responses');
    const boundary=core.speech({...review,end:30,responses:[{include:true,onset:19.99,correct:1},{include:true,onset:20,correct:1}]});
    assert(boundary.Num_20s_DT===1,'20-second boundary is exclusive');
    assert(core.csv([['=1+1','+SUM(A1)','@SUM(A1)',-2]]).includes("'=1+1"),'CSV formula injection is neutralized');
    const workbook=xlsx.workbook([{name:'Features',rows:[['feature','value'],['ACC_DT',.75],['stance_times',null]]},
      {name:'Review',rows:[['text','correct'],['七十八',0],['=HYPERLINK("bad")',1]]}]);
    assert(workbook[0]===0x50&&workbook[1]===0x4b,'XLSX ZIP header');
    return {checks:count,workbook,result};
  }
  root.PublicDemoChecks=run;
  if(typeof module!=='undefined')module.exports=run;
  if(typeof require!=='undefined'&&require.main===module){
    const fs=require('node:fs'),path=require('node:path'),base=path.join(__dirname,'..');
    const fixture=JSON.parse(fs.readFileSync(path.join(base,'public_demo/assets/fixture.js'),'utf8').split('window.PUBLIC_FIXTURE=')[1].trim().slice(0,-1));
    const tested=run(require('../public_demo/assets/core.js'),require('../public_demo/assets/xlsx.js'),fixture);
    if(process.argv[2])fs.writeFileSync(process.argv[2],tested.workbook);
    console.log('[DONE] Public demo pure-JS checks:',tested.checks);
  }
})(globalThis);
