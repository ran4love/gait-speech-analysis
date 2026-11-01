/* Pure, browser-local review calculations. No network, model or cohort access. */
((root) => {
  const gaitNames = ['average_step_length','average_stride_length','step_length_variability','stride_length_variability','average_step_time','average_stride_time','step_time_variability','stride_time_variability','average_step_assymetry','average_stride_assymetry','walking_speed','cadence','stance_times','swing_times','double_support_times'];
  const speechNames = ['Timecount_DT','Num_DT','Correct_Num_DT','Num_20s_DT','Correct_Num_20s_DT','Correct_persec_DT','ACC_DT','Timecount_DT_1stNum'];
  function validate(review, duration) {
    if (!Number.isFinite(review.start) || !Number.isFinite(review.end) || review.start < 0 || review.end > duration || review.end <= review.start) throw Error('window');
    if (!review.reviewer.trim() || !review.rule.trim()) throw Error('metadata');
    if (!review.gaitConfirmed || !review.speechConfirmed) throw Error('confirm');
    const seen = new Set();
    for (const row of review.responses) {
      if (!row.include) continue;
      if (!Number.isFinite(row.onset) || row.onset < review.start || row.onset >= review.end) throw Error('onset');
      if (!row.text.trim()) throw Error('text');
      if (row.correct !== 0 && row.correct !== 1) throw Error('correctness');
      const key = JSON.stringify([row.onset, row.text.trim()]);
      if (seen.has(key)) throw Error('duplicate');
      seen.add(key);
    }
  }
  function speech(review) {
    const rows = review.responses.filter(r => r.include).slice().sort((a,b) => a.onset-b.onset);
    const duration = review.end-review.start, early = rows.filter(r => r.onset < Math.min(review.end,review.start+20));
    const correct = rows.reduce((s,r) => s+r.correct,0);
    return {Timecount_DT:duration,Num_DT:rows.length,Correct_Num_DT:correct,Num_20s_DT:early.length,
      Correct_Num_20s_DT:early.reduce((s,r) => s+r.correct,0),Correct_persec_DT:correct/duration,
      ACC_DT:rows.length ? correct/rows.length : null,Timecount_DT_1stNum:rows.length ? rows[0].onset-review.start : null};
  }
  function result(fixture, review, revision) {
    validate(review, fixture.duration_s);
    const rows = gaitNames.map(feature => ({domain:'gait',feature,value:fixture.gait_reference[feature] ?? null,
      unit:feature.includes('length')?'image pixels':feature.includes('assymetry')?'%':feature==='cadence'?'steps/min':feature==='walking_speed'?'m/s (not estimated)':'s',
      source:'precomputed fixed synthetic gait reference; not recomputed online',
      status:['stance_times','swing_times','double_support_times'].includes(feature)?'not provided: no validated contact-phase definition':fixture.gait_reference[feature] == null?'missing or QC-failed reference':'experimental algorithm estimate'}));
    const live = speech(review);
    for (const feature of speechNames) rows.push({domain:'speech',feature,value:live[feature],
      unit:feature.startsWith('Timecount')?'s':feature==='ACC_DT'?'proportion':feature==='Correct_persec_DT'?'correct/s':'count',
      source:'computed in this browser from reviewed response annotations',status:live[feature] == null?'missing: no included responses':'human-reviewed annotation'});
    return {fixture_id:fixture.id,synthetic:true,revision,generated_at:new Date().toISOString(),
      features:rows,review:JSON.parse(JSON.stringify(review)),gait_qc:fixture.gait_reference,
      provenance:{...fixture.provenance,fixture_version:fixture.version,client_only:true,real_uploads_supported:false,
        neural_inference_performed:false,reviewer_confirmed:true,scoring_rule:review.rule,
        missing_values_are_not_imputed:true,features_sent_to_classifier:false}};
  }
  function csv(rows) {
    const cell = value => {
      if (value == null) return '';
      let text = String(value);
      if (typeof value === 'string' && /^[\s]*[=+@-]/.test(text)) text="'"+text;
      return '"'+text.replaceAll('"','""')+'"';
    };
    return '\ufeff'+rows.map(row => row.map(cell).join(',')).join('\r\n');
  }
  root.DemoCore = {gaitNames,speechNames,validate,speech,result,csv};
  if (typeof module !== 'undefined') module.exports = root.DemoCore;
})(globalThis);
