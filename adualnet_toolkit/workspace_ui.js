/* Presentation state follows existing jobs; it never fits, scores or edits data. */
const Workbench = (() => {
  const stages = {idle:0,running:1,awaiting_review:2,reviewed:3,features_ready:4,failed:-1};
  const labels = {idle:'等待上传',running:'正在处理',awaiting_review:'待人工核查',reviewed:'待导出特征',features_ready:'特征已生成',failed:'处理未完成'};
  const statuses = document.querySelectorAll('[data-workspace-stage]');
  const jump = document.getElementById('jump-review');
  let current = 'idle';
  function update(next) {
    current = Object.hasOwn(stages,next) ? next : 'idle';
    const phase = stages[current];
    for (const status of statuses) {
      status.dataset.state = current;
      I18n.bind(status,labels[current]);
    }
    jump.disabled = phase < 2;
    for (const [index, step] of [...document.querySelectorAll('[data-workflow-step]')].entries()) {
      const active = index === phase;
      step.dataset.progress = index < phase ? 'complete' : active ? 'current' : 'pending';
      if (active) step.setAttribute('aria-current','step');
      else step.removeAttribute('aria-current');
    }
  }
  jump.addEventListener('click',()=>{
    document.querySelector('[data-tab="video-panel"]').click();
    document.getElementById('video-review').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth',block:'start'});
  });
  document.querySelectorAll('[data-tab]').forEach(button=>button.addEventListener('click',()=>{
    const titles = {'video-panel':'步态与语音分析',demo:'合成模型演示',data:'队列特征分析',raw:'高级流程与配置'};
    I18n.bind('workspace-title',titles[button.dataset.tab]);
    const videoMode = button.dataset.tab === 'video-panel';
    document.querySelector('.heading-actions').hidden = !videoMode;
    document.querySelector('.protocol-strip').hidden = !videoMode;
  }));
  update(current);
  return {update};
})();
