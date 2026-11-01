/* Only registered interface text is translated. Never scan user data or logs. */
const I18n = (() => {
  const catalog = JSON.parse(document.getElementById('ui-translations').textContent);
  const staticText = [], staticAttributes = [], bindings = new Map();
  let language = 'en';
  try { language = localStorage.getItem('adualnet_language') === 'zh' ? 'zh' : 'en'; } catch (_) {}
  function tr(source, params = {}) {
    const template = language === 'en' ? (catalog[source] ?? source) : source;
    return template.replace(/\{(\w+)\}/g, (match, key) => Object.hasOwn(params, key) ? String(params[key]) : match);
  }
  function bind(element, source, params = {}, attribute = null) {
    if (typeof element === 'string') element = document.getElementById(element);
    let fields = bindings.get(element);
    if (!fields) bindings.set(element, fields = new Map());
    fields.set(attribute, {source, params});
    const value = tr(source, params);
    if (attribute) element.setAttribute(attribute, value);
    else element.textContent = value;
  }
  // Capture initial static nodes only: input values, transcripts, notes, code
  // and later model logs are never registered for automatic translation.
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (node.parentElement.closest('script,style,textarea,input,pre,code,[data-i18n-ignore]')) continue;
    const source = node.textContent.trim();
    if (Object.hasOwn(catalog, source)) staticText.push({node, source});
  }
  for (const element of document.querySelectorAll('[placeholder],[aria-label],[title]')) {
    for (const attribute of ['placeholder', 'aria-label', 'title']) {
      const source = element.getAttribute(attribute);
      if (source && Object.hasOwn(catalog, source)) staticAttributes.push({element, attribute, source});
    }
  }
  function setLanguage(next) {
    language = next === 'zh' ? 'zh' : 'en';
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    try { localStorage.setItem('adualnet_language', language); } catch (_) {}
    for (const {node, source} of staticText) node.textContent = tr(source);
    for (const {element, attribute, source} of staticAttributes) element.setAttribute(attribute, tr(source));
    for (const [element, fields] of bindings) {
      if (!element.isConnected) { bindings.delete(element); continue; }
      for (const [attribute, {source, params}] of fields) {
        if (attribute) element.setAttribute(attribute, tr(source, params));
        else element.textContent = tr(source, params);
      }
    }
    for (const button of document.querySelectorAll('[data-language]')) button.setAttribute('aria-pressed', String(button.dataset.language === language));
    document.dispatchEvent(new CustomEvent('adualnet:languagechange'));
  }
  for (const button of document.querySelectorAll('[data-language]')) button.addEventListener('click', () => setLanguage(button.dataset.language));
  setLanguage(language);
  return {tr, bind, setLanguage, get language() { return language; }};
})();
