// Turn results.json into report.html: one section per test, the three models side by side. Fully local.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DIR = path.dirname(fileURLToPath(import.meta.url));
const results = JSON.parse(fs.readFileSync(path.join(DIR, 'results.json'), 'utf8'));
const NAMES = { 'claude:sonnet': 'Claude Sonnet (what the apps use)', 'ollama:llama3.2:1b': 'Llama 1B, 8-bit (download)', 'ollama:llama3.2-1b-gptq-q4': 'Llama 1B, our 4-bit' };
const esc = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

function show(v) {                                    // readable view of any JSON answer
  if (v === null) return '<i class="muted">null</i>';
  if (Array.isArray(v)) return v.length ? `<ol>${v.map(x => `<li>${show(x)}</li>`).join('')}</ol>` : '<i class="muted">(empty)</i>';
  if (typeof v === 'object') return `<dl>${Object.entries(v).map(([k, x]) => `<dt>${esc(k)}</dt><dd>${show(x)}</dd>`).join('')}</dl>`;
  return esc(v).replace(/\n/g, '<br>');
}

const cases = [...new Set(results.map(r => r.case))];
const body = cases.map(id => {
  const rs = results.filter(r => r.case === id); const r0 = rs[0];
  return `<section class="case"><span class="label">${esc(r0.app)}</span><h2>${esc(r0.what)}</h2>
  <details><summary>What was sent (besides the app's instructions)</summary><pre>${esc(r0.input)}</pre></details>
  <div class="cols">${rs.map(r => `<div class="col"><h3>${esc(NAMES[r.model] || r.model)}</h3><p class="t">${r.secs.toFixed(1)} s</p>
    ${r.error ? `<p class="err">Error: ${esc(r.error.slice(0, 300))}</p>` : show(r.out)}</div>`).join('')}</div></section>`;
}).join('\n');

fs.writeFileSync(path.join(DIR, 'report.html'), `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Claude vs Local Llama</title><style>
:root{color-scheme:light;--ink:#1f2a33;--muted:#5b6863;--line:#d3dad4;--paper:#f3f5f1}
body{margin:0;padding:0 18px;background:var(--paper);color:var(--ink);font:15px/1.55 Georgia,serif}
.wrap{max-width:84rem;margin:0 auto;padding-block:1.5rem 4rem;display:grid;gap:1.6rem}
h1,h2,h3{font-family:"Segoe UI",system-ui,sans-serif;margin:0} h2{font-size:1.2rem} h3{font-size:.95rem}
.label{font:600 .72rem "Segoe UI",sans-serif;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.case{display:grid;gap:.6rem}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(17rem,1fr));gap:.8rem}
.col{background:#fff;border:1px solid var(--line);padding:.8rem 1rem;overflow-wrap:anywhere}
.col:first-child{border-top:4px solid #1e5a7a}.col:not(:first-child){border-top:4px solid #b26a12}
.t{margin:.1rem 0 .5rem;font:.8rem Consolas,monospace;color:var(--muted)}
dl{margin:0}dt{font:600 .75rem Consolas,monospace;color:var(--muted);margin-top:.4rem}dd{margin:0 0 0 .6rem}
ol{margin:.2rem 0;padding-left:1.2rem}.muted{color:var(--muted)}.err{color:#a33d6b}
pre{white-space:pre-wrap;background:#fff;border:1px solid var(--line);padding:.6rem;font:.82rem Consolas,monospace}
</style></head><body><main class="wrap"><header><span class="label">wave-net · app-compare</span><h1>Claude vs local Llama on Speak Up and Write Up</h1>
<p>Same instructions and same JSON format the apps use, sent to each model. Blue = what the apps use today; amber = local models in Ollama.</p></header>
${body}</main></body></html>`);
console.log('wrote report.html');
