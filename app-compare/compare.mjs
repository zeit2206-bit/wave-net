// Compare answers of Claude (as the apps call it) and local Ollama models, on the apps' own prompts and JSON schemas.
// The apps are only READ: their prompt texts and schemas are extracted from their server files at run time.
// Usage: node compare.mjs   -> results.json (+ report.html via report.mjs)
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(DIR, '..', '..');
const WORK = path.join(DIR, 'work'); fs.mkdirSync(WORK, { recursive: true });   // empty cwd for claude -p
const MODELS = ['claude:sonnet', 'ollama:llama3.2:1b', 'ollama:llama3.2-1b-gptq-q4'];

// Pull `const NAME = ...;` declarations out of a server file and evaluate them (template strings and object literals only).
function extract(file, names) {
  const src = fs.readFileSync(file, 'utf8'); const decls = [];
  for (const n of names) {
    const i = src.indexOf(`const ${n} = `); if (i < 0) throw new Error(`${n} not found in ${file}`);
    let j = i + `const ${n} = `.length, depth = 0, inStr = false;
    for (; j < src.length; j++) {
      const c = src[j];
      if (c === '`') inStr = !inStr;
      if (inStr) continue;
      if ('({['.includes(c)) depth++;
      if (')}]'.includes(c)) depth--;
      if (c === ';' && depth === 0) break;
    }
    decls.push(src.slice(i, j + 1));
  }
  return new Function(decls.join('\n') + `\nreturn {${names.join(',')}};`)();
}
const S = extract(path.join(ROOT, 'speak-up/server/server.mjs'), ['SYSTEM', 'EXAM', 'SCHEMA']);
const W = extract(path.join(ROOT, 'write-up/server.mjs'), ['COACH', 'COACH_SCHEMA', 'DRILL', 'DRILL_SCHEMA', 'PROMPT', 'PROMPT_SCHEMA', 'DRILLCHECK', 'DRILLCHECK_SCHEMA']);
const SPEAK_SCHEMA = JSON.parse(S.SCHEMA);

// Speak Up's user prompt, as in its prompt() for the "answer"/"start" modes.
const speak = (topic, level, history, answer) => [`Topic: ${topic}`, `Learner level: ${level}`, '', 'Conversation so far:',
  ...(history.length ? history.map(m => `${m.role === 'app' ? 'You' : 'Learner'}: ${m.text}`) : ['(nothing yet)']), '',
  answer ? `The learner answered in their own words: "${answer}"` : 'Start the conversation: greet briefly and ask a first question about the topic.'].join('\n');
const exam = (history, answer) => ['Part 2 cue card for this test: Describe something you own which is very important to you.', '', 'Test so far:',
  ...history.map(m => `${m.role === 'app' ? 'Examiner' : 'Candidate'}: ${m.text}`), '', `The candidate answered in their own words: "${answer}"`].join('\n');

// Drill input built from the learner's real saved mistakes (read only), like write-up's /api/drill.
const mistakes = JSON.parse(fs.readFileSync(path.join(ROOT, 'write-up/data/mistakes.json'), 'utf8'));
const top = Object.entries(mistakes).filter(([l]) => !/punctuation|comma|period|full stop|semicolon|capital|apostrophe/i.test(l))
  .sort((a, b) => b[1].count - a[1].count).slice(0, 4);
const drillDesc = top.map(([l, e]) => `- ${l} (${e.count}x), e.g. ${e.examples.slice(0, 2).map(x => `"${x.from}" → "${x.to}"`).join('; ')}`).join('\n');

const CASES = [
  { id: 'speak-start', app: 'Speak Up', what: 'Start a conversation (B1, travel)', system: S.SYSTEM, schema: SPEAK_SCHEMA,
    text: speak('Travel', 'B1', [], null) },
  { id: 'speak-b1', app: 'Speak Up', what: 'Reply + rephrase a B1 answer with mistakes', system: S.SYSTEM, schema: SPEAK_SCHEMA,
    text: speak('Travel', 'B1', [{ role: 'app', text: 'Hi! What was the last trip you really enjoyed?' }],
      'Last year I go to Cartagena with my family and we was very happy because the beach is very beautiful and the food also.') },
  { id: 'speak-a2', app: 'Speak Up', what: 'Reply + rephrase an A2 answer', system: S.SYSTEM, schema: SPEAK_SCHEMA,
    text: speak('Food', 'A2', [{ role: 'app', text: 'Hello! What do you like to cook?' }], 'I like cook pasta in the weekends, is easy for me.') },
  { id: 'speak-ielts', app: 'Speak Up', what: 'IELTS examiner, Part 1 answer', system: S.EXAM, schema: SPEAK_SCHEMA,
    text: exam([{ role: 'app', text: "Good morning. Let's talk about your home town. Where is your home town?" }],
      'My home town is Valencia, is a very big city in the east of Spain, it have many people and the weather is cold.') },
  { id: 'write-check-b1', app: 'Write Up', what: 'Correct a B1 text', system: W.COACH, schema: W.COACH_SCHEMA,
    text: 'Learner level: B1\nExplanation language: English\n\nLearner\'s text:\nYesterday I have gone to the supermarket for buy some things. When I arrived, I realised that I forget my wallet in home. I felt very embarrassing, so I returned to my house and I make the shopping later.' },
  { id: 'write-check-b2', app: 'Write Up', what: 'Correct a B2 opinion text', system: W.COACH, schema: W.COACH_SCHEMA,
    text: 'Learner level: B2\nExplanation language: English\n\nLearner\'s text:\nIn my opinion, working from home have more advantages than disadvantages. People can save the time they spend in the traffic, and they can organize better their day. However, some persons feel alone and it is difficult to separate the work and the personal life.' },
  { id: 'write-prompt', app: 'Write Up', what: 'Make a writing task + help (B1, work)', system: W.PROMPT, schema: W.PROMPT_SCHEMA,
    text: 'Learner level: B1\nTopic wish: work\n\nRecent questions:\n(none)' },
  { id: 'write-drill', app: 'Write Up', what: 'Drill from your real saved mistakes', system: W.DRILL, schema: W.DRILL_SCHEMA,
    text: `Learner level: B1\nExplanation language: English\n\nFrequent mistakes:\n${drillDesc}` },
  { id: 'write-drillcheck', app: 'Write Up', what: 'Grade a drill answer (a valid alternative fix)', system: W.DRILLCHECK, schema: W.DRILLCHECK_SCHEMA,
    text: 'Sentence with the mistake: I have finished the report yesterday.\nMistake type: present perfect vs past\nExpected fix: I finished the report yesterday.\nLearner\'s version: I completed the report yesterday.' },
];

function claude(model, c) {
  return new Promise((resolve, reject) => {
    const args = ['-p', '--model', model, '--tools', '', '--setting-sources', '', '--strict-mcp-config',
      '--no-session-persistence', '--output-format', 'json', '--system-prompt', c.system, '--json-schema', JSON.stringify(c.schema)];
    const p = spawn('claude', args, { cwd: WORK, shell: false, windowsHide: true });
    let out = '';
    const timer = setTimeout(() => { p.kill(); reject(new Error('timeout 120 s')); }, 120000);
    p.stdout.on('data', d => out += d);
    p.on('close', () => {
      clearTimeout(timer);
      try {
        const env = JSON.parse(out);
        if (env.is_error) return reject(new Error(env.result));
        resolve(env.structured_output || JSON.parse(String(env.result).replace(/^```(json)?|```$/g, '').trim()));
      } catch (e) { reject(new Error(out.slice(0, 300) || e.message)); }
    });
    p.stdin.end(c.text);
  });
}

async function ollama(model, c) {
  const r = await fetch('http://localhost:11434/api/chat', { method: 'POST', body: JSON.stringify({
    model, stream: false, format: c.schema, options: { num_ctx: 8192 },
    messages: [{ role: 'system', content: c.system }, { role: 'user', content: c.text }] }) });
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return JSON.parse(j.message.content);
}

const results = [];
for (const c of CASES) for (const m of MODELS) {
  const [kind, ...rest] = m.split(':'); const name = rest.join(':'); const t = Date.now();
  let out, error = null;
  try { out = await (kind === 'claude' ? claude(name, c) : ollama(name, c)); } catch (e) { error = e.message; }
  const secs = (Date.now() - t) / 1000;
  results.push({ case: c.id, app: c.app, what: c.what, input: c.text, model: m, secs, out, error });
  console.log(`${c.id.padEnd(18)} ${m.padEnd(28)} ${secs.toFixed(1).padStart(6)} s ${error ? 'ERROR ' + error.slice(0, 80) : 'ok'}`);
  fs.writeFileSync(path.join(DIR, 'results.json'), JSON.stringify(results, null, 2));
}
