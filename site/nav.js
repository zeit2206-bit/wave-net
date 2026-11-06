// Shared navigation: side menu with every page, and previous/next links at the bottom.
// Each page sets <body data-page="file.html">.
const PAGES = [
  ["index.html",      "Home: the whole journey"],
  ["ideas.html",      "1. The ideas we started from"],
  ["light.html",      "2. A network made of light"],
  ["compare.html",    "3. Light vs. a standard network"],
  ["hybrid.html",     "4. Light + camera"],
  ["gates.html",      "5. Using the computer's own unit: logic gates"],
  ["speed.html",      "6. Making gates fast"],
  ["bigger.html",     "7. A bigger gate network + the factory tricks"],
  ["fairness.html",   "8. Fairness check and verdict"],
  ["ternary.html",    "9. The middle ground: -1, 0, +1"],
  ["llama.html",      "10. A real model: Llama 3.2 1B"],
  ["llama-bits.html", "11. Llama with fewer bits"],
  ["llama-ternary.html", "12. Why Llama breaks as ternary"],
  ["llama-smaller.html", "13. Our own smaller Llama"],
  ["lessons.html",    "14. What we learned"],
];
(() => {
  const here = document.body.dataset.page, i = PAGES.findIndex(p => p[0] === here);
  const side = document.createElement("nav");
  side.className = "side"; side.id = "side"; side.setAttribute("aria-label", "All pages");
  side.innerHTML = `<a class="side-title" href="index.html">Wave-net<br>lab notebook</a><ol>${
    PAGES.map(([f, t]) => `<li><a href="${f}"${f === here ? ' class="here" aria-current="page"' : ""}>${t}</a></li>`).join("")}</ol>`;
  const btn = document.createElement("button");
  btn.className = "side-btn"; btn.textContent = "☰ Pages";
  btn.setAttribute("aria-controls", "side"); btn.setAttribute("aria-expanded", "false");
  const scrim = document.createElement("div"); scrim.className = "scrim";
  const toggle = open => { document.body.classList.toggle("side-open", open); btn.setAttribute("aria-expanded", open); };
  btn.addEventListener("click", () => toggle(!document.body.classList.contains("side-open")));
  scrim.addEventListener("click", () => toggle(false));
  document.addEventListener("keydown", e => { if (e.key === "Escape") toggle(false); });
  document.body.prepend(side, btn, scrim);
  const wrap = document.querySelector(".wrap");
  const pager = document.createElement("nav");
  pager.className = "pager";
  const prev = PAGES[i - 1], next = PAGES[i + 1];
  pager.innerHTML = (prev ? `<a href="${prev[0]}"><span class="label">← Previous</span><span class="t">${prev[1]}</span></a>` : "")
    + (next ? `<a class="next" href="${next[0]}"><span class="label">Next →</span><span class="t">${next[1]}</span></a>` : "");
  wrap.append(pager);
})();
