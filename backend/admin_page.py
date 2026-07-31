"""
The /admin query-log viewer — one self-contained HTML page, no build step.

Served by api.py as an HTMLResponse. Deliberately plain (vanilla JS, system
fonts, no CDN): it's an internal tool for reading the query log, not part of the
product UI, and keeping it dependency-free means it survives on the backend
container with nothing but FastAPI.

Auth: the page itself is public, but every API call it makes carries the
X-Admin-Token header (prompted for once, kept in localStorage). A 401 clears the
stored token and re-prompts. See `_require_admin` in api.py.
"""

ADMIN_HTML = r"""
<title>GeorgeBot — query log</title>
<style>
  :root { color-scheme: light dark; --bd:#d8d8de; --mut:#6b6b76; --bg:#fff;
          --card:#fafafa; --acc:#6d3fd4; }
  @media (prefers-color-scheme: dark) {
    :root { --bd:#33333c; --mut:#9a9aa6; --bg:#131316; --card:#1b1b20; --acc:#a98bf5; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); font:14px/1.5 -apple-system,BlinkMacSystemFont,
         "Segoe UI",Roboto,sans-serif; }
  header { display:flex; align-items:center; gap:12px; padding:12px 20px;
           border-bottom:1px solid var(--bd); flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:600; }
  .stats { display:flex; gap:18px; margin-left:auto; flex-wrap:wrap; }
  .stat b { display:block; font-size:16px; }
  .stat span { color:var(--mut); font-size:11px; text-transform:uppercase;
               letter-spacing:.04em; }
  .stat.warn b { color:#c2410c; }
  nav { display:flex; gap:8px; padding:12px 20px; align-items:center; flex-wrap:wrap; }
  button, select, input { font:inherit; padding:5px 10px; border:1px solid var(--bd);
           border-radius:6px; background:var(--card); color:inherit; }
  button { cursor:pointer; }
  button.on { background:var(--acc); color:#fff; border-color:var(--acc); }
  input[type=search] { min-width:240px; }
  main { padding:0 20px 60px; }
  table { border-collapse:collapse; width:100%; }
  th { text-align:left; font-size:11px; text-transform:uppercase; color:var(--mut);
       letter-spacing:.04em; border-bottom:1px solid var(--bd); padding:8px 10px; }
  td { border-bottom:1px solid var(--bd); padding:9px 10px; vertical-align:top; }
  tr.click { cursor:pointer; }
  tr.click:hover td { background:var(--card); }
  .mut { color:var(--mut); }
  .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }
  .pill { display:inline-block; padding:1px 7px; border-radius:999px; font-size:11px;
          border:1px solid var(--bd); margin:0 4px 4px 0; white-space:nowrap; }
  .pill.err { border-color:#dc2626; color:#dc2626; }
  .pill.warn { border-color:#c2410c; color:#c2410c; }
  .turn { border:1px solid var(--bd); border-radius:8px; padding:14px 16px;
          margin:12px 0; background:var(--card); }
  .turn .q { font-weight:600; }
  .turn .a { white-space:pre-wrap; margin-top:8px; }
  .lbl { font-size:11px; text-transform:uppercase; color:var(--mut);
         letter-spacing:.04em; margin-top:10px; }
  .empty { color:var(--mut); padding:40px 0; text-align:center; }
  a { color:var(--acc); }
</style>

<header>
  <h1>GeorgeBot — query log</h1>
  <div class="stats" id="stats"></div>
</header>

<nav>
  <button id="tab-sessions" class="on">Sessions</button>
  <button id="tab-turns">Turns</button>
  <input type="search" id="q" placeholder="Search questions & answers…">
  <select id="status">
    <option value="">any status</option>
    <option value="ok">ok</option>
    <option value="error">error</option>
    <option value="abandoned">abandoned</option>
    <option value="started">incomplete</option>
  </select>
  <label class="mut"><input type="checkbox" id="zero"> no sources only</label>
  <button id="reload">Reload</button>
  <a id="csv" href="#" download>Download CSV</a>
  <button id="forget" style="margin-left:auto">Forget token</button>
</nav>

<main id="main"><div class="empty">Loading…</div></main>

<script>
const $ = (id) => document.getElementById(id);
let token = localStorage.getItem("gb_admin_token") || "";
let view = "sessions";

function askToken() {
  const t = prompt("Admin token");
  if (t === null) return false;
  token = t.trim();
  localStorage.setItem("gb_admin_token", token);
  return true;
}

async function api(path) {
  if (!token && !askToken()) throw new Error("no token");
  const res = await fetch("/api/admin/" + path, { headers: { "X-Admin-Token": token } });
  if (res.status === 401) {
    localStorage.removeItem("gb_admin_token"); token = "";
    if (askToken()) return api(path);
    throw new Error("unauthorized");
  }
  if (res.status === 503) throw new Error("ADMIN_TOKEN is not set on the server");
  if (!res.ok) throw new Error("HTTP " + res.status);
  return res.json();
}

const esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const short = (s, n) => { s = String(s ?? ""); return s.length > n ? s.slice(0, n) + "…" : s; };
const when = (ts) => ts ? new Date(ts).toLocaleString() : "";

function filters() {
  const p = new URLSearchParams();
  if ($("q").value.trim()) p.set("q", $("q").value.trim());
  if ($("status").value) p.set("status", $("status").value);
  if ($("zero").checked) p.set("zero_sources", "1");
  return p;
}

function routePills(t) {
  let r = {};
  try { r = JSON.parse(t.route_json || "{}") || {}; } catch (e) {}
  const out = [];
  if (t.audience) out.push(["", t.audience]);
  if (t.mode) out.push(["", t.mode]);
  if (r.department) out.push(["", r.department]);
  (r.course_codes || []).forEach((c) => out.push(["", c]));
  if (r.program_query) out.push(["", "program: " + r.program_query]);
  if (r.instructor_query) out.push(["", "instr: " + r.instructor_query]);
  if (r.professor_query) out.push(["", "prof: " + r.professor_query]);
  (r.topic_families || []).forEach((f) => out.push(["", f]));
  // "started" = written on arrival and never completed: the client is still
  // streaming, or disconnected (sweep_stale relabels it "abandoned" later).
  if (t.status === "ok" && !t.n_sources) out.push(["warn", "no sources"]);
  if (t.status !== "ok") out.push(["err", t.status === "started" ? "incomplete" : t.status]);
  if (t.latency_ms) out.push(["", (t.latency_ms / 1000).toFixed(1) + "s"]);
  return out.map(([c, x]) => `<span class="pill ${c}">${esc(x)}</span>`).join("");
}

async function loadStats() {
  const s = await api("stats");
  $("stats").innerHTML = [
    `<div class="stat"><b>${s.n_turns || 0}</b><span>turns</span></div>`,
    `<div class="stat"><b>${s.n_sessions || 0}</b><span>sessions</span></div>`,
    `<div class="stat warn"><b>${s.n_no_sources || 0}</b><span>no sources</span></div>`,
    `<div class="stat"><b>${(s.n_error || 0) + (s.n_abandoned || 0)}</b><span>err/aband</span></div>`,
  ].join("");
}

async function loadSessions() {
  const rows = await api("sessions?limit=200");
  $("main").innerHTML = rows.length ? `<table><thead><tr>
      <th>Last active</th><th>First question</th><th>Turns</th><th>Flags</th></tr></thead>
    <tbody>${rows.map((r) => `<tr class="click" data-s="${r.session_id}">
      <td class="mut mono">${when(r.ended)}</td>
      <td>${esc(short(r.first_question, 110))}</td>
      <td>${r.n_turns}</td>
      <td>${r.n_no_sources ? `<span class="pill warn">${r.n_no_sources} no-source</span>` : ""}
          ${r.n_problem ? `<span class="pill err">${r.n_problem} err</span>` : ""}</td>
    </tr>`).join("")}</tbody></table>`
    : `<div class="empty">No sessions logged yet.</div>`;
  $("main").querySelectorAll("tr.click").forEach((tr) =>
    tr.onclick = () => loadTranscript(tr.dataset.s));
}

async function loadTranscript(sid) {
  const rows = await api("sessions/" + encodeURIComponent(sid));
  $("main").innerHTML = `<p><button id="back">← Sessions</button>
      <span class="mut mono" style="margin-left:10px">${esc(sid)}</span></p>` +
    rows.map((t) => `<div class="turn">
      <div class="mut mono">${when(t.ts)} · turn ${t.turn_index}</div>
      <div class="q">${esc(t.question)}</div>
      <div>${routePills(t)}</div>
      <div class="lbl">Rewritten query</div>
      <div class="mono">${esc(t.search_query) || "<span class=mut>—</span>"}</div>
      <div class="lbl">Answer</div>
      <div class="a">${esc(t.answer) || "<span class=mut>—</span>"}</div>
      ${t.error ? `<div class="lbl">Error</div><div class="mono">${esc(t.error)}</div>` : ""}
      ${sourceList(t)}
    </div>`).join("");
  $("back").onclick = loadSessions;
}

function sourceList(t) {
  let src = [];
  try { src = JSON.parse(t.source_titles || "[]") || []; } catch (e) {}
  if (!src.length) return "";
  return `<div class="lbl">Cited sources</div><div class="mut">` + src.map((s) =>
    `<div>${esc(s.source)} · ${s.url ? `<a href="${esc(s.url)}" target="_blank"
       rel="noreferrer">${esc(short(s.title || s.url, 90))}</a>`
      : esc(short(s.title, 90))}</div>`).join("") + `</div>`;
}

async function loadTurns() {
  const p = filters(); p.set("limit", "300");
  const rows = await api("logs?" + p.toString());
  $("main").innerHTML = rows.length ? `<table><thead><tr>
      <th>When</th><th>Question</th><th>Rewritten</th><th>Answer</th><th></th></tr></thead>
    <tbody>${rows.map((t) => `<tr class="click" data-s="${t.session_id}">
      <td class="mut mono">${when(t.ts)}</td>
      <td>${esc(short(t.question, 90))}</td>
      <td class="mono mut">${esc(short(t.search_query, 70))}</td>
      <td class="mut">${esc(short(t.answer, 110))}</td>
      <td>${routePills(t)}</td>
    </tr>`).join("")}</tbody></table>`
    : `<div class="empty">Nothing matches.</div>`;
  $("main").querySelectorAll("tr.click").forEach((tr) =>
    tr.onclick = () => loadTranscript(tr.dataset.s));
}

async function render() {
  $("main").innerHTML = `<div class="empty">Loading…</div>`;
  try {
    await loadStats();
    await (view === "sessions" ? loadSessions() : loadTurns());
    const p = filters(); p.set("token", token);
    $("csv").href = "/api/admin/logs.csv?" + p.toString();
  } catch (e) {
    $("main").innerHTML = `<div class="empty">${esc(e.message)}</div>`;
  }
}

$("tab-sessions").onclick = () => { view = "sessions";
  $("tab-sessions").classList.add("on"); $("tab-turns").classList.remove("on"); render(); };
$("tab-turns").onclick = () => { view = "turns";
  $("tab-turns").classList.add("on"); $("tab-sessions").classList.remove("on"); render(); };
$("reload").onclick = render;
$("status").onchange = () => { view = "turns"; $("tab-turns").click(); };
$("zero").onchange = () => { view = "turns"; $("tab-turns").click(); };
$("q").onkeydown = (e) => { if (e.key === "Enter") { view = "turns"; $("tab-turns").click(); } };
$("forget").onclick = () => { localStorage.removeItem("gb_admin_token");
  token = ""; render(); };

render();
</script>
"""
