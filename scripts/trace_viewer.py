#!/usr/bin/env python3
"""
Interactive trace viewer for MARLLLM experiments.

Serves a web UI that lets you browse experiments and three kinds of data:

  * Training rollouts  (traces/iter_*.json)
  * Checkpoint evals   (traces/ckpt_*/traces.json)
  * Behaviour snapshots(snapshots/iter_*.json)

Training curves are drawn alongside, with resizable charts and a hover
crosshair that re-uses cached chart backgrounds (no resort on every move).

Usage
-----
    python scripts/trace_viewer.py [output_root] [--port PORT] [--host HOST]

    output_root  : directory containing experiment sub-directories
                   (default: ./runs/cultural_emergence)
    --port       : port to listen on (default: 8765)
    --host       : host to bind (default: 0.0.0.0)

SSH port-forward
----------------
    On Isambard:  python scripts/trace_viewer.py --port 8765
    On laptop:    ssh -L 8765:localhost:8765 <isambard-login>
    Browser:      http://localhost:8765
"""
from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_ASSISTANT_TURN_RE = re.compile(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.DOTALL)


def _extract_assistant_turns(context_text: str) -> list[str]:
    """Return the full raw assistant output for each turn in this agent's context.

    With the new ``<post>``/``<python>`` format, anything outside the tagged
    blocks is private thinking — but rather than try to slice it out, we
    surface the entire generated turn so reviewers can see exactly what the
    model wrote (thinking, post, code, and any malformed/cut-off output).
    """
    return [t.strip() for t in _ASSISTANT_TURN_RE.findall(context_text or "")]


def _convert_forum_records(records: list, label: str) -> dict:
    """List-of-episode-records (forum env) → viewer payload."""
    episodes = []
    all_agents: list[str] = []
    for rec in records:
        env_t = rec.get("env_trace") or {}
        agents_raw = rec.get("agents") or {}
        ep_agents = list(env_t.get("agents") or list(agents_raw.keys()))
        for a in ep_agents:
            if a not in all_agents:
                all_agents.append(a)
        contexts: dict[str, str] = {}
        turns_by_agent: dict[str, list[str]] = {}
        for aid, at in agents_raw.items():
            ctx_text = at.get("context_text", "") if isinstance(at, dict) else str(at)
            contexts[aid] = ctx_text
            turns_by_agent[aid] = _extract_assistant_turns(ctx_text)
        thread = list(env_t.get("thread") or [])
        speaker_post_counts: dict[str, int] = {}
        for post in thread:
            # pybot/tool-output posts aren't agent turns — skip the lookup.
            if post.get("kind") == "tool_output":
                continue
            speaker = post.get("speaker")
            n = speaker_post_counts.get(speaker, 0)
            speaker_post_counts[speaker] = n + 1
            turns = turns_by_agent.get(speaker, [])
            if n < len(turns) and turns[n]:
                post["raw_turn"] = turns[n]
        ep_meta = {k: v for k, v in env_t.items() if k not in ("thread", "agents")}
        episodes.append({
            "episode": rec.get("episode", 0),
            "meta": ep_meta,
            "participating_agents": ep_agents,
            "thread": thread,
            "agent_contexts": contexts,
        })
    return {
        "format": "forum",
        "meta": {"label": label, "episodes": len(episodes)},
        "participating_agents": all_agents,
        "episodes": episodes,
    }


# ── Embedded HTML/JS/CSS ──────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MARLLLM Trace Viewer</title>
<style>
  :root {
    --bg:#1a1a2e; --panel:#16213e; --card:#0f3460; --card2:#0a2040;
    --accent:#e94560; --text:#e0e0e0; --muted:#777; --muted2:#555;
    --obs-bg:#0e1f0e; --act-bg:#1f0e1f; --prompt-bg:#0e0e1f;
    --obs-border:#1e4a1e; --act-border:#4a1e4a; --prompt-border:#1e1e4a;
    --font-mono:'Cascadia Code','Fira Code','Consolas',monospace;
    --font-ui: system-ui,-apple-system,sans-serif;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:var(--font-ui);
         display:flex; height:100vh; overflow:hidden; }

  /* Sidebar */
  #sidebar { width:260px; min-width:200px; background:var(--panel);
             border-right:1px solid #222; display:flex; flex-direction:column; flex-shrink:0; }
  #sidebar-title { padding:12px 14px; font-size:13px; font-weight:700; color:var(--accent);
                   border-bottom:1px solid #222; letter-spacing:1px; }
  #exp-list { flex:1; overflow-y:auto; padding:4px 0; }
  .exp-item { padding:7px 14px; cursor:pointer; font-size:12px; border-left:3px solid transparent;
              white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .exp-item:hover { background:#1e2d4a; }
  .exp-item.active { border-left-color:var(--accent); background:#1e2d4a; color:#fff; }

  #main { flex:1; display:flex; flex-direction:column; overflow:hidden; min-width:0; }

  /* Toolbar */
  #toolbar { background:var(--panel); border-bottom:1px solid #222; padding:6px 12px;
             display:flex; align-items:center; gap:8px; flex-wrap:wrap; flex-shrink:0; }
  #toolbar label { font-size:11px; color:var(--muted); }
  #toolbar select, #toolbar input[type=range] { background:var(--card); border:1px solid #333;
              color:var(--text); padding:3px 7px; border-radius:4px; font-size:12px; }
  #toolbar select { min-width:130px; }
  .tb-sep { width:1px; height:18px; background:#333; }
  .tb-btn { background:var(--card2); border:1px solid #333; color:var(--muted);
            padding:3px 9px; border-radius:4px; cursor:pointer; font-size:11px; white-space:nowrap; }
  .tb-btn:hover { color:var(--text); border-color:#555; }
  .tb-btn.active { background:var(--accent); border-color:var(--accent); color:#fff; }
  .agent-tab { background:var(--card2); border:1px solid #333; color:var(--muted);
               padding:3px 9px; border-radius:4px; cursor:pointer; font-size:11px; }
  .agent-tab.active { border-color:#6a9; color:#6a9; }
  #agent-tabs, #thread-tabs { display:none; gap:5px; flex-wrap:wrap; }
  .tb-right { margin-left:auto; font-size:11px; color:var(--muted); }
  .source-pill { background:var(--card2); border:1px solid #333; color:var(--muted);
                 padding:3px 10px; border-radius:12px; cursor:pointer; font-size:11px; }
  .source-pill.active { background:#1e3a5f; border-color:#4a7ab8; color:#cfe; }

  /* Metrics panel */
  #metrics-panel { display:none; border-bottom:1px solid #222; background:var(--card2); flex-shrink:0; }
  #metrics-inner { display:flex; gap:8px; padding:8px 12px; overflow-x:auto; align-items:flex-start; }
  #metrics-inner::-webkit-scrollbar { height:6px; }
  #metrics-inner::-webkit-scrollbar-thumb { background:#333; border-radius:3px; }

  .chart-wrap { flex-shrink:0; display:flex; flex-direction:column; align-items:center; position:relative; }
  .chart-stack { position:relative; }
  .chart-stack canvas { display:block; border-radius:4px; }
  .chart-stack canvas.bg { position:absolute; left:0; top:0; }
  .chart-stack canvas.fg { position:relative; cursor:crosshair; background:transparent; }
  .chart-legend { display:flex; gap:6px; flex-wrap:wrap; justify-content:center; margin-top:3px; }
  .legend-item { display:flex; align-items:center; gap:3px; font-size:9px; color:var(--muted); white-space:nowrap; }
  .legend-dot { width:8px; height:3px; border-radius:1px; }

  #chart-tooltip { position:fixed; background:#0a1825; border:1px solid #335;
                   border-radius:4px; padding:5px 8px; font-size:11px; font-family:var(--font-mono);
                   pointer-events:none; display:none; z-index:999; max-width:240px; line-height:1.6; }
  .tt-row { display:flex; gap:6px; align-items:center; }
  .tt-dot { width:8px; height:8px; border-radius:50%; }
  .tt-iter { color:var(--muted); margin-bottom:3px; font-size:10px; }

  /* Content */
  #content { flex:1; overflow-y:auto; padding:12px 16px; }
  #placeholder { color:var(--muted); text-align:center; margin-top:80px; font-size:13px; }

  #meta-card { background:var(--card); border-radius:5px; padding:10px 14px;
               margin-bottom:12px; display:flex; flex-wrap:wrap; gap:14px; }
  .meta-item { font-size:11px; }
  .meta-key { color:var(--muted); margin-right:4px; }
  .meta-val { color:#fff; font-weight:600; }

  .section-hdr { font-size:10px; font-weight:700; letter-spacing:1.5px;
                 color:var(--muted); text-transform:uppercase; margin:14px 0 7px; }

  .env-event { margin-bottom:10px; border-radius:5px; overflow:hidden; }
  .env-hdr { padding:5px 10px; font-size:11px; font-family:var(--font-mono); font-weight:600;
             display:flex; align-items:center; gap:8px; }
  .env-body { padding:7px 10px; font-family:var(--font-mono); font-size:12px;
              white-space:pre-wrap; word-break:break-word; }
  .ev-act { background:var(--act-bg); border:1px solid var(--act-border); }
  .ev-act .env-hdr { background:#2a0e2a; color:#bb77bb; }
  .ev-obs-env { background:var(--obs-bg); border:1px solid var(--obs-border); }
  .ev-obs-env .env-hdr { background:#0e2a0e; color:#77bb77; }
  .ev-obs-routed { background:#0e1e2a; border:1px solid #1e4a5a; }
  .ev-obs-routed .env-hdr { background:#0e1e2a; color:#77bbcc; }
  .routing-arrow { color:var(--accent); }
  .prefix-detail { font-size:10px; color:var(--muted); padding:3px 10px 7px; font-family:var(--font-mono); }

  .turn { margin-bottom:7px; border-radius:5px; overflow:hidden; }
  .turn-lbl { padding:2px 9px; font-size:9px; font-family:var(--font-mono);
              font-weight:700; letter-spacing:0.5px; text-transform:uppercase; }
  .turn-body { padding:7px 10px; font-family:var(--font-mono); font-size:12px;
               white-space:pre-wrap; word-break:break-word; line-height:1.5; }
  .turn-prompt { background:var(--prompt-bg); border:1px solid var(--prompt-border); }
  .turn-prompt .turn-lbl { background:#12123a; color:#7777cc; }
  .turn-obs { background:var(--obs-bg); border:1px solid var(--obs-border); }
  .turn-obs .turn-lbl { background:#0e2a0e; color:#77bb77; }
  .turn-act { background:var(--act-bg); border:1px solid var(--act-border); }
  .turn-act .turn-lbl { background:#2a0e2a; color:#bb77bb; }
  #raw-view, .raw { font-family:var(--font-mono); font-size:12px; white-space:pre-wrap;
              word-break:break-word; line-height:1.5; background:var(--card); padding:14px; border-radius:5px; }

  #agents-bar { display:flex; gap:7px; flex-wrap:wrap; margin-bottom:10px; }
  .agent-chip { font-size:11px; padding:2px 9px; border-radius:10px;
                background:var(--card); color:var(--text); display:inline-flex; align-items:center; gap:5px; }
  .agent-chip .agent-swatch { width:9px; height:9px; border-radius:50%; }

  #forum-view { display:flex; flex-direction:column; gap:10px; max-width:980px; }
  .forum-post { border-radius:6px; overflow:hidden; border:1px solid; background:rgba(255,255,255,0.02); }
  .forum-post-hdr { padding:6px 12px; font-size:12px; font-weight:600;
                    display:flex; align-items:center; gap:10px; color:#fff; }
  .forum-post-hdr .post-idx { font-family:var(--font-mono); font-size:10px; opacity:0.7; font-weight:400; }
  .forum-post-body { padding:10px 14px; font-family:var(--font-mono); font-size:13px;
                     white-space:pre-wrap; word-break:break-word; line-height:1.55;
                     color:var(--text); background:rgba(0,0,0,0.25); }
  .forum-think { margin:0; background:rgba(255,255,255,0.03); border-bottom:1px dashed rgba(255,255,255,0.12); }
  .forum-think > summary { padding:4px 12px; cursor:pointer; font-size:10px;
                           font-weight:600; letter-spacing:1px; text-transform:uppercase;
                           color:#aab; list-style:none; user-select:none; }
  .forum-think > summary::-webkit-details-marker { display:none; }
  .forum-think > summary::before { content:'▸ '; display:inline-block; width:1em; color:#889; }
  .forum-think[open] > summary::before { content:'▾ '; }
  .forum-think-body { padding:8px 14px 10px; font-family:var(--font-mono); font-size:12px;
                      white-space:pre-wrap; word-break:break-word; line-height:1.5;
                      color:#99a; font-style:italic; }

  /* Snapshot view */
  #snap-view { display:grid; gap:12px; grid-template-columns:repeat(auto-fill,minmax(420px,1fr)); }
  .snap-agent { background:var(--card); border-radius:6px; overflow:hidden; }
  .snap-agent-hdr { padding:6px 12px; font-size:12px; font-weight:700;
                    background:rgba(255,255,255,0.05); color:#fff; display:flex; align-items:center; gap:10px; }
  .snap-agent-hdr .swatch { width:10px; height:10px; border-radius:50%; }
  .snap-agent-hdr .count { color:var(--muted); font-weight:400; font-size:10px; margin-left:auto; }
  .snap-sample { border-top:1px solid rgba(255,255,255,0.05); padding:8px 12px;
                 font-family:var(--font-mono); font-size:12px; line-height:1.5;
                 white-space:pre-wrap; word-break:break-word; }
  .snap-sample .tok { color:var(--muted); font-size:10px; margin-right:6px; }

  ::-webkit-scrollbar { width:6px; height:6px; }
  ::-webkit-scrollbar-thumb { background:#333; border-radius:3px; }
</style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-title">MARLLLM Traces</div>
  <div id="exp-list"><div style="padding:14px;color:var(--muted);font-size:11px">Loading…</div></div>
</div>

<div id="main">
  <div id="toolbar">
    <span class="source-pill active" id="src-iter"  onclick="setSource('iter')">Training</span>
    <span class="source-pill"        id="src-ckpt"  onclick="setSource('ckpt')">Checkpoints</span>
    <span class="source-pill"        id="src-snap"  onclick="setSource('snap')">Snapshots</span>
    <div class="tb-sep"></div>
    <label>Iter</label>
    <select id="iter-select"><option>—</option></select>
    <span id="ep-wrap" style="display:none"><label>Ep</label>
      <select id="ep-select"></select>
    </span>
    <div class="tb-sep"></div>
    <label>View</label>
    <button class="tb-btn"        id="btn-forum"   onclick="setMode('forum')">Forum</button>
    <button class="tb-btn"        id="btn-env"     onclick="setMode('env')">Env log</button>
    <button class="tb-btn"        id="btn-context" onclick="setMode('context')">Context</button>
    <div id="thread-tabs"></div>
    <div id="agent-tabs"></div>
    <div class="tb-sep"></div>
    <button class="tb-btn" id="btn-metrics" onclick="toggleMetrics()">📈 Curves</button>
    <label>Size</label>
    <input type="range" id="chart-size" min="160" max="520" step="20" value="240"/>
    <div class="tb-right" id="tb-info"></div>
  </div>

  <div id="metrics-panel">
    <div id="metrics-inner"><span style="color:var(--muted);font-size:11px">No metrics loaded.</span></div>
  </div>

  <div id="content"><div id="placeholder">← Select an experiment</div></div>
</div>

<div id="chart-tooltip"></div>

<script>
// ── Constants ─────────────────────────────────────────────────────────────────
const AGENT_COLORS = ['#7a9cff','#ff9a7a','#7affaa','#ffdd7a','#ff7adc','#7adcff','#ffaa44','#aa7aff'];
const SCALAR_COLORS = {
  total_loss:'#e94560', success_rate:'#7affaa', wrong_rate:'#ffaa44',
  mean_correct:'#7adcff', n_episodes:'#aaa', unique_pairings:'#888',
};
const FORUM_PALETTE = [
  {bg:'#2d4a7a',border:'#4a7ab8'},{bg:'#7a4a2d',border:'#b87a4a'},
  {bg:'#2d7a4a',border:'#4ab87a'},{bg:'#7a2d6a',border:'#b84aa0'},
  {bg:'#7a702d',border:'#b8a84a'},{bg:'#2d6a7a',border:'#4aa0b8'},
  {bg:'#5a2d7a',border:'#8a4ab8'},{bg:'#7a2d3a',border:'#b84a5a'},
];
const SKIP_KEYS   = new Set(['iteration','wall_time','n_episodes','unique_pairings']);
const SKIP_PREFIX = ['env_episodes/'];
const ORDER = ['mean_return','success_rate','total_loss','act_loss','perc_loss',
               'kl','entropy','mean_advantage','mean_surprise','wrong_rate','mean_correct'];

// ── State ─────────────────────────────────────────────────────────────────────
let currentExp=null, currentSource='iter';
let currentTrace=null, currentSnapshot=null;
let currentMode='env', currentAgent=null, currentEpisode=0, currentThread=null;
let annotate=true;
let metricsRows=[], metricsGroups=null;   // pre-sorted, cached
let metricsVisible=false;
let chartSize = parseInt(localStorage.getItem('mvChartSize') || '240', 10);
let exIndex = { iters:[], ckpts:[], snaps:[] };
let traceCache = new Map();   // key → payload

document.getElementById('chart-size').value = chartSize;
document.getElementById('chart-size').oninput = function() {
  chartSize = parseInt(this.value,10);
  localStorage.setItem('mvChartSize', String(chartSize));
  if (metricsVisible) renderMetrics();
};

// ── API ───────────────────────────────────────────────────────────────────────
async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

function agentColor(agent, agentList) {
  const idx = agentList.indexOf(agent);
  if (idx < 0) return FORUM_PALETTE[0];
  return FORUM_PALETTE[idx % FORUM_PALETTE.length];
}

// ── Experiments ───────────────────────────────────────────────────────────────
async function loadExperiments() {
  const data = await api('/api/experiments');
  const el = document.getElementById('exp-list');
  el.innerHTML = '';
  if (!data.experiments.length) {
    el.innerHTML = '<div style="padding:14px;color:var(--muted);font-size:11px">No experiments found.</div>';
    return;
  }
  data.experiments.forEach(exp => {
    const d = document.createElement('div');
    d.className='exp-item'; d.textContent=exp; d.title=exp;
    d.onclick = () => selectExp(exp, d);
    el.appendChild(d);
  });
}

async function selectExp(name, el) {
  document.querySelectorAll('.exp-item').forEach(e => e.classList.remove('active'));
  el.classList.add('active');
  currentExp=name; currentTrace=null; currentSnapshot=null;
  metricsRows=[]; metricsGroups=null; traceCache.clear();

  const [idx, metricsResp] = await Promise.all([
    api('/api/index?exp='+encodeURIComponent(name)),
    api('/api/metrics?exp='+encodeURIComponent(name)),
  ]);
  exIndex = idx;
  metricsRows = metricsResp.rows;
  metricsGroups = buildChartGroups(metricsRows);
  if (metricsVisible) renderMetrics();

  populateIterSelect();
  if (firstAvailableKey()) loadCurrent();
  else document.getElementById('content').innerHTML = '<div id="placeholder">No data found.</div>';
}

function setSource(s){
  currentSource=s;
  document.querySelectorAll('.source-pill').forEach(p=>p.classList.remove('active'));
  document.getElementById('src-'+s).classList.add('active');
  populateIterSelect();
  if (firstAvailableKey()) loadCurrent();
}

function listForSource(){
  if (currentSource==='iter') return exIndex.iters;
  if (currentSource==='ckpt') return exIndex.ckpts;
  return exIndex.snaps;
}
function firstAvailableKey(){
  const list = listForSource();
  return list.length ? list[0] : null;
}

function populateIterSelect(){
  const list = listForSource();
  const sel = document.getElementById('iter-select');
  sel.innerHTML='';
  list.forEach(name => {
    const o=document.createElement('option');
    o.value=name; o.textContent=name; sel.appendChild(o);
  });
}

document.getElementById('iter-select').onchange = function(){
  if (this.value) loadCurrent(this.value);
};

function loadCurrent(name){
  if (!name) name = document.getElementById('iter-select').value || firstAvailableKey();
  if (!name) return;
  if (currentSource==='snap') return loadSnapshot(name);
  return loadTrace(name);
}

// ── Trace loading (training rollout / checkpoint trace) ───────────────────────
async function loadTrace(name) {
  currentTrace=null; currentSnapshot=null;
  const cacheKey = currentSource+':'+name;
  document.getElementById('content').innerHTML =
    '<div style="padding:40px;color:var(--muted);text-align:center;font-size:12px">Loading…</div>';
  try {
    let data = traceCache.get(cacheKey);
    if (!data) {
      const url = currentSource==='ckpt'
        ? '/api/ckpt_trace?exp='+encodeURIComponent(currentExp)+'&ckpt='+encodeURIComponent(name)
        : '/api/trace?exp='+encodeURIComponent(currentExp)+'&trace='+encodeURIComponent(name);
      data = await api(url);
      traceCache.set(cacheKey, data);
    }
    currentTrace = data;
    currentEpisode = 0;

    const isForum = data.format === 'forum';
    document.getElementById('btn-forum').style.display   = isForum ? 'inline-block' : 'none';
    document.getElementById('btn-env').style.display     = isForum ? 'none' : 'inline-block';
    document.getElementById('btn-context').style.display = 'inline-block';
    if (isForum) currentMode = 'forum';
    else if (currentMode==='forum') currentMode='env';

    const epWrap = document.getElementById('ep-wrap');
    const epSel  = document.getElementById('ep-select');
    if (isForum && data.episodes && data.episodes.length > 1) {
      epWrap.style.display='inline-flex';
      epSel.innerHTML='';
      data.episodes.forEach((ep,i) => {
        const o=document.createElement('option');
        o.value=i;
        const pairing = (ep.meta && ep.meta.pairing) ? ep.meta.pairing : ('episode '+ep.episode);
        o.textContent = pairing; epSel.appendChild(o);
      });
      epSel.value=0;
    } else { epWrap.style.display='none'; }

    document.getElementById('thread-tabs').style.display='none';
    const agents = currentEpisodeAgents();
    currentAgent = agents[0] || null;
    buildAgentTabs(agents);
    refreshModeButtons();
    document.getElementById('iter-select').value = name;
    if (metricsVisible) updateMetricsMarker();
    render();
    const meta = data.meta || {};
    document.getElementById('tb-info').textContent = currentSource+' • '+(meta.label || meta.iteration || name);
  } catch(e){
    document.getElementById('content').innerHTML =
      '<div style="padding:40px;color:#c44;font-size:12px">Error: '+esc(e.message)+'</div>';
  }
}

async function loadSnapshot(name) {
  currentTrace=null; currentSnapshot=null;
  const cacheKey = 'snap:'+name;
  document.getElementById('content').innerHTML =
    '<div style="padding:40px;color:var(--muted);text-align:center;font-size:12px">Loading…</div>';
  try {
    let data = traceCache.get(cacheKey);
    if (!data) {
      data = await api('/api/snapshot?exp='+encodeURIComponent(currentExp)+'&iter='+encodeURIComponent(name));
      traceCache.set(cacheKey, data);
    }
    currentSnapshot = data;
    document.getElementById('btn-forum').style.display='none';
    document.getElementById('btn-env').style.display='none';
    document.getElementById('btn-context').style.display='none';
    document.getElementById('agent-tabs').style.display='none';
    document.getElementById('ep-wrap').style.display='none';

    const threads = data.threads || [];
    if (!threads.includes(currentThread)) currentThread = threads[0] || null;
    buildThreadTabs(threads);
    document.getElementById('iter-select').value = name;
    if (metricsVisible) updateMetricsMarker();
    renderSnapshot();
    document.getElementById('tb-info').textContent = 'snapshot • iter '+(data.iteration ?? name);
  } catch(e){
    document.getElementById('content').innerHTML =
      '<div style="padding:40px;color:#c44;font-size:12px">Error: '+esc(e.message)+'</div>';
  }
}

function currentEpisodeData() {
  if (!currentTrace) return null;
  if (currentTrace.format==='forum') return (currentTrace.episodes||[])[currentEpisode] || null;
  return null;
}
function currentEpisodeAgents() {
  if (!currentTrace) return [];
  if (currentTrace.format==='forum') {
    const ep = currentEpisodeData();
    return ep ? ep.participating_agents : currentTrace.participating_agents;
  }
  return currentTrace.participating_agents || [];
}

document.getElementById('ep-select').onchange = function(){
  currentEpisode = parseInt(this.value,10) || 0;
  const agents = currentEpisodeAgents();
  currentAgent = agents[0] || null;
  buildAgentTabs(agents);
  render();
};

// ── Metrics: build groups ONCE, sort series ONCE, cache ───────────────────────
function buildChartGroups(rows) {
  const groups = new Map();
  rows.forEach(row => {
    const iter = row.iteration;
    if (iter == null) return;
    Object.entries(row).forEach(([k,v]) => {
      if (SKIP_KEYS.has(k)) return;
      if (SKIP_PREFIX.some(p => k.startsWith(p))) return;
      if (typeof v !== 'number' || !isFinite(v)) return;
      let groupKey, agentName;
      const slash = k.indexOf('/');
      if (slash !== -1) { agentName = k.slice(0,slash); groupKey = k.slice(slash+1); }
      else              { agentName = '__scalar__';      groupKey = k; }
      if (!groups.has(groupKey)) groups.set(groupKey, new Map());
      const g = groups.get(groupKey);
      if (!g.has(agentName)) g.set(agentName, []);
      g.get(agentName).push({x:iter,y:v});
    });
  });

  const agentOrder = [];
  groups.forEach(am => am.forEach((_,a) => {
    if (a !== '__scalar__' && !agentOrder.includes(a)) agentOrder.push(a);
  }));

  const out = Array.from(groups.entries()).map(([key, am]) => {
    const isScalar = am.size===1 && am.has('__scalar__');
    const series = Array.from(am.entries()).map(([agent,data]) => {
      data.sort((a,b)=>a.x-b.x);   // sort once
      const idx = agentOrder.indexOf(agent);
      const color = isScalar ? (SCALAR_COLORS[key] || '#aaaaaa')
                             : AGENT_COLORS[idx % AGENT_COLORS.length];
      return { name:isScalar?key:agent, color, data };
    });
    return { key, title:key, isScalar, series };
  });

  out.sort((a,b) => {
    const ia=ORDER.indexOf(a.key), ib=ORDER.indexOf(b.key);
    if (ia===-1 && ib===-1) return a.key < b.key ? -1 : 1;
    if (ia===-1) return 1;
    if (ib===-1) return -1;
    return ia - ib;
  });
  return out;
}

// ── Metrics panel ─────────────────────────────────────────────────────────────
function toggleMetrics() {
  metricsVisible = !metricsVisible;
  document.getElementById('btn-metrics').classList.toggle('active', metricsVisible);
  document.getElementById('metrics-panel').style.display = metricsVisible ? 'block' : 'none';
  if (metricsVisible) renderMetrics();
}

function iterFromName(name) {
  const m = String(name).match(/(\d+)$/);
  return m ? parseInt(m[1],10) : null;
}

function nearestKey(targetIter, list) {
  if (!list.length) return null;
  let best=list[0], bestD=Infinity;
  list.forEach(t => {
    const it=iterFromName(t); if (it==null) return;
    const d=Math.abs(it-targetIter);
    if (d<bestD){bestD=d; best=t;}
  });
  return best;
}

function currentIter() {
  if (currentTrace) {
    const m = currentTrace.meta || {};
    if (typeof m.iteration === 'number') return m.iteration;
    return iterFromName(m.label || m.iteration);
  }
  if (currentSnapshot) return currentSnapshot.iteration;
  return null;
}

function renderMetrics() {
  const inner = document.getElementById('metrics-inner');
  if (!metricsRows.length || !metricsGroups) {
    inner.innerHTML = '<span style="color:var(--muted);font-size:11px;padding:4px">No metrics data.</span>';
    return;
  }
  inner.innerHTML='';
  const iter = currentIter();
  const traceIters = listForSource().map(iterFromName).filter(x => x!=null);
  const W = chartSize, H = Math.round(chartSize * 0.78);

  metricsGroups.forEach(group => {
    const wrap = document.createElement('div'); wrap.className='chart-wrap';
    const stack = document.createElement('div'); stack.className='chart-stack';
    stack.style.width=W+'px'; stack.style.height=H+'px';

    const bg = document.createElement('canvas'); bg.className='bg'; bg.width=W; bg.height=H;
    const fg = document.createElement('canvas'); fg.className='fg'; fg.width=W; fg.height=H;
    fg.dataset.groupKey = group.key;
    stack.appendChild(bg); stack.appendChild(fg);

    drawChartBackground(bg, group);
    drawChartOverlay(fg, group, iter, traceIters, null);
    setupChartEvents(fg, group, traceIters);

    const legend = document.createElement('div'); legend.className='chart-legend';
    legend.style.maxWidth = W+'px';
    if (!group.isScalar) {
      group.series.forEach(s => {
        legend.innerHTML += '<div class="legend-item">'
          + '<div class="legend-dot" style="background:'+s.color+'"></div>'+esc(s.name)+'</div>';
      });
    }
    wrap.appendChild(stack); wrap.appendChild(legend);
    inner.appendChild(wrap);
  });
}

function updateMetricsMarker() {
  if (!metricsGroups) return;
  const iter = currentIter();
  const traceIters = listForSource().map(iterFromName).filter(x=>x!=null);
  document.querySelectorAll('#metrics-inner canvas.fg').forEach(fg => {
    const key = fg.dataset.groupKey;
    const group = metricsGroups.find(g => g.key === key);
    if (group) drawChartOverlay(fg, group, iter, traceIters, null);
  });
}

// ── Chart drawing ─────────────────────────────────────────────────────────────
const PL=42, PR=10, PT=20, PB=26;

function chartScale(canvas, group) {
  const IW = canvas.width-PL-PR, IH = canvas.height-PT-PB;
  let minX=Infinity, maxX=-Infinity, minY=Infinity, maxY=-Infinity;
  group.series.forEach(s => s.data.forEach(d => {
    if (d.x<minX) minX=d.x; if (d.x>maxX) maxX=d.x;
    if (d.y<minY) minY=d.y; if (d.y>maxY) maxY=d.y;
  }));
  const padY = Math.max((maxY-minY)*0.08, Math.abs(maxY)*0.01, 1e-6);
  const yLo=minY-padY, yHi=maxY+padY;
  const sx = x => PL + (maxX>minX ? (x-minX)/(maxX-minX)*IW : IW/2);
  const sy = y => PT + IH - (yHi>yLo ? (y-yLo)/(yHi-yLo)*IH : IH/2);
  return { IW, IH, minX, maxX, yLo, yHi, sx, sy };
}

function drawChartBackground(canvas, group) {
  const ctx = canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;
  const sc = chartScale(canvas, group);
  canvas._sc = sc; canvas._group = group;

  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#0a1c2e'; ctx.fillRect(0,0,W,H);

  ctx.fillStyle='#667'; ctx.font='10px system-ui'; ctx.textAlign='center';
  ctx.fillText(group.title, W/2, 13);

  const nY=4;
  for (let i=0; i<=nY; i++) {
    const gy = PT + i*sc.IH/nY;
    ctx.strokeStyle='#122030'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(PL,gy); ctx.lineTo(W-PR,gy); ctx.stroke();
    const val = sc.yHi - i*(sc.yHi-sc.yLo)/nY;
    ctx.fillStyle='#445'; ctx.font='8px monospace'; ctx.textAlign='right';
    const fmt = Math.abs(val)<0.01 ? val.toExponential(1)
              : Math.abs(val)<10   ? val.toFixed(3)
              : Math.abs(val)<1000 ? val.toFixed(1) : Math.round(val).toString();
    ctx.fillText(fmt, PL-3, gy+3);
  }

  const nX=4;
  ctx.fillStyle='#445'; ctx.font='8px monospace';
  for (let i=0; i<=nX; i++) {
    const gx = PL + i*sc.IW/nX;
    const it = Math.round(sc.minX + i*(sc.maxX-sc.minX)/nX);
    ctx.textAlign='center'; ctx.fillText(it, gx, H-4);
  }

  group.series.forEach(s => {
    if (!s.data.length) return;
    ctx.strokeStyle=s.color; ctx.lineWidth=1.5; ctx.beginPath();
    s.data.forEach((d,i) => {
      const px=sc.sx(d.x), py=sc.sy(d.y);
      i===0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
    });
    ctx.stroke();
  });
}

function drawChartOverlay(canvas, group, currentIter, traceIters, hoverX) {
  // Mirror the bg's scale by setting it from the bg canvas (same dimensions)
  const ctx = canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;
  const sc = chartScale(canvas, group);
  canvas._sc = sc; canvas._group = group;
  ctx.clearRect(0,0,W,H);

  // tick marks for available iters on x-axis
  ctx.fillStyle='#334';
  traceIters.forEach(it => {
    if (it<sc.minX || it>sc.maxX) return;
    const tx = sc.sx(it);
    ctx.fillRect(tx-0.5, PT+sc.IH, 1, 4);
  });

  if (hoverX != null) {
    ctx.strokeStyle='rgba(255,255,255,0.18)'; ctx.lineWidth=1; ctx.setLineDash([2,3]);
    ctx.beginPath(); ctx.moveTo(hoverX, PT); ctx.lineTo(hoverX, PT+sc.IH); ctx.stroke();
    ctx.setLineDash([]);
  }
  if (currentIter != null) {
    const mx = sc.sx(currentIter);
    ctx.strokeStyle='#e94560'; ctx.lineWidth=1.5; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(mx, PT); ctx.lineTo(mx, PT+sc.IH); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='#e94560';
    ctx.beginPath(); ctx.moveTo(mx,PT+3); ctx.lineTo(mx-4,PT-4); ctx.lineTo(mx+4,PT-4); ctx.closePath(); ctx.fill();
  }
}

// ── Chart interaction (rAF-throttled, no resorts) ─────────────────────────────
const tooltip = document.getElementById('chart-tooltip');

function setupChartEvents(canvas, group, traceIters) {
  let rafId = 0, pendingEv = null;

  function xToIter(canvasX) {
    const sc = canvas._sc; if (!sc) return null;
    const t = (canvasX-PL)/sc.IW;
    return sc.minX + t*(sc.maxX-sc.minX);
  }

  function valuesAtIter(targetIter) {
    const out=[];
    group.series.forEach(s => {
      // s.data already sorted; binary search for nearest
      const arr = s.data;
      if (!arr.length) return;
      let lo=0, hi=arr.length-1;
      while (lo<hi) { const mid=(lo+hi)>>1; if (arr[mid].x<targetIter) lo=mid+1; else hi=mid; }
      let best = arr[lo];
      if (lo>0 && Math.abs(arr[lo-1].x-targetIter) < Math.abs(best.x-targetIter)) best=arr[lo-1];
      out.push({name:s.name, color:s.color, x:best.x, y:best.y});
    });
    return out;
  }

  function handle() {
    rafId = 0;
    const e = pendingEv; if (!e) return;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const sc = canvas._sc;
    if (!sc || cx < PL || cx > PL+sc.IW) {
      tooltip.style.display='none';
      drawChartOverlay(canvas, group, currentIter(), traceIters, null);
      return;
    }
    drawChartOverlay(canvas, group, currentIter(), traceIters, cx);
    const targetIter = xToIter(cx);
    const vals = valuesAtIter(targetIter);
    if (!vals.length) { tooltip.style.display='none'; return; }
    let html = '<div class="tt-iter">iter '+vals[0].x+'</div>';
    vals.forEach(v => {
      const fmt = Math.abs(v.y)<0.001 ? v.y.toExponential(2) : v.y.toFixed(4);
      html += '<div class="tt-row"><div class="tt-dot" style="background:'+v.color+'"></div>'
            + esc(v.name)+' <b>'+fmt+'</b></div>';
    });
    tooltip.innerHTML = html;
    tooltip.style.display = 'block';
    const tx = e.clientX+12, ty = e.clientY-10;
    const ow = tooltip.offsetWidth, oh = tooltip.offsetHeight;
    tooltip.style.left = (tx+ow > window.innerWidth ? tx-ow-16 : tx) + 'px';
    tooltip.style.top  = (ty+oh > window.innerHeight ? ty-oh    : ty) + 'px';
  }

  canvas.addEventListener('mousemove', e => {
    pendingEv = e;
    if (!rafId) rafId = requestAnimationFrame(handle);
  });
  canvas.addEventListener('mouseleave', () => {
    pendingEv = null; if (rafId) { cancelAnimationFrame(rafId); rafId=0; }
    tooltip.style.display='none';
    drawChartOverlay(canvas, group, currentIter(), traceIters, null);
  });
  canvas.addEventListener('click', e => {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const sc = canvas._sc;
    if (!sc || cx<PL || cx>PL+sc.IW) return;
    const targetIter = xToIter(cx);
    const list = listForSource();
    const name = nearestKey(targetIter, list);
    if (name) loadCurrent(name);
  });
}

// ── Mode/agent/thread tabs ────────────────────────────────────────────────────
function setMode(m) { currentMode=m; refreshModeButtons(); render(); }
function refreshModeButtons() {
  ['forum','env','context'].forEach(id => {
    const el = document.getElementById('btn-'+id);
    if (el) el.classList.toggle('active', id===currentMode);
  });
  document.getElementById('agent-tabs').style.display = currentMode==='context' ? 'flex' : 'none';
}
function buildAgentTabs(agents) {
  const bar = document.getElementById('agent-tabs');
  bar.innerHTML='';
  agents.forEach(a => {
    const b=document.createElement('button');
    b.className='agent-tab'+(a===currentAgent?' active':'');
    b.textContent=a;
    b.onclick = () => { currentAgent=a; document.querySelectorAll('.agent-tab').forEach(e=>e.classList.remove('active'));
                        b.classList.add('active'); render(); };
    bar.appendChild(b);
  });
  bar.style.display = currentMode==='context' ? 'flex' : 'none';
}
function buildThreadTabs(threads) {
  const bar = document.getElementById('thread-tabs');
  bar.innerHTML='';
  threads.forEach(t => {
    const b=document.createElement('button');
    b.className='agent-tab'+(t===currentThread?' active':'');
    b.textContent=t.replace(/^thread_/,'');
    b.onclick = () => { currentThread=t; document.querySelectorAll('#thread-tabs .agent-tab').forEach(e=>e.classList.remove('active'));
                        b.classList.add('active'); renderSnapshot(); };
    bar.appendChild(b);
  });
  bar.style.display = threads.length ? 'flex' : 'none';
}

// ── Render ────────────────────────────────────────────────────────────────────
function renderMeta(meta) {
  let html = '<div id="meta-card">';
  Object.entries(meta||{}).forEach(([k,v]) => {
    html += '<div class="meta-item"><span class="meta-key">'+esc(k)+'</span>'
          + '<span class="meta-val">'+esc(v)+'</span></div>';
  });
  return html + '</div>';
}

function renderEnvLog(trace) {
  let html = renderMeta(trace.meta);
  html += '<div class="section-hdr">Environment log</div>';
  html += '<div id="agents-bar">'
        + (trace.participating_agents||[]).map(a => '<span class="agent-chip">'+esc(a)+'</span>').join('')
        + '</div>';
  (trace.environment_log||[]).forEach(ev => {
    if (ev.type==='act') {
      let lbl = '['+esc(ev.agent)+' → ENV]';
      if (ev.routed_to) lbl += ' &nbsp;<span class="routing-arrow">↓ routed to '+esc(ev.routed_to)+'</span>';
      html += '<div class="env-event ev-act"><div class="env-hdr">'+lbl+'</div>'
            + '<div class="env-body">'+esc(ev.text)+'</div></div>';
    } else {
      let cls,lbl;
      if (ev.source_type==='env') { cls='ev-obs-env'; lbl='[ENV → '+esc(ev.agent)+']'; }
      else if (ev.source_type==='agent') { cls='ev-obs-routed'; lbl='['+esc(ev.source_agent)+' → '+esc(ev.agent)+']'; }
      else { cls='ev-obs-routed'; lbl='['+esc(ev.source_agent)+'+ENV → '+esc(ev.agent)+']'; }
      html += '<div class="env-event '+cls+'"><div class="env-hdr">'+lbl+'</div>'
            + '<div class="env-body">'+esc(ev.text)+'</div>';
      if (ev.source_type==='agent+env' && ev.env_prefix)
        html += '<div class="prefix-detail">env prefix: '+esc(ev.env_prefix)+'</div>';
      html += '</div>';
    }
  });
  return html;
}

function renderForumView(trace) {
  const ep = currentEpisodeData();
  if (!ep) return '<div style="color:var(--muted)">No episode data.</div>';
  const agents = ep.participating_agents || [];
  let html = renderMeta({...(trace.meta||{}), ...(ep.meta||{})});
  html += '<div id="agents-bar">';
  agents.forEach(a => {
    const c = agentColor(a, agents);
    html += '<span class="agent-chip"><span class="agent-swatch" style="background:'+c.border+'"></span>'+esc(a)+'</span>';
  });
  html += '</div>';
  html += '<div class="section-hdr">Forum thread</div>';
  // Preserve the env's append order. Posts with a numeric ``post_index``
  // sort by that; pybot/tool-output posts (``post_index: null``) carry the
  // array index of the post they followed so they stay attached to it
  // instead of collapsing to the top of the thread.
  const _raw = (ep.thread||[]);
  let _lastNumeric = -1;
  const thread = _raw.map((p, i) => {
    let key;
    if (typeof p.post_index === 'number') {
      _lastNumeric = p.post_index;
      key = p.post_index * 1000 + i;
    } else {
      key = _lastNumeric * 1000 + i;
    }
    return {post: p, _key: key};
  }).sort((a, b) => a._key - b._key).map(x => x.post);
  if (!thread.length) return html + '<div style="color:var(--muted)">No posts.</div>';
  html += '<div id="forum-view">';
  thread.forEach(post => {
    const speaker = post.speaker || '(unknown)';
    const c = agentColor(speaker, agents);
    let thinkHtml = '';
    const rawTurn = post.raw_turn || post.thinking;
    if (rawTurn) {
      const label = post.raw_turn ? 'full output' : 'thinking';
      thinkHtml = '<details class="forum-think"><summary>'+label+'</summary>'
                + '<div class="forum-think-body">'+esc(rawTurn)+'</div></details>';
    }
    const idxChip = (typeof post.post_index === 'number')
      ? '<span class="post-idx">#'+esc(post.post_index)+'</span>'
      : (post.kind === 'tool_output' ? '<span class="post-idx">tool</span>' : '');
    html += '<div class="forum-post" style="border-color:'+c.border+'">'
          + '<div class="forum-post-hdr" style="background:'+c.bg+'">'+esc(speaker)
          + idxChip+'</div>'
          + thinkHtml
          + '<div class="forum-post-body">'+esc(post.text||'')+'</div></div>';
  });
  html += '</div>';
  return html;
}

function renderContextAnnotated(turns) {
  let html='';
  turns.forEach(t => {
    if (t.type==='prompt') {
      html += '<div class="turn turn-prompt"><div class="turn-lbl">System prompt</div>'
            + '<div class="turn-body">'+esc(t.text)+'</div></div>';
    } else if (t.type==='obs') {
      let src = t.source_type==='env' ? 'env'
              : t.source_type==='agent' ? '← '+t.source_agent
              : '← '+t.source_agent+' + env';
      html += '<div class="turn turn-obs"><div class="turn-lbl">Observation &nbsp; '+esc(src)+'</div>'
            + '<div class="turn-body">'+esc(t.text)+'</div></div>';
    } else {
      html += '<div class="turn turn-act"><div class="turn-lbl">Action</div>'
            + '<div class="turn-body">'+esc(t.text)+'</div></div>';
    }
  });
  return html;
}

function render() {
  if (!currentTrace) return;
  const content = document.getElementById('content');
  if (currentMode==='forum')   { content.innerHTML = renderForumView(currentTrace); return; }
  if (currentMode==='env')     { content.innerHTML = renderEnvLog(currentTrace);    return; }

  let contextSource;
  if (currentTrace.format==='forum') {
    const ep = currentEpisodeData();
    contextSource = ep ? ep.agent_contexts : {};
  } else {
    contextSource = currentTrace.agent_contexts || {};
  }
  const ctx = contextSource[currentAgent];
  let html = renderMeta(currentTrace.meta);
  html += '<div style="display:flex;gap:7px;margin-bottom:10px;align-items:center">'
        + '<div class="section-hdr" style="margin:0">Context: '+esc(currentAgent)+'</div></div>';
  if (Array.isArray(ctx)) {
    html += '<div style="display:flex;gap:5px;margin-bottom:10px;justify-content:flex-end">'
          + '<button class="tb-btn'+(annotate?' active':'')+'" onclick="setAnnotate(true)">Annotated</button>'
          + '<button class="tb-btn'+(!annotate?' active':'')+'" onclick="setAnnotate(false)">Raw</button></div>';
    html += annotate ? renderContextAnnotated(ctx) : ('<div id="raw-view">'+esc(ctx.map(t=>t.text).join(''))+'</div>');
  } else if (typeof ctx === 'string') {
    html += '<div id="raw-view">'+esc(ctx)+'</div>';
  } else {
    html += '<div style="color:var(--muted)">No context for this agent.</div>';
  }
  content.innerHTML = html;
}
function setAnnotate(v){ annotate=v; render(); }

// ── Snapshot rendering ────────────────────────────────────────────────────────
function renderSnapshot() {
  if (!currentSnapshot) return;
  const content = document.getElementById('content');
  const snap = currentSnapshot;
  const agents = snap.agents_order || Object.keys(snap.agents||{});
  if (!currentThread) {
    content.innerHTML = renderMeta({iteration:snap.iteration, eval_set:snap.eval_set,
                                    temperature:snap.temperature, samples_per_ctx:snap.samples_per_ctx})
      + '<div style="color:var(--muted)">No threads in this snapshot.</div>';
    return;
  }
  let html = renderMeta({iteration:snap.iteration, thread:currentThread, eval_set:snap.eval_set,
                         temperature:snap.temperature, samples_per_ctx:snap.samples_per_ctx});
  html += '<div class="section-hdr">'+esc(currentThread)+'</div>';
  html += '<div id="snap-view">';
  agents.forEach(a => {
    const samples = ((snap.agents[a]||{})[currentThread]) || [];
    const c = agentColor(a, agents);
    html += '<div class="snap-agent">'
          + '<div class="snap-agent-hdr"><span class="swatch" style="background:'+c.border+'"></span>'
          + esc(a)+'<span class="count">'+samples.length+' samples</span></div>';
    samples.forEach((s,i) => {
      const tok = s.tokens != null ? '<span class="tok">#'+(i+1)+' • '+s.tokens+' tok</span>' : '<span class="tok">#'+(i+1)+'</span>';
      html += '<div class="snap-sample">'+tok+esc(s.text||'')+'</div>';
    });
    html += '</div>';
  });
  html += '</div>';
  content.innerHTML = html;
}

// ── Boot ──────────────────────────────────────────────────────────────────────
loadExperiments();
</script>
</body>
</html>
"""


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    root: Path = Path("runs/cultural_emergence")

    def log_message(self, fmt, *args):
        pass

    def _respond(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data):
        self._respond(200, "application/json", json.dumps(data).encode())

    def _error(self, code, msg):
        self._respond(code, "text/plain", msg.encode())

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs   = parse_qs(parsed.query)

        if path in ("/", ""):
            self._respond(200, "text/html; charset=utf-8", _HTML.encode())

        elif path == "/api/experiments":
            exps = []
            if self.root.is_dir():
                for d in sorted(self.root.iterdir()):
                    if not d.is_dir():
                        continue
                    if (d / "traces").is_dir() or (d / "snapshots").is_dir() or (d / "metrics.jsonl").exists():
                        exps.append(d.name)
            self._json({"experiments": exps})

        elif path == "/api/index":
            exp = qs.get("exp", [""])[0]
            base = self.root / exp
            traces_dir = base / "traces"
            snaps_dir  = base / "snapshots"
            iters, ckpts, snaps = [], [], []
            if traces_dir.is_dir():
                iters = sorted(p.stem for p in traces_dir.glob("iter_*.json"))
                if not iters:
                    iters = sorted(p.stem for p in traces_dir.glob("iter_*.txt"))
                ckpts = sorted(p.name for p in traces_dir.iterdir()
                               if p.is_dir() and p.name.startswith("ckpt_"))
            if snaps_dir.is_dir():
                snaps = sorted(p.stem for p in snaps_dir.glob("iter_*.json"))
            self._json({"iters": iters, "ckpts": ckpts, "snaps": snaps})

        elif path == "/api/trace":
            exp   = qs.get("exp",   [""])[0]
            trace = qs.get("trace", [""])[0]
            json_path = self.root / exp / "traces" / f"{trace}.json"
            txt_path  = self.root / exp / "traces" / f"{trace}.txt"
            if json_path.exists():
                with open(json_path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    payload = _convert_forum_records(data, trace)
                    payload["meta"]["iteration"] = _iter_num(trace)
                    self._json(payload)
                else:
                    self._json(data)
            elif txt_path.exists():
                with open(txt_path) as f:
                    text = f.read()
                self._json({
                    "meta": {"iteration": _iter_num(trace), "label": trace},
                    "participating_agents": ["(raw)"],
                    "environment_log": [],
                    "agent_contexts": {"(raw)": [{"type": "prompt", "text": text}]},
                })
            else:
                self._error(404, "Trace not found")

        elif path == "/api/ckpt_trace":
            exp  = qs.get("exp",  [""])[0]
            ckpt = qs.get("ckpt", [""])[0]
            ckpt_dir = self.root / exp / "traces" / ckpt
            json_path = ckpt_dir / "traces.json"
            txt_path  = ckpt_dir / "traces.txt"
            if json_path.exists():
                with open(json_path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    payload = _convert_forum_records(data, ckpt)
                    payload["meta"]["iteration"] = _iter_num(ckpt)
                    self._json(payload)
                else:
                    self._json(data)
            elif txt_path.exists():
                with open(txt_path) as f:
                    text = f.read()
                self._json({
                    "meta": {"iteration": _iter_num(ckpt), "label": ckpt},
                    "participating_agents": ["(raw)"],
                    "environment_log": [],
                    "agent_contexts": {"(raw)": [{"type": "prompt", "text": text}]},
                })
            else:
                self._error(404, "Checkpoint trace not found")

        elif path == "/api/snapshot":
            exp = qs.get("exp", [""])[0]
            it  = qs.get("iter", [""])[0]
            snap_path = self.root / exp / "snapshots" / f"{it}.json"
            if not snap_path.exists():
                self._error(404, "Snapshot not found"); return
            with open(snap_path) as f:
                data = json.load(f)
            agents = data.get("agents", {}) or {}
            agents_order = list(agents.keys())
            thread_set: list[str] = []
            for a in agents_order:
                for k in (agents[a] or {}).keys():
                    if k not in thread_set:
                        thread_set.append(k)
            self._json({
                "iteration":       data.get("iteration", _iter_num(it)),
                "temperature":     data.get("temperature"),
                "samples_per_ctx": data.get("samples_per_ctx"),
                "eval_set":        data.get("eval_set"),
                "agents_order":    agents_order,
                "threads":         thread_set,
                "agents":          agents,
            })

        elif path == "/api/metrics":
            exp = qs.get("exp", [""])[0]
            metrics_path = self.root / exp / "metrics.jsonl"
            rows: list[dict] = []
            if metrics_path.exists():
                with open(metrics_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        try: rows.append(json.loads(line))
                        except json.JSONDecodeError: pass
            self._json({"rows": rows})

        else:
            self._error(404, "Not found")


def _iter_num(label: str):
    m = re.search(r"(\d+)$", label or "")
    return int(m.group(1)) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("output_root", nargs="?", default="runs/cultural_emergence",
                        help="Root directory of experiment outputs (default: runs/cultural_emergence)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    Handler.root = root

    print(f"Serving experiments from:  {root}")
    print(f"Listening on:              http://{args.host}:{args.port}")
    print(f"SSH port-forward:          ssh -L {args.port}:localhost:{args.port} <host>")
    print("Ctrl-C to stop.")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
