#!/usr/bin/env python3
"""
Interactive trace viewer for MARLLLM experiments.

Serves a web UI that lets you browse experiments, view episode traces in
multiple display modes, and inspect per-agent context windows alongside
live training curves.

Usage
-----
    python scripts/trace_viewer.py [output_root] [--port PORT] [--host HOST]

    output_root  : directory containing experiment sub-directories
                   (default: ./outputs)
    --port       : port to listen on (default: 8765)
    --host       : host to bind (default: 0.0.0.0, needed for SSH port-forward)

SSH port-forward access
-----------------------
    On Isambard:  python scripts/trace_viewer.py --port 8765
    On laptop:    ssh -L 8765:localhost:8765 <isambard-login>
    Browser:      http://localhost:8765
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ── Embedded HTML/JS/CSS ──────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MARLLLM Trace Viewer</title>
<style>
  :root {
    --bg: #1a1a2e; --panel: #16213e; --card: #0f3460; --card2: #0a2040;
    --accent: #e94560; --text: #e0e0e0; --muted: #777; --muted2: #555;
    --obs-bg: #0e1f0e; --act-bg: #1f0e1f; --prompt-bg: #0e0e1f;
    --obs-border: #1e4a1e; --act-border: #4a1e4a; --prompt-border: #1e1e4a;
    --font-mono: 'Cascadia Code','Fira Code','Consolas',monospace;
    --font-ui: system-ui,-apple-system,sans-serif;
    --chart-bg: #0a1e30;
    --chart-grid: #122030;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font-ui);
         display: flex; height: 100vh; overflow: hidden; }

  /* ── Sidebar ── */
  #sidebar { width: 260px; min-width: 200px; background: var(--panel);
             border-right: 1px solid #222; display: flex; flex-direction: column; flex-shrink: 0; }
  #sidebar-title { padding: 12px 14px; font-size: 13px; font-weight: 700;
                   color: var(--accent); border-bottom: 1px solid #222; letter-spacing: 1px; }
  #exp-list { flex: 1; overflow-y: auto; padding: 4px 0; }
  .exp-item { padding: 7px 14px; cursor: pointer; font-size: 12px; border-left: 3px solid transparent;
              white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .exp-item:hover { background: #1e2d4a; }
  .exp-item.active { border-left-color: var(--accent); background: #1e2d4a; color: #fff; }

  /* ── Main column ── */
  #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }

  /* ── Toolbar ── */
  #toolbar { background: var(--panel); border-bottom: 1px solid #222; padding: 6px 12px;
             display: flex; align-items: center; gap: 8px; flex-wrap: wrap; flex-shrink: 0; }
  #toolbar label { font-size: 11px; color: var(--muted); }
  #iter-select { background: var(--card); border: 1px solid #333; color: var(--text);
                 padding: 3px 7px; border-radius: 4px; font-size: 12px; min-width: 150px; }
  .tb-sep { width: 1px; height: 18px; background: #333; }
  .tb-btn { background: var(--card2); border: 1px solid #333; color: var(--muted);
            padding: 3px 9px; border-radius: 4px; cursor: pointer; font-size: 11px;
            white-space: nowrap; }
  .tb-btn:hover { color: var(--text); border-color: #555; }
  .tb-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; }
  #agent-tabs { display: none; gap: 5px; flex-wrap: wrap; }
  .agent-tab { background: var(--card2); border: 1px solid #333; color: var(--muted);
               padding: 3px 9px; border-radius: 4px; cursor: pointer; font-size: 11px; }
  .agent-tab.active { border-color: #6a9; color: #6a9; }
  .tb-right { margin-left: auto; font-size: 11px; color: var(--muted); }

  /* ── Metrics panel ── */
  #metrics-panel { display: none; border-bottom: 1px solid #222; background: var(--card2);
                   flex-shrink: 0; }
  #metrics-inner { display: flex; gap: 8px; padding: 8px 12px; overflow-x: auto;
                   align-items: flex-start; }
  #metrics-inner::-webkit-scrollbar { height: 5px; }
  #metrics-inner::-webkit-scrollbar-track { background: transparent; }
  #metrics-inner::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }

  .chart-wrap { flex-shrink: 0; display: flex; flex-direction: column; align-items: center; }
  .chart-wrap canvas { display: block; border-radius: 4px; cursor: crosshair; }
  .chart-legend { display: flex; gap: 6px; flex-wrap: wrap; justify-content: center;
                  margin-top: 3px; max-width: 220px; }
  .legend-item { display: flex; align-items: center; gap: 3px; font-size: 9px;
                 color: var(--muted); white-space: nowrap; }
  .legend-dot { width: 8px; height: 3px; border-radius: 1px; flex-shrink: 0; }

  /* Tooltip */
  #chart-tooltip { position: fixed; background: #0a1825; border: 1px solid #335;
                   border-radius: 4px; padding: 5px 8px; font-size: 11px; font-family: var(--font-mono);
                   pointer-events: none; display: none; z-index: 999; max-width: 200px;
                   line-height: 1.6; }
  .tt-row { display: flex; gap: 6px; align-items: center; }
  .tt-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .tt-iter { color: var(--muted); margin-bottom: 3px; font-size: 10px; }

  /* ── Trace content ── */
  #content { flex: 1; overflow-y: auto; padding: 12px 16px; }
  #placeholder { color: var(--muted); text-align: center; margin-top: 80px; font-size: 13px; }

  /* Meta */
  #meta-card { background: var(--card); border-radius: 5px; padding: 10px 14px;
               margin-bottom: 12px; display: flex; flex-wrap: wrap; gap: 14px; }
  .meta-item { font-size: 11px; }
  .meta-key { color: var(--muted); margin-right: 4px; }
  .meta-val { color: #fff; font-weight: 600; }

  /* Section header */
  .section-hdr { font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
                 color: var(--muted); text-transform: uppercase; margin: 14px 0 7px; }

  /* Env log */
  .env-event { margin-bottom: 10px; border-radius: 5px; overflow: hidden; }
  .env-hdr { padding: 5px 10px; font-size: 11px; font-family: var(--font-mono);
             font-weight: 600; display: flex; align-items: center; gap: 8px; }
  .env-body { padding: 7px 10px; font-family: var(--font-mono); font-size: 12px;
              white-space: pre-wrap; word-break: break-word; }
  .ev-act { background: var(--act-bg); border: 1px solid var(--act-border); }
  .ev-act .env-hdr { background: #2a0e2a; color: #bb77bb; }
  .ev-obs-env { background: var(--obs-bg); border: 1px solid var(--obs-border); }
  .ev-obs-env .env-hdr { background: #0e2a0e; color: #77bb77; }
  .ev-obs-routed { background: #0e1e2a; border: 1px solid #1e4a5a; }
  .ev-obs-routed .env-hdr { background: #0e1e2a; color: #77bbcc; }
  .routing-arrow { color: var(--accent); }
  .prefix-detail { font-size: 10px; color: var(--muted); padding: 3px 10px 7px;
                   font-family: var(--font-mono); }

  /* Context turns */
  .turn { margin-bottom: 7px; border-radius: 5px; overflow: hidden; }
  .turn-lbl { padding: 2px 9px; font-size: 9px; font-family: var(--font-mono);
              font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; }
  .turn-body { padding: 7px 10px; font-family: var(--font-mono); font-size: 12px;
               white-space: pre-wrap; word-break: break-word; line-height: 1.5; }
  .turn-prompt { background: var(--prompt-bg); border: 1px solid var(--prompt-border); }
  .turn-prompt .turn-lbl { background: #12123a; color: #7777cc; }
  .turn-obs { background: var(--obs-bg); border: 1px solid var(--obs-border); }
  .turn-obs .turn-lbl { background: #0e2a0e; color: #77bb77; }
  .turn-act { background: var(--act-bg); border: 1px solid var(--act-border); }
  .turn-act .turn-lbl { background: #2a0e2a; color: #bb77bb; }
  #raw-view { font-family: var(--font-mono); font-size: 12px; white-space: pre-wrap;
              word-break: break-word; line-height: 1.5; background: var(--card);
              padding: 14px; border-radius: 5px; }

  /* Agents chips */
  #agents-bar { display: flex; gap: 7px; flex-wrap: wrap; margin-bottom: 10px; }
  .agent-chip { font-size: 11px; padding: 2px 9px; border-radius: 10px;
                background: var(--card); color: var(--text); }

  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
</style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-title">MARLLLM Traces</div>
  <div id="exp-list"><div style="padding:14px;color:var(--muted);font-size:11px">Loading…</div></div>
</div>

<div id="main">
  <div id="toolbar">
    <label>Episode</label>
    <select id="iter-select"><option>—</option></select>
    <div class="tb-sep"></div>
    <label>View</label>
    <button class="tb-btn active" id="btn-env"     onclick="setMode('env')">Environment log</button>
    <button class="tb-btn"        id="btn-context" onclick="setMode('context')">Agent context</button>
    <div id="agent-tabs"></div>
    <div class="tb-sep"></div>
    <button class="tb-btn" id="btn-metrics" onclick="toggleMetrics()">📈 Training curves</button>
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
  total_loss: '#e94560', success_rate: '#7affaa', wrong_rate: '#ffaa44',
  mean_correct: '#7adcff', n_episodes: '#aaa', unique_pairings: '#888',
};
const CHART_W = 220, CHART_H = 185;
const SKIP_KEYS   = new Set(['iteration','wall_time','n_episodes','unique_pairings']);
const SKIP_PREFIX = ['env_episodes/'];

// ── State ─────────────────────────────────────────────────────────────────────
let currentExp   = null;
let currentTrace = null;
let currentMode  = 'env';
let currentAgent = null;
let annotate     = true;
let metricsData  = [];
let metricsVisible = false;
let tracesList   = [];    // array of trace stem names, e.g. ["iter_000000", ...]

// ── API ───────────────────────────────────────────────────────────────────────
async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
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
    d.className = 'exp-item'; d.textContent = exp; d.title = exp;
    d.onclick = () => selectExp(exp, d);
    el.appendChild(d);
  });
}

async function selectExp(name, el) {
  document.querySelectorAll('.exp-item').forEach(e => e.classList.remove('active'));
  el.classList.add('active');
  currentExp = name; currentTrace = null; metricsData = [];

  const [tracesData, metricsResp] = await Promise.all([
    api('/api/traces?exp=' + encodeURIComponent(name)),
    api('/api/metrics?exp=' + encodeURIComponent(name)),
  ]);

  tracesList = tracesData.traces;
  metricsData = metricsResp.rows;

  const sel = document.getElementById('iter-select');
  sel.innerHTML = '';
  tracesList.forEach(t => {
    const o = document.createElement('option');
    o.value = t; o.textContent = t; sel.appendChild(o);
  });

  if (metricsVisible) renderMetrics();
  if (tracesList.length) loadTrace(tracesList[0]);
  else document.getElementById('content').innerHTML = '<div id="placeholder">No traces found.</div>';
}

document.getElementById('iter-select').onchange = function() {
  if (this.value) loadTrace(this.value);
};

async function loadTrace(name) {
  currentTrace = null;
  document.getElementById('content').innerHTML =
    '<div style="padding:40px;color:var(--muted);text-align:center;font-size:12px">Loading…</div>';
  try {
    const data = await api('/api/trace?exp=' + encodeURIComponent(currentExp) + '&trace=' + encodeURIComponent(name));
    currentTrace = data;
    currentAgent = data.participating_agents[0] || null;
    buildAgentTabs(data.participating_agents);
    document.getElementById('iter-select').value = name;
    if (metricsVisible) updateMetricsMarker();
    render();
    document.getElementById('tb-info').textContent = 'iter ' + data.meta.iteration;
  } catch(e) {
    document.getElementById('content').innerHTML =
      '<div style="padding:40px;color:#c44;font-size:12px">Error: ' + esc(e.message) + '</div>';
  }
}

// ── Metrics panel ─────────────────────────────────────────────────────────────
function toggleMetrics() {
  metricsVisible = !metricsVisible;
  const btn = document.getElementById('btn-metrics');
  const panel = document.getElementById('metrics-panel');
  btn.classList.toggle('active', metricsVisible);
  panel.style.display = metricsVisible ? 'block' : 'none';
  if (metricsVisible) renderMetrics();
}

function iterFromName(name) {
  const m = name.match(/(\d+)$/);
  return m ? parseInt(m[1], 10) : null;
}

function nearestTrace(targetIter) {
  if (!tracesList.length) return null;
  let best = tracesList[0], bestDist = Infinity;
  tracesList.forEach(t => {
    const it = iterFromName(t);
    if (it == null) return;
    const d = Math.abs(it - targetIter);
    if (d < bestDist) { bestDist = d; best = t; }
  });
  return best;
}

function buildChartGroups(rows) {
  // Map: groupKey → Map<agentName, [{x,y}]>
  const groups = new Map();

  rows.forEach(row => {
    const iter = row.iteration;
    if (iter == null) return;
    Object.entries(row).forEach(([k, v]) => {
      if (SKIP_KEYS.has(k)) return;
      if (SKIP_PREFIX.some(p => k.startsWith(p))) return;
      if (typeof v !== 'number' || !isFinite(v)) return;

      let groupKey, agentName;
      const slash = k.indexOf('/');
      if (slash !== -1) {
        agentName = k.slice(0, slash);
        groupKey  = k.slice(slash + 1);
      } else {
        agentName = '__scalar__';
        groupKey  = k;
      }

      if (!groups.has(groupKey)) groups.set(groupKey, new Map());
      const g = groups.get(groupKey);
      if (!g.has(agentName)) g.set(agentName, []);
      g.get(agentName).push({ x: iter, y: v });
    });
  });

  const agentOrder = [];
  groups.forEach(agentMap => {
    agentMap.forEach((_, agent) => {
      if (agent !== '__scalar__' && !agentOrder.includes(agent)) agentOrder.push(agent);
    });
  });

  return Array.from(groups.entries()).map(([key, agentMap]) => {
    const isScalar = agentMap.size === 1 && agentMap.has('__scalar__');
    const series = Array.from(agentMap.entries()).map(([agent, data]) => {
      const colorIdx = agentOrder.indexOf(agent);
      const color = isScalar
        ? (SCALAR_COLORS[key] || '#aaaaaa')
        : AGENT_COLORS[colorIdx % AGENT_COLORS.length];
      return { name: isScalar ? key : agent, color, data };
    });
    return { key, title: key, isScalar, series };
  });
}

function renderMetrics() {
  const inner = document.getElementById('metrics-inner');
  if (!metricsData.length) {
    inner.innerHTML = '<span style="color:var(--muted);font-size:11px;padding:4px">No metrics data.</span>';
    return;
  }

  const groups = buildChartGroups(metricsData);
  inner.innerHTML = '';

  // Priority ordering: put the most useful charts first
  const ORDER = ['mean_return','success_rate','total_loss','act_loss','perc_loss',
                 'kl','entropy','mean_advantage','mean_surprise','wrong_rate','mean_correct'];
  groups.sort((a, b) => {
    const ia = ORDER.indexOf(a.key), ib = ORDER.indexOf(b.key);
    if (ia === -1 && ib === -1) return a.key < b.key ? -1 : 1;
    if (ia === -1) return 1;
    if (ib === -1) return -1;
    return ia - ib;
  });

  const currentIter = currentTrace ? currentTrace.meta.iteration : null;
  const traceIters  = tracesList.map(t => iterFromName(t)).filter(x => x != null);

  groups.forEach(group => {
    const wrap = document.createElement('div');
    wrap.className = 'chart-wrap';

    const canvas = document.createElement('canvas');
    canvas.width  = CHART_W;
    canvas.height = CHART_H;
    canvas.dataset.groupKey = group.key;
    drawChart(canvas, group, currentIter, traceIters);
    setupChartEvents(canvas, group, traceIters);

    const legend = document.createElement('div');
    legend.className = 'chart-legend';
    if (!group.isScalar) {
      group.series.forEach(s => {
        legend.innerHTML +=
          '<div class="legend-item">' +
          '<div class="legend-dot" style="background:' + s.color + '"></div>' +
          esc(s.name) + '</div>';
      });
    }

    wrap.appendChild(canvas);
    wrap.appendChild(legend);
    inner.appendChild(wrap);
  });
}

function updateMetricsMarker() {
  const currentIter = currentTrace ? currentTrace.meta.iteration : null;
  const traceIters  = tracesList.map(t => iterFromName(t)).filter(x => x != null);
  const groups = buildChartGroups(metricsData);
  const groupMap = new Map(groups.map(g => [g.key, g]));

  document.querySelectorAll('#metrics-inner canvas').forEach(canvas => {
    const key = canvas.dataset.groupKey;
    const group = groupMap.get(key);
    if (group) drawChart(canvas, group, currentIter, traceIters);
  });
}

// ── Chart drawing (canvas) ────────────────────────────────────────────────────
const PL = 42, PR = 10, PT = 20, PB = 26;

function chartScale(canvas, group) {
  const IW = canvas.width - PL - PR;
  const IH = canvas.height - PT - PB;
  const allX = group.series.flatMap(s => s.data.map(d => d.x));
  const allY = group.series.flatMap(s => s.data.map(d => d.y));
  const minX = Math.min(...allX), maxX = Math.max(...allX);
  const minY = Math.min(...allY), maxY = Math.max(...allY);
  const padY = Math.max((maxY - minY) * 0.08, Math.abs(maxY) * 0.01, 1e-6);
  const yLo = minY - padY, yHi = maxY + padY;
  const sx = x => PL + (maxX > minX ? (x - minX) / (maxX - minX) * IW : IW / 2);
  const sy = y => PT + IH - (yHi > yLo ? (y - yLo) / (yHi - yLo) * IH : IH / 2);
  return { IW, IH, minX, maxX, yLo, yHi, sx, sy };
}

function drawChart(canvas, group, currentIter, traceIters, hoverX) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const sc = chartScale(canvas, group);
  const { IW, IH, minX, maxX, yLo, yHi, sx, sy } = sc;
  canvas._sc = sc; canvas._group = group;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0a1c2e';
  ctx.fillRect(0, 0, W, H);

  // Title
  ctx.fillStyle = '#667';
  ctx.font = '10px system-ui';
  ctx.textAlign = 'center';
  ctx.fillText(group.title, W / 2, 13);

  // Grid lines + Y axis labels
  const nY = 4;
  for (let i = 0; i <= nY; i++) {
    const gy = PT + i * IH / nY;
    ctx.strokeStyle = '#122030';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PL, gy); ctx.lineTo(W - PR, gy); ctx.stroke();
    const val = yHi - i * (yHi - yLo) / nY;
    ctx.fillStyle = '#445';
    ctx.font = '8px monospace';
    ctx.textAlign = 'right';
    const fmt = Math.abs(val) < 0.01 ? val.toExponential(1)
              : Math.abs(val) < 10   ? val.toFixed(3)
              : Math.abs(val) < 1000 ? val.toFixed(1)
              : Math.round(val).toString();
    ctx.fillText(fmt, PL - 3, gy + 3);
  }

  // X axis labels
  const nX = 4;
  ctx.fillStyle = '#445';
  ctx.font = '8px monospace';
  for (let i = 0; i <= nX; i++) {
    const gx = PL + i * IW / nX;
    const it = Math.round(minX + i * (maxX - minX) / nX);
    ctx.textAlign = 'center';
    ctx.fillText(it, gx, H - 4);
  }

  // Trace tick marks on x-axis
  ctx.fillStyle = '#334';
  traceIters.forEach(it => {
    if (it < minX || it > maxX) return;
    const tx = sx(it);
    ctx.fillRect(tx - 0.5, PT + IH, 1, 4);
  });

  // Series lines
  group.series.forEach(s => {
    if (!s.data.length) return;
    const sorted = [...s.data].sort((a, b) => a.x - b.x);
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    sorted.forEach((d, i) => {
      const px = sx(d.x), py = sy(d.y);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.stroke();
  });

  // Hover crosshair
  if (hoverX != null) {
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(hoverX, PT); ctx.lineTo(hoverX, PT + IH); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Current iteration marker
  if (currentIter != null) {
    const mx = sx(currentIter);
    ctx.strokeStyle = '#e94560';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(mx, PT); ctx.lineTo(mx, PT + IH); ctx.stroke();
    ctx.setLineDash([]);
    // Triangle pointer
    ctx.fillStyle = '#e94560';
    ctx.beginPath();
    ctx.moveTo(mx, PT + 3);
    ctx.lineTo(mx - 4, PT - 4);
    ctx.lineTo(mx + 4, PT - 4);
    ctx.closePath();
    ctx.fill();
  }
}

// ── Chart interaction ─────────────────────────────────────────────────────────
const tooltip = document.getElementById('chart-tooltip');

function setupChartEvents(canvas, group, traceIters) {
  function xToIter(canvasX) {
    const sc = canvas._sc;
    if (!sc) return null;
    const t = (canvasX - PL) / sc.IW;
    return sc.minX + t * (sc.maxX - sc.minX);
  }

  function valuesAtIter(targetIter) {
    const out = [];
    group.series.forEach(s => {
      const sorted = [...s.data].sort((a, b) => a.x - b.x);
      let best = null, bestD = Infinity;
      sorted.forEach(d => { const dd = Math.abs(d.x - targetIter); if (dd < bestD) { bestD = dd; best = d; } });
      if (best) out.push({ name: s.name, color: s.color, x: best.x, y: best.y });
    });
    return out;
  }

  canvas.addEventListener('mousemove', function(e) {
    const rect = this.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const sc = this._sc;
    if (!sc || cx < PL || cx > PL + sc.IW) {
      tooltip.style.display = 'none';
      return;
    }
    const currentIter = currentTrace ? currentTrace.meta.iteration : null;
    drawChart(this, this._group, currentIter, traceIters, cx);

    const targetIter = xToIter(cx);
    const vals = valuesAtIter(targetIter);
    if (!vals.length) { tooltip.style.display = 'none'; return; }
    const nearestIter = vals[0].x;
    let html = '<div class="tt-iter">iter ' + nearestIter + '</div>';
    vals.forEach(v => {
      const fmt = Math.abs(v.y) < 0.001 ? v.y.toExponential(2) : v.y.toFixed(4);
      html += '<div class="tt-row"><div class="tt-dot" style="background:' + v.color + '"></div>'
            + esc(v.name) + ' <b>' + fmt + '</b></div>';
    });
    tooltip.innerHTML = html;
    tooltip.style.display = 'block';
    const tx = e.clientX + 12, ty = e.clientY - 10;
    const ow = tooltip.offsetWidth, oh = tooltip.offsetHeight;
    tooltip.style.left = (tx + ow > window.innerWidth ? tx - ow - 16 : tx) + 'px';
    tooltip.style.top  = (ty + oh > window.innerHeight ? ty - oh    : ty) + 'px';
  });

  canvas.addEventListener('mouseleave', function() {
    tooltip.style.display = 'none';
    const currentIter = currentTrace ? currentTrace.meta.iteration : null;
    drawChart(this, this._group, currentIter, traceIters);
  });

  canvas.addEventListener('click', function(e) {
    const rect = this.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const sc = this._sc;
    if (!sc || cx < PL || cx > PL + sc.IW) return;
    const targetIter = xToIter(cx);
    const name = nearestTrace(targetIter);
    if (name) loadTrace(name);
  });
}

// ── Mode / view controls ──────────────────────────────────────────────────────
function setMode(m) {
  currentMode = m;
  ['env','context'].forEach(id => {
    document.getElementById('btn-' + id).classList.toggle('active', id === m);
  });
  document.getElementById('agent-tabs').style.display = m === 'context' ? 'flex' : 'none';
  render();
}

function buildAgentTabs(agents) {
  const bar = document.getElementById('agent-tabs');
  bar.innerHTML = '';
  agents.forEach(a => {
    const b = document.createElement('button');
    b.className = 'agent-tab' + (a === currentAgent ? ' active' : '');
    b.textContent = a;
    b.onclick = () => { currentAgent = a; setAgentActive(b); render(); };
    bar.appendChild(b);
  });
  bar.style.display = currentMode === 'context' ? 'flex' : 'none';
}

function setAgentActive(el) {
  document.querySelectorAll('.agent-tab').forEach(e => e.classList.remove('active'));
  el.classList.add('active');
}

// ── Render ────────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function renderMeta(trace) {
  const m = trace.meta;
  let html = '<div id="meta-card">';
  Object.entries(m).forEach(([k, v]) => {
    html += '<div class="meta-item"><span class="meta-key">' + esc(k) + '</span>'
          + '<span class="meta-val">' + esc(v) + '</span></div>';
  });
  return html + '</div>';
}

function renderEnvLog(trace) {
  let html = renderMeta(trace);
  html += '<div class="section-hdr">Environment log</div>';
  html += '<div id="agents-bar">'
        + trace.participating_agents.map(a =>
            '<span class="agent-chip">' + esc(a) + '</span>').join('') + '</div>';

  trace.environment_log.forEach(ev => {
    if (ev.type === 'act') {
      let lbl = '[' + esc(ev.agent) + ' → ENV]';
      if (ev.routed_to)
        lbl += ' &nbsp;<span class="routing-arrow">↓ routed to ' + esc(ev.routed_to) + '</span>';
      html += '<div class="env-event ev-act">'
            + '<div class="env-hdr">' + lbl + '</div>'
            + '<div class="env-body">' + esc(ev.text) + '</div></div>';
    } else {
      let cls, lbl;
      if (ev.source_type === 'env') {
        cls = 'ev-obs-env';
        lbl = '[ENV → ' + esc(ev.agent) + ']';
      } else if (ev.source_type === 'agent') {
        cls = 'ev-obs-routed';
        lbl = '[' + esc(ev.source_agent) + ' → ' + esc(ev.agent) + ']';
      } else {
        cls = 'ev-obs-routed';
        lbl = '[' + esc(ev.source_agent) + '+ENV → ' + esc(ev.agent) + ']';
      }
      html += '<div class="env-event ' + cls + '">'
            + '<div class="env-hdr">' + lbl + '</div>'
            + '<div class="env-body">' + esc(ev.text) + '</div>';
      if (ev.source_type === 'agent+env' && ev.env_prefix)
        html += '<div class="prefix-detail">env prefix: ' + esc(ev.env_prefix) + '</div>';
      html += '</div>';
    }
  });
  return html;
}

function renderContextAnnotated(turns) {
  let html = '';
  turns.forEach(t => {
    if (t.type === 'prompt') {
      html += '<div class="turn turn-prompt">'
            + '<div class="turn-lbl">System prompt</div>'
            + '<div class="turn-body">' + esc(t.text) + '</div></div>';
    } else if (t.type === 'obs') {
      let src = t.source_type === 'env' ? 'env'
              : t.source_type === 'agent' ? '← ' + t.source_agent
              : '← ' + t.source_agent + ' + env';
      html += '<div class="turn turn-obs">'
            + '<div class="turn-lbl">Observation &nbsp; ' + esc(src) + '</div>'
            + '<div class="turn-body">' + esc(t.text) + '</div></div>';
    } else {
      html += '<div class="turn turn-act">'
            + '<div class="turn-lbl">Action</div>'
            + '<div class="turn-body">' + esc(t.text) + '</div></div>';
    }
  });
  return html;
}

function renderContextRaw(turns) {
  let raw = '';
  turns.forEach(t => { raw += t.text; });
  return '<div id="raw-view">' + esc(raw) + '</div>';
}

function render() {
  if (!currentTrace) return;
  const content = document.getElementById('content');
  if (currentMode === 'env') {
    content.innerHTML = renderEnvLog(currentTrace);
  } else {
    const turns = currentTrace.agent_contexts[currentAgent] || [];
    let html = renderMeta(currentTrace);
    html += '<div style="display:flex;gap:7px;margin-bottom:10px;align-items:center">';
    html += '<div class="section-hdr" style="margin:0">Context: ' + esc(currentAgent) + '</div>';
    html += '<div style="margin-left:auto;display:flex;gap:5px">';
    html += '<button class="tb-btn' + (annotate ? ' active' : '') + '" onclick="setAnnotate(true)">Annotated</button>';
    html += '<button class="tb-btn' + (!annotate ? ' active' : '') + '" onclick="setAnnotate(false)">Raw</button>';
    html += '</div></div>';
    html += annotate ? renderContextAnnotated(turns) : renderContextRaw(turns);
    content.innerHTML = html;
  }
}

function setAnnotate(val) { annotate = val; render(); }

// ── Boot ──────────────────────────────────────────────────────────────────────
loadExperiments();
</script>
</body>
</html>
"""


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    root: Path = Path("outputs")

    def log_message(self, fmt, *args):
        pass

    def _respond(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict) -> None:
        self._respond(200, "application/json", json.dumps(data).encode())

    def _error(self, code: int, msg: str) -> None:
        self._respond(code, "text/plain", msg.encode())

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path   = parsed.path
        qs     = parse_qs(parsed.query)

        if path in ("/", ""):
            self._respond(200, "text/html; charset=utf-8", _HTML.encode())

        elif path == "/api/experiments":
            exps = sorted(
                d.name for d in self.root.iterdir()
                if d.is_dir() and (d / "traces").is_dir()
            ) if self.root.is_dir() else []
            self._json({"experiments": exps})

        elif path == "/api/traces":
            exp = qs.get("exp", [""])[0]
            traces_dir = self.root / exp / "traces"
            names = sorted(p.stem for p in traces_dir.glob("iter_*.json")) \
                    if traces_dir.is_dir() else []
            # Fall back to .txt if no json traces yet
            if not names and traces_dir.is_dir():
                names = sorted(p.stem for p in traces_dir.glob("iter_*.txt"))
            self._json({"traces": names})

        elif path == "/api/trace":
            exp   = qs.get("exp",   [""])[0]
            trace = qs.get("trace", [""])[0]
            json_path = self.root / exp / "traces" / f"{trace}.json"
            txt_path  = self.root / exp / "traces" / f"{trace}.txt"

            if json_path.exists():
                with open(json_path) as f:
                    self._json(json.load(f))
            elif txt_path.exists():
                with open(txt_path) as f:
                    text = f.read()
                self._json({
                    "meta": {"iteration": trace},
                    "participating_agents": ["(raw)"],
                    "environment_log": [],
                    "agent_contexts": {"(raw)": [{"type": "prompt", "text": text}]},
                })
            else:
                self._error(404, "Trace not found")

        elif path == "/api/metrics":
            exp = qs.get("exp", [""])[0]
            metrics_path = self.root / exp / "metrics.jsonl"
            rows: list[dict] = []
            if metrics_path.exists():
                with open(metrics_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                rows.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            self._json({"rows": rows})

        else:
            self._error(404, "Not found")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("output_root", nargs="?", default="outputs",
                        help="Root directory of experiment outputs (default: outputs)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind host (0.0.0.0 for SSH port-forward access)")
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    Handler.root = root

    print(f"Serving experiments from:  {root}")
    print(f"Listening on:              http://{args.host}:{args.port}")
    print(f"SSH port-forward:          ssh -L {args.port}:localhost:{args.port} <host>")
    print("Ctrl-C to stop.")

    server = HTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
