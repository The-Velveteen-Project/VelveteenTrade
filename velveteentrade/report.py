"""Public track-record page generator — the manifesto's commitment #2 made real.

    python -m velveteentrade report          # writes docs/index.html

Serve it with GitHub Pages (Settings → Pages → main /docs) and every
`report` + commit + push updates the public page. Self-contained HTML:
no external assets, light/dark aware, losses included by design.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

from .memory import Journal

# Palette roles (validated reference palette; swap for brand values later).
_CSS = """
:root {
  color-scheme: light dark;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --series: #2a78d6;
  --good: #006300; --bad: #d03b3b; --border: rgba(11,11,11,0.10);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835; --series: #3987e5;
    --good: #0ca30c; --bad: #e66767; --border: rgba(255,255,255,0.10);
  }
}
:root[data-theme="dark"] {
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --axis: #383835; --series: #3987e5;
  --good: #0ca30c; --bad: #e66767; --border: rgba(255,255,255,0.10);
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 880px; margin: 0 auto; padding: 40px 20px 80px; }
h1 { font-size: 26px; margin: 0 0 4px; }
.sub { color: var(--ink-2); margin: 0 0 28px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr));
  gap: 12px; margin-bottom: 28px; }
.tile { background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px; }
.tile .k { font-size: 12px; color: var(--muted); text-transform: uppercase;
  letter-spacing: .04em; }
.tile .v { font-size: 24px; font-weight: 650; margin-top: 2px; }
.tile .d { font-size: 13px; margin-top: 2px; }
.up { color: var(--good); } .down { color: var(--bad); }
.card { background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 18px; margin-bottom: 28px; }
.card h2 { font-size: 16px; margin: 0 0 12px; }
.empty { color: var(--muted); font-size: 14px; padding: 24px 0; text-align: center; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th { text-align: left; color: var(--muted); font-weight: 500; font-size: 12px;
  text-transform: uppercase; letter-spacing: .04em; padding: 6px 8px;
  border-bottom: 1px solid var(--grid); }
td { padding: 8px; border-bottom: 1px solid var(--grid); vertical-align: top;
  font-variant-numeric: tabular-nums; }
.tblwrap { overflow-x: auto; }
details summary { cursor: pointer; color: var(--ink-2); }
details p { margin: 6px 0 0; color: var(--ink-2); }
.cite { display: inline-block; background: color-mix(in srgb, var(--series) 12%, transparent);
  color: var(--series); border-radius: 4px; padding: 0 6px; font-size: 12px;
  margin: 2px 4px 0 0; }
.act-BUY { color: var(--good); font-weight: 600; }
.act-SELL { color: var(--bad); font-weight: 600; }
.act-HOLD { color: var(--muted); }
.foot { color: var(--muted); font-size: 13px; border-top: 1px solid var(--grid);
  padding-top: 16px; }
svg text { font: 11px system-ui, sans-serif; fill: var(--muted); }
.tooltip { position: absolute; pointer-events: none; background: var(--surface);
  border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px;
  font-size: 12px; display: none; box-shadow: 0 2px 8px rgba(0,0,0,.12); }
"""

_CHART_JS = """
(function () {
  const svg = document.getElementById('eq-svg');
  if (!svg) return;
  const pts = JSON.parse(svg.dataset.points);
  const tip = document.getElementById('eq-tip');
  const wrap = document.getElementById('eq-wrap');
  svg.addEventListener('mousemove', (e) => {
    const r = svg.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width * 720;
    let best = pts[0];
    for (const p of pts) if (Math.abs(p.x - x) < Math.abs(best.x - x)) best = p;
    tip.style.display = 'block';
    tip.style.left = (best.x / 720 * r.width + 12) + 'px';
    tip.style.top = (best.y / 220 * r.height - 8) + 'px';
    tip.textContent = best.label;
  });
  svg.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
})();
"""


def _fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def _equity_chart(history: list[tuple[str, float]]) -> str:
    if len(history) < 2:
        return ('<div class="empty">La curva de equity se dibuja a partir del segundo ciclo — '
                'vuelve mañana. La transparencia empieza vacía, no fabricada.</div>')
    W, H, PAD = 720, 220, 36
    values = [v for _, v in history]
    lo, hi = min(values), max(values)
    span = (hi - lo) or max(hi * 0.01, 1.0)
    lo -= span * 0.08
    hi += span * 0.08
    n = len(history)
    pts = []
    for i, (ts, v) in enumerate(history):
        x = PAD + (W - 2 * PAD) * (i / (n - 1))
        y = H - PAD - (H - 2 * PAD) * ((v - lo) / (hi - lo))
        pts.append({"x": round(x, 1), "y": round(y, 1),
                    "label": f"{ts[:10]} · {_fmt_money(v)}"})
    poly = " ".join(f"{p['x']},{p['y']}" for p in pts)
    grid = ""
    for frac in (0.0, 0.5, 1.0):
        y = H - PAD - (H - 2 * PAD) * frac
        val = lo + (hi - lo) * frac
        grid += (f'<line x1="{PAD}" y1="{y:.1f}" x2="{W - PAD}" y2="{y:.1f}" '
                 f'stroke="var(--grid)" stroke-width="1"/>'
                 f'<text x="{PAD - 6}" y="{y + 4:.1f}" text-anchor="end">{val:,.0f}</text>')
    last = pts[-1]
    x_first, x_last = history[0][0][:10], history[-1][0][:10]
    return (
        f'<div id="eq-wrap" style="position:relative">'
        f'<svg id="eq-svg" viewBox="0 0 {W} {H}" width="100%" role="img" '
        f"aria-label=\"Curva de equity de {x_first} a {x_last}\" "
        f"data-points='{json.dumps(pts)}'>"
        f"{grid}"
        f'<line x1="{PAD}" y1="{H - PAD}" x2="{W - PAD}" y2="{H - PAD}" '
        f'stroke="var(--axis)" stroke-width="1"/>'
        f'<text x="{PAD}" y="{H - 8}">{x_first}</text>'
        f'<text x="{W - PAD}" y="{H - 8}" text-anchor="end">{x_last}</text>'
        f'<polyline points="{poly}" fill="none" stroke="var(--series)" '
        f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{last["x"]}" cy="{last["y"]}" r="4" fill="var(--series)" '
        f'stroke="var(--surface)" stroke-width="2"/>'
        f'</svg><div class="tooltip" id="eq-tip"></div></div>'
        f"<script>{_CHART_JS}</script>"
    )


def _decision_rows(journal: Journal, limit: int = 60) -> str:
    rows = []
    for r in journal.recent(limit):
        d = r.get("decision") or {}
        v = r.get("verdict") or {}
        if not d and not v:
            continue
        action = d.get("action", v.get("action", "—"))
        thesis = html.escape(d.get("thesis", ""))
        invalidation = html.escape(d.get("invalidation", ""))
        cites = "".join(f'<span class="cite">{html.escape(c)}</span>'
                        for c in (d.get("canon_citations") or []))
        reason = html.escape("; ".join(v.get("reasons", []))[:220])
        detail = ""
        if thesis:
            detail = (f"<details><summary>tesis</summary><p>{thesis}</p>"
                      + (f"<p><b>Invalidación:</b> {invalidation}</p>" if invalidation else "")
                      + (f"<p>{cites}</p>" if cites else "")
                      + (f"<p><b>Gate:</b> {reason}</p>" if reason else "")
                      + "</details>")
        elif reason:
            detail = f'<span style="color:var(--ink-2)">{reason}</span>'
        status = "✅ ejecutada" if r.get("executed") else (
            "✋ vetada" if v and not v.get("approved") else "—")
        rows.append(
            f"<tr><td>{r['ts'][:10]}</td><td>{html.escape(r['symbol'])}</td>"
            f'<td class="act-{html.escape(str(action))}">{html.escape(str(action))}</td>'
            f"<td>{d.get('conviction', '—')}</td><td>{status}</td><td>{detail}</td></tr>"
        )
    return "".join(rows)


def generate(journal: Journal, capital: float | None) -> str:
    history = journal.equity_history()
    equity_now = history[-1][1] if history else (capital or 0.0)
    equity_start = history[0][1] if history else equity_now
    pnl = equity_now - equity_start
    pnl_pct = (pnl / equity_start * 100) if equity_start else 0.0
    pnl_cls = "up" if pnl >= 0 else "down"
    decisions = [r for r in journal.recent(500) if r.get("decision")]
    executed = sum(1 for r in journal.recent(500) if r.get("executed"))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    tiles = f"""
    <div class="tiles">
      <div class="tile"><div class="k">Equity</div>
        <div class="v">{_fmt_money(equity_now)}</div>
        <div class="d {pnl_cls}">{'+' if pnl >= 0 else ''}{_fmt_money(pnl)} ({pnl_pct:+.2f}%)</div></div>
      <div class="tile"><div class="k">Ciclos registrados</div>
        <div class="v">{len(history)}</div></div>
      <div class="tile"><div class="k">Decisiones analizadas</div>
        <div class="v">{len(decisions)}</div></div>
      <div class="tile"><div class="k">Órdenes ejecutadas</div>
        <div class="v">{executed}</div></div>
    </div>"""

    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VelveteenTrade — Track Record</title>
<style>{_CSS}</style></head><body>
<div class="wrap">
  <h1>VelveteenTrade · Track Record</h1>
  <p class="sub">Cada operación del sistema, con su tesis y su evidencia — pérdidas incluidas.
  Cuenta de práctica (paper trading), capital de trabajo {_fmt_money(capital) if capital else 'n/d'}.
  Actualizado {now}.</p>
  {tiles}
  <div class="card"><h2>Curva de equity</h2>{_equity_chart(history)}</div>
  <div class="card"><h2>Diario de decisiones</h2>
    <div class="tblwrap"><table>
      <thead><tr><th>Fecha</th><th>Símbolo</th><th>Acción</th><th>Conv.</th>
      <th>Resultado</th><th>Detalle</th></tr></thead>
      <tbody>{_decision_rows(journal) or '<tr><td colspan="6" class="empty">Aún sin decisiones registradas.</td></tr>'}</tbody>
    </table></div></div>
  <p class="foot">VelveteenTrade es un sistema experimental en fase de evidencia.
  Nada aquí es asesoría de inversión ni promesa de rentabilidad — publicamos el
  historial completo precisamente porque nadie honesto puede prometer retornos.
  Un producto de The Velveteen Project.</p>
</div></body></html>"""


def write_report(journal: Journal, capital: float | None, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(generate(journal, capital))
    return out_path
