/* FlashGuard v4 — app.js
   TradingView-style terminal. lightweight-charts candlestick. Upstox-first. */

const API = "";
const UPSTOX_TOKEN_KEY = "fg_upstox_token";

let currentInterval  = "1d";
let currentTicker    = "RELIANCE";
let priceChart       = null;   // LightweightCharts instance
let priceSeries      = null;   // candlestick series
let volumeSeries     = null;   // histogram series
let historyChart     = null;   // Chart.js history
let autoInterval     = null;
let autoActive       = false;
let selectedFile     = null;
let selectedInstrumentKey = null;
let searchDebounceTimer   = null;
let timelinePeriod   = "6mo";
const scanHistory    = [];

// ── Interval → period mapping ─────────────────────────────────────────────────
const PERIOD_MAP = {
    "1m":  "5d",
    "5m":  "1mo",
    "15m": "3mo",
    "30m": "3mo",
    "1h":  "6mo",
    "1d":  "2y",
    "1wk": "10y",
    "1mo": "10y",
};

const INTRADAY = new Set(["1m","5m","15m","30m","1h"]);

// ── Watchlist ─────────────────────────────────────────────────────────────────
const WATCHLIST = [
    {sym:"NIFTY50",  label:"NIFTY 50"},
    {sym:"BANKNIFTY",label:"BANK NIFTY"},
    {sym:"RELIANCE", label:"RELIANCE"},
    {sym:"TCS",      label:"TCS"},
    {sym:"HDFCBANK", label:"HDFC BANK"},
    {sym:"INFY",     label:"INFOSYS"},
    {sym:"ICICIBANK",label:"ICICI BANK"},
    {sym:"SBIN",     label:"SBI"},
    {sym:"WIPRO",    label:"WIPRO"},
    {sym:"MARUTI",   label:"MARUTI"},
];

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initClock();
    initTabs();
    initIntervalPills();
    initWatchlist();
    initTokenInput();
    initAutoRefresh();
    initCSV();
    initStockSearch();
    loadModels();

    document.getElementById("btnScan").addEventListener("click", () => runScan());
    document.getElementById("btnPortfolio").addEventListener("click", runPortfolio);

    refreshWatchlistPrices();
    setInterval(refreshWatchlistPrices, 15000);

    // Resize chart on window resize
    window.addEventListener("resize", () => {
        if (priceChart) priceChart.applyOptions({ width: document.getElementById("priceChartContainer").clientWidth });
    });
});

// ── Clock ─────────────────────────────────────────────────────────────────────
function initClock() {
    const el = document.getElementById("clock");
    const tick = () => {
        const n = new Date(), p = v => String(v).padStart(2,"0");
        el.textContent = `${p(n.getHours())}:${p(n.getMinutes())}:${p(n.getSeconds())}`;
    };
    tick(); setInterval(tick, 1000);
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function initTabs() {
    document.querySelectorAll(".tab").forEach(tab => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".tab-body").forEach(b => b.classList.remove("active"));
            tab.classList.add("active");
            const body = document.querySelector(`.tab-body[data-tab="${tab.dataset.tab}"]`);
            if (body) body.classList.add("active");
            if (tab.dataset.tab === "chart" && priceChart) {
                setTimeout(() => {
                    const c = document.getElementById("priceChartContainer");
                    if (c) priceChart.applyOptions({ width: c.clientWidth, height: c.clientHeight });
                }, 50);
            }
        });
    });
}

// ── Interval pills ────────────────────────────────────────────────────────────
function initIntervalPills() {
    document.querySelectorAll(".pill").forEach(p => {
        p.addEventListener("click", () => {
            document.querySelectorAll(".pill").forEach(b => b.classList.remove("active"));
            p.classList.add("active");
            currentInterval = p.dataset.iv;
            setText("bInterval", currentInterval.toUpperCase());
        });
    });
}

// ── Token ─────────────────────────────────────────────────────────────────────
function initTokenInput() {
    const inp = document.getElementById("tokenInput");
    const saved = localStorage.getItem(UPSTOX_TOKEN_KEY);
    if (saved) { inp.value = saved; updateTokenStatus(saved); }
    inp.addEventListener("input", () => {
        const val = inp.value.trim();
        if (val) localStorage.setItem(UPSTOX_TOKEN_KEY, val);
        updateTokenStatus(val);
    });
}

function updateTokenStatus(token) {
    const el = document.getElementById("tokenStatus");
    if (!el) return;
    if (token && token.length > 20) {
        el.textContent = "✓ Token set — using Upstox live data";
        el.className   = "token-status ok";
    } else {
        el.textContent = "Enter token for live data";
        el.className   = "token-status";
    }
}

function getToken() {
    return document.getElementById("tokenInput")?.value.trim() || localStorage.getItem(UPSTOX_TOKEN_KEY) || "";
}

// ── Watchlist ─────────────────────────────────────────────────────────────────
function initWatchlist() {
    const el = document.getElementById("watchlist");
    el.innerHTML = WATCHLIST.map(w => `
        <div class="watch-item" data-sym="${w.sym}" onclick="selectWatchItem('${w.sym}')">
            <span class="watch-name">${w.label}</span>
            <span class="watch-price" id="wp-${w.sym}">—</span>
            <span class="watch-chg"   id="wc-${w.sym}">—</span>
        </div>`).join("");
    document.querySelector(`.watch-item[data-sym="${currentTicker}"]`)?.classList.add("active");
}

function selectWatchItem(sym) {
    document.querySelectorAll(".watch-item").forEach(i => i.classList.remove("active"));
    document.querySelector(`.watch-item[data-sym="${sym}"]`)?.classList.add("active");
    currentTicker = sym;
    document.getElementById("tickerInput").value = sym;
    document.getElementById("stockSearchInput").value = sym;
    selectedInstrumentKey = null;
    document.getElementById("instrumentKeyInput").value = "";
    runScan();
}

async function refreshWatchlistPrices() {
    const token = getToken();
    if (!token) return;
    try {
        const res  = await fetch(`${API}/api/market-overview`, {
            method:"POST", headers:{"Content-Type":"application/json"},
            body: JSON.stringify({token})
        });
        const data = await res.json();
        if (data.data && data.data.length) {
            data.data.forEach(q => {
                WATCHLIST.forEach(w => {
                    if (q.name.includes(w.label.split(" ")[0]) || q.name === w.label) {
                        const pEl = document.getElementById(`wp-${w.sym}`);
                        const cEl = document.getElementById(`wc-${w.sym}`);
                        if (pEl) pEl.textContent = "₹" + (q.ltp||0).toLocaleString("en-IN");
                        if (cEl) {
                            const up = (q.change_pct||0) >= 0;
                            cEl.textContent = (up?"+":"") + (q.change_pct||0).toFixed(2) + "%";
                            cEl.className   = "watch-chg " + (up?"up":"dn");
                        }
                    }
                });
            });
        }
    } catch(e) {}
}

// ── Stock Search ──────────────────────────────────────────────────────────────
function initStockSearch() {
    const input   = document.getElementById("stockSearchInput");
    const results = document.getElementById("stockSearchResults");
    let activeIdx = -1;

    input.addEventListener("input", () => {
        clearTimeout(searchDebounceTimer);
        const q = input.value.trim();
        if (q.length < 1) { results.innerHTML = ""; results.classList.remove("open"); return; }
        searchDebounceTimer = setTimeout(() => fetchStockResults(q), 300);
    });

    input.addEventListener("keydown", e => {
        const items = results.querySelectorAll(".stock-result-item");
        if      (e.key === "ArrowDown") { e.preventDefault(); activeIdx = Math.min(activeIdx+1, items.length-1); highlightItem(items, activeIdx); }
        else if (e.key === "ArrowUp")   { e.preventDefault(); activeIdx = Math.max(activeIdx-1, 0); highlightItem(items, activeIdx); }
        else if (e.key === "Enter")     { e.preventDefault(); if (activeIdx >= 0 && items[activeIdx]) items[activeIdx].click(); else if (input.value.trim()) runScan(); }
        else if (e.key === "Escape")    { results.innerHTML = ""; results.classList.remove("open"); activeIdx = -1; }
    });

    document.addEventListener("click", e => {
        if (!document.getElementById("tickerSearchWrap").contains(e.target)) {
            results.innerHTML = ""; results.classList.remove("open"); activeIdx = -1;
        }
    });
}

function highlightItem(items, idx) {
    items.forEach((el, i) => el.classList.toggle("active", i === idx));
    if (items[idx]) items[idx].scrollIntoView({ block:"nearest" });
}

async function fetchStockResults(query) {
    const results = document.getElementById("stockSearchResults");
    try {
        const res    = await fetch(`${API}/api/search-stocks?q=${encodeURIComponent(query)}`);
        const stocks = await res.json();
        if (!stocks.length) {
            results.innerHTML = `<div class="stock-result-empty">No results for "${query}"</div>`;
            results.classList.add("open"); return;
        }
        results.innerHTML = stocks.map(s => `
            <div class="stock-result-item" data-symbol="${s.symbol}" data-key="${s.instrument_key}" data-name="${s.name}">
                <span class="stock-result-symbol">${s.symbol}</span>
                <span class="stock-result-name">${s.name}</span>
            </div>`).join("");
        results.classList.add("open");
        results.querySelectorAll(".stock-result-item").forEach(item => {
            item.addEventListener("click", () => selectStock(item.dataset.symbol, item.dataset.key, item.dataset.name));
        });
    } catch {
        results.innerHTML = `<div class="stock-result-empty">Search error</div>`;
        results.classList.add("open");
    }
}

function selectStock(symbol, instrumentKey, name) {
    document.getElementById("stockSearchInput").value = `${symbol} — ${name}`;
    document.getElementById("tickerInput").value      = symbol;
    document.getElementById("instrumentKeyInput").value = instrumentKey;
    selectedInstrumentKey = instrumentKey;
    currentTicker = symbol;
    const r = document.getElementById("stockSearchResults");
    r.innerHTML = ""; r.classList.remove("open");
    document.querySelectorAll(".watch-item").forEach(i => i.classList.remove("active"));
    document.querySelector(`.watch-item[data-sym="${symbol}"]`)?.classList.add("active");
}

// ── Models ────────────────────────────────────────────────────────────────────
async function loadModels() {
    try {
        const models = await (await fetch(`${API}/api/models`)).json();
        const sel    = document.getElementById("modelSelect");
        sel.innerHTML = "";
        const valid  = models.filter(m => !m.error);
        if (!valid.length) { sel.innerHTML = "<option>No models loaded</option>"; setConn(false); return; }
        valid.sort((a,b) => a.name.includes("minute") ? -1 : 1).forEach(m => {
            const o = document.createElement("option");
            o.value = m.name;
            o.textContent = m.name.replace(".keras","").replace(".h5","") + ` (${m.timesteps}×${m.features})`;
            sel.appendChild(o);
        });
        setConn(true);
    } catch {
        showToast("Cannot connect to API — run api_server.py first");
        setConn(false);
    }
}

function setConn(on) {
    const dot = document.getElementById("connDot");
    const lbl = document.getElementById("connLabel");
    if (dot) dot.className = "conn-dot" + (on?" live":"");
    if (lbl) lbl.textContent = on ? "LIVE" : "OFFLINE";
}

// ── Auto Refresh ──────────────────────────────────────────────────────────────
function initAutoRefresh() {
    const btn = document.getElementById("btnAuto");
    btn.addEventListener("click", () => {
        autoActive = !autoActive;
        if (autoActive) {
            btn.style.background = "#3fb950"; btn.style.color = "#000"; btn.textContent = "AUTO ✓";
            autoInterval = setInterval(() => runScan(true), 30000);
        } else {
            btn.style.background = ""; btn.style.color = ""; btn.textContent = "AUTO";
            clearInterval(autoInterval);
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN SCAN
// ═══════════════════════════════════════════════════════════════════════════════
async function runScan(silent = false) {
    const ticker = document.getElementById("tickerInput")?.value.trim() || currentTicker;
    const model  = document.getElementById("modelSelect")?.value;
    const token  = getToken();
    const instrumentKey = selectedInstrumentKey || document.getElementById("instrumentKeyInput")?.value || null;
    if (!ticker) return showToast("Search and select a stock first");
    if (!model)  return showToast("No model loaded");

    currentTicker = ticker;
    const period  = PERIOD_MAP[currentInterval] || "6mo";

    if (!silent) {
        document.getElementById("chartEmpty").classList.add("hidden");
        document.getElementById("priceChartContainer").classList.add("hidden");
        setText("chartTicker", ticker);
        setText("chartPrice", "Loading…");
    }

    try {
        const res  = await fetch(`${API}/api/predict`, {
            method:"POST", headers:{"Content-Type":"application/json"},
            body: JSON.stringify({ ticker, model, token, period, interval: currentInterval, threshold:0.20, instrument_key: instrumentKey })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        renderScan(data);
        setConn(true);
    } catch(e) {
        showToast("Scan failed: " + e.message);
        document.getElementById("chartEmpty").classList.remove("hidden");
        setConn(false);
    }
}

function renderScan(data) {
    const pct  = data.risk_pct  ?? 0;
    const prob = data.probability ?? 0;
    const band = data.band ?? "STABLE";
    const ohlc = data.ohlc ?? [];

    // Price & change
    const last  = ohlc.length ? ohlc[ohlc.length-1] : null;
    const prev  = ohlc.length > 1 ? ohlc[ohlc.length-2] : null;
    const price = data.live_price || data.latest_close || (last?.close ?? 0);
    const chg   = last && prev ? last.close - prev.close : 0;
    const chgP  = prev && prev.close ? (chg / prev.close * 100) : 0;
    const up    = chg >= 0;

    setText("chartTicker", data.ticker?.replace(".NS","") || currentTicker);
    setText("chartPrice",  "₹" + price.toLocaleString("en-IN", {minimumFractionDigits:2, maximumFractionDigits:2}));

    const chgEl = document.getElementById("chartChange");
    chgEl.textContent = `${up?"+":""}${chg.toFixed(2)} (${up?"+":""}${chgP.toFixed(2)}%)`;
    chgEl.className   = "chart-change " + (up?"up":"dn");

    if (last) {
        setText("ohlcO", "₹"+last.open.toLocaleString("en-IN"));
        setText("ohlcH", "₹"+last.high.toLocaleString("en-IN"));
        setText("ohlcL", "₹"+last.low.toLocaleString("en-IN"));
        setText("ohlcC", "₹"+last.close.toLocaleString("en-IN"));
        setText("ohlcV", (last.volume||0).toLocaleString());
    }
    setText("chartMeta", `${ohlc.length} bars · ${currentInterval.toUpperCase()} · ${data.latest_date??""}`);

    // Bottom strip
    const srcMap = {upstox:"Upstox ●", yfinance:"yFinance", demo:"Demo"};
    setText("bClose",    "₹" + price.toLocaleString("en-IN"));
    setText("bInterval", currentInterval.toUpperCase());
    setText("bBars",     ohlc.length.toString());
    setText("bSource",   srcMap[data.source] || data.source);
    setText("bUpdated",  new Date().toLocaleTimeString("en-IN",{hour:"2-digit",minute:"2-digit",second:"2-digit"}));

    // Draw chart
    drawCandleChart(ohlc);

    // Risk gauge
    drawRiskRing(prob);
    animateNum(document.getElementById("gaugePct"), pct);
    const bandEl = document.getElementById("riskBand");
    bandEl.textContent = band;
    bandEl.className   = "risk-band " + bandClass(band);
    setText("riskDesc",
        band === "HIGH RISK" ? "⚠️ High crash probability detected!" :
        band === "ELEVATED"  ? "Elevated volatility — monitor closely" :
                               "No significant crash risk detected");

    // Alert system
    const alertOver = document.getElementById("alertOverlay");
    const alertInd  = document.getElementById("alertIndicator");
    if (band === "HIGH RISK") {
        alertOver.classList.add("show");
        alertInd.classList.add("show");
    } else {
        alertOver.classList.remove("show");
        alertInd.classList.remove("show");
    }

    // Info panel
    setText("infoTicker", data.ticker?.replace(".NS","") || currentTicker);
    setText("infoModel",  (data.model||"").replace(".keras","").replace(".h5",""));
    setText("infoTs",     data.timesteps?.toString() || "—");
    setText("infoFeat",   data.features?.toString()  || "—");
    setText("infoDate",   data.latest_date || "—");
    const srcEl  = document.getElementById("infoSource");
    const srcCls = {upstox:"upstox",yfinance:"yfinance",demo:"demo"}[data.source]||"demo";
    srcEl.innerHTML = `<span class="source-pill ${srcCls}">${srcMap[data.source]||data.source}</span>`;

    // Feature Snapshot
    renderFeatureSnapshot(data.feature_values || {});

    // Scan history
    scanHistory.push({ t: new Date().toLocaleTimeString("en-IN",{hour:"2-digit",minute:"2-digit",second:"2-digit"}), v: pct });
    if (scanHistory.length > 30) scanHistory.shift();
    drawHistory();

    // Watchlist price update
    const wPEl = document.getElementById(`wp-${currentTicker}`);
    if (wPEl) wPEl.textContent = "₹" + price.toLocaleString("en-IN");
}

// ── Feature Snapshot ──────────────────────────────────────────────────────────
function renderFeatureSnapshot(feats) {
    const section = document.getElementById("featSnapSection");
    const grid    = document.getElementById("featSnapGrid");
    if (!feats || !Object.keys(feats).length) { section.style.display = "none"; return; }
    section.style.display = "block";
    const LABELS = {
        "return":"Return","log_return":"Log Return","volatility_5":"Vol 5",
        "volatility_10":"Vol 10","volatility_20":"Vol 20","momentum_5":"Mom 5",
        "momentum_10":"Mom 10","high_low_spread":"HL Spread",
        "open_close_return":"OC Return","price_acceleration":"Accel",
    };
    grid.innerHTML = Object.entries(feats).map(([k, v]) => {
        const cls = v > 0.001 ? "pos" : v < -0.001 ? "neg" : "neu";
        const fmt = Math.abs(v) < 0.01 ? v.toFixed(6) : v.toFixed(4);
        return `<div class="feat-snap-item">
            <div class="fsi-key">${LABELS[k]||k}</div>
            <div class="fsi-val ${cls}">${v >= 0 ? "+" : ""}${fmt}</div>
        </div>`;
    }).join("");
}

// ═══════════════════════════════════════════════════════════════════════════════
// CANDLESTICK CHART (lightweight-charts)
// ═══════════════════════════════════════════════════════════════════════════════
function toLwTime(dateStr) {
    if (INTRADAY.has(currentInterval)) {
        // Parse as UTC to avoid timezone shift issues
        const d = new Date(dateStr.endsWith('Z') || dateStr.includes('+') ? dateStr : dateStr + 'Z');
        return Math.floor(d.getTime() / 1000);
    }
    return dateStr.slice(0, 10);
}

function drawCandleChart(ohlc) {
    document.getElementById("chartEmpty").classList.add("hidden");
    const container = document.getElementById("priceChartContainer");
    container.classList.remove("hidden");

    // Destroy old chart
    if (priceChart) { priceChart.remove(); priceChart = null; priceSeries = null; volumeSeries = null; }
    if (!ohlc || !ohlc.length) return;

    priceChart = LightweightCharts.createChart(container, {
        autoSize: true,
        layout: {
            background: { color: "#020408" },
            textColor:   "#3d6080",
        },
        grid: {
            vertLines: { color: "rgba(0,210,255,0.05)" },
            horzLines: { color: "rgba(0,210,255,0.05)" },
        },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: "rgba(0,210,255,0.1)" },
        leftPriceScale:  { visible: false },
        timeScale: {
            borderColor: "rgba(0,210,255,0.1)",
            timeVisible: INTRADAY.has(currentInterval),
            secondsVisible: false,
        },
        watermark: { visible: false },
        localization: {
            timeFormatter: (time) => {
                const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
                if (INTRADAY.has(currentInterval)) {
                    // time is a Unix timestamp (seconds)
                    const d  = new Date(time * 1000);
                    const dd = String(d.getUTCDate()).padStart(2,'0');
                    const hh = String(d.getUTCHours()).padStart(2,'0');
                    const mm = String(d.getUTCMinutes()).padStart(2,'0');
                    return `${dd} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}  ${hh}:${mm}`;
                }
                // time is "YYYY-MM-DD" string for daily+
                const parts = String(time).split('-');
                if (parts.length === 3) {
                    return `${parseInt(parts[2])} ${MONTHS[parseInt(parts[1])-1]} ${parts[0]}`;
                }
                return String(time);
            },
        },
    });

    // Candlestick series
    priceSeries = priceChart.addCandlestickSeries({
        upColor:        "#00ff9d",
        downColor:      "#ff3d2e",
        borderUpColor:  "#00ff9d",
        borderDownColor:"#ff3d2e",
        wickUpColor:    "#00ff9d",
        wickDownColor:  "#ff3d2e",
    });

    // Volume histogram
    volumeSeries = priceChart.addHistogramSeries({
        priceFormat: { type: "volume" },
        priceScaleId: "vol",
        color: "rgba(0,210,255,0.2)",
    });
    priceChart.priceScale("vol").applyOptions({
        scaleMargins: { top: 0.82, bottom: 0 },
    });

    // Format data — deduplicate & sort by time
    const seen  = new Set();
    const candleData = [];
    const volData    = [];

    ohlc.forEach(d => {
        const t = toLwTime(d.date);
        const key = String(t);
        if (seen.has(key)) return;
        seen.add(key);
        candleData.push({ time: t, open: d.open, high: d.high, low: d.low, close: d.close });
        volData.push({
            time:  t,
            value: d.volume || 0,
            color: d.close >= d.open ? "rgba(0,255,157,0.25)" : "rgba(255,61,46,0.25)",
        });
    });

    // Sort ascending
    candleData.sort((a, b) => (a.time > b.time ? 1 : -1));
    volData.sort((a, b)    => (a.time > b.time ? 1 : -1));

    priceSeries.setData(candleData);
    volumeSeries.setData(volData);

    // Default zoom: show last N bars so daily view is shown initially
    const DEFAULT_BARS = {
        "1m":  390,  // ~1 trading day
        "5m":  78,   // ~1 trading day
        "15m": 26,   // ~1 trading day
        "30m": 13,   // ~1 trading day
        "1h":  30,   // ~1 month
        "1d":  30,   // ~1 month
        "1wk": 26,   // ~6 months
        "1mo": 24,   // ~2 years
    };
    const barsToShow = DEFAULT_BARS[currentInterval] || 30;

    if (candleData.length > barsToShow) {
        const lastBar  = candleData[candleData.length - 1];
        const firstBar = candleData[candleData.length - barsToShow];
        priceChart.timeScale().setVisibleRange({
            from: firstBar.time,
            to:   lastBar.time,
        });
    } else {
        priceChart.timeScale().fitContent();
    }

    // Update crosshair tooltip in chart header
    priceChart.subscribeCrosshairMove(param => {
        if (!param || !param.time || !param.seriesData) return;
        const bar = param.seriesData.get(priceSeries);
        if (!bar) return;
        setText("ohlcO", "₹" + bar.open.toLocaleString("en-IN"));
        setText("ohlcH", "₹" + bar.high.toLocaleString("en-IN"));
        setText("ohlcL", "₹" + bar.low.toLocaleString("en-IN"));
        setText("ohlcC", "₹" + bar.close.toLocaleString("en-IN"));
        const vBar = param.seriesData.get(volumeSeries);
        if (vBar) setText("ohlcV", (vBar.value||0).toLocaleString());
    });
}

// ── Risk Gauge Ring ───────────────────────────────────────────────────────────
function drawRiskRing(prob) {
    const canvas = document.getElementById("riskRing");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const sz  = 130;
    canvas.width = sz*dpr; canvas.height = sz*dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = sz+"px"; canvas.style.height = sz+"px";

    const cx=sz/2, cy=sz/2, r=52, lw=7, sa=-Math.PI/2;
    ctx.beginPath(); ctx.arc(cx,cy,r,0,2*Math.PI);
    ctx.lineWidth=lw; ctx.strokeStyle="rgba(0,210,255,0.08)"; ctx.stroke();

    if (prob > 0) {
        let arcStyle;
        try {
            if (typeof ctx.createConicGradient === "function") {
                const g = ctx.createConicGradient(sa,cx,cy);
                g.addColorStop(0,"#00ff9d"); g.addColorStop(0.35,"#ffc72c");
                g.addColorStop(0.65,"#ff3d2e"); g.addColorStop(1,"#ff3d2e");
                arcStyle = g;
            } else {
                arcStyle = prob < 0.13 ? "#00ff9d" : prob < 0.20 ? "#ffc72c" : "#ff3d2e";
            }
        } catch { arcStyle = "#00d2ff"; }
        const ea = sa + prob * 2 * Math.PI;
        ctx.beginPath(); ctx.arc(cx,cy,r,sa,ea);
        ctx.lineWidth=lw; ctx.strokeStyle=arcStyle; ctx.lineCap="round"; ctx.stroke();
        if (prob > 0.01) {
            const ex=cx+r*Math.cos(ea), ey=cy+r*Math.sin(ea);
            const gd = ctx.createRadialGradient(ex,ey,0,ex,ey,10);
            const rgb = prob<0.13?"0,255,157":prob<0.20?"255,199,44":"255,61,46";
            gd.addColorStop(0,`rgba(${rgb},.7)`); gd.addColorStop(1,`rgba(${rgb},0)`);
            ctx.beginPath(); ctx.arc(ex,ey,10,0,2*Math.PI); ctx.fillStyle=gd; ctx.fill();
        }
    }
}

// ── Scan History Chart ────────────────────────────────────────────────────────
function drawHistory() {
    const canvas = document.getElementById("historyChart");
    if (!canvas || !scanHistory.length) return;
    const ctx = canvas.getContext("2d");
    if (historyChart) { historyChart.destroy(); historyChart = null; }
    const grad = ctx.createLinearGradient(0,0,0,90);
    grad.addColorStop(0,"rgba(0,210,255,0.3)"); grad.addColorStop(1,"rgba(0,210,255,0)");
    historyChart = new Chart(ctx, {
        type:"line",
        data: {
            labels: scanHistory.map(s=>s.t),
            datasets:[{
                data:              scanHistory.map(s=>s.v),
                borderColor:       "#00d2ff",
                backgroundColor:   grad,
                borderWidth:       1.5,
                pointRadius:       2,
                pointBackgroundColor:"#00d2ff",
                fill:    true,
                tension: 0.3,
            }]
        },
        options:{
            responsive:true, maintainAspectRatio:false,
            plugins:{legend:{display:false},tooltip:{
                backgroundColor:"#080f1a",titleColor:"#e8f4ff",bodyColor:"#7ba8cc",
                cornerRadius:4,padding:8,
                titleFont:{family:"'Share Tech Mono',monospace",size:10},
                bodyFont: {family:"'Share Tech Mono',monospace",size:10},
                callbacks:{label:c=>`Risk: ${c.raw.toFixed(1)}%`}
            }},
            scales:{
                x:{display:false},
                y:{min:0,max:100,grid:{color:"rgba(0,210,255,0.06)"},
                   ticks:{color:"#1e3a52",font:{size:9}}}
            }
        }
    });
}



// ═══════════════════════════════════════════════════════════════════════════════
// PORTFOLIO
// ═══════════════════════════════════════════════════════════════════════════════
async function runPortfolio() {
    const raw    = document.getElementById("portTickers")?.value.trim();
    const model  = document.getElementById("modelSelect")?.value;
    const token  = getToken();
    if (!raw) return showToast("Enter tickers");
    const tickers   = raw.split(",").map(t=>t.trim()).filter(Boolean);
    const period    = PERIOD_MAP[currentInterval] || "6mo";
    const body_el   = document.getElementById("portBody");
    body_el.innerHTML = `<tr><td colspan="4" style="text-align:center;padding:18px"><div class="spinner" style="margin:0 auto"></div></td></tr>`;
    try {
        const res  = await fetch(`${API}/api/portfolio`, {
            method:"POST", headers:{"Content-Type":"application/json"},
            body: JSON.stringify({ tickers, model, token, period, interval:currentInterval, threshold:0.20 })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        body_el.innerHTML = data.results.map(r => {
            if (r.error) return `<tr><td style="font-weight:600">${r.ticker}</td><td colspan="3" style="color:var(--danger)">${r.error}</td></tr>`;
            const color = bandHex(r.band), pct = r.risk_pct;
            return `<tr>
                <td style="font-weight:600;font-family:var(--font-m)">${r.ticker.replace(".NS","")}</td>
                <td style="font-family:var(--font-m)">₹${(r.latest_close||0).toLocaleString("en-IN")}</td>
                <td>
                    <span style="color:${color};font-family:var(--font-m);font-weight:600">${pct.toFixed(1)}%</span>
                    <div class="risk-bar-bg"><div class="risk-bar-fill" style="width:${Math.min(pct*5,100)}%;background:${color}"></div></div>
                </td>
                <td><span style="color:${color};font-size:11px;font-weight:600">${r.band}</span></td>
            </tr>`;
        }).join("");
    } catch(e) {
        body_el.innerHTML = `<tr><td colspan="4" style="color:var(--danger);padding:14px">${e.message}</td></tr>`;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CSV ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════════
function initCSV() {
    const zone  = document.getElementById("dropZone");
    const input = document.getElementById("csvInput");
    zone.addEventListener("click",    () => input.click());
    zone.addEventListener("dragover",  e => { e.preventDefault(); zone.classList.add("drag"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
    zone.addEventListener("drop", e => {
        e.preventDefault(); zone.classList.remove("drag");
        if (e.dataTransfer.files[0]) setCSVFile(e.dataTransfer.files[0]);
    });
    input.addEventListener("change", () => { if (input.files[0]) setCSVFile(input.files[0]); });
    document.getElementById("btnCsv").addEventListener("click", runCSV);
}

function setCSVFile(f) {
    selectedFile = f;
    document.getElementById("dropZone").classList.add("has-file");
    document.getElementById("dropZone").innerHTML =
        `<div class="drop-icon">✅</div><div class="drop-text">${f.name}</div><div class="drop-hint">${(f.size/1024).toFixed(1)} KB</div>`;
}

async function runCSV() {
    if (!selectedFile) return showToast("Select a CSV file first");
    const model  = document.getElementById("modelSelect")?.value;
    const res_el = document.getElementById("csvResult");
    res_el.innerHTML = `<div style="display:flex;align-items:center;gap:8px;color:var(--text-2)"><div class="spinner"></div>Analyzing…</div>`;
    try {
        const fd = new FormData();
        fd.append("file", selectedFile); fd.append("model", model); fd.append("threshold", "0.20");
        const data = await (await fetch(`${API}/api/upload`, {method:"POST", body:fd})).json();
        if (data.error) throw new Error(data.error);
        const color = bandHex(data.band);
        res_el.innerHTML = `
            <div style="background:var(--panel);border:1px solid var(--border-2);border-radius:0;padding:14px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                    <span style="font-size:22px;font-weight:700;font-family:var(--font-d);color:${color}">${data.risk_pct.toFixed(1)}%</span>
                    <span style="color:${color};font-size:11px;font-weight:600;background:rgba(255,255,255,.05);padding:3px 10px">${data.band}</span>
                </div>
                <div style="font-size:11px;color:var(--text-2)">Rows: ${data.rows_loaded} · Model: ${(data.model||"").replace(".keras","")}</div>
            </div>`;
    } catch(e) {
        res_el.innerHTML = `<div style="color:var(--danger);font-size:12px">${e.message}</div>`;
    }
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function bandHex(b)  { return b==="STABLE"?"#00ff9d":b==="ELEVATED"?"#ffc72c":"#ff3d2e"; }
function bandClass(b){ return b==="STABLE"?"stable":b==="ELEVATED"?"elevated":"high"; }

function animateNum(el, target) {
    if (!el) return;
    const start = parseFloat(el.textContent)||0, diff = target-start, t0 = performance.now();
    const tick = now => {
        const p = Math.min((now-t0)/600,1), e = 1-Math.pow(1-p,3);
        el.textContent = (start+diff*e).toFixed(1);
        if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
}

function setText(id, txt) { const el = document.getElementById(id); if (el) el.textContent = txt; }

function showToast(msg) {
    const t = document.createElement("div"); t.className="toast"; t.textContent=msg;
    document.body.appendChild(t);
    setTimeout(() => { t.style.opacity="0"; t.style.transition="opacity .3s"; setTimeout(()=>t.remove(),300); }, 4000);
}
