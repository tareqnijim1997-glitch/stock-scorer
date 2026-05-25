"""Stock Explosion Scorer Pro+ - Mit Prognosen, Top Picks, Erweiterte Analyse & Smart-Filter"""
from __future__ import annotations
import sys, json, logging, math, time
from datetime import datetime, timedelta
from functools import lru_cache
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("scorer")

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# Cache fuer Top Picks (max 1 Stunde)
_top_picks_cache = {"data": None, "timestamp": 0}
_CACHE_TTL = 3600  # 1 Stunde

# Beliebte Aktien fuer den Top Picks Scanner
POPULAR_TICKERS = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "AMD",
    "AVGO", "ORCL", "CRM", "ADBE", "NFLX", "INTC", "QCOM", "TXN",
    "PLTR", "COIN", "MSTR", "SHOP", "SQ", "PYPL", "UBER", "ABNB",
    "JPM", "BAC", "GS", "V", "MA", "AXP", "BRK-B",
    "UNH", "LLY", "JNJ", "PFE", "MRK", "ABBV",
    "WMT", "COST", "HD", "MCD", "NKE", "SBUX",
    "XOM", "CVX", "BA", "CAT", "GE",
    "DIS", "T", "VZ", "TMUS",
    "SMCI", "ARM", "MU"
]


class MarketData:
    SECTOR_ETF_MAP = {
        "technology": "XLK", "financial services": "XLF", "healthcare": "XLV",
        "consumer cyclical": "XLY", "consumer defensive": "XLP",
        "industrials": "XLI", "energy": "XLE", "utilities": "XLU",
        "real estate": "XLRE", "basic materials": "XLB",
        "communication services": "XLC",
    }

    def __init__(self, ticker):
        self.ticker = ticker.upper().strip()
        self._yf = yf.Ticker(self.ticker)
        log.info(f"Loading {self.ticker}")
        self.hist = self._yf.history(period="14mo", auto_adjust=True)
        if self.hist.empty:
            raise ValueError(f"Keine Preisdaten fuer '{self.ticker}'")
        try: self.info = self._yf.info or {}
        except: self.info = {}
        try: self.q_earnings = self._yf.quarterly_income_stmt
        except: self.q_earnings = pd.DataFrame()
        self._spy = self._qqq = self._vix = self._tnx = self._dxy = self._sec = None
        self._sector_symbol = None

    @property
    def close(self): return self.hist["Close"]
    @property
    def volume(self): return self.hist["Volume"]
    def ma(self, n): return self.close.rolling(n).mean()
    def ma_was(self, n, days_ago):
        s = self.ma(n)
        return float(s.iloc[-1-days_ago]) if len(s) > days_ago else float("nan")

    def _safe(self, sym, period):
        try: return yf.Ticker(sym).history(period=period, auto_adjust=True)
        except: return pd.DataFrame()

    @property
    def spy(self):
        if self._spy is None: self._spy = self._safe("SPY", "14mo")
        return self._spy
    @property
    def qqq(self):
        if self._qqq is None: self._qqq = self._safe("QQQ", "6mo")
        return self._qqq
    @property
    def vix(self):
        if self._vix is None: self._vix = self._safe("^VIX", "3mo")
        return self._vix
    @property
    def tnx(self):
        if self._tnx is None: self._tnx = self._safe("^TNX", "6mo")
        return self._tnx
    @property
    def dxy(self):
        if self._dxy is None:
            self._dxy = self._safe("DX-Y.NYB", "6mo")
            if self._dxy.empty: self._dxy = self._safe("UUP", "6mo")
        return self._dxy
    @property
    def sector_etf(self):
        if self._sec is None:
            sec = (self.info.get("sector") or "").lower()
            etf = self.SECTOR_ETF_MAP.get(sec, "SPY")
            self._sector_symbol = etf
            self._sec = self._safe(etf, "14mo")
        return self._sec


def _step(v, thrs):
    for t, p in thrs:
        if v >= t: return p
    return 0.0

def _ret(s, lb):
    return float((s.iloc[-1]/s.iloc[-1-lb]) - 1) if len(s) > lb else float("nan")

def _r(cid, score, mx, detail, evaluable=True):
    return {"id": cid, "score": round(float(score), 2), "max": float(mx),
            "detail": detail, "evaluable": evaluable}

def _eps(d):
    if d.q_earnings.empty: return []
    rows = d.q_earnings.index
    for cand in ["Diluted EPS", "Basic EPS", "Net Income"]:
        if cand in rows:
            return [float(d.q_earnings.loc[cand, c]) for c in d.q_earnings.columns
                    if pd.notna(d.q_earnings.loc[cand, c])]
    return []


# === 19 BEDINGUNGEN ===

def c1(d):
    c = float(d.close.iloc[-1])
    ma50, ma150, ma200 = float(d.ma(50).iloc[-1]), float(d.ma(150).iloc[-1]), float(d.ma(200).iloc[-1])
    ma200_1m = d.ma_was(200, 21)
    h52 = float(d.close.iloc[-252:].max()); l52 = float(d.close.iloc[-252:].min())
    chk = [("P>MA150",c>ma150),("P>MA200",c>ma200),("MA150>MA200",ma150>ma200),
           ("MA200up1m",ma200>ma200_1m if not np.isnan(ma200_1m) else False),
           ("MA50>MA150",ma50>ma150),("MA50>MA200",ma50>ma200),("P>MA50",c>ma50),
           ("P>1.25L52",c>=1.25*l52),("P>=0.75H52",c>=0.75*h52)]
    p = sum(1 for _,ok in chk if ok)
    return _r(1, p/9*10, 10, f"{p}/9 conditions met")

def c2(d):
    h = float(d.close.iloc[-252:].max()); c = float(d.close.iloc[-1]); pct = c/h
    return _r(2, _step(pct, [(.95,5),(.90,4),(.85,3),(.75,2),(.65,1)]), 5,
              f"{pct*100:.1f}% of 52w high")

def c3(d):
    if d.spy.empty or len(d.spy)<252: return _r(3, 0, 10, "SPY data unavailable", False)
    sr, br = _ret(d.close,252), _ret(d.spy["Close"],252)
    if np.isnan(sr) or np.isnan(br): return _r(3, 0, 10, "Insufficient", False)
    diff = sr-br
    return _r(3, _step(diff, [(.50,10),(.30,8),(.15,6),(.05,4),(0,2)]), 10,
              f"Stock 12m: {sr*100:+.1f}% vs SPY: {br*100:+.1f}%")

def c4(d):
    c = float(d.close.iloc[-1]); m = float(d.ma(150).iloc[-1])
    m1, m3 = d.ma_was(150,21), d.ma_was(150,63)
    if np.isnan(m1) or np.isnan(m3): return _r(4, 0, 5, "Insufficient", False)
    if c>m and m>m1 and m>m3: return _r(4, 5, 5, "Stage 2 confirmed")
    if c>m and m>m1: return _r(4, 3, 5, "Stage 1->2 transition")
    if abs(c-m)/m < .10 and not (m>m3): return _r(4, 1, 5, "Stage 1 basing")
    return _r(4, 0, 5, "Stage 3 or 4")

def c5(d):
    if len(d.close)<60: return _r(5, 0, 5, "Insufficient", False)
    rngs = []
    for i in range(3):
        end = -(20*i) if i>0 else None
        seg = d.close.iloc[-(20*(i+1)):end]
        if len(seg)<5 or seg.mean()==0: return _r(5, 0, 5, "Bad segment", False)
        rngs.append((seg.max()-seg.min())/seg.mean())
    contr, tight = rngs[0]<rngs[1]<rngs[2], rngs[0]<.10
    if contr and tight: pts,m = 5, "Contracting & tight"
    elif contr: pts,m = 3, "Contracting"
    elif tight: pts,m = 2, "Tight"
    else: pts,m = 0, "No contraction"
    return _r(5, pts, 5, m)

def c6(d):
    if len(d.volume)<50: return _r(6, 0, 5, "Insufficient", False)
    v50 = d.volume.iloc[-50:].mean()
    if v50 == 0: return _r(6, 0, 5, "Zero volume", False)
    r = d.volume.iloc[-5:].mean() / v50
    rets = d.close.pct_change().iloc[-50:]; vols = d.volume.iloc[-50:]
    uv = vols[rets>0].mean() if (rets>0).any() else 0
    dv = vols[rets<0].mean() if (rets<0).any() else 1
    ud = uv/dv if dv>0 else 0
    if r>1.5 and ud>1.3: pts=5
    elif r>1.2 and ud>1.0: pts=3
    elif ud>1.0: pts=2
    else: pts=0
    return _r(6, pts, 5, f"5d/50d: {r:.2f}x")

def c7(d):
    e = _eps(d)
    if len(e)<5: return _r(7, 0, 8, "Quarterly data unavailable", False)
    if e[4]==0: return _r(7, 0, 8, "YA EPS=0", False)
    g = (e[0]-e[4])/abs(e[4])
    if e[4]<0 and e[0]>0: g = max(g, 1.0)
    return _r(7, _step(g,[(.50,8),(.30,6),(.20,4),(.10,2),(0,1)]), 8,
              f"EPS YoY: {g*100:+.1f}%")

def c8(d):
    e = _eps(d)
    if len(e)<7: return _r(8, 0, 7, "Insufficient", False)
    gs = []
    for i in range(3):
        ya = e[i+4] if i+4<len(e) else None
        if ya is None or ya==0: return _r(8, 0, 7, "Data gaps", False)
        gs.append((e[i]-ya)/abs(ya))
    allp = all(g>0 for g in gs); ax = gs[0]>gs[1]>gs[2]; al = gs[0]>gs[1]
    if allp and ax: pts,m = 7, "All 3 accelerating"
    elif allp and al: pts,m = 5, "Latest Q accelerated"
    elif allp: pts,m = 2, "All positive"
    else: pts,m = 0, "Negative present"
    return _r(8, pts, 7, m)

def c9(d):
    g = d.info.get("revenueGrowth")
    if g is None: return _r(9, 0, 5, "Unavailable", False)
    return _r(9, _step(g,[(.30,5),(.20,4),(.15,3),(.10,2),(.05,1)]), 5,
              f"Revenue growth: {g*100:+.1f}%")

def c10(d):
    om = d.info.get("operatingMargins"); pm = d.info.get("profitMargins")
    m = om if om is not None else pm
    if m is None: return _r(10, 0, 5, "Unavailable", False)
    label = "Operating" if om is not None else "Profit"
    return _r(10, _step(m,[(.30,5),(.20,4),(.15,3),(.10,2),(.05,1)]), 5,
              f"{label} margin: {m*100:.1f}%")

def c11(d):
    roe = d.info.get("returnOnEquity")
    if roe is None: return _r(11, 0, 5, "Unavailable", False)
    return _r(11, _step(roe,[(.30,5),(.20,4),(.17,3),(.10,2),(.05,1)]), 5,
              f"ROE: {roe*100:.1f}%")

def c12(d):
    if d.sector_etf.empty or d.spy.empty: return _r(12, 0, 5, "Unavailable", False)
    sr, br = _ret(d.sector_etf["Close"],252), _ret(d.spy["Close"],252)
    if np.isnan(sr) or np.isnan(br): return _r(12, 0, 5, "Insufficient", False)
    diff = sr-br
    pts = 5 if diff>=.15 else 4 if diff>=.08 else 3 if diff>=.03 else 2 if diff>=0 else 1 if diff>=-.05 else 0
    return _r(12, pts, 5, f"Sector vs SPY: {diff*100:+.1f}%")

def c13(d):
    if d.spy.empty or len(d.spy)<200: return _r(13, 0, 5, "Unavailable", False)
    s = d.spy["Close"]
    m50, m200 = float(s.rolling(50).mean().iloc[-1]), float(s.rolling(200).mean().iloc[-1])
    last = float(s.iloc[-1]); pts=0; parts=[]
    if last>m50: pts+=2; parts.append("SPY>MA50")
    if last>m200: pts+=2; parts.append("SPY>MA200")
    if m50>m200: pts+=1; parts.append("Golden Cross")
    return _r(13, pts, 5, ", ".join(parts) if parts else "Bearish")

def c14(d):
    if d.vix.empty: return _r(14, 0, 3, "Unavailable", False)
    v = float(d.vix["Close"].iloc[-1])
    pts = 3 if v<15 else 2 if v<20 else 1 if v<25 else 0
    return _r(14, pts, 3, f"VIX: {v:.2f}")

def c15(d):
    if d.tnx.empty or len(d.tnx)<60: return _r(15, 0, 3, "Unavailable", False)
    cur = float(d.tnx["Close"].iloc[-1]); m60 = float(d.tnx["Close"].rolling(60).mean().iloc[-1])
    if m60==0: return _r(15, 0, 3, "Bad data", False)
    diff = (cur-m60)/m60
    pts = 3 if diff<-.05 else 2 if diff<0 else 1 if diff<=.05 else 0
    return _r(15, pts, 3, f"10Y vs 60d-MA: {diff*100:+.1f}%")

def c16(d):
    if d.dxy.empty or len(d.dxy)<60: return _r(16, 0, 2, "Unavailable", False)
    cur = float(d.dxy["Close"].iloc[-1]); m60 = float(d.dxy["Close"].rolling(60).mean().iloc[-1])
    if m60==0: return _r(16, 0, 2, "Bad data", False)
    diff = (cur-m60)/m60
    pts = 2 if diff<-.03 else 1 if diff<0 else 0
    return _r(16, pts, 2, f"DXY vs 60d-MA: {diff*100:+.1f}%")

def c17(d):
    if d.qqq.empty or d.spy.empty: return _r(17, 0, 2, "Unavailable", False)
    qr, sr = _ret(d.qqq["Close"],63), _ret(d.spy["Close"],63)
    if np.isnan(qr) or np.isnan(sr): return _r(17, 0, 2, "Insufficient", False)
    risk_on = qr > sr
    sec = (d.info.get("sector") or "").lower()
    growth = sec in {"technology","communication services","consumer cyclical"}
    defensive = sec in {"consumer defensive","utilities","healthcare"}
    if risk_on and growth: return _r(17, 2, 2, f"Risk-on & growth")
    if risk_on or defensive: return _r(17, 1, 2, f"Risk-on OR defensive")
    return _r(17, 0, 2, f"Risk-off & growth")

def c18(d):
    e = _eps(d)
    if len(e)<5: return _r(18, 0, 5, "Unavailable", False)
    if e[4]==0: return _r(18, 0, 5, "YA=0", False)
    g = (e[0]-e[4])/abs(e[4])
    pts = 5 if g>=.30 else 3 if g>=.15 else 1 if g>=0 else 0
    return _r(18, pts, 5, f"Latest Q: {g*100:+.1f}% YoY")

def c19(d):
    if len(d.volume)<50: return _r(19, 0, 5, "Insufficient", False)
    v50 = d.volume.iloc[-50:].mean()
    if v50==0: return _r(19, 0, 5, "Zero volume", False)
    r = d.volume.iloc[-10:].mean() / v50
    return _r(19, _step(r,[(1.5,5),(1.2,3),(1.0,2),(.8,1)]), 5, f"10d/50d volume: {r:.2f}x")


ALL = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19]
CAT_OF = {1:0,2:0,3:0,4:0,5:0,6:0, 7:1,8:1,9:1,10:1,11:1,12:1,
          13:2,14:2,15:2,16:2,17:2, 18:3,19:3}


# === EXISTING HELPER FUNCTIONS ===

def get_chart_data(d):
    def slice_data(days):
        s = d.close.iloc[-days:] if len(d.close) >= days else d.close
        step = max(1, len(s) // 60)
        sampled = s.iloc[::step]
        return {
            "labels": [t.strftime("%d.%m") for t in sampled.index],
            "values": [round(float(v), 2) for v in sampled.values],
        }
    return {"1m": slice_data(21), "3m": slice_data(63), "12m": slice_data(252)}


def get_news(ticker_obj, max_items=8):
    try:
        raw = ticker_obj.news or []
    except Exception as e:
        log.warning(f"News fetch failed: {e}")
        return []
    items = []
    pos_words = {"beat", "beats", "surge", "rally", "growth", "upgrade", "raises",
                 "exceeds", "strong", "record", "high", "boost", "gains", "soar"}
    neg_words = {"miss", "drop", "fall", "downgrade", "weak", "loss", "concern",
                 "risk", "decline", "cut", "warning", "lawsuit", "probe", "plunge"}
    for entry in raw[:max_items]:
        c = entry.get("content") or entry
        title = c.get("title") or entry.get("title", "")
        publisher = (c.get("provider", {}) or {}).get("displayName") or entry.get("publisher", "Unknown")
        url = (c.get("clickThroughUrl") or {}).get("url") or (c.get("canonicalUrl") or {}).get("url") or entry.get("link", "")
        ts = c.get("pubDate") or entry.get("providerPublishTime")
        if isinstance(ts, (int, float)):
            published = datetime.fromtimestamp(ts).isoformat()
        elif isinstance(ts, str):
            published = ts
        else:
            published = datetime.now().isoformat()
        text = title.lower()
        pos = sum(1 for w in pos_words if w in text)
        neg = sum(1 for w in neg_words if w in text)
        if pos > neg: sentiment = "positive"
        elif neg > pos: sentiment = "negative"
        else: sentiment = "neutral"
        if title:
            items.append({"title": title, "publisher": publisher, "url": url,
                          "published": published, "sentiment": sentiment})
    return items


def get_expectations(d, total_score):
    info = d.info
    current = float(d.close.iloc[-1])
    target_mean = info.get("targetMeanPrice")
    target_high = info.get("targetHighPrice")
    target_low = info.get("targetLowPrice")
    num_analysts = info.get("numberOfAnalystOpinions")
    rec_key = info.get("recommendationKey", "none")
    upside = None
    if target_mean and current > 0:
        upside = round((target_mean / current - 1) * 100, 1)
    rets = d.close.pct_change().dropna()
    vol_ann = float(rets.std() * math.sqrt(252)) if len(rets) > 30 else 0.30
    score_factor = total_score / 100
    bull = 0.10 + score_factor * 0.45
    base = 0.30 + (1 - abs(score_factor - 0.5) * 2) * 0.10
    side = 0.20 - score_factor * 0.10
    bear = 1 - bull - base - side
    vol_adj = max(0, (vol_ann - 0.25) / 0.5)
    bull += vol_adj * 0.05; bear += vol_adj * 0.05
    base -= vol_adj * 0.05; side -= vol_adj * 0.05
    total = bull + base + side + bear
    bull, base, side, bear = [max(0.02, x/total) for x in (bull, base, side, bear)]
    total = bull + base + side + bear
    bull, base, side, bear = [round(x/total*100, 1) for x in (bull, base, side, bear)]
    return {
        "analyst": {"recommendation": rec_key,
            "target_mean": round(target_mean, 2) if target_mean else None,
            "target_high": round(target_high, 2) if target_high else None,
            "target_low": round(target_low, 2) if target_low else None,
            "num_analysts": num_analysts, "upside_pct": upside},
        "earnings_estimate": {"next_eps": info.get("forwardEps"),
            "trailing_eps": info.get("trailingEps"),
            "earnings_growth": info.get("earningsGrowth"),
            "revenue_growth": info.get("revenueGrowth")},
        "volatility": {"annualized": round(vol_ann * 100, 1)},
        "scenarios": {
            "bull": {"probability": bull},
            "base": {"probability": base},
            "sideways": {"probability": side},
            "bear": {"probability": bear}}}


# === NEW: PAKET 1 — MULTI-TIMEFRAME FORECAST ===

def get_forecast(d, total_score, categories):
    """Berechnet konkrete %-Prognosen fuer 1 Woche, 1 Monat, Jahresende.
    Mit Bereichen und Konfidenz."""
    info = d.info
    current = float(d.close.iloc[-1])

    # Faktoren sammeln
    score_factor = total_score / 100.0  # 0..1

    # Analyst Upside
    target_mean = info.get("targetMeanPrice")
    analyst_upside = 0.0
    if target_mean and current > 0:
        analyst_upside = (target_mean / current) - 1
        analyst_upside = max(-0.5, min(0.5, analyst_upside))  # cap at +-50%

    # Momentum (12M Return)
    momentum_12m = _ret(d.close, 252)
    if np.isnan(momentum_12m):
        momentum_12m = 0.0
    momentum_factor = max(-0.5, min(0.5, momentum_12m))  # cap

    # Sector Strength
    sector_factor = 0.0
    if not d.sector_etf.empty and not d.spy.empty:
        sr = _ret(d.sector_etf["Close"], 252)
        br = _ret(d.spy["Close"], 252)
        if not (np.isnan(sr) or np.isnan(br)):
            sector_factor = max(-0.2, min(0.2, sr - br))

    # Market Trend
    market_factor = 0.0
    if not d.spy.empty:
        spy_ret = _ret(d.spy["Close"], 252)
        if not np.isnan(spy_ret):
            market_factor = max(-0.3, min(0.3, spy_ret))

    # Volatilitaet
    rets = d.close.pct_change().dropna()
    vol_ann = float(rets.std() * math.sqrt(252)) if len(rets) > 30 else 0.30

    # === Berechne erwartete 12M-Rendite ===
    # Wenn Score schwach (< 30) -> negative Rendite
    # Wenn Score stark (> 70) -> positive Rendite
    score_contribution = (score_factor - 0.5) * 0.6  # -0.3 bis +0.3

    expected_12m = (
        0.35 * score_contribution +
        0.25 * analyst_upside +
        0.20 * momentum_factor +
        0.10 * sector_factor +
        0.10 * market_factor
    )

    # Skaliere auf die Zeitrahmen
    # 1 Woche = etwa 1/52 des Jahres, aber wir nehmen 0.18 weil Setups oft in der Anfangswoche stark performen
    # 1 Monat = etwa 0.40, weil das meiste Setup-Movement in den ersten Wochen passiert
    # Jahresende: skaliert nach verbleibenden Tagen im Jahr
    today = datetime.now()
    days_to_year_end = (datetime(today.year, 12, 31) - today).days
    days_to_year_end = max(30, min(365, days_to_year_end))
    yearend_scale = days_to_year_end / 365.0

    forecasts = {
        "1w":  {"days": 7,   "scale": 0.18},
        "1m":  {"days": 30,  "scale": 0.40},
        "yearend": {"days": days_to_year_end, "scale": yearend_scale * 0.85},
    }

    result = {}
    for key, cfg in forecasts.items():
        expected_pct = expected_12m * cfg["scale"]
        # Range basierend auf Volatilitaet und Zeitrahmen
        time_factor = math.sqrt(cfg["days"] / 252.0)
        std_dev = vol_ann * time_factor
        # 1-sigma Range
        low_pct = expected_pct - std_dev
        high_pct = expected_pct + std_dev

        # Probability up (basierend auf erwartetem Wert und Vol)
        if std_dev > 0:
            z = expected_pct / std_dev
            # Approximation Normalverteilung CDF
            prob_up = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        else:
            prob_up = 0.5 if expected_pct == 0 else (1.0 if expected_pct > 0 else 0.0)
        prob_up = max(0.05, min(0.95, prob_up))

        # Target price
        target_price = current * (1 + expected_pct)
        target_date = (today + timedelta(days=cfg["days"])).strftime("%Y-%m-%d")

        # Direction signal
        if expected_pct > 0.10:
            direction = "strong_up"
        elif expected_pct > 0.03:
            direction = "up"
        elif expected_pct > -0.03:
            direction = "neutral"
        elif expected_pct > -0.10:
            direction = "down"
        else:
            direction = "strong_down"

        result[key] = {
            "expected_pct": round(expected_pct * 100, 2),
            "low_pct": round(low_pct * 100, 2),
            "high_pct": round(high_pct * 100, 2),
            "target_price": round(target_price, 2),
            "target_date": target_date,
            "probability_up": round(prob_up * 100, 1),
            "direction": direction,
            "days": cfg["days"],
        }

    # Data quality flag
    data_quality = "high"
    missing = 0
    if target_mean is None: missing += 1
    if d.q_earnings.empty: missing += 1
    if not info.get("revenueGrowth"): missing += 1
    if missing >= 2:
        data_quality = "low"
    elif missing == 1:
        data_quality = "medium"

    return {
        "forecasts": result,
        "data_quality": data_quality,
        "factors": {
            "score": round(score_factor * 100, 1),
            "analyst_upside": round(analyst_upside * 100, 1),
            "momentum_12m": round(momentum_12m * 100, 1),
            "sector_strength": round(sector_factor * 100, 1),
            "market_trend": round(market_factor * 100, 1),
            "volatility_annual": round(vol_ann * 100, 1),
        }
    }


def get_explanation(d, conditions, categories):
    info = d.info
    summary = info.get("longBusinessSummary") or info.get("summary") or ""
    drivers_pos = []; drivers_neg = []
    rev_growth = info.get("revenueGrowth")
    if rev_growth and rev_growth >= 0.20:
        drivers_pos.append({"title": "Starkes Umsatzwachstum",
            "text": f"Umsatz waechst um {rev_growth*100:.1f}% YoY."})
    elif rev_growth and rev_growth < 0:
        drivers_neg.append({"title": "Umsatz schrumpft",
            "text": f"Umsatz faellt um {rev_growth*100:.1f}% YoY."})
    margin = info.get("operatingMargins") or info.get("profitMargins")
    if margin and margin >= 0.25:
        drivers_pos.append({"title": "Hohe Margen",
            "text": f"Margin von {margin*100:.1f}% zeigt Preissetzungsmacht."})
    elif margin and margin < 0.05:
        drivers_neg.append({"title": "Schwache Margen",
            "text": f"Margin von {margin*100:.1f}% - wenig Puffer."})
    roe = info.get("returnOnEquity")
    if roe and roe >= 0.20:
        drivers_pos.append({"title": "Starke Kapitalrendite",
            "text": f"ROE von {roe*100:.1f}%."})
    debt_eq = info.get("debtToEquity")
    if debt_eq and debt_eq > 200:
        drivers_neg.append({"title": "Hohe Verschuldung",
            "text": f"Debt-to-Equity von {debt_eq:.0f}."})
    if categories["technical"] >= 30:
        drivers_pos.append({"title": "Technisches Setup intakt",
            "text": f"Technische Bedingungen: {categories['technical']:.0f}/40 Punkte."})
    if categories["macro"] <= 5:
        drivers_neg.append({"title": "Schwieriges Makro-Umfeld",
            "text": f"Makro-Score nur {categories['macro']:.0f}/15."})
    pe = info.get("forwardPE") or info.get("trailingPE")
    if pe and pe > 50:
        drivers_neg.append({"title": "Hohe Bewertung",
            "text": f"KGV von {pe:.0f} preist Wachstum bereits ein."})
    beta = info.get("beta")
    if beta and beta > 1.5:
        drivers_neg.append({"title": "Hohe Volatilitaet",
            "text": f"Beta {beta:.2f}."})
    return {"summary": summary,
        "drivers_positive": drivers_pos[:4],
        "drivers_negative": drivers_neg[:4],
        "key_metrics": {
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume")}}


# === NEW: PAKET 3 — EXTENDED ANALYSIS ===

def get_extended_analysis(d):
    """Erweiterte Analyse: Insider, Short-Interest, Earnings Date, etc."""
    info = d.info
    yf_obj = d._yf

    # Short Interest
    short_data = {
        "short_ratio": info.get("shortRatio"),
        "short_percent_of_float": info.get("shortPercentOfFloat"),
        "shares_short": info.get("sharesShort"),
        "shares_short_prior_month": info.get("sharesShortPriorMonth"),
    }
    if short_data["short_percent_of_float"]:
        short_data["short_percent_of_float"] = round(short_data["short_percent_of_float"] * 100, 2)

    # Insider Activity
    insider_data = {
        "insider_ownership_pct": info.get("heldPercentInsiders"),
        "institution_ownership_pct": info.get("heldPercentInstitutions"),
        "insider_transactions": None,
    }
    if insider_data["insider_ownership_pct"]:
        insider_data["insider_ownership_pct"] = round(insider_data["insider_ownership_pct"] * 100, 2)
    if insider_data["institution_ownership_pct"]:
        insider_data["institution_ownership_pct"] = round(insider_data["institution_ownership_pct"] * 100, 2)

    # Cash Flow
    cashflow_data = {
        "operating_cashflow": info.get("operatingCashflow"),
        "free_cashflow": info.get("freeCashflow"),
        "total_cash": info.get("totalCash"),
        "total_debt": info.get("totalDebt"),
    }

    # Earnings Date
    earnings_date = None
    try:
        cal = yf_obj.calendar
        if cal is not None and not (hasattr(cal, 'empty') and cal.empty):
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed and isinstance(ed, list) and len(ed) > 0:
                    earnings_date = str(ed[0]) if hasattr(ed[0], 'strftime') else None
                    if hasattr(ed[0], 'strftime'):
                        earnings_date = ed[0].strftime("%Y-%m-%d")
    except Exception as e:
        log.warning(f"Earnings date fetch failed: {e}")

    # Competitors (basierend auf Sektor)
    competitors = []
    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()
    competitor_map = {
        "semiconductors": ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "TSM"],
        "software": ["MSFT", "ORCL", "CRM", "ADBE", "NOW"],
        "internet content": ["GOOGL", "META", "PINS", "SNAP"],
        "auto manufacturers": ["TSLA", "F", "GM", "RIVN", "LCID"],
        "consumer electronics": ["AAPL", "SONY", "HPQ"],
    }
    matched_industry = None
    for ind_key in competitor_map:
        if ind_key in industry:
            matched_industry = ind_key
            break
    if matched_industry:
        comps = [c for c in competitor_map[matched_industry] if c != d.ticker][:3]
        for comp_ticker in comps:
            try:
                comp = yf.Ticker(comp_ticker)
                comp_info = comp.info or {}
                comp_hist = comp.history(period="1y", auto_adjust=True)
                if not comp_hist.empty:
                    ret_1y = (comp_hist["Close"].iloc[-1] / comp_hist["Close"].iloc[0] - 1) * 100
                    competitors.append({
                        "ticker": comp_ticker,
                        "name": comp_info.get("shortName", comp_ticker),
                        "price": round(float(comp_hist["Close"].iloc[-1]), 2),
                        "return_1y": round(float(ret_1y), 2),
                        "market_cap": comp_info.get("marketCap"),
                    })
            except Exception as e:
                log.warning(f"Competitor {comp_ticker} failed: {e}")

    return {
        "short_interest": short_data,
        "insider": insider_data,
        "cashflow": cashflow_data,
        "earnings_date": earnings_date,
        "competitors": competitors,
    }


def score_ticker(ticker):
    d = MarketData(ticker)
    results = []
    for fn in ALL:
        try: results.append(fn(d))
        except Exception as e:
            log.exception(f"{fn.__name__} failed")
            results.append({"id":0,"score":0,"max":0,"detail":f"Error: {e}","evaluable":False})
    cats = [0.0,0.0,0.0,0.0]
    for r in results:
        if r["id"] in CAT_OF: cats[CAT_OF[r["id"]]] += r["score"]
    total = round(sum(cats), 2)
    rec = "STRONG" if total>=80 else "WATCH" if total>=65 else "NEUTRAL" if total>=50 else "AVOID"
    categories = {"technical":round(cats[0],2),"fundamental":round(cats[1],2),
                  "macro":round(cats[2],2),"catalyst":round(cats[3],2)}
    prev_close = float(d.close.iloc[-2]) if len(d.close) > 1 else float(d.close.iloc[-1])
    current = float(d.close.iloc[-1])
    change = current - prev_close
    change_pct = (change / prev_close * 100) if prev_close > 0 else 0
    return {
        "ticker": d.ticker,
        "company": d.info.get("longName") or d.info.get("shortName") or d.ticker,
        "sector": d.info.get("sector") or "Unknown",
        "industry": d.info.get("industry") or "",
        "currency": d.info.get("currency") or "USD",
        "exchange": d.info.get("exchange") or "",
        "current_price": round(current, 2),
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_score": total,
        "categories": categories,
        "recommendation": rec,
        "conditions": results,
        "chart": get_chart_data(d),
        "news": get_news(d._yf),
        "expectations": get_expectations(d, total),
        "explanation": get_explanation(d, results, categories),
        "forecast": get_forecast(d, total, categories),
        "extended": get_extended_analysis(d),
    }


def quick_score(ticker):
    """Schnellere Version fuer Top Picks Scan - nur Total Score und Basis-Info."""
    try:
        d = MarketData(ticker)
        results = []
        for fn in ALL:
            try: results.append(fn(d))
            except: results.append({"id":0,"score":0,"max":0,"detail":"","evaluable":False})
        cats = [0.0,0.0,0.0,0.0]
        for r in results:
            if r["id"] in CAT_OF: cats[CAT_OF[r["id"]]] += r["score"]
        total = round(sum(cats), 2)
        current = float(d.close.iloc[-1])
        prev = float(d.close.iloc[-2]) if len(d.close) > 1 else current
        change_pct = (current/prev - 1) * 100 if prev > 0 else 0
        forecast = get_forecast(d, total, {"technical":round(cats[0],2),"fundamental":round(cats[1],2),"macro":round(cats[2],2),"catalyst":round(cats[3],2)})
        return {
            "ticker": d.ticker,
            "company": d.info.get("shortName") or d.info.get("longName") or d.ticker,
            "sector": d.info.get("sector") or "Unknown",
            "price": round(current, 2),
            "change_pct": round(change_pct, 2),
            "score": total,
            "tech": round(cats[0],1),
            "fund": round(cats[1],1),
            "macro": round(cats[2],1),
            "cat": round(cats[3],1),
            "forecast_1w": forecast["forecasts"]["1w"]["expected_pct"],
            "forecast_1m": forecast["forecasts"]["1m"]["expected_pct"],
            "forecast_year": forecast["forecasts"]["yearend"]["expected_pct"],
        }
    except Exception as e:
        log.warning(f"quick_score failed for {ticker}: {e}")
        return None


# === ROUTES ===

@app.route("/")
def index(): return send_from_directory(".", "index.html")

@app.route("/api/score/<ticker>")
def api_score(ticker):
    try: return jsonify(score_ticker(ticker))
    except ValueError as e: return jsonify({"error": str(e)}), 404
    except Exception as e:
        log.exception("Failed")
        return jsonify({"error": f"Scoring failed: {e}"}), 500

@app.route("/api/top-picks")
def api_top_picks():
    """Scannt populaere Aktien und gibt Top Picks zurueck."""
    global _top_picks_cache
    now = time.time()
    # Cache check
    if _top_picks_cache["data"] and (now - _top_picks_cache["timestamp"] < _CACHE_TTL):
        return jsonify(_top_picks_cache["data"])

    log.info("Scanning popular tickers for Top Picks...")
    results = []
    for tk in POPULAR_TICKERS:
        r = quick_score(tk)
        if r:
            results.append(r)
        time.sleep(0.05)  # rate limit protection

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    response = {
        "scanned": len(POPULAR_TICKERS),
        "evaluated": len(results),
        "top_picks": results[:20],
        "timestamp": datetime.now().isoformat(),
    }
    _top_picks_cache = {"data": response, "timestamp": now}
    return jsonify(response)

@app.route("/api/health")
def health(): return jsonify({"status":"ok","time":datetime.now().isoformat()})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Stock Explosion Scorer Pro+ -> http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
