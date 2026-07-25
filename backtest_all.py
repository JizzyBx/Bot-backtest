"""
30m TREND SIGNAL — WIDE SYMBOL TEST (100+ Binance USDT-M futures coins)
Same signal as the confirmed-better 30m bot: ADX>=22 + 50EMA slope +
9/21 EMA crossover, TP=3x/SL=2x ATR. Tests a much broader coin list than
the original 30, with the same split-half consistency check, to see if
the edge is broad or concentrated in a small set of coins.
Writes to CSV incrementally after every symbol so a timeout doesn't
lose everything.
"""

import requests, time, io, zipfile, csv, os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

ORIGINAL_30 = {
    'ETHUSDT','DOGEUSDT','DOTUSDT','ARBUSDT','1000BONKUSDT','1000PEPEUSDT',
    '1000SHIBUSDT','ADAUSDT','APTUSDT','LINKUSDT','SOLUSDT','SUIUSDT',
    '1000FLOKIUSDT','WIFUSDT','BTCUSDT','BNBUSDT','NEARUSDT','XRPUSDT',
    'AVAXUSDT','LTCUSDT','ATOMUSDT','OPUSDT','INJUSDT','UNIUSDT','AAVEUSDT',
    'HBARUSDT','TRUMPUSDT','BOMEUSDT','WLDUSDT','NEIROUSDT',
}

EXTRA_SYMBOLS = [
    'TONUSDT','TRXUSDT','FILUSDT','GRTUSDT','MEWUSDT','PEOPLEUSDT','STXUSDT',
    'TURBOUSDT','ETCUSDT','XLMUSDT','ICPUSDT','RENDERUSDT','FETUSDT','TIAUSDT',
    'SEIUSDT','ORDIUSDT','JUPUSDT','PYTHUSDT','STRKUSDT','ENAUSDT','WUSDT',
    'PENDLEUSDT','NOTUSDT','IOUSDT','ZKUSDT','RUNEUSDT','GALAUSDT','SANDUSDT',
    'MANAUSDT','AXSUSDT','EGLDUSDT','FLOWUSDT','THETAUSDT','KAVAUSDT','MINAUSDT',
    'ROSEUSDT','CHZUSDT','ENJUSDT','ZILUSDT','ALGOUSDT','IOTAUSDT','XTZUSDT',
    'EOSUSDT','KSMUSDT','DYDXUSDT','GMTUSDT','APEUSDT','LDOUSDT','CRVUSDT',
    'SNXUSDT','COMPUSDT','MKRUSDT','YFIUSDT','SUSHIUSDT','ZRXUSDT','BATUSDT',
    'ANKRUSDT','CELRUSDT','CTSIUSDT','SKLUSDT','OCEANUSDT','RSRUSDT','STORJUSDT',
    'BANDUSDT','KNCUSDT','LRCUSDT','C98USDT','MASKUSDT','HOOKUSDT','HIGHUSDT',
    'AGIXUSDT','RLCUSDT','WOOUSDT','JOEUSDT','CFXUSDT','ARKMUSDT','SSVUSDT',
    'LQTYUSDT','IDUSDT','ARUSDT','ASTRUSDT','GASUSDT','POLYXUSDT','PHBUSDT',
    'HOTUSDT','JASMYUSDT','TWTUSDT','DUSKUSDT','VETUSDT','QNTUSDT','FLMUSDT',
    'ALPHAUSDT','CKBUSDT',
]

SYMBOLS = sorted(ORIGINAL_30) + EXTRA_SYMBOLS

INTERVAL      = '30m'
DAYS_BACK     = 90
ATR_PERIOD    = 14
ADX_PERIOD    = 14
ADX_MIN       = 22
TP_MULT       = 3.0
SL_MULT       = 2.0
COOLDOWN_BARS = 2
FEE_PCT       = 0.0008

VISION_BASE = "https://data.binance.vision/data/futures/um"
OUT_CSV = "wide_test_results.csv"
OUT_SUMMARY = "wide_test_summary.txt"

def _download_month(symbol, interval, year_month):
    url = f"{VISION_BASE}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year_month}.zip"
    try:
        r = requests.get(url, timeout=25)
        if r.status_code != 200: return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        fname = z.namelist()[0]
        df = pd.read_csv(z.open(fname), header=None, names=[
            'open_time','open','high','low','close','volume','close_time',
            'qav','trades','tbb','tbq','ignore'])
        df = df[pd.to_numeric(df['open_time'], errors='coerce').notna()]
        return df
    except Exception:
        return None

def fetch_klines(symbol, interval, days_back=DAYS_BACK):
    months_needed = max(2, (days_back // 30) + 2)
    frames = []
    month_dt = datetime.utcnow().replace(day=1)
    for m in range(months_needed):
        ym = month_dt.strftime('%Y-%m')
        df = _download_month(symbol, interval, ym)
        if df is not None and len(df) > 0: frames.append(df)
        month_dt = (month_dt - timedelta(days=1)).replace(day=1)
        time.sleep(0.1)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df[['open_time','open','high','low','close','volume']].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True)
    cutoff = df['open_time'].max() - pd.Timedelta(days=days_back)
    df = df[df['open_time'] >= cutoff].reset_index(drop=True)
    return df

def add_indicators(df):
    c, h, l = df['close'], df['high'], df['low']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/ATR_PERIOD, adjust=False).mean()

    up_move = h.diff(); down_move = -l.diff()
    plus_dm  = np.where((up_move>down_move)&(up_move>0), up_move, 0.0)
    minus_dm = np.where((down_move>up_move)&(down_move>0), down_move, 0.0)
    atr14 = tr.ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
    plus_di  = 100*pd.Series(plus_dm, index=df.index).ewm(alpha=1/ADX_PERIOD, adjust=False).mean()/atr14
    minus_di = 100*pd.Series(minus_dm, index=df.index).ewm(alpha=1/ADX_PERIOD, adjust=False).mean()/atr14
    dx = (100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,np.nan)).fillna(0)
    df['adx'] = dx.ewm(alpha=1/ADX_PERIOD, adjust=False).mean()

    df['ema9']  = c.ewm(span=9, adjust=False).mean()
    df['ema21'] = c.ewm(span=21, adjust=False).mean()
    df['ema50'] = c.ewm(span=50, adjust=False).mean()
    df['slope_pct'] = (df['ema50'] - df['ema50'].shift(10)) / df['ema50'].shift(10) * 100
    return df

def make_signal(df):
    trend_up = df['slope_pct'] > 0.05; trend_dn = df['slope_pct'] < -0.05
    cross_up = (df['ema9']>df['ema21']) & (df['ema9'].shift()<=df['ema21'].shift())
    cross_dn = (df['ema9']<df['ema21']) & (df['ema9'].shift()>=df['ema21'].shift())
    adx_ok = df['adx'] >= ADX_MIN
    s = pd.Series(None, index=df.index, dtype=object)
    s[adx_ok & trend_up & cross_up] = 'buy'
    s[adx_ok & trend_dn & cross_dn] = 'sell'
    return s

def backtest(df, signal_series):
    trades = []
    n = len(df)
    cooldown_until = -1
    opens=df['open'].values; highs=df['high'].values; lows=df['low'].values; atrs=df['atr'].values
    for i in range(60, n-2):
        if i <= cooldown_until: continue
        sig = signal_series.iat[i]
        if sig is None or pd.isna(atrs[i]): continue
        entry_idx = i+1
        if entry_idx >= n: continue
        entry = opens[entry_idx]; atr = atrs[i]
        tp_dist = atr*TP_MULT; sl_dist = atr*SL_MULT
        if sig=='buy':
            tp_price = entry+tp_dist; sl_price = entry-sl_dist
        else:
            tp_price = entry-tp_dist; sl_price = entry+sl_dist
        outcome=None; exit_bar=entry_idx
        max_fwd = min(entry_idx+192, n)
        for j in range(entry_idx, max_fwd):
            hi,lo = highs[j], lows[j]
            if sig=='buy':
                hit_tp = hi>=tp_price; hit_sl = lo<=sl_price
            else:
                hit_tp = lo<=tp_price; hit_sl = hi>=sl_price
            if hit_tp and hit_sl: outcome='loss'; exit_bar=j; break
            elif hit_tp: outcome='win'; exit_bar=j; break
            elif hit_sl: outcome='loss'; exit_bar=j; break
        if outcome is None: continue
        pnl_pct = (tp_dist/entry*100 - FEE_PCT*100) if outcome=='win' else -(sl_dist/entry*100 + FEE_PCT*100)
        trades.append({'entry_time':df['open_time'].iat[i],'side':sig,'outcome':outcome,'pnl_pct':pnl_pct})
        cooldown_until = exit_bar + COOLDOWN_BARS
    return pd.DataFrame(trades)

def summarize(trades_df):
    if len(trades_df)==0:
        return dict(total_trades=0,wins=0,losses=0,win_rate=None,expectancy_pct=None,profit_factor=None)
    wins=(trades_df['outcome']=='win').sum(); losses=(trades_df['outcome']=='loss').sum()
    win_rate=wins/len(trades_df)*100
    gross_win=trades_df.loc[trades_df['outcome']=='win','pnl_pct'].sum()
    gross_loss=-trades_df.loc[trades_df['outcome']=='loss','pnl_pct'].sum()
    pf=(gross_win/gross_loss) if gross_loss>0 else np.nan
    return dict(total_trades=len(trades_df),wins=int(wins),losses=int(losses),
                win_rate=round(win_rate,1),expectancy_pct=round(trades_df['pnl_pct'].mean(),4),
                profit_factor=round(pf,2) if not np.isnan(pf) else None)

def write_header_if_needed():
    if not os.path.exists(OUT_CSV):
        with open(OUT_CSV, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['symbol','in_original_30','total_trades','wins','losses','win_rate',
                        'expectancy_pct','profit_factor','half1_pf','half2_pf','consistent_both_halves'])

def append_row(row):
    with open(OUT_CSV, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow(row)

def main():
    write_header_if_needed()
    done_symbols = set()
    if os.path.exists(OUT_CSV):
        try:
            existing = pd.read_csv(OUT_CSV)
            done_symbols = set(existing['symbol'].tolist())
        except Exception:
            pass

    for idx, symbol in enumerate(SYMBOLS):
        if symbol in done_symbols:
            print(f"[{idx+1}/{len(SYMBOLS)}] {symbol} already done, skipping")
            continue
        print(f"[{idx+1}/{len(SYMBOLS)}] Fetching {symbol} (30m)...")
        df = fetch_klines(symbol, INTERVAL)
        if df is None or len(df) < 250:
            print(f"  skip {symbol} (no/insufficient data)")
            continue
        df = add_indicators(df)
        sig_full = make_signal(df)
        stats_full = summarize(backtest(df, sig_full))

        midpoint = df['open_time'].iloc[len(df)//2]
        first_half = df[df['open_time'] < midpoint].reset_index(drop=True)
        second_half = df[df['open_time'] >= midpoint].reset_index(drop=True)
        first_half = add_indicators(first_half[['open_time','open','high','low','close','volume']].copy())
        second_half = add_indicators(second_half[['open_time','open','high','low','close','volume']].copy())
        h1_stats = summarize(backtest(first_half, make_signal(first_half)))
        h2_stats = summarize(backtest(second_half, make_signal(second_half)))
        h1_pf = h1_stats['profit_factor'] or 0
        h2_pf = h2_stats['profit_factor'] or 0
        consistent = (h1_pf >= 1.0) and (h2_pf >= 1.0)

        row = [symbol, symbol in ORIGINAL_30, stats_full['total_trades'], stats_full['wins'],
               stats_full['losses'], stats_full['win_rate'], stats_full['expectancy_pct'],
               stats_full['profit_factor'], h1_pf, h2_pf, consistent]
        append_row(row)
        print(f"  done {symbol} — {stats_full['total_trades']} trades | PF={stats_full['profit_factor']} | consistent={consistent}")

    results_df = pd.read_csv(OUT_CSV)
    valid = results_df[results_df['total_trades'] > 0]
    orig = valid[valid['in_original_30'] == True]
    new = valid[valid['in_original_30'] == False]

    def agg(d):
        if len(d)==0: return (0,0,0,0,0)
        t = d['total_trades'].sum(); w = d['wins'].sum()
        wr = w/t*100 if t>0 else 0
        exp = d['expectancy_pct'].mean(); pf = d['profit_factor'].mean()
        cons = d['consistent_both_halves'].sum()
        return (t, wr, exp, pf, cons)

    lines = []
    lines.append("WIDE SYMBOL TEST — 30m TREND SIGNAL")
    lines.append("="*70)
    for label, d in [("ALL COINS", valid), ("ORIGINAL 30", orig), ("NEW/EXTRA COINS", new)]:
        t, wr, exp, pf, cons = agg(d)
        lines.append(f"{label:<18} coins={len(d):<4} trades={t:<7} WR={wr:.1f}% exp={exp:.4f}% PF={pf:.2f} consistent={cons}")
    lines.append("\n" + "="*70)
    lines.append("CONSISTENT COINS (PF>=1.0 both halves):")
    cons_df = valid[valid['consistent_both_halves']==True].sort_values('expectancy_pct', ascending=False)
    lines.append(cons_df.to_string(index=False))
    lines.append("\n" + "="*70)
    lines.append("FULL BREAKDOWN, sorted by expectancy:")
    lines.append(valid.sort_values('expectancy_pct', ascending=False).to_string(index=False))

    with open(OUT_SUMMARY, 'w') as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved: {OUT_CSV}, {OUT_SUMMARY}")

if __name__ == '__main__':
    main()
