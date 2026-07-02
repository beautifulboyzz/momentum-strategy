import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import unicodedata
from datetime import datetime, date
import re
import sys
from pathlib import Path

_LS_EXEC_DIR = Path(__file__).resolve().parent / "轮动策略"
for _p in (_LS_EXEC_DIR, _LS_EXEC_DIR / "轮动策略优化版本"):
    if (_p / "ls_execution.py").exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from ls_execution import apply_pending_at_open, gap_pnl_breakdown, pnl_base_price

# ================= 1. 系统配置 =================
st.set_page_config(page_title="Enhanced Dual Momentum - Long & Short", layout="wide", page_icon="🚀")

# --- A. 字体适配 ---
FONT_FILE = "Ubuntu_18.04_SimHei.ttf"
if os.path.exists(FONT_FILE):
    my_font = fm.FontProperties(fname=FONT_FILE)
    plt.rcParams['font.sans-serif'] = ['SimHei']
else:
    # 优先尝试微软雅黑，然后是黑体，再是苹果的 PingFang 和 Arial
    my_font = fm.FontProperties(family=['Microsoft YaHei', 'SimHei', 'PingFang SC'])
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Arial Unicode MS', 'sans-serif']

plt.rcParams['axes.unicode_minus'] = False  # 修复坐标轴负号显示问题

# --- B. 路径适配 ---
local_absolute_path = r"D:\全市场品种"
relative_path = "data"

if os.path.exists(local_absolute_path):
    DEFAULT_DATA_FOLDER = local_absolute_path
elif os.path.exists(relative_path):
    DEFAULT_DATA_FOLDER = relative_path
else:
    DEFAULT_DATA_FOLDER = "."


# ================= 2. 全局映射字典 =================

CN_NAME_MAP = {
    '沪金': 'au', '黄金': 'au', '沪银': 'ag', '白银': 'ag',
    '沪铜': 'cu', '铜': 'cu', '沪铝': 'al', '铝': 'al',
    '沪锌': 'zn', '锌': 'zn', '沪铅': 'pb', '铅': 'pb',
    '沪镍': 'ni', '镍': 'ni', '沪锡': 'sn', '锡': 'sn',
    '氧化铝': 'ao', '不锈钢': 'ss', '国际铜': 'bc',
    '螺纹': 'rb', '螺纹钢': 'rb', '热卷': 'hc', '铁矿': 'i', '铁矿石': 'i',
    '焦炭': 'j', '焦煤': 'jm', '硅铁': 'sf', '锰硅': 'sm', '线材': 'wr',
    '原油': 'sc', '燃油': 'fu', '低硫燃油': 'lu', '沥青': 'bu',
    '橡胶': 'ru', '20号胶': 'nr', '合成橡胶': 'br', '纸浆': 'sp',
    '塑料': 'l', '聚乙烯': 'l', 'PVC': 'v', 'PP': 'pp', '苯乙烯': 'eb', '乙二醇': 'eg',
    'LPG': 'pg', '液化气': 'pg', '纯苯': 'bz',
    '甲醇': 'ma', 'PTA': 'ta', '短纤': 'pf', '纯碱': 'sa', '玻璃': 'fg',
    '尿素': 'ur', '烧碱': 'sh', '对二甲苯': 'px', '瓶片': 'pr', 'LU燃油': 'lu',
    '豆一': 'a', '豆二': 'b', '豆粕': 'm', '豆油': 'y', '棕榈': 'p',
    '玉米': 'c', '淀粉': 'cs', '鸡蛋': 'jd', '生猪': 'lh',
    '白糖': 'sr', '棉花': 'cf', '棉纱': 'cy', '菜油': 'oi', '菜粕': 'rm', '花生': 'pk',
    '粳米': 'rr', '油菜籽': 'rs', '苹果': 'ap', '红枣': 'cj',
    '碳酸锂': 'lc', '工业硅': 'si', '多晶硅': 'ps',
    '1000股指': 'IM', '500股指': 'IC', '300股指': 'IF', '50股指': 'IH',
    '棕榈油': 'p', '菜籽油': 'oi', '菜籽粕': 'rm', '聚丙烯': 'pp', '丁二烯橡胶': 'br',
    '集运欧线': 'ec', '原木': 'lg', '钯金': 'pd', '铂金': 'pt',
}

# 数据文件名别名 → 统一品种名（避免同一合约被加载两次）
CANONICAL_NAME_MAP = {
    '菜籽油': '菜油',
    '菜籽粕': '菜粕',
    '棕榈油': '棕榈',
    '螺纹钢': '螺纹',
    '铁矿石': '铁矿',
    '聚乙烯': '塑料',
    '液化气': 'LPG',
    '黄金': '沪金',
    '白银': '沪银',
}

CONTRACT_MULTIPLIERS = {
    'rb': 10, 'hc': 10, 'i': 100, 'j': 100, 'jm': 60, 'zc': 100, 'sf': 5, 'sm': 5, 'ss': 5, 'wr': 10,
    'cu': 5, 'al': 5, 'zn': 5, 'pb': 5, 'ni': 1, 'sn': 1, 'bc': 5, 'ao': 20,
    'au': 1000, 'ag': 15,
    'ru': 10, 'bu': 10, 'sp': 10, 'fu': 10, 'sc': 1000, 'pg': 20, 'l': 5, 'pp': 5, 'v': 5, 'eg': 10,
    'ta': 5, 'ma': 10, 'ur': 20, 'sa': 20, 'lu': 10, 'eb': 5, 'pf': 5, 'px': 5, 'nr': 10, 'br': 5,
    'sh': 30, 'pr': 15, 'cy': 5,
    'c': 10, 'cs': 10, 'a': 10, 'b': 10, 'm': 10, 'y': 10, 'p': 10, 'oi': 10, 'rm': 10,
    'cf': 5, 'sr': 10, 'jd': 5, 'ap': 10, 'cj': 5, 'lh': 16, 'pk': 5, 'si': 5, 'lc': 1, 'ps': 3,
    'bz': 5, 'rr': 10, 'rs': 10,
    'fg': 20, 'ec': 50, 'lg': 90, 'pt': 1000, 'pd': 1000,
    'IF': 300, 'IH': 300, 'IC': 200, 'IM': 200
}

SECTOR_DEF = {
    '贵金属': ['au', 'ag'],
    '有色': ['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'bc', 'ao'],
    '黑色': ['rb', 'hc', 'i', 'j', 'jm', 'sm', 'sf', 'ss', 'wr'],
    '化工': ['sc', 'fu', 'lu', 'bu', 'ru', 'nr', 'br', 'l', 'v', 'pp', 'ta', 'ma', 'eg', 'eb', 'pg', 'sa', 'ur', 'fg', 'sh', 'px', 'pr', 'pf', 'sp', 'bz'],
    '农产品': ['m', 'y', 'p', 'oi', 'rm', 'c', 'cs', 'a', 'b', 'jd', 'lh', 'sr', 'cf', 'cy', 'ap', 'cj', 'pk', 'rr', 'rs'],
    '中金所': ['IF', 'IH', 'IC', 'IM'],
}

CODE_TO_SECTOR = {code: sec for sec, codes in SECTOR_DEF.items() for code in codes}

# 近似保证金率表：用于回测阶段估算账户保证金占用，不参与成交成本计算。
# 实盘保证金率会随交易所、期货公司、节假日和行情波动调整，这里只做保守风控模拟。
DEFAULT_MARGIN_RATE = 0.12
MARGIN_RATE_MAP = {
    'au': 0.12, 'ag': 0.14,
    'cu': 0.12, 'al': 0.12, 'zn': 0.12, 'pb': 0.12, 'ni': 0.16, 'sn': 0.14, 'bc': 0.12, 'ao': 0.13,
    'rb': 0.13, 'hc': 0.13, 'i': 0.15, 'j': 0.20, 'jm': 0.20, 'sm': 0.16, 'sf': 0.16, 'ss': 0.13,
    'sc': 0.15, 'fu': 0.15, 'lu': 0.15, 'bu': 0.15, 'ru': 0.15, 'nr': 0.15, 'br': 0.15,
    'l': 0.12, 'v': 0.12, 'pp': 0.12, 'ta': 0.12, 'ma': 0.13, 'eg': 0.13, 'eb': 0.13,
    'pg': 0.14, 'sa': 0.16, 'ur': 0.12, 'fg': 0.16, 'sh': 0.16, 'px': 0.13, 'pr': 0.13, 'pf': 0.12, 'sp': 0.12, 'bz': 0.13,
    'm': 0.12, 'y': 0.12, 'p': 0.13, 'oi': 0.12, 'rm': 0.12, 'c': 0.10, 'cs': 0.10,
    'a': 0.12, 'b': 0.12, 'jd': 0.12, 'lh': 0.15, 'sr': 0.12, 'cf': 0.12, 'cy': 0.12,
    'ap': 0.15, 'cj': 0.15, 'pk': 0.12, 'rr': 0.10, 'rs': 0.12,
    'lc': 0.18, 'si': 0.15, 'ps': 0.15,
    'IF': 0.12, 'IH': 0.12, 'IC': 0.12, 'IM': 0.12,
    'ec': 0.18, 'lg': 0.15, 'pt': 0.14, 'pd': 0.14,
}

MIN_ACTIVE_VOLUME = 1
MIN_ACTIVE_TRADE_DAYS = 120
FFILL_GAP_LIMIT = 10
FILE_SKIP_KEYWORDS = ["纤维板", "胶合板", "线材", "强麦", "早籼稻", "二年债", "五年债", "十年债", "三十债"]
ZOMBIE_FILE_RECOMMEND_DELETE = ["动力煤", "晚籼稻", "粳稻", "普麦", "早籼稻"]
# 日更脚本备份副本：原油_20260623_170549.csv → 跳过，只保留 原油.csv
BACKUP_FILE_PATTERN = re.compile(r'_\d{8}_\d{6}$')


# ================= 3. 数据清洗工具函数 =================

def should_skip_asset(df):
    vol = df["volume"].fillna(0)
    active = vol >= MIN_ACTIVE_VOLUME
    n_active = int(active.sum())
    if n_active == 0: return True, "无有效成交"
    if n_active < MIN_ACTIVE_TRADE_DAYS: return True, f"有效成交日仅 {n_active} 天"
    last_active = df.index[active][-1]
    ghost_tail = int((df.index > last_active).sum())
    if ghost_tail >= 30: return True, f"末尾 {ghost_tail} 行零成交续行"
    recent_cut = df.index[-1] - pd.Timedelta(days=400)
    recent_active = int((vol.loc[df.index >= recent_cut] >= MIN_ACTIVE_VOLUME).sum())
    if recent_active == 0: return True, f"近 400 自然日无成交 (最后成交 {last_active.date()})"
    return False, None

def sanitize_trading_df(df):
    df = df.sort_index()
    vol = df["volume"].fillna(0)
    active = vol >= MIN_ACTIVE_VOLUME
    if not active.any(): return None
    last_active = df.index[active][-1]
    df = df.loc[:last_active].copy()
    inactive = df["volume"].fillna(0) < MIN_ACTIVE_VOLUME
    for col in ["close", "open", "high", "low", "amount"]:
        if col in df.columns: df.loc[inactive, col] = np.nan
    df.loc[inactive, "volume"] = 0
    return df

def align_panel(series_dict, ffill_limit=FFILL_GAP_LIMIT):
    if not series_dict: return pd.DataFrame()
    return pd.DataFrame(series_dict).ffill(limit=ffill_limit)

def canonicalize_asset_name(name):
    clean = name.replace("主连", "").replace("指数", "").replace("连续", "").replace("日线", "").replace(".csv", "").strip()
    return CANONICAL_NAME_MAP.get(clean, clean)

def merge_trading_df(df_a, df_b):
    combined = pd.concat([df_a, df_b])
    combined = combined[~combined.index.duplicated(keep='last')]
    return combined.sort_index()

def preferred_display_name(*names):
    """同合约多别名时，优先更短的规范名（如 菜油 优于 菜籽油）。"""
    canon = [canonicalize_asset_name(n) for n in names if n]
    if not canon:
        return ""
    return min(canon, key=len)

def dedupe_series_dict_by_contract_code(series_dict, skip_log=None):
    """按合约代码合并重复品种列，防止 菜油/菜籽油 等别名同时进入截面池。"""
    if skip_log is None:
        skip_log = []
    merged = {}
    code_to_name = {}
    for name, series in series_dict.items():
        code = _resolve_contract_code(name).lower()
        canon = canonicalize_asset_name(name)
        if code not in code_to_name:
            code_to_name[code] = canon
            merged[canon] = series.copy() if hasattr(series, 'copy') else series
            continue
        keep_name = preferred_display_name(code_to_name[code], canon)
        drop_name = code_to_name[code] if keep_name != code_to_name[code] else canon
        if drop_name in merged and keep_name != drop_name:
            base = merged.pop(drop_name)
            other = series
            combined = pd.concat([base, other], axis=1).ffill(axis=1).iloc[:, -1]
            merged[keep_name] = combined
        else:
            base = merged[keep_name]
            combined = pd.concat([base, series], axis=1).ffill(axis=1).iloc[:, -1]
            merged[keep_name] = combined
        code_to_name[code] = keep_name
        skip_log.append(f"{name} → 与 {keep_name} 同合约({code})，已合并")
    return merged, skip_log

def _normalize_asset_label(asset_name):
    return canonicalize_asset_name(
        asset_name.replace("主连", "").replace("指数", "").replace("连续", "").replace("日线", "").replace(".csv", "").strip()
    )

def _resolve_contract_code(asset_name):
    clean_name = _normalize_asset_label(asset_name)
    if clean_name in CN_NAME_MAP:
        return CN_NAME_MAP[clean_name].lower()
    match = re.match(r"([a-zA-Z]+)", clean_name)
    if match:
        return match.group(1).lower()
    return clean_name.lower()

def _group_columns_by_contract_code(columns):
    groups = {}
    for col in columns:
        code = _resolve_contract_code(col)
        groups.setdefault(code, []).append(col)
    return groups

def canonicalize_panel_columns(*dfs):
    """将面板列名统一为规范名，并按合约代码合并重复列（加载缓存未刷新时的兜底）。"""
    ref = next((d for d in dfs if d is not None and not d.empty), None)
    if ref is None:
        return dfs
    groups = _group_columns_by_contract_code(ref.columns)
    keep_map = {}
    for code, cols in groups.items():
        keep = preferred_display_name(*[_normalize_asset_label(c) for c in cols])
        keep_map[code] = (keep, cols)
    out = []
    for df in dfs:
        if df is None or df.empty:
            out.append(df)
            continue
        merged = {}
        for code, (keep, cols) in keep_map.items():
            avail = [c for c in cols if c in df.columns]
            if not avail:
                continue
            if len(avail) == 1:
                merged[keep] = df[avail[0]]
            else:
                merged[keep] = df[avail].ffill(axis=1).iloc[:, -1]
        out.append(pd.DataFrame(merged, index=df.index))
    return tuple(out)

def get_sector(asset_name):
    code = _resolve_contract_code(asset_name)
    return CODE_TO_SECTOR.get(code.lower(), CODE_TO_SECTOR.get(code.upper(), '其他'))


def _pick_with_rank_protect(ranked_series, hold_num, max_per_sector, current_holdings, rank_threshold):
    """已持仓品种得分排名≤阈值则续持，不足 hold_num 时再从当日排名补齐。"""
    rank_map = {a: i + 1 for i, a in enumerate(ranked_series.index)}
    sec_counts, picked, seen_codes = {}, [], set()

    for asset, w in (current_holdings or {}).items():
        if w <= 0:
            continue
        if asset not in rank_map or rank_map[asset] > rank_threshold:
            continue
        code = _resolve_contract_code(asset)
        if code in seen_codes:
            continue
        sec = get_sector(asset)
        if sec_counts.get(sec, 0) >= max_per_sector:
            continue
        picked.append(asset)
        seen_codes.add(code)
        sec_counts[sec] = sec_counts.get(sec, 0) + 1

    for asset in ranked_series.index:
        if len(picked) >= hold_num:
            break
        if asset in picked:
            continue
        code = _resolve_contract_code(asset)
        if code in seen_codes:
            continue
        sec = get_sector(asset)
        if sec_counts.get(sec, 0) >= max_per_sector:
            continue
        picked.append(asset)
        seen_codes.add(code)
        sec_counts[sec] = sec_counts.get(sec, 0) + 1
    return picked


def compute_holding_wave_stats(cycle_details, nav_index):
    """按回测交易日切分连续持仓波段，与「持仓周期精细画像」口径一致。"""
    history_records = []
    for day in cycle_details:
        d_date = day['date']
        for asset, w in day.get('next_day_hold_long', {}).items():
            if w > 0:
                history_records.append({'date': d_date, 'asset': asset, 'direction': '做多'})
        for asset, w in day.get('next_day_hold_short', {}).items():
            if w > 0:
                history_records.append({'date': d_date, 'asset': asset, 'direction': '做空'})
    if not history_records:
        return pd.DataFrame(), pd.DataFrame()

    df_bh = pd.DataFrame(history_records)
    all_backtest_dates = sorted(list(nav_index))
    date_to_idx = {d: idx for idx, d in enumerate(all_backtest_dates)}
    df_bh['date_idx'] = df_bh['date'].map(date_to_idx)
    df_bh['asset_label'] = df_bh['asset'] + ' (' + df_bh['direction'] + ')'
    df_bh.sort_values(['asset', 'direction', 'date'], inplace=True)
    df_bh['is_new_wave'] = df_bh.groupby(['asset', 'direction'])['date_idx'].diff() != 1
    df_bh['wave_id'] = df_bh.groupby(['asset', 'direction'])['is_new_wave'].cumsum()

    wave_stats = df_bh.groupby(['asset', 'direction', 'wave_id']).agg(
        Start_Date=('date', 'min'), End_Date=('date', 'max'),
        Start_Idx=('date_idx', 'min'), End_Idx=('date_idx', 'max'), Hold_Days=('date', 'count'),
    ).reset_index()
    wave_stats['asset_label'] = wave_stats['asset'] + ' (' + wave_stats['direction'] + ')'

    wave_returns = []
    for _, row in wave_stats.iterrows():
        origin_asset = row['asset']
        is_short_side = row['direction'] == '做空'
        s_idx, e_idx = int(row['Start_Idx']), int(row['End_Idx'])
        daily_pnls = []
        for idx in range(s_idx + 1, e_idx + 2):
            if idx < len(cycle_details):
                day_ret = cycle_details[idx].get('asset_rets', {}).get(origin_asset, 0.0)
                daily_pnls.append(-day_ret if is_short_side else day_ret)
        wave_returns.append(np.prod([1 + r for r in daily_pnls]) - 1 if daily_pnls else 0.0)
    wave_stats['Wave_Return'] = wave_returns
    return df_bh, wave_stats


def get_multiplier(asset_name):
    code = _resolve_contract_code(asset_name)
    return CONTRACT_MULTIPLIERS.get(code.lower(), CONTRACT_MULTIPLIERS.get(code.upper(), 1))

def get_margin_rate(asset_name):
    code = _resolve_contract_code(asset_name)
    return MARGIN_RATE_MAP.get(code.lower(), MARGIN_RATE_MAP.get(code.upper(), DEFAULT_MARGIN_RATE))

def estimate_margin_usage(long_weights, short_weights):
    total = 0.0
    detail = {}
    for asset, weight in long_weights.items():
        usage = abs(weight) * get_margin_rate(asset)
        detail[(asset, 'long')] = usage
        total += usage
    for asset, weight in short_weights.items():
        usage = abs(weight) * get_margin_rate(asset)
        detail[(asset, 'short')] = usage
        total += usage
    return total, detail


def compute_day_allocation_breakdown(
    curr_date, df_p, df_atr_norm, df_vol, debug_data,
    hold_num, max_per_sector, net_exposure_target, max_gross_exposure, max_margin_usage,
    max_single_weight=0.30, use_vol_scaling=True, banned_assets=None,
):
    """复刻单日仓位分配全链路，供白盒拆解 UI 展示。"""
    if banned_assets is None:
        banned_assets = set()
    long_count = max(hold_num, 3)
    short_count = max(hold_num, 3)

    filter_cond = debug_data['ma_filter'].loc[curr_date] & debug_data['momentum_filter'].loc[curr_date]
    short_filter_cond = df_p.loc[curr_date] < df_p.rolling(60, min_periods=1).mean().loc[curr_date]
    long_scores = debug_data['long_score'].loc[curr_date].dropna()
    short_scores = debug_data['short_score'].loc[curr_date].dropna()
    common_score_idx = long_scores.index.intersection(short_scores.index)

    valid_pool = [
        a for a in common_score_idx
        if a not in banned_assets
        and pd.notna(df_p.loc[curr_date, a])
        and (a not in df_vol.columns or df_vol.loc[curr_date, a] >= MIN_ACTIVE_VOLUME)
    ]

    ideal_long, ideal_short = [], []
    sec_counts_l, sec_counts_s = {}, {}

    if valid_pool:
        ranked_long = long_scores.loc[valid_pool].sort_values(ascending=False)
        ranked_short = short_scores.loc[valid_pool].sort_values(ascending=False)
        for a in ranked_long.index:
            if not filter_cond.get(a, False):
                continue
            if long_scores[a] <= 0.5:
                continue
            sec = get_sector(a)
            if sec_counts_l.get(sec, 0) < max_per_sector:
                ideal_long.append(a)
                sec_counts_l[sec] = sec_counts_l.get(sec, 0) + 1
            if len(ideal_long) == long_count:
                break
        for a in ranked_short.index:
            if not short_filter_cond.get(a, False):
                continue
            if short_scores[a] <= 0.5:
                continue
            sec = get_sector(a)
            if sec_counts_s.get(sec, 0) < max_per_sector:
                ideal_short.append(a)
                sec_counts_s[sec] = sec_counts_s.get(sec, 0) + 1
            if len(ideal_short) == short_count:
                break

    vol_scaler = debug_data['vol_scaler']
    pos_multiplier = vol_scaler.loc[curr_date] if use_vol_scaling and curr_date in vol_scaler.index else 1.0

    def _risk_parity_weights(pool):
        if not pool:
            return {}, 0.0, False
        vols = df_atr_norm.loc[curr_date, pool].clip(lower=0.00001)
        inv_v = 1.0 / vols
        sum_inv = float(inv_v.sum())
        w = (inv_v / inv_v.sum()) * pos_multiplier
        w_clipped = w.clip(upper=max_single_weight)
        clip_triggered = bool((w > max_single_weight).any())
        if w_clipped.sum() > 0:
            w_clipped = (w_clipped / w_clipped.sum()) * pos_multiplier
        detail = {
            a: {
                'atr_norm': float(vols[a]),
                'inv_vol': float(inv_v[a]),
                'pool_share': float(inv_v[a] / sum_inv),
                'raw_weight': float(w[a]),
                'clipped_weight': float(w_clipped[a]),
            }
            for a in pool
        }
        return detail, sum_inv, clip_triggered

    long_detail, long_sum_inv, long_clip = _risk_parity_weights(ideal_long)
    short_detail, short_sum_inv, short_clip = _risk_parity_weights(ideal_short)
    raw_w_long = {a: d['clipped_weight'] for a, d in long_detail.items()}
    raw_w_short = {a: d['clipped_weight'] for a, d in short_detail.items()}

    tot_l = sum(raw_w_long.values())
    tot_s = sum(raw_w_short.values())
    net_exp = tot_l - tot_s
    net_scale_side, net_scale_factor = None, 1.0
    w_long_after_net = raw_w_long.copy()
    w_short_after_net = raw_w_short.copy()
    if net_exp > net_exposure_target:
        net_scale_side = 'long'
        net_scale_factor = net_exposure_target / max(net_exp, 0.001)
        w_long_after_net = {a: w * net_scale_factor for a, w in raw_w_long.items()}
    elif net_exp < -net_exposure_target:
        net_scale_side = 'short'
        net_scale_factor = net_exposure_target / max(abs(net_exp), 0.001)
        w_short_after_net = {a: w * net_scale_factor for a, w in raw_w_short.items()}

    gross_pre = sum(w_long_after_net.values()) + sum(w_short_after_net.values())
    gross_scale_factor = 1.0
    w_long_after_gross = w_long_after_net.copy()
    w_short_after_gross = w_short_after_net.copy()
    if gross_pre > max_gross_exposure:
        gross_scale_factor = max_gross_exposure / max(gross_pre, 0.001)
        w_long_after_gross = {a: w * gross_scale_factor for a, w in w_long_after_net.items()}
        w_short_after_gross = {a: w * gross_scale_factor for a, w in w_short_after_net.items()}

    margin_pre, _ = estimate_margin_usage(w_long_after_gross, w_short_after_gross)
    margin_scale_factor = 1.0
    w_long_final = w_long_after_gross.copy()
    w_short_final = w_short_after_gross.copy()
    if margin_pre > max_margin_usage:
        margin_scale_factor = max_margin_usage / max(margin_pre, 0.001)
        w_long_final = {a: w * margin_scale_factor for a, w in w_long_after_gross.items()}
        w_short_final = {a: w * margin_scale_factor for a, w in w_short_after_gross.items()}

    short_rank_map = {}
    if valid_pool:
        ranked_short_all = short_scores.loc[valid_pool].sort_values(ascending=False)
        for rk, a in enumerate(ranked_short_all.index, start=1):
            short_rank_map[a] = rk
    long_rank_map = {}
    if valid_pool:
        ranked_long_all = long_scores.loc[valid_pool].sort_values(ascending=False)
        for rk, a in enumerate(ranked_long_all.index, start=1):
            long_rank_map[a] = rk

    return {
        'valid_pool_size': len(valid_pool),
        'ideal_long': ideal_long,
        'ideal_short': ideal_short,
        'long_detail': long_detail,
        'short_detail': short_detail,
        'long_sum_inv': long_sum_inv,
        'short_sum_inv': short_sum_inv,
        'long_clip_triggered': long_clip,
        'short_clip_triggered': short_clip,
        'raw_w_long': raw_w_long,
        'raw_w_short': raw_w_short,
        'pos_multiplier': float(pos_multiplier),
        'tot_long_raw': tot_l,
        'tot_short_raw': tot_s,
        'net_exp_raw': net_exp,
        'gross_raw': tot_l + tot_s,
        'net_scale_side': net_scale_side,
        'net_scale_factor': net_scale_factor,
        'w_long_after_net': w_long_after_net,
        'w_short_after_net': w_short_after_net,
        'gross_pre': gross_pre,
        'gross_scale_factor': gross_scale_factor,
        'w_long_after_gross': w_long_after_gross,
        'w_short_after_gross': w_short_after_gross,
        'margin_pre': margin_pre,
        'margin_scale_factor': margin_scale_factor,
        'w_long_final': w_long_final,
        'w_short_final': w_short_final,
        'short_rank_map': short_rank_map,
        'long_rank_map': long_rank_map,
        'short_scores': short_scores,
        'long_scores': long_scores,
    }


def read_robust_csv(f):
    for enc in ['gbk', 'utf-8', 'gb18030', 'cp936']:
        try:
            df = pd.read_csv(f, encoding=enc, engine='python')
            rename_map = {}
            for c in df.columns:
                c_str = str(c).strip()
                if c_str in ['日期', '日期/时间', 'date', 'Date', 'trade_date']: rename_map[c] = 'date'
                if c_str in ['收盘价', '收盘', 'close', 'price', 'Close']: rename_map[c] = 'close'
                if c_str in ['最高价', '最高', 'high', 'High']: rename_map[c] = 'high'
                if c_str in ['最低价', '最低', 'low', 'Low']: rename_map[c] = 'low'
                if c_str in ['开盘价', '开盘', 'open', 'Open']: rename_map[c] = 'open'
                if c_str in ['成交量', 'volume', 'Volume', 'vol']: rename_map[c] = 'volume'
                if c_str in ['成交额', 'amount', 'Amount']: rename_map[c] = 'amount'
                if c_str in ['持仓量', 'open_interest', 'oi']: rename_map[c] = 'open_interest'
            df.rename(columns=rename_map, inplace=True)
            if 'date' in df.columns and 'close' in df.columns: return df
        except: continue
    return None

@st.cache_data(ttl=3600)
def load_data_and_calc_metrics(folder, atr_window=20, _panel_dedupe_version=3):
    if not os.path.exists(folder): return (None,)*10 + (f"路径不存在: {folder}",)
    try: files = sorted([f for f in os.listdir(folder) if f.endswith('.csv')])
    except: return (None,)*10 + ("无法读取目录",)
    if not files: return (None,)*10 + ("无CSV文件",)

    price_dict, low_dict, open_dict, high_dict = {}, {}, {}, {}
    atr_dict, atr_norm_dict = {}, {}
    vol_dict, vol_ma_dict = {}, {}
    amount_dict, liquidity_score_dict = {}, {}
    raw_df_dict = {}
    skip_log = []

    progress_bar = st.progress(0, text="正在加载品种数据...")
    for i, file in enumerate(files):
        file_norm = unicodedata.normalize('NFC', file)
        if any(x in file_norm for x in FILE_SKIP_KEYWORDS):
            skip_log.append(f"{file_norm}: 静态黑名单")
            continue
        raw_name = file_norm.split('.')[0].replace("主连", "").replace("日线", "")
        name = canonicalize_asset_name(raw_name)
        if BACKUP_FILE_PATTERN.search(raw_name):
            skip_log.append(f"{file_norm}: 日更备份副本(已跳过)")
            continue
        path = os.path.join(folder, file)
        df = read_robust_csv(path)
        if df is None:
            skip_log.append(f"{raw_name}: CSV 解析失败")
            continue
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            if 'volume' not in df.columns: df['volume'] = 0
            df.dropna(subset=['date'], inplace=True)
            df['date'] = df['date'].dt.normalize()
            df.sort_values('date', inplace=True)
            df.drop_duplicates(subset=['date'], keep='last', inplace=True)
            df.set_index('date', inplace=True)

            skip, reason = should_skip_asset(df)
            if skip:
                skip_log.append(f"{raw_name}: {reason}")
                continue

            df = sanitize_trading_df(df)
            if df is None:
                skip_log.append(f"{raw_name}: 清洗后无有效成交")
                continue

            if name in raw_df_dict:
                raw_df_dict[name] = merge_trading_df(raw_df_dict[name], df)
                skip_log.append(f"{raw_name} → 合并至 {name}")
            else:
                raw_df_dict[name] = df
                if raw_name != name:
                    skip_log.append(f"{raw_name} → 统一命名为 {name}")
        except:
            skip_log.append(f"{raw_name}: 指标计算异常")
            continue
        if i % 10 == 0: progress_bar.progress((i+1)/len(files), text=f"加载: {name}")

    for name, df in raw_df_dict.items():
        try:
            multiplier = get_multiplier(name)
            df['amount'] = df['close'] * df['volume'] * multiplier
            df.loc[df['volume'] < MIN_ACTIVE_VOLUME, 'amount'] = np.nan

            prev_close = df['close'].shift(1)
            tr = pd.concat([df['high'] - df['low'], (df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()], axis=1).max(axis=1)
            atr = tr.rolling(atr_window, min_periods=1).mean()
            natr = atr / df['close'].abs().replace(0, np.nan)
            vol_ma = df['volume'].rolling(20, min_periods=1).mean()
            liquidity_score = df['amount'].rolling(20, min_periods=1).mean()

            price_dict[name] = df['close']
            low_dict[name] = df['low']
            open_dict[name] = df['open']
            high_dict[name] = df['high']
            atr_dict[name] = atr
            atr_norm_dict[name] = natr
            vol_dict[name] = df['volume']
            vol_ma_dict[name] = vol_ma
            amount_dict[name] = df['amount']
            liquidity_score_dict[name] = liquidity_score
        except:
            skip_log.append(f"{name}: 指标汇总异常")
            continue

    price_dict, skip_log = dedupe_series_dict_by_contract_code(price_dict, skip_log)
    low_dict, _ = dedupe_series_dict_by_contract_code(low_dict, skip_log)
    open_dict, _ = dedupe_series_dict_by_contract_code(open_dict, skip_log)
    high_dict, _ = dedupe_series_dict_by_contract_code(high_dict, skip_log)
    atr_dict, _ = dedupe_series_dict_by_contract_code(atr_dict, skip_log)
    atr_norm_dict, _ = dedupe_series_dict_by_contract_code(atr_norm_dict, skip_log)
    vol_dict, _ = dedupe_series_dict_by_contract_code(vol_dict, skip_log)
    vol_ma_dict, _ = dedupe_series_dict_by_contract_code(vol_ma_dict, skip_log)
    amount_dict, _ = dedupe_series_dict_by_contract_code(amount_dict, skip_log)
    liquidity_score_dict, _ = dedupe_series_dict_by_contract_code(liquidity_score_dict, skip_log)

    progress_bar.empty()

    st.session_state['data_skip_log'] = skip_log
    panels = (
        align_panel(price_dict), align_panel(atr_norm_dict), align_panel(low_dict), align_panel(open_dict),
        align_panel(high_dict), align_panel(atr_dict), pd.DataFrame(vol_dict).fillna(0),
        align_panel(vol_ma_dict).fillna(0), align_panel(amount_dict), align_panel(liquidity_score_dict),
    )
    return (*canonicalize_panel_columns(*panels), None)


@st.cache_data(ttl=3600)
def load_index_close_panel(folder):
    """加载文华板块/商品指数收盘价面板，用于基准对比。"""
    if not os.path.exists(folder):
        return None, f"指数路径不存在: {folder}"
    files = sorted([f for f in os.listdir(folder) if f.endswith('.csv')])
    if not files:
        return None, f"[{folder}] 中无 CSV 文件"
    price_dict = {}
    for file in files:
        name = file.split('.')[0].replace("主连", "").replace("日线", "").replace("指数", "").strip()
        path = os.path.join(folder, file)
        df = read_robust_csv(path)
        if df is None:
            continue
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date', 'close'], inplace=True)
            df['date'] = df['date'].dt.normalize()
            df.sort_values('date', inplace=True)
            df.drop_duplicates(subset=['date'], keep='last', inplace=True)
            df.set_index('date', inplace=True)
            price_dict[name] = df['close']
        except Exception:
            continue
    if not price_dict:
        return None, "未能解析任何指数 CSV"
    return pd.DataFrame(price_dict).ffill(), None


# ================= 4. 优化因子类 =================

class EnhancedFactors:
    @staticmethod
    def calculate_multi_period_momentum(df_p, periods=[5,10,20,60], weights=[0.4,0.3,0.2,0.1], smooth_window=3, mom_tolerance=0):
        if smooth_window > 1: smoothed_p = df_p.rolling(window=smooth_window, min_periods=1).mean()
        else: smoothed_p = df_p
        if len(weights) != len(periods): weights = [1.0/len(periods)]*len(periods)
        else:
            total_weight = sum(weights)
            weights = [w/total_weight for w in weights]
        weighted_avg_rank = pd.DataFrame(0.0, index=df_p.index, columns=df_p.columns)
        mom_sign_matrix = pd.DataFrame(0.0, index=df_p.index, columns=df_p.columns)
        debug_period_roc, debug_period_rank = {}, {}
        for i, p in enumerate(periods):
            roc = smoothed_p.diff(p) / smoothed_p.shift(p).abs().replace(0, np.nan)
            current_rank = roc.rank(axis=1, pct=True)
            w = weights[i]
            weighted_avg_rank = weighted_avg_rank.add(current_rank * w, fill_value=0)
            mom_sign = (roc > 0).astype(int)
            mom_sign_matrix = mom_sign_matrix.add(mom_sign, fill_value=0)
            debug_period_roc[p] = roc
            debug_period_rank[p] = current_rank
        momentum_score = weighted_avg_rank.rank(axis=1, pct=True)
        required_pass_count = len(periods) - mom_tolerance
        momentum_filter = (mom_sign_matrix >= required_pass_count)
        mom_debug_info = {
            'roc': debug_period_roc, 'rank': debug_period_rank, 'avg_rank': weighted_avg_rank,
            'weights': dict(zip(periods, weights)), 'mom_sign_matrix': mom_sign_matrix,
        }
        return momentum_score, momentum_filter, mom_debug_info

    @staticmethod
    def calculate_volatility_adjustment(df_p, target_vol=0.25):
        returns = df_p.diff() / df_p.shift(1).abs().replace(0, np.nan)
        all_assets_vol = returns.rolling(60, min_periods=20).std() * np.sqrt(252)
        market_vol_avg = all_assets_vol.mean(axis=1)
        vol_scaler = target_vol / (market_vol_avg + 1e-8)
        return pd.Series(vol_scaler.clip(0.3, 2.0), index=df_p.index)

    @staticmethod
    def calculate_liquidity_score(df_amount, window=20):
        # 🛠️ 对齐 dm_pro 的时序平均后进行截面百分比排名
        liquidity = df_amount.rolling(window, min_periods=1).mean()
        return liquidity.rank(axis=1, pct=True)

    @staticmethod
    def calculate_steady_growth_filter(df_p, filter_ma=60, er_window=20, er_threshold=0.3):
        ma = df_p.rolling(window=filter_ma, min_periods=1).mean()
        ma_rising = ma > ma.shift(1)
        net_change = df_p.diff(er_window).abs()
        path_length = df_p.diff().abs().rolling(window=er_window, min_periods=1).sum()
        efficiency_ratio = net_change / path_length.replace(0, np.nan)
        er_pass = efficiency_ratio >= er_threshold
        return ma_rising & er_pass, {'er': efficiency_ratio, 'ma_rising': ma_rising}


# ================= 5. 高级止损类 =================

class AdvancedStopLoss:
    def __init__(self):
        self.asset_records = {}
    def update_and_check(self, asset, entry_price, current_price, days_held, atr_value, today_high, today_low):
        records = self.asset_records.get(asset, {'entry_price': entry_price, 'max_profit': 0.0, 'stop_reason': None})
        profit_ratio = (current_price - entry_price) / abs(entry_price) if entry_price != 0 else 0
        if profit_ratio > records['max_profit']: records['max_profit'] = profit_ratio
        stop_reason = None
        suggested_exit_price = current_price
        if today_high > 0:
            drop_from_high = (today_high - current_price) / abs(today_high)
            if drop_from_high > 0.04 and current_price > entry_price:
                stop_reason = f"动态止盈(高点回撤{drop_from_high:.1%})"
                suggested_exit_price = today_high * (1 - 0.04)
                suggested_exit_price = max(suggested_exit_price, current_price)
        if not stop_reason:
            if days_held > 20 and profit_ratio < 0.03:
                stop_reason = "时间止损(持仓僵化)"
                suggested_exit_price = current_price
        if stop_reason: records['stop_reason'] = stop_reason
        self.asset_records[asset] = records
        return (stop_reason, suggested_exit_price) if stop_reason else None


# ================= 6. 核心多空策略逻辑 =================

def build_weekly_summary(cycle_details: list) -> pd.DataFrame:
    """按 ISO 自然周汇总日收益，与 generate_weekly_log 口径一致。"""
    if not cycle_details:
        return pd.DataFrame(columns=['周次', '起始', '结束', '周收益', '周末净值'])
    rows = []
    bucket = []
    week_no = 1
    last_iso = None
    for d in cycle_details:
        iso = pd.Timestamp(d['date']).isocalendar()[:2]
        if last_iso is not None and iso != last_iso:
            c_ret = float(np.prod([1 + x['ret'] for x in bucket]) - 1)
            rows.append({
                '周次': week_no,
                '起始': pd.Timestamp(bucket[0]['date']).date(),
                '结束': pd.Timestamp(bucket[-1]['date']).date(),
                '周收益': c_ret,
                '周末净值': bucket[-1]['nav'],
            })
            week_no += 1
            bucket = []
        last_iso = iso
        bucket.append(d)
    if bucket:
        c_ret = float(np.prod([1 + x['ret'] for x in bucket]) - 1)
        rows.append({
            '周次': week_no,
            '起始': pd.Timestamp(bucket[0]['date']).date(),
            '结束': pd.Timestamp(bucket[-1]['date']).date(),
            '周收益': c_ret,
            '周末净值': bucket[-1]['nav'],
        })
    return pd.DataFrame(rows)


def run_long_short_strategy(df_p, df_atr_norm, df_l, df_o, df_h, df_atr_abs, df_vol, df_vol_ma, df_amount, df_liquidity, params):
    df_p, df_atr_norm, df_l, df_o, df_h, df_atr_abs, df_vol, df_vol_ma, df_amount, df_liquidity = canonicalize_panel_columns(
        df_p, df_atr_norm, df_l, df_o, df_h, df_atr_abs, df_vol, df_vol_ma, df_amount, df_liquidity
    )
    lookback_periods = params['periods']
    hold_num = params['hold_num']
    filter_ma = params['ma']
    stop_loss_trail = params['stop_loss_trail']
    stop_loss_hard = params['stop_loss_hard']
    commission_rate = params.get('commission', 0.0)
    start_date = pd.to_datetime(params['start_date'])
    end_date = pd.to_datetime(params['end_date'])
    use_vol_scaling = params.get('use_vol_scaling', True)
    target_volatility = params.get('target_volatility', 0.25)
    max_per_sector = params.get('max_per_sector', 2)
    net_exposure_target = params.get('net_exposure_target', 1.0)
    max_gross_exposure = params.get('max_gross_exposure', 1.0)
    max_margin_usage = params.get('max_margin_usage', 0.35)
    max_single_weight = params.get('max_single_weight', 0.30)
    mom_tolerance = params.get('mom_tolerance', 0)
    rank_protect_enabled = params.get('rank_protect_enabled', False)
    rank_protect_threshold = params.get('rank_protect_threshold', 8)
    execution_mode = params.get('execution_mode', 'close')  # close | next_open

    long_count = max(hold_num, 3)
    short_count = max(hold_num, 3)

    factors = EnhancedFactors()
    momentum_score, momentum_filter, mom_debug_info = factors.calculate_multi_period_momentum(
        df_p, lookback_periods, smooth_window=3, mom_tolerance=mom_tolerance
    )
    liquidity_score = factors.calculate_liquidity_score(df_amount)
    
    if use_vol_scaling: vol_scaler = factors.calculate_volatility_adjustment(df_p, target_volatility)
    else: vol_scaler = pd.Series(1.0, index=df_p.index)
        
    ma_filter = df_p > df_p.rolling(filter_ma, min_periods=1).mean()
    short_ma_win = 60
    ma_filter_short = df_p < df_p.rolling(short_ma_win, min_periods=1).mean()
    use_steady_filter = params.get('use_steady_filter', True)
    if use_steady_filter:
        er_win = params.get('er_win', 20)
        er_thresh = params.get('er_thresh', 0.3)
        steady_filter, steady_debug = factors.calculate_steady_growth_filter(df_p, filter_ma, er_win, er_thresh)
        combined_ma_filter = ma_filter & steady_filter
    else:
        combined_ma_filter = ma_filter

    # 多空分开评分：
    # - 多头：动量越强越好，流动性越好越好
    # - 空头：动量越弱越好，但仍要求流动性越好越好，避免低流动性品种因为低综合分被误选为空头
    long_score = (momentum_score * 0.7 + liquidity_score * 0.3).clip(0, 1)
    short_score = ((1 - momentum_score) * 0.7 + liquidity_score * 0.3).clip(0, 1)

    debug_factors = {
        'momentum_score': momentum_score, 'momentum_filter': momentum_filter,
        'liquidity_score': liquidity_score, 'long_score': long_score, 'short_score': short_score,
        'ma_filter': combined_ma_filter,
        'vol_scaler': vol_scaler, 'mom_debug_info': mom_debug_info
    }

    dates = df_p.index
    try: start_idx = dates.get_indexer([start_date], method='bfill')[0]
    except: start_idx = 0
    if start_idx < 1: start_idx = 1

    capital = 1.0
    capital_gross = 1.0
    nav_record = []
    asset_contribution = {}
    logs = []
    current_holdings_long = {}
    current_holdings_short = {}
    entry_prices = {}  
    entry_dates = {}   
    banned_assets = set()
    stop_manager = AdvancedStopLoss()
    cycle_details = []
    all_daily_details = []
    last_iso_week = None
    cycle_count = 1
    pending_long, pending_short = {}, {}
    has_pending = False

    def generate_weekly_log(details, count, current_nav):
        if not details: return []
        block_logs = []
        c_ret = (np.prod([1 + d['ret'] for d in details]) - 1)
        start_d_str = details[0]['date'].date()
        end_d_str = details[-1]['date'].date()
        header = f"第{count}周：{start_d_str} ~ {end_d_str} 收益: {c_ret * 100:+.2f}% | 净值: {current_nav:.4f}"
        block_logs.append(header)
        block_logs.append("")
        for d in details:
            date_str = f"[{d['date'].date()}]"
            block_logs.append(date_str)
            block_logs.append(f"📊 当天总收益率：{d['ret'] * 100:+.2f}%")
            if d.get('start_hold_long') or d.get('start_hold_short'):
                block_logs.append("📈 当天持仓明细：")
                for asset, weight in d.get('start_hold_long', {}).items():
                    if weight > 0:
                        ret = d['asset_rets'].get(asset, 0.0) * 100
                        block_logs.append(f"  {asset} (做多) ➔ 仓位: {weight:.1%}, 今日涨跌: {ret:+.2f}%")
                for asset, weight in d.get('start_hold_short', {}).items():
                    if weight > 0:
                        ret = d['asset_rets'].get(asset, 0.0) * 100
                        block_logs.append(f"  {asset} (做空) ➔ 仓位: {weight:.1%}, 今日涨跌: {-ret:+.2f}%")
            else:
                block_logs.append("📈 当天持仓：空仓")
            if d['stops']:
                stop_texts = [f"{s['asset']}({s['dir']}) {s['reason']}" for s in d['stops']]
                block_logs.append(f"⚠️ 当天止损：{'，'.join(stop_texts)} (进入隔离小黑屋🚫)")
            next_long = d.get('next_day_hold_long', {})
            next_short = d.get('next_day_hold_short', {})
            if next_long or next_short:
                block_logs.append("🔮 隔夜持仓：")
                for asset, weight in next_long.items():
                    if weight > 0:
                        block_logs.append(f"  {asset} (做多)：(仓位：{weight:.0%})")
                for asset, weight in next_short.items():
                    if weight > 0:
                        block_logs.append(f"  {asset} (做空)：(仓位：{weight:.0%})")
            else:
                block_logs.append("🔮 隔夜持仓：空仓")
            block_logs.append("")
        block_logs.append("=" * 60)
        return block_logs

    for i in range(start_idx, len(dates)):
        curr_date = dates[i]
        if curr_date > end_date: break
        prev_date = dates[i-1]

        curr_iso = curr_date.isocalendar()[:2]
        if last_iso_week is not None and curr_iso != last_iso_week:
            logs.extend(generate_weekly_log(cycle_details, cycle_count, cycle_details[-1]['nav'] if cycle_details else capital))
            cycle_count += 1
            cycle_details = []
        last_iso_week = curr_iso

        assets_to_unban = []
        for banned_asset in list(banned_assets):
            try:
                if banned_asset not in df_h.columns: continue
                h_hist = df_h.loc[:prev_date, banned_asset]
                if len(h_hist) >= 5 and h_hist.iloc[-1] > h_hist.iloc[-4:-1].min():
                    assets_to_unban.append(banned_asset)
            except: continue
        for asset in assets_to_unban: banned_assets.remove(asset)

        daily_gross_pnl = 0.0
        daily_cost = 0.0
        gap_long_pnl, gap_short_pnl = 0.0, 0.0
        if execution_mode == 'next_open':
            if current_holdings_long or current_holdings_short:
                gap_long_pnl, gap_short_pnl, gap_total = gap_pnl_breakdown(
                    current_holdings_long, current_holdings_short, prev_date, curr_date, df_p, df_o,
                )
                daily_gross_pnl += gap_total
            if has_pending:
                daily_cost += apply_pending_at_open(
                    pending_long, pending_short,
                    current_holdings_long, current_holdings_short,
                    curr_date, df_o, commission_rate, entry_prices, entry_dates,
                )

        start_long = current_holdings_long.copy()
        start_short = current_holdings_short.copy()
        
        stopped_assets_info = []
        daily_asset_rets = {}

        # 1. 多头止损检测
        for asset, w in list(start_long.items()):
            if w == 0: continue
            prev_close = df_p.loc[prev_date, asset]
            today_open = df_o.loc[curr_date, asset]
            today_low = df_l.loc[curr_date, asset]
            today_high = df_h.loc[curr_date, asset]
            today_close = df_p.loc[curr_date, asset]
            if pd.isna(prev_close) or pd.isna(today_close):
                current_holdings_long[asset] = 0
                continue
            
            ref_entry = entry_prices.get((asset, 'long'), prev_close)
            stop_trail = prev_close * (1 - stop_loss_trail)
            stop_hard = ref_entry * (1 - stop_loss_hard)
            atr_val = df_atr_abs.loc[prev_date, asset] if asset in df_atr_abs.columns else 0
            stop_atr = prev_close - (3 * atr_val)
            effective_stop = max(stop_trail, stop_hard, stop_atr)
            
            triggered = False; exit_price = today_close; stop_reason = ""
            if today_open < effective_stop:
                triggered = True; exit_price = today_open; stop_reason = "多头跳空硬止损" if today_open < stop_hard else "多头跳空移动止损"
            elif today_low < effective_stop:
                triggered = True
                if effective_stop == stop_atr: exit_price = stop_atr; stop_reason = "多头ATR熔断"
                elif effective_stop == stop_hard: exit_price = stop_hard; stop_reason = "多头硬止损"
                else: exit_price = stop_trail; stop_reason = "多头移动止损"
                
            if not triggered:
                days_held = (curr_date - entry_dates.get((asset, 'long'), curr_date)).days
                adv_res = stop_manager.update_and_check(asset + "_L", ref_entry, today_close, days_held, atr_val, today_high, today_low)
                if adv_res:
                    triggered = True; stop_reason = f"多头{adv_res[0]}"; exit_price = max(adv_res[1], today_low)
                    
            if triggered:
                base_px = pnl_base_price(execution_mode, prev_date, curr_date, asset, df_p, df_o)
                actual_ret = (exit_price - base_px) / base_px if pd.notna(base_px) and base_px != 0 else 0.0
                current_holdings_long[asset] = 0
                entry_prices.pop((asset, 'long'), None); entry_dates.pop((asset, 'long'), None)
                banned_assets.add(asset)
                stopped_assets_info.append({'asset': asset, 'ret': actual_ret, 'reason': stop_reason, 'dir': '多头'})
            else:
                base_px = pnl_base_price(execution_mode, prev_date, curr_date, asset, df_p, df_o)
                actual_ret = (today_close - base_px) / base_px if pd.notna(base_px) and base_px != 0 else 0.0
                
            daily_gross_pnl += w * actual_ret
            asset_contribution[asset] = asset_contribution.get(asset, 0.0) + w * actual_ret
            daily_asset_rets[asset] = actual_ret

        # 2. 空头止损检测
        for asset, w in list(start_short.items()):
            if w == 0: continue
            prev_close = df_p.loc[prev_date, asset]
            today_open = df_o.loc[curr_date, asset]
            today_low = df_l.loc[curr_date, asset]
            today_high = df_h.loc[curr_date, asset]
            today_close = df_p.loc[curr_date, asset]
            if pd.isna(prev_close) or pd.isna(today_close):
                current_holdings_short[asset] = 0
                continue
            
            ref_entry = entry_prices.get((asset, 'short'), prev_close)
            stop_trail = prev_close * (1 + stop_loss_trail)
            stop_hard = ref_entry * (1 + stop_loss_hard)
            atr_val = df_atr_abs.loc[prev_date, asset] if asset in df_atr_abs.columns else 0
            stop_atr = prev_close + (3 * atr_val)
            effective_stop = min(stop_trail, stop_hard, stop_atr)
            
            triggered = False; exit_price = today_close; stop_reason = ""
            if today_open > effective_stop:
                triggered = True; exit_price = today_open; stop_reason = "空头跳空硬止损" if today_open > stop_hard else "空头跳空移动止损"
            elif today_high > effective_stop:
                triggered = True
                if effective_stop == stop_atr: exit_price = stop_atr; stop_reason = "空头ATR熔断"
                elif effective_stop == stop_hard: exit_price = stop_hard; stop_reason = "空头硬止损"
                else: exit_price = stop_trail; stop_reason = "空头移动止损"
                
            if not triggered:
                days_held = (curr_date - entry_dates.get((asset, 'short'), curr_date)).days
                adv_res = stop_manager.update_and_check(asset + "_S", -ref_entry, -today_close, days_held, atr_val, -today_low, -today_high)
                if adv_res:
                    triggered = True; stop_reason = f"空头{adv_res[0]}"; exit_price = min(-adv_res[1], today_high)
                    
            if triggered:
                base_px = pnl_base_price(execution_mode, prev_date, curr_date, asset, df_p, df_o)
                actual_ret = -(exit_price - base_px) / base_px if pd.notna(base_px) and base_px != 0 else 0.0
                current_holdings_short[asset] = 0
                entry_prices.pop((asset, 'short'), None); entry_dates.pop((asset, 'short'), None)
                banned_assets.add(asset)
                stopped_assets_info.append({'asset': asset, 'ret': actual_ret, 'reason': stop_reason, 'dir': '空头'})
            else:
                base_px = pnl_base_price(execution_mode, prev_date, curr_date, asset, df_p, df_o)
                actual_ret = -(today_close - base_px) / base_px if pd.notna(base_px) and base_px != 0 else 0.0
                
            daily_gross_pnl += w * actual_ret
            asset_contribution[asset] = asset_contribution.get(asset, 0.0) + w * actual_ret
            daily_asset_rets[asset] = -actual_ret

        # 3. 截面非对称信号生成
        next_long, next_short = {}, {}
        
        try:
            filter_cond = combined_ma_filter.loc[curr_date] & momentum_filter.loc[curr_date]
            long_scores = long_score.loc[curr_date].dropna()
            short_scores = short_score.loc[curr_date].dropna()
            common_score_idx = long_scores.index.intersection(short_scores.index)
            long_scores = long_scores.loc[common_score_idx]
            short_scores = short_scores.loc[common_score_idx]
                
            valid_pool = [a for a in common_score_idx if a not in banned_assets and pd.notna(df_p.loc[curr_date, a]) and (a not in df_vol.columns or df_vol.loc[curr_date, a] >= MIN_ACTIVE_VOLUME)]
            
            if valid_pool:
                ranked_long = long_scores.loc[valid_pool].sort_values(ascending=False)
                ranked_short = short_scores.loc[valid_pool].sort_values(ascending=False)
                short_filter_cond = ma_filter_short.loc[curr_date]

                long_candidates = ranked_long.loc[
                    [a for a in ranked_long.index if filter_cond.get(a, False) and long_scores[a] > 0.5]
                ]
                short_candidates = ranked_short.loc[
                    [a for a in ranked_short.index if short_filter_cond.get(a, False) and short_scores[a] > 0.5]
                ]

                if rank_protect_enabled:
                    ideal_long = _pick_with_rank_protect(
                        long_candidates, long_count, max_per_sector, start_long, rank_protect_threshold,
                    )
                    ideal_short = _pick_with_rank_protect(
                        short_candidates, short_count, max_per_sector, start_short, rank_protect_threshold,
                    )
                else:
                    sec_counts_l = {}
                    ideal_long = []
                    seen_long_codes = set()
                    for a in ranked_long.index:
                        code = _resolve_contract_code(a)
                        if code in seen_long_codes:
                            continue
                        if not filter_cond.get(a, False): continue
                        if long_scores[a] <= 0.5: continue
                        sec = get_sector(a)
                        if sec_counts_l.get(sec, 0) < max_per_sector:
                            ideal_long.append(a)
                            seen_long_codes.add(code)
                            sec_counts_l[sec] = sec_counts_l.get(sec, 0) + 1
                        if len(ideal_long) == long_count: break

                    sec_counts_s = {}
                    ideal_short = []
                    seen_short_codes = set()
                    for a in ranked_short.index:
                        code = _resolve_contract_code(a)
                        if code in seen_short_codes:
                            continue
                        if not short_filter_cond.get(a, False): continue
                        if short_scores[a] <= 0.5: continue
                        sec = get_sector(a)
                        if sec_counts_s.get(sec, 0) < max_per_sector:
                            ideal_short.append(a)
                            seen_short_codes.add(code)
                            sec_counts_s[sec] = sec_counts_s.get(sec, 0) + 1
                        if len(ideal_short) == short_count: break

                pos_multiplier = vol_scaler.loc[curr_date] if use_vol_scaling else 1.0

                # 风险平价配资 - 多头
                raw_w_long = {}
                if ideal_long:
                    vols = df_atr_norm.loc[curr_date, ideal_long].clip(lower=0.00001)
                    inv_v = 1.0 / vols
                    w_l = (inv_v / inv_v.sum()) * pos_multiplier
                    w_l = w_l.clip(upper=max_single_weight)
                    if w_l.sum() > 0: w_l = (w_l / w_l.sum()) * pos_multiplier
                    raw_w_long = w_l.to_dict()

                # 风险平价配资 - 空头
                raw_w_short = {}
                if ideal_short:
                    vols = df_atr_norm.loc[curr_date, ideal_short].clip(lower=0.00001)
                    inv_v = 1.0 / vols
                    w_s = (inv_v / inv_v.sum()) * pos_multiplier
                    w_s = w_s.clip(upper=max_single_weight)
                    if w_s.sum() > 0: w_s = (w_s / w_s.sum()) * pos_multiplier
                    raw_w_short = w_s.to_dict()

                # 组合净敞口风控截断
                tot_l = sum(raw_w_long.values())
                tot_s = sum(raw_w_short.values())
                net_exp = tot_l - tot_s
                
                if net_exp > net_exposure_target:
                    scale = net_exposure_target / max(net_exp, 0.001)
                    raw_w_long = {a: w * scale for a, w in raw_w_long.items()}
                elif net_exp < -net_exposure_target:
                    scale = net_exposure_target / max(abs(net_exp), 0.001)
                    raw_w_short = {a: w * scale for a, w in raw_w_short.items()}

                # 杠杆限制：杠杆 = 多头总名义仓位 + 空头总名义仓位。
                # 超过上限时，多头和空头同时等比例降仓，控制组合总名义杠杆。
                gross_exposure_pre_margin = sum(raw_w_long.values()) + sum(raw_w_short.values())
                if gross_exposure_pre_margin > max_gross_exposure:
                    gross_scale = max_gross_exposure / max(gross_exposure_pre_margin, 0.001)
                    raw_w_long = {a: w * gross_scale for a, w in raw_w_long.items()}
                    raw_w_short = {a: w * gross_scale for a, w in raw_w_short.items()}

                # 保证金占用限制：估算保证金 = Σ(|名义仓位| × 品种保证金率)
                # 超过上限时，多头和空头同时等比例降仓，保留原始多空结构与品种相对权重。
                margin_usage, _ = estimate_margin_usage(raw_w_long, raw_w_short)
                if margin_usage > max_margin_usage:
                    margin_scale = max_margin_usage / max(margin_usage, 0.001)
                    raw_w_long = {a: w * margin_scale for a, w in raw_w_long.items()}
                    raw_w_short = {a: w * margin_scale for a, w in raw_w_short.items()}
                    margin_usage, _ = estimate_margin_usage(raw_w_long, raw_w_short)

                # 多头调仓摩擦
                for asset, target_w in raw_w_long.items():
                    curr_w = start_long.get(asset, 0.0)
                    if execution_mode == 'close':
                        if abs(target_w - curr_w) > 0.02:
                            daily_cost += abs(target_w - curr_w) * commission_rate
                            if curr_w == 0:
                                entry_prices[(asset, 'long')] = df_p.loc[curr_date, asset]
                                entry_dates[(asset, 'long')] = curr_date
                    next_long[asset] = target_w
                    
                # 空头调仓摩擦
                for asset, target_w in raw_w_short.items():
                    curr_w = start_short.get(asset, 0.0)
                    if execution_mode == 'close':
                        if abs(target_w - curr_w) > 0.02:
                            daily_cost += abs(target_w - curr_w) * commission_rate
                            if curr_w == 0:
                                entry_prices[(asset, 'short')] = df_p.loc[curr_date, asset]
                                entry_dates[(asset, 'short')] = curr_date
                    next_short[asset] = target_w

                if execution_mode == 'close':
                    for a in start_long:
                        if a not in next_long: daily_cost += start_long[a] * commission_rate
                    for a in start_short:
                        if a not in next_short: daily_cost += start_short[a] * commission_rate
        except: pass

        if execution_mode == 'next_open':
            pending_long, pending_short = next_long.copy(), next_short.copy()
            has_pending = bool(pending_long or pending_short)
        else:
            current_holdings_long = next_long.copy()
            current_holdings_short = next_short.copy()
        
        daily_net_pnl = daily_gross_pnl - daily_cost
        capital *= (1 + daily_net_pnl)
        capital_gross *= (1 + daily_gross_pnl)
        nav_record.append({
            'date': curr_date, 'nav': capital, 'nav_gross': capital_gross,
            'daily_cost': daily_cost,
        })
        
        long_exposure = sum(next_long.values())
        short_exposure = sum(next_short.values())
        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure
        margin_usage, margin_detail = estimate_margin_usage(next_long, next_short)

        cycle_details.append({
            'date': curr_date, 'ret': daily_net_pnl,
            'start_hold_long': start_long, 'start_hold_short': start_short,
            'asset_rets': daily_asset_rets, 'stops': stopped_assets_info,
            'next_day_hold_long': next_long, 'next_day_hold_short': next_short,
            'banned_list': list(banned_assets), 'entry_prices': entry_prices.copy(), 'nav': capital,
            'long_exposure': long_exposure, 'short_exposure': short_exposure,
            'gross_exposure': gross_exposure, 'net_exposure': net_exposure,
            'margin_usage': margin_usage, 'margin_detail': margin_detail,
            'vol_scaler': vol_scaler.loc[curr_date] if curr_date in vol_scaler.index else np.nan,
            'gap_long_pnl': gap_long_pnl, 'gap_short_pnl': gap_short_pnl,
        })
        all_daily_details.append(cycle_details[-1])

    nav_df = pd.DataFrame(nav_record).set_index('date')
    if cycle_details:
        logs.extend(generate_weekly_log(cycle_details, cycle_count, capital))
    return nav_df, logs, debug_factors, asset_contribution, all_daily_details


# ================= 7. UI 主控制渲染 =================

with st.sidebar:
    st.header("多空双向趋势动量面板")
    st.caption("Enhanced Long-Short Momentum System")
    data_folder = st.text_input("品种数据路径", value=DEFAULT_DATA_FOLDER)
    index_folder = st.text_input("指数数据目录 (基准)", value=r"D:\数据集文件\日线")
    bench_name_input = st.text_input("基准识别名", value="文华商品")
    st.divider()
    col1, col2 = st.columns(2)
    start_d = col1.date_input("开始日期", value=pd.to_datetime("2026-01-01"))
    end_d = col2.date_input("结束日期", value=pd.to_datetime("2026-12-31"))
    
    st.subheader("🎯 核心多空配比控制")
    c1, c2 = st.columns(2)
    hold_num = c1.number_input("单向目标持仓", 1, 20, 4)
    max_sector = c2.number_input("板块风控上限", 1, 10, 2)
    net_exp_target = st.slider("最大净敞口", 0.0, 1.0, 1.00, step=0.05, help="净敞口 = 多头总仓位 - 空头总仓位；超过上限时对偏多/偏空一侧等比例降仓。")
    max_gross_exposure_ui = st.slider("最大杠杆", 0.50, 4.00, 1.00, step=0.10, help="杠杆 = 多头总仓位 + 空头总仓位；超过上限时多空同时等比例降仓。")
    max_single_weight_ui = st.slider("单品种仓位上限", 0.05, 0.50, 0.30, step=0.05, help="风险平价初配后，单个品种权重不超过此上限，超出部分重归一化分配。")
    max_margin_usage_ui = st.slider("最大估算保证金占用", 0.05, 1.00, 0.35, step=0.05, help="按 品种名义仓位×近似保证金率 估算账户保证金占用；超过上限时多空同时等比例降仓。")

    st.markdown("**📌 排名保护（续持）**")
    rank_protect_ui = st.checkbox(
        "启用排名保护", value=True,
        help="已持仓品种综合得分排名未跌出阈值则续持；跌出后才换仓，可减少频繁换手。",
    )
    rank_protect_threshold_ui = st.select_slider(
        "续持排名阈值 (Top N)", options=[6, 8, 10], value=8, disabled=not rank_protect_ui,
        help="例：阈值8 = 排名第8以内续持，第9名起才换出；目标持仓数仍由上方「单向目标持仓」控制。",
    )

    st.markdown("**🛡️ 动量突围防守过滤器（仅做多侧）**")
    mom_tolerance_ui = st.slider("动量周期允许微跌/收绿缺口", 0, 3, 1, help="0=4周期全红（严苛）；1=允许1个周期微跌；仅影响做多滤网")
    
    st.write("🛑 **风控安全网设定**")
    s1, s2 = st.columns(2)
    stop_trail = s1.number_input("移动止损(%)", 0.0, 20.0, 4.0, step=0.5)
    stop_hard = s2.number_input("硬核止损(%)", 0.0, 20.0, 4.0, step=0.5)
    
    st.subheader("💸 摩擦摩擦成本设置")
    cc1, cc2 = st.columns(2)
    comm_bp = cc1.number_input("手续费(万)", 0.0, 5.0, 1.5, step=0.1,
                               help="单边手续费，万1.5 = 0.015% = 1.5bp")
    
    with st.expander("🛠️ 多因子底层引擎深度微调"):
        periods_input = st.text_input("交叉动量时序列表", value="5,10,20,60")
        periods = [int(p.strip()) for p in periods_input.split(",") if p.strip().isdigit()]
        if not periods: periods = [5,10,20,60]
        ma_win = st.number_input("绝对多头过滤器均线", value=60)
        atr_win = st.number_input("ATR时序平滑周期", value=20)
        target_volatility = st.slider("目标组合年化波动率%", 5, 50, 25) / 100
        er_win = st.number_input("ER效率计算时段", value=20)
        er_thresh = st.slider("最低信噪比ER判定线", 0.0, 1.0, 0.30, step=0.05)

    run_btn = st.button("🚀 启动多空回测模拟", type="primary", use_container_width=True)
    if st.button("🔄 清除数据缓存", use_container_width=True, help="修改去重逻辑后若仍见重复品种，点此强制重载 CSV"):
        load_data_and_calc_metrics.clear()
        st.session_state.pop('has_run', None)

if run_btn: st.session_state['has_run'] = True

if st.session_state.get('has_run', False):
    with st.spinner("数据加载中..."):
        df_p, df_atr_norm, df_l, df_o, df_h, df_atr_abs, df_vol, df_vol_ma, df_amount, df_liquidity, err = load_data_and_calc_metrics(
            data_folder, atr_win, _panel_dedupe_version=3
        )
        idx_p, err_idx = load_index_close_panel(index_folder)

    if err: st.error(err)
    else:
        dup_codes = [code for code, cols in _group_columns_by_contract_code(df_p.columns).items() if len(cols) > 1]
        skip_log = st.session_state.get('data_skip_log', [])
        merge_lines = [ln for ln in skip_log if '合并' in ln or '同合约' in ln]
        if merge_lines:
            with st.expander(f"📋 品种去重记录 ({len(merge_lines)} 条)", expanded=False):
                st.code("\n".join(merge_lines[:80]))
        if dup_codes:
            st.error(f"数据面板仍有同合约重复列（{len(dup_codes)} 组），请先点侧栏「清除数据缓存」再回测。")
        if err_idx:
            st.warning(f"基准指数加载提示：{err_idx}")
        if start_d >= end_d:
            st.error("日期设置错误")
        elif not dup_codes:
            params = {
                'periods': periods, 'hold_num': hold_num, 'max_per_sector': max_sector, 'ma': ma_win,
                'stop_loss_trail': stop_trail/100.0, 'stop_loss_hard': stop_hard/100.0,
                'start_date': start_d, 'end_date': end_d, 'commission': comm_bp/10000,
                'target_volatility': target_volatility, 'use_vol_scaling': True,
                'use_steady_filter': True, 'er_win': er_win, 'er_thresh': er_thresh,
                'net_exposure_target': net_exp_target, 'max_gross_exposure': max_gross_exposure_ui,
                'max_margin_usage': max_margin_usage_ui, 'max_single_weight': max_single_weight_ui,
                'mom_tolerance': mom_tolerance_ui,
                'rank_protect_enabled': rank_protect_ui,
                'rank_protect_threshold': rank_protect_threshold_ui,
            }
            with st.spinner("策略回测中..."):
                res_nav, res_logs, debug_data, res_contrib, res_cycle_details = run_long_short_strategy(df_p, df_atr_norm, df_l, df_o, df_h, df_atr_abs, df_vol, df_vol_ma, df_amount, df_liquidity, params)
                res_contrib_df = pd.DataFrame(list(res_contrib.items()), columns=['Asset', 'Contribution'])
                
            if res_nav.empty: st.warning("结果为空，请检查日期或数据")
            else:
                # 8大核心指标卡片文案
                res_contrib_df.sort_values('Contribution', ascending=False, inplace=True)
                tot_ret = res_nav['nav'].iloc[-1] - 1
                d_rets = res_nav['nav'].pct_change().dropna()
                calendar_days = max((res_nav.index[-1] - res_nav.index[0]).days, 1)
                years = calendar_days / 365.25
                # CAGR = (终值/初值)^(1/年数)-1；整年回测时 years≈1，与累计收益一致
                ann_ret = (1 + tot_ret) ** (1 / years) - 1 if years > 0 else 0
                max_dd = ((res_nav['nav'] - res_nav['nav'].cummax()) / res_nav['nav'].cummax()).min()
                sharpe = (d_rets.mean() / d_rets.std()) * np.sqrt(252) if not d_rets.empty and d_rets.std() != 0 else 0
                calmar = abs(ann_ret/max_dd) if max_dd != 0 else 0
                win_rate = (d_rets>0).mean()*100 if len(d_rets)>0 else 0
                
                # 统计每日实际持仓的总个数
                daily_holding_counts = [
                    len([k for k, v in d.get('start_hold_long', {}).items() if v > 0]) +
                    len([k for k, v in d.get('start_hold_short', {}).items() if v > 0])
                    for d in res_cycle_details
                ]
                avg_daily_holdings = np.mean(daily_holding_counts) if daily_holding_counts else 0.0
                risk_df = pd.DataFrame(res_cycle_details)
                avg_long_exp = risk_df['long_exposure'].mean() if 'long_exposure' in risk_df else 0.0
                avg_short_exp = risk_df['short_exposure'].mean() if 'short_exposure' in risk_df else 0.0
                max_gross_exp = risk_df['gross_exposure'].max() if 'gross_exposure' in risk_df else 0.0
                max_margin_used = risk_df['margin_usage'].max() if 'margin_usage' in risk_df else 0.0

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("多空独立累计总净收益", f"{tot_ret*100:.2f}%")
                col2.metric("年化复合增长率(CAGR)", f"{ann_ret*100:.2f}%")
                col3.metric("平均每天持仓个数", f"{avg_daily_holdings:.1f} 个") 
                col4.metric("动态全时最大回撤", f"{max_dd*100:.2f}%")
                col5, col6, col7, col8 = st.columns(4)
                col5.metric("夏普表现比率 (Sharpe)", f"{sharpe:.2f}")
                col6.metric("卡玛风险收益比 (Calmar)", f"{calmar:.2f}")
                col7.metric("每日胜率偏置", f"{win_rate:.1f}%")
                col8.metric("运行结算总天数", f"{len(res_nav)} 天")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("平均多头仓位", f"{avg_long_exp:.1%}")
                r2.metric("平均空头仓位", f"{avg_short_exp:.1%}")
                r3.metric("最高杠杆", f"{max_gross_exp:.1%}")
                r4.metric("最高估算保证金占用", f"{max_margin_used:.1%}")

                nav_gross_final = res_nav['nav_gross'].iloc[-1] if 'nav_gross' in res_nav.columns else res_nav['nav'].iloc[-1]
                nav_net_final = res_nav['nav'].iloc[-1]
                fee_drag_nav = nav_gross_final - nav_net_final
                fee_drag_pct = fee_drag_nav / nav_gross_final * 100 if nav_gross_final > 0 else 0.0
                f1, f2, f3 = st.columns(3)
                f1.metric("手续费率(单边)", f"万{comm_bp:g}")
                f2.metric("手续费侵蚀净值", f"-{fee_drag_nav:.4f}", help="无手续费净值 − 含手续费净值（复利）")
                f3.metric("手续费侵蚀幅度", f"{fee_drag_pct:.2f}%", help="相对无手续费净值的最终侵蚀比例（复利）")

                holding_df_bh, holding_wave_stats = compute_holding_wave_stats(res_cycle_details, res_nav.index)
                if not holding_wave_stats.empty:
                    avg_hold_trading_days = float(holding_wave_stats['Hold_Days'].mean())
                    holding_wave_count = len(holding_wave_stats)
                else:
                    avg_hold_trading_days = 0.0
                    holding_wave_count = 0

                if rank_protect_ui:
                    rp1, rp2, rp3, rp4 = st.columns(4)
                    rp1.metric("排名保护", f"Top{rank_protect_threshold_ui} 续持")
                    rp2.metric(
                        "平均持仓天数", f"{avg_hold_trading_days:.1f} 个交易日",
                        help="与下方「持仓周期精细画像」同口径：按回测交易日统计连续持仓波段。",
                    )
                    rp3.metric("完整持仓段数", f"{holding_wave_count}")
                    rp4.metric("目标持仓 Top", f"{hold_num} / 侧")

                bench_nav_series = None
                actual_bench_name = None
                if idx_p is not None:
                    actual_bench_name = next((c for c in idx_p.columns if bench_name_input in c), None)
                    if actual_bench_name:
                        try:
                            bench_slice = idx_p.loc[res_nav.index[0]:res_nav.index[-1], actual_bench_name]
                            bench_nav_series = bench_slice / bench_slice.iloc[0]
                        except Exception:
                            pass

                # 9大看板名字
                t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
                    "📈 资产组合净值曲线", 
                    "📊 个体盈亏贡献分布", 
                    "📝 自动化穿透日志", 
                    "🔬 每日全市场因子透视", 
                    "🧮 独立品种数学解构", 
                    "🏅 时序动量截面排序", 
                    "⏱️ 持仓周期精细画像",
                    "🧭 多空收益拆解",
                    "🧩 策略框架诊断"
                ])
                
                with t1:
                    fig, ax1 = plt.subplots(figsize=(12, 4.5))
                    ax1.plot(res_nav.index, res_nav['nav'], color='#2ca02c', lw=2,
                             label=f'净值 ({res_nav["nav"].iloc[-1]:.2f})')

                    fee_drag = None
                    if comm_bp > 0 and 'nav_gross' in res_nav.columns:
                        fee_drag = res_nav['nav_gross'] - res_nav['nav']

                    excess_curve = None
                    if bench_nav_series is not None:
                        ax1.plot(bench_nav_series.index, bench_nav_series, color='#1f77b4', lw=1.5, alpha=0.85,
                                 label=f'基准 ({actual_bench_name})')
                        aligned_bench = bench_nav_series.reindex(res_nav.index).ffill()
                        strat_norm = res_nav['nav'] / res_nav['nav'].iloc[0]
                        bench_norm = aligned_bench / aligned_bench.iloc[0]
                        excess_curve = strat_norm - bench_norm

                    ax2 = ax1.twinx()
                    if fee_drag is not None:
                        ax2.plot(fee_drag.index, fee_drag, color='#d62728', lw=1.5, linestyle='-.',
                                 label=f'手续费 ({fee_drag.iloc[-1]:.4f})')
                    if excess_curve is not None:
                        ax2.plot(excess_curve.index, excess_curve, color='#ff7f0e', lw=1.5, linestyle='--',
                                 label='超额')
                        ax2.set_ylabel("超额 / 手续费", fontproperties=my_font, color='#666666')
                    elif fee_drag is not None:
                        ax2.set_ylabel("手续费侵蚀", fontproperties=my_font, color='#d62728')

                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, prop=my_font, loc='upper left')
                    ax1.set_title("趋势策略_多空日频", fontproperties=my_font, fontsize=14)
                    ax1.set_ylabel("累计净值", fontproperties=my_font)
                    ax1.grid(True, alpha=0.3)
                    fig.tight_layout(); st.pyplot(fig)

                with t2:
                    st.markdown("### 📊 品种贡献驾驶舱")
                    st.caption("先看谁贡献最大、谁拖累最大，再看贡献集中度。这个页面用来判断策略是否过度依赖少数品种。")
                    if not res_contrib_df.empty:
                        contrib_view = res_contrib_df.copy()
                        contrib_view['Contribution_pct'] = contrib_view['Contribution'] / tot_ret if tot_ret != 0 else 0
                        contrib_view['方向'] = np.where(contrib_view['Contribution'] >= 0, '贡献', '拖累')
                        pos_sum = contrib_view.loc[contrib_view['Contribution'] > 0, 'Contribution'].sum()
                        neg_sum = contrib_view.loc[contrib_view['Contribution'] < 0, 'Contribution'].sum()
                        top_asset = contrib_view.iloc[0]['Asset'] if not contrib_view.empty else '-'
                        worst_asset = contrib_view.sort_values('Contribution').iloc[0]['Asset'] if not contrib_view.empty else '-'
                        top5_share = contrib_view.head(5)['Contribution'].sum() / pos_sum if pos_sum != 0 else 0

                        c2m1, c2m2, c2m3, c2m4 = st.columns(4)
                        c2m1.metric("正贡献合计", f"{pos_sum:+.2%}")
                        c2m2.metric("负贡献合计", f"{neg_sum:+.2%}")
                        c2m3.metric("最大贡献品种", str(top_asset))
                        c2m4.metric("Top5贡献占比", f"{top5_share:.1%}")

                        c2_left, c2_right = st.columns([1.1, 1])
                        with c2_left:
                            plot_n = st.slider("贡献图显示品种数量", 5, min(40, len(contrib_view)), min(20, len(contrib_view)), key='t2_plot_n')
                            bar_df = pd.concat([contrib_view.head(plot_n // 2), contrib_view.tail(plot_n // 2)]).drop_duplicates('Asset')
                            bar_df = bar_df.sort_values('Contribution')
                            fig_c2, ax_c2 = plt.subplots(figsize=(8, max(4, len(bar_df) * 0.28)))
                            colors = ['#2ca02c' if v >= 0 else '#d62728' for v in bar_df['Contribution']]
                            ax_c2.barh(bar_df['Asset'], bar_df['Contribution'], color=colors, alpha=0.78)
                            ax_c2.axvline(0, color='black', lw=0.8, alpha=0.5)
                            ax_c2.set_title("头尾品种贡献分布", fontproperties=my_font)
                            ax_c2.grid(True, axis='x', alpha=0.25)
                            fig_c2.tight_layout(); st.pyplot(fig_c2)
                        with c2_right:
                            st.markdown("#### 🎯 使用者结论")
                            st.info(f"最大贡献：{top_asset}；最大拖累：{worst_asset}。如果 Top5 占比长期过高，说明策略收益依赖少数品种，需要关注集中度风险。")
                            view_mode_t2 = st.radio("查看", ['全部', '只看贡献', '只看拖累'], horizontal=True, key='t2_view_mode')
                            show_contrib = contrib_view.copy()
                            if view_mode_t2 == '只看贡献': show_contrib = show_contrib[show_contrib['Contribution'] > 0]
                            if view_mode_t2 == '只看拖累': show_contrib = show_contrib[show_contrib['Contribution'] < 0]
                            st.dataframe(
                                show_contrib.style.format({'Contribution': '{:+.2%}', 'Contribution_pct': '{:+.1%}'}).background_gradient(cmap='RdYlGn', subset=['Contribution']),
                                use_container_width=True, height=520
                            )
                    else:
                        st.info("暂无品种贡献数据。")

                with t3:
                    st.markdown("### 📝 自动化穿透日志工作台")
                    st.caption("按自然周汇总收益，并可搜索、定位、统计、下载。")
                    log_text = "\n".join(res_logs)
                    weekly_df = build_weekly_summary(res_cycle_details)
                    week_header_lines = [line for line in res_logs if line.startswith('第') and '周：' in line and '收益' in line]
                    if not weekly_df.empty:
                        latest_week = weekly_df.iloc[-1]
                        prev_week_ret = weekly_df.iloc[-2]['周收益'] if len(weekly_df) > 1 else np.nan
                        week_win_rate = (weekly_df['周收益'] > 0).mean()
                        w1, w2, w3, w4 = st.columns(4)
                        w1.metric(
                            "本周收益",
                            f"{latest_week['周收益'] * 100:+.2f}%",
                            help=f"{latest_week['起始']} ~ {latest_week['结束']}",
                        )
                        w2.metric(
                            "上周收益",
                            f"{prev_week_ret * 100:+.2f}%" if pd.notna(prev_week_ret) else "—",
                        )
                        w3.metric("周胜率", f"{week_win_rate:.1%}")
                        w4.metric("统计周数", f"{len(weekly_df)} 周")
                        st.dataframe(
                            weekly_df.sort_values('周次', ascending=False).style.format({
                                '周收益': '{:+.2%}',
                                '周末净值': '{:.4f}',
                            }).background_gradient(cmap='RdYlGn', subset=['周收益']),
                            use_container_width=True,
                            height=min(35 + 35 * len(weekly_df), 420),
                        )
                        st.divider()
                    log_col1, log_col2, log_col3, log_col4 = st.columns(4)
                    stop_lines = [line for line in res_logs if '止损' in line]
                    holding_lines = [line for line in res_logs if '仓位:' in line]
                    loss_day_lines = [line for line in res_logs if '当天总收益率：-' in line]
                    log_col1.metric("日志行数", f"{len(res_logs)}")
                    log_col2.metric("持仓记录行", f"{len(holding_lines)}")
                    log_col3.metric("止损记录行", f"{len(stop_lines)}")
                    log_col4.metric("亏损日记录", f"{len(loss_day_lines)}")

                    q1, q2, q3 = st.columns([2, 1, 1])
                    with q1:
                        log_keyword = st.text_input("搜索关键词（品种/止损/日期/做多/做空）", value="", key='t3_log_keyword')
                    with q2:
                        quick_filter = st.selectbox("快捷过滤", ['全部日志', '只看周汇总', '只看止损', '只看持仓', '只看亏损日'], key='t3_quick_filter')
                    with q3:
                        newest_first = st.checkbox("倒序显示", value=False, key='t3_reverse')

                    filtered_logs = res_logs.copy()
                    if quick_filter == '只看周汇总':
                        filtered_logs = week_header_lines
                    elif quick_filter == '只看止损':
                        filtered_logs = stop_lines
                    elif quick_filter == '只看持仓':
                        filtered_logs = holding_lines
                    elif quick_filter == '只看亏损日':
                        filtered_logs = loss_day_lines
                    if log_keyword.strip():
                        filtered_logs = [line for line in filtered_logs if log_keyword.strip().lower() in line.lower()]
                    if newest_first:
                        filtered_logs = list(reversed(filtered_logs))
                    st.text_area("筛选后日志", "\n".join(filtered_logs), height=560)
                    dl1, dl2 = st.columns(2)
                    if log_text: dl1.download_button("📥 下载完整日志", log_text, "long_short_strategy_log.txt", "text/plain")
                    if filtered_logs: dl2.download_button("📥 下载筛选日志", "\n".join(filtered_logs), "filtered_strategy_log.txt", "text/plain")

                with t4:
                    st.markdown("### 🔬 每日全品种得分透视")
                    st.caption("选择一个日期，查看当天所有品种的因子得分、排名以及过滤状态。这是排查'为什么没买它'的神器。")

                    valid_dates = res_nav.index
                    default_date = valid_dates[-1] if len(valid_dates) > 0 else date.today()

                    c_sel1, c_sel2 = st.columns([1, 3])
                    with c_sel1:
                        target_date_input = st.date_input("选择查看日期", value=default_date, min_value=valid_dates[0], max_value=valid_dates[-1], key="t4_date_input")

                    target_date = pd.to_datetime(target_date_input)

                    if target_date not in valid_dates:
                        st.warning("该日期无交易数据（可能是周末或节假日），请选择临近日期。")
                    else:
                        try:
                            day_detail = next((d for d in res_cycle_details if d['date'] == target_date), None)
                            held_assets = list(day_detail['next_day_hold_long'].keys()) + list(day_detail['next_day_hold_short'].keys()) if day_detail else []
                            banned_now = day_detail['banned_list'] if day_detail and 'banned_list' in day_detail else []
                            entry_prices_now = day_detail['entry_prices'] if day_detail and 'entry_prices' in day_detail else {}

                            mom_s = debug_data['momentum_score'].loc[target_date]
                            liq_s = debug_data['liquidity_score'].loc[target_date]
                            ma_pass = debug_data['ma_filter'].loc[target_date]
                            mom_pass = debug_data['momentum_filter'].loc[target_date]
                            sign_mat = debug_data['mom_debug_info']['mom_sign_matrix'].loc[target_date]
                            closes = df_p.loc[target_date]

                            stop_prices = pd.Series(np.nan, index=closes.index)
                            stop_trail_pct = stop_trail / 100.0
                            stop_hard_pct = stop_hard / 100.0

                            for asset in held_assets:
                                if asset in closes:
                                    prev_c = closes[asset]
                                    s_trail = prev_c * (1 - stop_trail_pct)
                                    s_hard = entry_prices_now.get((asset, 'long'), entry_prices_now.get((asset, 'short'), prev_c)) * (1 - stop_hard_pct)
                                    s_atr = prev_c - 3 * df_atr_abs.loc[target_date, asset] if asset in df_atr_abs.columns else 0
                                    stop_prices[asset] = max(s_trail, s_hard, s_atr)

                            long_final_score = debug_data['long_score'].loc[target_date].fillna(-1)
                            short_final_score = debug_data['short_score'].loc[target_date].fillna(-1)
                            short_momentum_weak_score = (1 - mom_s).fillna(-1)

                            df_debug = pd.DataFrame({
                                '价格': closes,
                                '次日止损': stop_prices,
                                '多头综合分': long_final_score,
                                '空头综合分': short_final_score,
                                '动量分': mom_s,
                                '空头弱势分': short_momentum_weak_score,
                                '流动性分': liq_s,
                                '趋势滤网': ma_pass,
                                '放宽动量网': mom_pass,
                                '收红周期数': sign_mat,
                            })

                            df_debug['状态'] = '观察'
                            df_debug.loc[df_debug.index.isin(banned_now), '状态'] = '🚫熔断黑名单'
                            if day_detail:
                                df_debug.loc[df_debug.index.isin(list(day_detail.get('next_day_hold_long', {}).keys())), '状态'] = '✅ 多头持仓'
                                df_debug.loc[df_debug.index.isin(list(day_detail.get('next_day_hold_short', {}).keys())), '状态'] = '📉 空头持仓'

                            df_debug.dropna(subset=['价格'], inplace=True)
                            df_debug['多头排名'] = df_debug['多头综合分'].rank(ascending=False, method='min')
                            df_debug['空头排名'] = df_debug['空头综合分'].rank(ascending=False, method='min')
                            df_debug.sort_values(by=['多头综合分'], ascending=False, inplace=True)

                            t4m1, t4m2, t4m3, t4m4, t4m5 = st.columns(5)
                            t4m1.metric("全市场有效品种", f"{len(df_debug)}")
                            t4m2.metric("趋势通过", f"{int(df_debug['趋势滤网'].sum())}")
                            t4m3.metric("动量通过", f"{int(df_debug['放宽动量网'].sum())}")
                            t4m4.metric("多头持仓", f"{(df_debug['状态'] == '✅ 多头持仓').sum()}")
                            t4m5.metric("空头持仓", f"{(df_debug['状态'] == '📉 空头持仓').sum()}")

                            s4a, s4b, s4c = st.columns([1, 1, 2])
                            with s4a:
                                t4_view = st.radio("视角", ['多头排行', '空头排行', '最终持仓', '黑名单/异常', '全部'], horizontal=False, key='t4_view')
                            with s4b:
                                t4_topn = st.number_input("显示Top N", 5, 100, 30, key='t4_topn')
                            with s4c:
                                t4_asset_kw = st.text_input("品种搜索", value="", key='t4_asset_kw')

                            show_debug = df_debug.copy()
                            if t4_view == '多头排行':
                                show_debug = show_debug.sort_values('多头综合分', ascending=False).head(int(t4_topn))
                            elif t4_view == '空头排行':
                                show_debug = show_debug.sort_values('空头综合分', ascending=False).head(int(t4_topn))
                            elif t4_view == '最终持仓':
                                show_debug = show_debug[show_debug['状态'].isin(['✅ 多头持仓', '📉 空头持仓'])].sort_values('状态')
                            elif t4_view == '黑名单/异常':
                                show_debug = show_debug[show_debug['状态'].str.contains('黑名单', na=False)]
                            if t4_asset_kw.strip():
                                show_debug = show_debug[show_debug.index.astype(str).str.contains(t4_asset_kw.strip(), case=False, regex=False)]

                            def highlight_status(val):
                                if '多头' in str(val): return 'background-color: #e2f0d9; color: black; font-weight: bold'
                                if '空头' in str(val): return 'background-color: #fce4d6; color: black; font-weight: bold'
                                if '黑名单' in str(val): return 'background-color: #ffcccb; color: black'
                                return ''

                            def color_bool(val):
                                color = '#90ee90' if val else '#ffcccb'
                                return f'background-color: {color}; color: black'

                            st.dataframe(
                                df_debug.style
                                .map(highlight_status, subset=['状态'])
                                .map(color_bool, subset=['趋势滤网', '放宽动量网'])
                                .format({
                                    '价格': '{:.2f}',
                                    '次日止损': '{:.2f}',
                                    '多头综合分': '{:.4f}',
                                    '空头综合分': '{:.4f}',
                                    '动量分': '{:.4f}',
                                    '空头弱势分': '{:.4f}',
                                    '流动性分': '{:.4f}',
                                    '收红周期数': '{:.0f}',
                                })
                                .bar(subset=['多头综合分'], color='#2ca02c', vmin=0, vmax=1)
                                .bar(subset=['空头综合分'], color='#d62728', vmin=0, vmax=1),
                                use_container_width=True,
                                height=800
                            )
                        except Exception as e:
                            st.error(f"无法生成当日透视表 (可能数据缺失): {str(e)}")

                with t5:
                    st.markdown("### 🧮 独立品种数学解构")
                    st.caption("选择日期和特定品种，如同用显微镜查看系统底层是如何用数学公式一步步算出最终信号和权重的。")
                    c_t5_1, c_t5_2 = st.columns(2)
                    with c_t5_1:
                        target_t5_date_input = st.date_input("选择回测日期", value=default_date, min_value=valid_dates[0], max_value=valid_dates[-1], key="t5_date")
                    target_t5_date = pd.to_datetime(target_t5_date_input)

                    if target_t5_date in valid_dates:
                        available_assets = df_p.loc[target_t5_date].dropna().index.tolist()
                        with c_t5_2:
                            target_asset = st.selectbox("选择要拆解的品种", options=available_assets, key="t5_asset")

                        if target_asset:
                            try:
                                p_today = df_p.loc[target_t5_date, target_asset]
                                st.markdown(f"#### 🔎 【{target_asset}】 @ {target_t5_date.date()} 因子链路白盒拆解")
                                st.divider()

                                short_ma_win = 60
                                price_hist = df_p[target_asset].loc[:target_t5_date]

                                past_prices = price_hist.tail(int(ma_win))
                                ma_val = past_prices.mean()
                                is_ma_pass = p_today > ma_val
                                ma_series = price_hist.rolling(int(ma_win), min_periods=1).mean()
                                ma_prev = ma_series.iloc[-2] if len(ma_series) >= 2 else ma_val
                                is_ma_rising = ma_val > ma_prev

                                er_prices = price_hist.tail(int(er_win) + 1)
                                net_change = abs(er_prices.iloc[-1] - er_prices.iloc[0]) if len(er_prices) > 1 else 0
                                path_length = er_prices.diff().abs().sum() if len(er_prices) > 1 else 1
                                er_val = net_change / path_length if path_length != 0 else 0
                                is_er_pass = er_val >= er_thresh
                                is_long_trend_pass = is_ma_pass and is_er_pass and is_ma_rising

                                past_prices_short = price_hist.tail(short_ma_win)
                                ma_val_short = past_prices_short.mean()
                                is_short_ma_pass = p_today < ma_val_short

                                st.markdown("##### 1. 核心趋势滤网（多空非对称）")
                                st.markdown("**1A. 做多侧 — MA & ER Steady Filter**")
                                st.latex(r"MA_{" + str(int(ma_win)) + r"} = \frac{1}{" + str(int(ma_win)) + r"}\sum_{i=0}^{" + str(int(ma_win) - 1) + r"} P_{t-i}")
                                st.write(f"- **均线上方**：$P = {p_today:.2f}$，$MA_{{{int(ma_win)}}} = {ma_val:.2f}$ ➔ **{'✅ 达标 (P > MA)' if is_ma_pass else '❌ 破位 (多头不可买入)'}**")
                                st.write(f"- **ER 效率判定**：净位移 `{net_change:.2f}` / 路径总长 `{path_length:.2f}` = **`{er_val:.3f}`** ➔ **{'✅ 达标' if is_er_pass else '❌ 震荡过滤'}**")
                                st.write(f"- **均线上行**：$MA_t = {ma_val:.2f}$，$MA_{{t-1}} = {ma_prev:.2f}$ ➔ **{'✅ 达标' if is_ma_rising else '❌ 均线走平/下行'}**")
                                st.write(f"- **做多趋势滤网汇总**：{'✅ 全部通过' if is_long_trend_pass else '❌ 未通过（做多剔除）'}")

                                st.markdown("**1B. 做空侧 — MA60 压制滤网**")
                                st.latex(r"Short_{Pass} \Leftrightarrow P < MA_{60}")
                                st.write(f"- **均线下方**：$P = {p_today:.2f}$，$MA_{{{short_ma_win}}} = {ma_val_short:.2f}$ ➔ **{'✅ 达标 (P < MA，空头可候选)' if is_short_ma_pass else '❌ 仍在均线上方 (空头不可入选)'}**")
                                st.caption("空头侧不要求 ER 效率与均线上行，仅需价格跌破 60 日均线。")

                                st.markdown("##### 2. 动量容忍过滤（多空非对称）")
                                sign_count = debug_data['mom_debug_info']['mom_sign_matrix'].loc[target_t5_date, target_asset]
                                req_count = len(periods) - mom_tolerance_ui
                                is_mom_pass = sign_count >= req_count
                                green_count = len(periods) - sign_count

                                st.markdown("**2A. 做多侧 — 收红周期容忍**")
                                st.latex(r"Momentum_{Pass}^{long} = \sum_{i=1}^{N} I(ROC_{n\_i} > 0) \ge N - \text{Tolerance}")
                                st.write(f"- 动量周期数 $N={len(periods)}$，容忍缺口 $\\text{{Tolerance}}={mom_tolerance_ui}$，至少需 **`{req_count}`** 个周期收红。")
                                st.write(f"- 实际收红周期数 **`{sign_count:.0f}`** ➔ **{'✅ 通过' if is_mom_pass else '❌ 未通过（做多剔除）'}**")

                                st.markdown("**2B. 做空侧 — 无收红滤网（动量弱势由截面得分体现）**")
                                st.latex(r"Momentum_{Pass}^{short} \equiv \text{True} \quad (\text{不施加收红容忍约束})")
                                st.write(f"- 实际收绿周期数 **`{green_count:.0f}`** / {len(periods)}（供参考，收绿越多动量越弱，有利于空头得分）")
                                st.write("- **做空动量滤网**：✅ 豁免（空头入选不依赖收红周期，弱动量通过 `Score_short` 截面排名体现）")
                                is_long_filter_pass = is_long_trend_pass and is_mom_pass

                                st.markdown("##### 3. 多空独立截面排名得分合成 (Cross-Sectional Scoring)")
                                mom_s = debug_data['momentum_score'].loc[target_t5_date, target_asset]
                                liq_s = debug_data['liquidity_score'].loc[target_t5_date, target_asset]
                                weak_s = 1 - mom_s if not pd.isna(mom_s) else 0.0
                                long_final_score = debug_data['long_score'].loc[target_t5_date, target_asset]
                                short_final_score = debug_data['short_score'].loc[target_t5_date, target_asset]
                                is_long_score_pass = (not pd.isna(long_final_score)) and long_final_score > 0.5
                                is_short_score_pass = (not pd.isna(short_final_score)) and short_final_score > 0.5
                                st.latex(r"Score_{long} = (Mom_{rank} \times 0.7) + (Liq_{rank} \times 0.3)")
                                st.latex(r"Score_{short} = ((1 - Mom_{rank}) \times 0.7) + (Liq_{rank} \times 0.3)")
                                st.write(f"- **代入数据**：动量全市场击败了 `{mom_s*100:.1f}%` 的品种；空头弱势分为 `{weak_s*100:.1f}%`；流动性击败了 `{liq_s*100:.1f}%` 的品种。")
                                st.write(f"- **多头得分**：`({mom_s:.4f} * 0.7) + ({liq_s:.4f} * 0.3) = {long_final_score:.4f}` ➔ **{'✅ > 0.5 候选' if is_long_score_pass else '❌ ≤ 0.5 剔除'}**")
                                st.write(f"- **空头得分**：`({weak_s:.4f} * 0.7) + ({liq_s:.4f} * 0.3) = {short_final_score:.4f}` ➔ **{'✅ > 0.5 候选' if is_short_score_pass else '❌ ≤ 0.5 剔除'}**")

                                st.markdown("##### 4. 组合分配与最终仓位 (Portfolio Allocation)")
                                day_detail_t5 = next((d for d in res_cycle_details if d['date'] == target_t5_date), None)
                                h_long = day_detail_t5.get('next_day_hold_long', {}) if day_detail_t5 else {}
                                h_short = day_detail_t5.get('next_day_hold_short', {}) if day_detail_t5 else {}
                                banned_t5 = set(day_detail_t5.get('banned_list', [])) if day_detail_t5 else set()
                                is_short_filter_pass = is_short_ma_pass and is_short_score_pass

                                col_long, col_short = st.columns(2)
                                with col_long:
                                    st.markdown("**做多链路复盘**")
                                    st.write(f"- 趋势滤网：{'✅' if is_long_trend_pass else '❌'}")
                                    st.write(f"- 动量滤网：{'✅' if is_mom_pass else '❌'}")
                                    st.write(f"- 综合分 > 0.5：{'✅' if is_long_score_pass else '❌'}")
                                    st.write(f"- **汇总**：{'✅ 可入选做多池' if is_long_filter_pass and is_long_score_pass else '❌ 未入选做多池'}")
                                with col_short:
                                    st.markdown("**做空链路复盘**")
                                    st.write(f"- MA60 压制：{'✅' if is_short_ma_pass else '❌'}")
                                    st.write(f"- 动量滤网：✅ 豁免")
                                    st.write(f"- 综合分 > 0.5：{'✅' if is_short_score_pass else '❌'}")
                                    st.write(f"- **汇总**：{'✅ 可入选做空池' if is_short_filter_pass else '❌ 未入选做空池'}")

                                alloc = compute_day_allocation_breakdown(
                                    target_t5_date, df_p, df_atr_norm, df_vol, debug_data,
                                    hold_num, max_sector, net_exp_target, max_gross_exposure_ui, max_margin_usage_ui,
                                    max_single_weight=max_single_weight_ui, use_vol_scaling=True, banned_assets=banned_t5,
                                )
                                in_long_pool = target_asset in alloc['ideal_long']
                                in_short_pool = target_asset in alloc['ideal_short']
                                side = 'long' if target_asset in h_long else ('short' if target_asset in h_short else None)

                                st.markdown("##### 5. 仓位分配数学推导 (Position Sizing)")
                                st.markdown("**5A. 截面候选池入选（Top N + 板块上限）**")
                                short_rank = alloc['short_rank_map'].get(target_asset)
                                long_rank = alloc['long_rank_map'].get(target_asset)
                                st.write(
                                    f"- 有效品种池 **`{alloc['valid_pool_size']}`** 个；"
                                    f"单向目标持仓 **`{max(hold_num, 3)}`**；板块上限 **`{max_sector}`** / 板块"
                                )
                                st.write(f"- **{target_asset}** 空头截面排名 **`#{short_rank}`** / {alloc['valid_pool_size']}" if short_rank else f"- **{target_asset}** 不在空头有效排名池")
                                st.write(f"- **{target_asset}** 多头截面排名 **`#{long_rank}`** / {alloc['valid_pool_size']}" if long_rank else f"- **{target_asset}** 不在多头有效排名池")
                                if in_short_pool:
                                    st.write(f"- ✅ 入选当日 **做空候选池**：`{', '.join(alloc['ideal_short'])}`")
                                elif is_short_filter_pass:
                                    st.write(f"- ❌ 未入选做空候选池（排名未进 Top 或同板块已满 `{max_sector}` 席）")
                                if in_long_pool:
                                    st.write(f"- ✅ 入选当日 **做多候选池**：`{', '.join(alloc['ideal_long'])}`")

                                st.markdown("**5B. 风险平价初配（逆 ATR 加权 × 波动率缩放）**")
                                st.latex(r"w_i^{(0)} = \frac{1 / ATR_i}{\sum_j 1 / ATR_j} \times \lambda_{vol}, \quad w_i \le " + f"{max_single_weight_ui:.2f}" + r" \text{ 后重归一化}")
                                st.write(f"- 当日波动率缩放因子 $\\lambda_{{vol}} = {alloc['pos_multiplier']:.4f}$")

                                def _render_risk_parity(side_key, pool, detail, scores, sum_inv, clip_triggered):
                                    if not pool:
                                        st.write(f"- **{side_key}侧**：候选池为空，跳过配仓")
                                        return
                                    inv_terms = " + ".join(f"1/{detail[a]['atr_norm']:.6f}" for a in pool)
                                    st.write(f"- **{side_key}侧** $\\sum_j 1/ATR_j$ = {inv_terms} = **`{sum_inv:.4f}`**")
                                    rows = []
                                    for a in pool:
                                        d = detail[a]
                                        rows.append({
                                            '品种': a,
                                            'ATR归一化': d['atr_norm'],
                                            '1/ATR': d['inv_vol'],
                                            '池内占比': d['pool_share'],
                                            '初配权重': d['raw_weight'],
                                            '单品种上限后': d['clipped_weight'],
                                            '综合分': float(scores.get(a, np.nan)),
                                        })
                                    st.write(f"- **{side_key}侧** 候选池风险平价明细{'（⚠️ 有品种触发单品种上限并重归一化）' if clip_triggered else '（均未触发单品种上限）'}：")
                                    st.dataframe(
                                        pd.DataFrame(rows).style.format({
                                            'ATR归一化': '{:.6f}',
                                            '1/ATR': '{:.4f}',
                                            '池内占比': '{:.2%}',
                                            '初配权重': '{:.2%}',
                                            '单品种上限后': '{:.2%}',
                                            '综合分': '{:.4f}',
                                        }),
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                                def _format_asset_weight_steps(asset_name, td, sum_inv, pos_mult, clip_triggered, side_label, cap=max_single_weight_ui):
                                    share = td['pool_share']
                                    cap_pct = f"{cap:.0%}"
                                    lines = [
                                        f"**{asset_name} {side_label}初配逐步代入**：",
                                        f"1. $1/ATR = 1/{td['atr_norm']:.6f} = {td['inv_vol']:.4f}$",
                                        f"2. 池内占比 $= {td['inv_vol']:.4f} / {sum_inv:.4f} = {share:.4f}$（{share:.2%}）",
                                        f"3. 初配权重 $= {share:.4f} \\times \\lambda_{{vol}}({pos_mult:.4f}) = {td['raw_weight']:.4f}$（**{td['raw_weight']:.2%}**）",
                                    ]
                                    if clip_triggered and abs(td['clipped_weight'] - td['raw_weight']) > 1e-6:
                                        lines.append(f"4. 单品种 {cap_pct} 上限后重归一化 → **{td['clipped_weight']:.2%}**")
                                    else:
                                        lines.append(f"4. 未触发单品种 {cap_pct} 上限，权重不变")
                                    return "\n\n".join(lines)

                                _render_risk_parity('做空', alloc['ideal_short'], alloc['short_detail'], alloc['short_scores'], alloc['short_sum_inv'], alloc['short_clip_triggered'])
                                _render_risk_parity('做多', alloc['ideal_long'], alloc['long_detail'], alloc['long_scores'], alloc['long_sum_inv'], alloc['long_clip_triggered'])

                                if side == 'short' and target_asset in alloc['short_detail']:
                                    td = alloc['short_detail'][target_asset]
                                    st.info(_format_asset_weight_steps(
                                        target_asset, td, alloc['short_sum_inv'], alloc['pos_multiplier'],
                                        alloc['short_clip_triggered'], '做空',
                                    ))
                                elif side == 'long' and target_asset in alloc['long_detail']:
                                    td = alloc['long_detail'][target_asset]
                                    st.info(_format_asset_weight_steps(
                                        target_asset, td, alloc['long_sum_inv'], alloc['pos_multiplier'],
                                        alloc['long_clip_triggered'], '做多',
                                    ))

                                st.markdown("**5C. 组合级风控缩放（净敞口 → 杠杆 → 保证金）**")
                                st.latex(r"Net = \sum w_{long} - \sum w_{short}, \quad Leverage = \sum w_{long} + \sum w_{short}")
                                st.write(
                                    f"- **风险平价初配汇总**：多头 `{alloc['tot_long_raw']:.2%}` + 空头 `{alloc['tot_short_raw']:.2%}`"
                                    f" → 净敞口 **`{alloc['net_exp_raw']:+.2%}`**，杠杆 **`{alloc['gross_raw']:.2%}`**"
                                )
                                if alloc['net_scale_side']:
                                    st.write(
                                        f"- **净敞口截断**：`|Net| = {abs(alloc['net_exp_raw']):.2%}` > 上限 `{net_exp_target:.2%}`，"
                                        f"对 **{alloc['net_scale_side']}** 侧乘以 `{alloc['net_scale_factor']:.4f}`"
                                    )
                                else:
                                    st.write(f"- **净敞口截断**：`|Net| = {abs(alloc['net_exp_raw']):.2%}` ≤ 上限 `{net_exp_target:.2%}`，无需缩放")
                                st.write(f"- 净敞口缩放后杠杆：**`{alloc['gross_pre']:.2%}`**")
                                if alloc['gross_scale_factor'] < 1.0:
                                    st.write(
                                        f"- **杠杆截断**：`Leverage = {alloc['gross_pre']:.2%}` > 上限 `{max_gross_exposure_ui:.2%}`，"
                                        f"多空同时 × `{alloc['gross_scale_factor']:.4f}`"
                                    )
                                else:
                                    st.write(f"- **杠杆截断**：`Leverage = {alloc['gross_pre']:.2%}` ≤ 上限 `{max_gross_exposure_ui:.2%}`，无需缩放")
                                if alloc['margin_scale_factor'] < 1.0:
                                    st.write(
                                        f"- **保证金截断**：估算占用 `{alloc['margin_pre']:.2%}` > 上限 `{max_margin_usage_ui:.2%}`，"
                                        f"多空同时 × `{alloc['margin_scale_factor']:.4f}`"
                                    )
                                else:
                                    st.write(f"- **保证金截断**：估算占用 `{alloc['margin_pre']:.2%}` ≤ 上限 `{max_margin_usage_ui:.2%}`，无需缩放")

                                st.markdown("**5D. 最终隔夜仓位**")
                                if side == 'short':
                                    w_raw = alloc['raw_w_short'].get(target_asset, 0.0)
                                    w_net = alloc['w_short_after_net'].get(target_asset, 0.0)
                                    w_gross = alloc['w_short_after_gross'].get(target_asset, 0.0)
                                    w_final = alloc['w_short_final'].get(target_asset, h_short.get(target_asset, 0.0))
                                    cum_scale = (w_final / w_raw) if w_raw > 1e-8 else 1.0
                                    st.write(
                                        f"- **{target_asset} 做空权重链路**："
                                        f"初配 **`{w_raw:.2%}`**"
                                        + (f" → 净敞口后 `{w_net:.2%}`" if abs(w_net - w_raw) > 1e-6 else "")
                                        + (f" → 杠杆缩放后 `{w_gross:.2%}`" if abs(w_gross - w_net) > 1e-6 else "")
                                        + (f" → 保证金后 `{w_final:.2%}`" if abs(w_final - w_gross) > 1e-6 else "")
                                        + f" → **最终 `{w_final:.2%}`**"
                                    )
                                    if abs(cum_scale - 1.0) > 1e-4:
                                        st.caption(
                                            f"组合级风控对该品种累计缩放 × `{cum_scale:.4f}`"
                                            f"（初配 {w_raw:.2%} × {cum_scale:.4f} ≈ 最终 {w_final:.2%}）"
                                        )
                                    elif w_raw > 0:
                                        st.caption("初配后未触发组合级净敞口/杠杆/保证金缩放，最终权重等于初配。")
                                elif side == 'long':
                                    w_raw = alloc['raw_w_long'].get(target_asset, 0.0)
                                    w_net = alloc['w_long_after_net'].get(target_asset, 0.0)
                                    w_gross = alloc['w_long_after_gross'].get(target_asset, 0.0)
                                    w_final = alloc['w_long_final'].get(target_asset, h_long.get(target_asset, 0.0))
                                    cum_scale = (w_final / w_raw) if w_raw > 1e-8 else 1.0
                                    st.write(
                                        f"- **{target_asset} 做多权重链路**："
                                        f"初配 **`{w_raw:.2%}`**"
                                        + (f" → 净敞口后 `{w_net:.2%}`" if abs(w_net - w_raw) > 1e-6 else "")
                                        + (f" → 杠杆缩放后 `{w_gross:.2%}`" if abs(w_gross - w_net) > 1e-6 else "")
                                        + (f" → 保证金后 `{w_final:.2%}`" if abs(w_final - w_gross) > 1e-6 else "")
                                        + f" → **最终 `{w_final:.2%}`**"
                                    )
                                    if abs(cum_scale - 1.0) > 1e-4:
                                        st.caption(
                                            f"组合级风控对该品种累计缩放 × `{cum_scale:.4f}`"
                                            f"（初配 {w_raw:.2%} × {cum_scale:.4f} ≈ 最终 {w_final:.2%}）"
                                        )
                                    elif w_raw > 0:
                                        st.caption("初配后未触发组合级净敞口/杠杆/保证金缩放，最终权重等于初配。")

                                if target_asset in h_long:
                                    st.success(f"🎉 **{target_asset}** 成功闯过所有多头滤网，入选做多持仓，配额：**`{h_long[target_asset]:.2%}`**")
                                elif target_asset in h_short:
                                    st.warning(
                                        f"📉 **{target_asset}** 空头链路通过：跌破 MA60 + 空头综合分 `{short_final_score:.4f}` 截面靠前"
                                        f"（动量弱 `{weak_s*100:.1f}%` + 流动性 `{liq_s*100:.1f}%`），入选做空持仓，配额：**`{h_short[target_asset]:.2%}`**"
                                    )
                                else:
                                    if is_short_filter_pass and not in_short_pool:
                                        st.info("🔍 品种通过空头基础滤网且综合分达标，但截面排名未进入 Top 池或受板块上限约束，今日保持空仓。")
                                    elif is_long_filter_pass and is_long_score_pass and not in_long_pool:
                                        st.info("🔍 品种通过多头基础滤网，但截面排名未进入 Top 池或受板块上限约束，今日保持空仓。")
                                    elif is_short_filter_pass or (is_long_filter_pass and is_long_score_pass):
                                        st.info("🔍 品种进入候选池逻辑但未获最终仓位，可能受组合敞口/保证金缩放后权重归零或调仓阈值影响。")
                                    else:
                                        st.info("🔍 品种处于中性观察区间，多空两侧均未满足入选条件，今日保持空仓。")
                            except Exception as e:
                                st.error(f"公式细节解析异常: {str(e)}")

                with t6:
                    st.markdown("### 🏅 时序动量截面排序")
                    st.caption("展示【先排名，再时间加权】的计算链路。既破成了长线‘幅度霸权’，又赋予了近期爆发力更高的权重。")
                    target_t6_date = pd.to_datetime(st.date_input("📅 选择回测日期", value=default_date, min_value=valid_dates[0], max_value=valid_dates[-1], key="t6_date_pick"))
                    if target_t6_date in valid_dates:
                        mom_info = debug_data.get('mom_debug_info')
                        if mom_info and 'roc' in mom_info:
                            available_assets_t6 = df_p.loc[target_t6_date].dropna().index.tolist()
                            target_asset_t6 = st.selectbox("📦 选择要拆解的品种", options=available_assets_t6, key="t6_asset")
                            if target_asset_t6:
                                records = []
                                w_dict = mom_info['weights']
                                for p in periods:
                                    r_val = mom_info['roc'][p].loc[target_t6_date, target_asset_t6]
                                    rank_val = mom_info['rank'][p].loc[target_t6_date, target_asset_t6]
                                    records.append({
                                        "观察周期": f"{p} 日",
                                        "平滑绝对涨幅": f"{r_val * 100:.2f}%" if not pd.isna(r_val) else "N/A",
                                        "是否收红(ROC>0)": "🔴 满足" if (not pd.isna(r_val) and r_val > 0) else "🍏 微跌",
                                        "截面名次(0~1)": rank_val if not pd.isna(rank_val) else 0.0,
                                        "时间权重分配": f"{w_dict.get(p, 0)*100:.0f}%",
                                        "折算后贡献分": (rank_val * w_dict.get(p, 0)) if not pd.isna(rank_val) else 0.0
                                    })
                                st.dataframe(pd.DataFrame(records).style.format({"截面名次(0~1)": "{:.4f}", "折算后贡献分": "{:.4f}"}).bar(subset=['折算后贡献分'], color='#5fba7d'), use_container_width=True)
                        else: st.warning("未检测到动量调试数据。")

                with t7:
                    st.markdown("### ⏱️ 持仓周期精细画像")
                    st.caption("基于回测每日实盘留存仓位，穿透计算各个品种的【单次连续持仓天数】及【历史平均生命周期】。")
                    if not res_cycle_details:
                        st.warning("暂无持仓细节数据。")
                    elif holding_wave_stats.empty:
                        st.info("回测期间未发生任何实际持仓。")
                    else:
                        wave_stats = holding_wave_stats
                        df_bh = holding_df_bh

                        col_ha1, col_ha2 = st.columns(2)
                        col_ha1.metric("⚖️ 全市场平均持仓期", f"{wave_stats['Hold_Days'].mean():.1f} 个交易日")
                        col_ha2.metric("🔄 发生的总调仓次数", f"{len(wave_stats)} 次")
                        st.divider()

                        asset_profile = wave_stats.groupby('asset_label').agg(
                            Total_Days=('Hold_Days', 'sum'), Avg_Days=('Hold_Days', 'mean'), Max_Days=('Hold_Days', 'max'), Total_Waves=('Hold_Days', 'count'), Net_Return=('Wave_Return', lambda s: np.prod(1+s)-1)
                        ).reset_index()
                        asset_profile.columns = ['多空品种标签', '总持有天数(日)', '平均持仓天数(日)', '单次最长持仓(日)', '回测期间持有次数', '持仓期总收益']
                        asset_profile.sort_values(by='总持有天数(日)', ascending=False, inplace=True)

                        def color_pnl_col(s): return [f"color: {'#d32f2f' if v > 0 else ('#388e3c' if v < 0 else 'black')}; font-weight: bold;" for v in s]

                        sub_col1, sub_col2 = st.columns([1, 1])
                        with sub_col1:
                            st.markdown("#### 📊 各品种持仓时限画像")
                            st.dataframe(asset_profile.style.format({'平均持仓天数(日)': '{:2.1f}', '持仓期总收益': '{:+.2%}'}).background_gradient(cmap='YlOrRd', subset=['总持有天数(日)']).apply(color_pnl_col, subset=['持仓期总收益']), use_container_width=True, height=350)
                        with sub_col2:
                            st.markdown("#### 🔎 连续持仓波段穿透明细流水")
                            wave_print = wave_stats.copy()
                            wave_print['Start_Date_Show'] = wave_print['Start_Date'].dt.date
                            wave_print['End_Date_Show'] = wave_print['End_Date'].dt.date
                            wave_print.rename(columns={'asset_label': '多空标识', 'Hold_Days': '连续持有天数', 'Wave_Return': '区间损益'}, inplace=True)
                            st.dataframe(wave_print[['多空标识', 'Start_Date_Show', 'End_Date_Show', '连续持有天数', '区间损益']].style.format({'区间损益': '{:+.2%}'}).apply(color_pnl_col, subset=['区间损益']).bar(subset=['连续持有天数'], color='#a6c8e0'), use_container_width=True, height=350)

                        st.divider()
                        st.markdown("#### 🔬 单特定品种历史全波段时序放大镜")
                        st.caption("选一个品种即可按时间线查看其所有做多/做空波段，无需分开选择。")
                        held_assets = sorted(df_bh['asset'].unique().tolist())

                        def _compound_wave_returns(waves_df):
                            if waves_df.empty:
                                return 0.0
                            return float(np.prod(1 + waves_df['Wave_Return'].astype(float)) - 1)

                        mag_col_sel, mag_col_tot, mag_col_long, mag_col_short = st.columns([2.4, 1, 1, 1])
                        with mag_col_sel:
                            selected_asset = st.selectbox(
                                "🎯 选择品种以全面复盘其过往所有持仓波段：",
                                options=held_assets,
                                key="t7_asset_magnifier",
                            )

                        if selected_asset:
                            sub_waves = wave_stats[wave_stats['asset'] == selected_asset].copy().sort_values(by='Start_Date', ascending=True)
                            sub_waves = sub_waves.reset_index(drop=True)
                            sub_waves['seq'] = sub_waves.index + 1
                            n_long = int((sub_waves['direction'] == '做多').sum())
                            n_short = int((sub_waves['direction'] == '做空').sum())

                            asset_total_ret = _compound_wave_returns(sub_waves)
                            asset_long_ret = _compound_wave_returns(sub_waves[sub_waves['direction'] == '做多'])
                            asset_short_ret = _compound_wave_returns(sub_waves[sub_waves['direction'] == '做空'])

                            mag_col_tot.metric(f"{selected_asset} 总收益", f"{asset_total_ret:+.2%}")
                            mag_col_long.metric("做多收益", f"{asset_long_ret:+.2%}")
                            mag_col_short.metric("做空收益", f"{asset_short_ret:+.2%}")

                            st.write(f"共 **{len(sub_waves)}** 个波段（做多 {n_long} / 做空 {n_short}）")

                            for _, wave_item in sub_waves.iterrows():
                                e_idx = int(wave_item['End_Idx'])
                                w_id = int(wave_item['seq'])
                                w_pnl = wave_item['Wave_Return']
                                direction = wave_item['direction']
                                dir_color = "#d62728" if direction == '做多' else "#2ca02c"
                                expected_stop_dir = '多头' if direction == '做多' else '空头'

                                reason_str = "正常截面动量轮动淘汰（综合得分跌出池子或被更优品种替换）"
                                b_color = "#1f77b4"

                                if e_idx + 1 < len(res_cycle_details):
                                    exit_day = res_cycle_details[e_idx + 1]
                                    my_stop = next(
                                        (s for s in exit_day.get('stops', [])
                                         if s['asset'] == selected_asset and s.get('dir') == expected_stop_dir),
                                        None,
                                    )
                                    if my_stop:
                                        reason_str = f"🚨 盘中触发止损：{my_stop['reason']}"
                                        b_color = "#d62728"

                                pnl_display = f"🌟 区间累计收益: {'+' if w_pnl >= 0 else ''}{w_pnl*100:.2f}%"
                                wave_pnl_color = "#d32f2f" if w_pnl >= 0 else "#388e3c"

                                with st.container():
                                    st.markdown(
                                        f"#### 🌊 波段 {w_id} · "
                                        f"<span style='color:{dir_color}; font-weight:bold;'>{direction}</span>",
                                        unsafe_allow_html=True,
                                    )
                                    cl1, cl2, cl3 = st.columns(3)
                                    cl1.write(f"**🟢 进场建仓日**：`{wave_item['Start_Date'].strftime('%Y-%m-%d')}`")
                                    cl2.write(f"**🔴 实际清仓日**：`{wave_item['End_Date'].strftime('%Y-%m-%d')}`")
                                    cl3.write(f"**⏱️ 连续持有天数**：`{wave_item['Hold_Days']} 个交易日`")

                                    d_col1, d_col2 = st.columns([1, 2])
                                    with d_col1:
                                        st.markdown(
                                            f"<div style='padding:12px; background-color:#f1f3f6; border-left:5px solid {b_color}; border-radius:4px; height: 100%;'>"
                                            f"<b>📈 盈亏综合表现 ({direction}):</b><br/>"
                                            f"<span style='color:{wave_pnl_color}; font-size:15px; font-weight:bold;'>{pnl_display}</span>"
                                            f"</div>",
                                            unsafe_allow_html=True
                                        )
                                    with d_col2:
                                        w_style = "color:#856404;" if b_color != "#d62728" else "color:#d62728;"
                                        b_inner_style = "background-color:#fff3cd;" if b_color != "#d62728" else "background-color:#f8d7da;"
                                        exit_label = "平多离场" if direction == '做多' else "平空离场"
                                        st.markdown(
                                            f"<div style='padding:12px; {b_inner_style} border-left:5px solid {b_color}; border-radius:4px; height: 100%;'>"
                                            f"<b>❓ {exit_label}诱因诊断:</b><br/>"
                                            f"<span style='{w_style} font-size:14px; font-weight:bold;'>{reason_str}</span>"
                                            f"</div>",
                                            unsafe_allow_html=True
                                        )
                                    st.write("")

                with t8:
                    st.markdown("### 🧭 多空收益拆解驾驶舱")
                    st.caption("把每日持仓按做多/做空方向拆成独立收益流，快速定位是哪一边、哪个品种、哪一段时间贡献了收益或回撤。")

                    pnl_records = []
                    all_backtest_dates_t8 = sorted(list(res_nav.index))
                    date_to_idx_t8 = {d: idx for idx, d in enumerate(all_backtest_dates_t8)}
                    for day in res_cycle_details:
                        d_date = day['date']
                        d_idx = date_to_idx_t8.get(d_date, np.nan)
                        for asset, weight in day.get('start_hold_long', {}).items():
                            if weight > 0:
                                asset_ret = day.get('asset_rets', {}).get(asset, 0.0)
                                pnl_records.append({
                                    'date': d_date, 'date_idx': d_idx, 'asset': asset, '方向': '做多',
                                    '仓位': weight, '品种涨跌': asset_ret, '收益贡献': weight * asset_ret,
                                    '当日组合收益': day.get('ret', 0.0)
                                })
                        for asset, weight in day.get('start_hold_short', {}).items():
                            if weight > 0:
                                asset_ret = day.get('asset_rets', {}).get(asset, 0.0)
                                short_ret = -asset_ret
                                pnl_records.append({
                                    'date': d_date, 'date_idx': d_idx, 'asset': asset, '方向': '做空',
                                    '仓位': weight, '品种涨跌': asset_ret, '收益贡献': weight * short_ret,
                                    '当日组合收益': day.get('ret', 0.0)
                                })

                    if not pnl_records:
                        st.info("暂无多空持仓收益记录。")
                    else:
                        pnl_df = pd.DataFrame(pnl_records)
                        daily_side = pnl_df.pivot_table(index='date', columns='方向', values='收益贡献', aggfunc='sum').fillna(0.0)
                        for col in ['做多', '做空']:
                            if col not in daily_side.columns:
                                daily_side[col] = 0.0
                        daily_side = daily_side[['做多', '做空']]
                        daily_side['多空合计'] = daily_side['做多'] + daily_side['做空']
                        daily_side['多头净值'] = (1 + daily_side['做多']).cumprod()
                        daily_side['空头净值'] = (1 + daily_side['做空']).cumprod()
                        daily_side['多空拆解净值'] = (1 + daily_side['多空合计']).cumprod()

                        long_total = daily_side['多头净值'].iloc[-1] - 1
                        short_total = daily_side['空头净值'].iloc[-1] - 1
                        combo_total = daily_side['多空拆解净值'].iloc[-1] - 1
                        long_win = (daily_side['做多'] > 0).mean()
                        short_win = (daily_side['做空'] > 0).mean()
                        best_side = '做多' if long_total >= short_total else '做空'

                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("做多累计贡献", f"{long_total:+.2%}")
                        k2.metric("做空累计贡献", f"{short_total:+.2%}")
                        k3.metric("拆解合计贡献", f"{combo_total:+.2%}")
                        k4.metric("主贡献方向", best_side)
                        k5, k6, k7, k8 = st.columns(4)
                        k5.metric("做多日胜率", f"{long_win:.1%}")
                        k6.metric("做空日胜率", f"{short_win:.1%}")
                        k7.metric("做多最大单日贡献", f"{daily_side['做多'].max():+.2%}")
                        k8.metric("做空最大单日贡献", f"{daily_side['做空'].max():+.2%}")

                        fig_t8, ax_t8 = plt.subplots(figsize=(12, 4.8))
                        ax_t8.plot(daily_side.index, daily_side['多头净值'], color='#d62728', lw=2, label='做多收益净值')
                        ax_t8.plot(daily_side.index, daily_side['空头净值'], color='#2ca02c', lw=2, label='做空收益净值')
                        ax_t8.plot(daily_side.index, daily_side['多空拆解净值'], color='#1f77b4', lw=2.2, label='多空合计拆解净值')
                        ax_t8.axhline(y=1, color='gray', linestyle='--', alpha=0.45)
                        ax_t8.set_title("做多 vs 做空 独立收益曲线", fontproperties=my_font, fontsize=14)
                        ax_t8.grid(True, alpha=0.25)
                        ax_t8.legend(prop=my_font)
                        fig_t8.tight_layout()
                        st.pyplot(fig_t8)

                        st.divider()
                        left_rank, right_rank = st.columns(2)
                        asset_side_summary = pnl_df.groupby(['方向', 'asset']).agg(
                            总收益贡献=('收益贡献', 'sum'),
                            平均日贡献=('收益贡献', 'mean'),
                            持仓天数=('date', 'count'),
                            平均仓位=('仓位', 'mean'),
                            胜率=('收益贡献', lambda s: (s > 0).mean()),
                            最大单日赚=('收益贡献', 'max'),
                            最大单日亏=('收益贡献', 'min'),
                        ).reset_index()
                        asset_side_summary['贡献效率/天'] = asset_side_summary['总收益贡献'] / asset_side_summary['持仓天数'].replace(0, np.nan)

                        def pnl_text_color(s):
                            return [f"color: {'#d32f2f' if v > 0 else ('#388e3c' if v < 0 else 'black')}; font-weight: bold;" for v in s]

                        with left_rank:
                            st.markdown("#### 📈 做多品种贡献排行")
                            long_rank = asset_side_summary[asset_side_summary['方向'] == '做多'].sort_values('总收益贡献', ascending=False)
                            st.dataframe(
                                long_rank.style.format({
                                    '总收益贡献': '{:+.2%}', '平均日贡献': '{:+.3%}', '平均仓位': '{:.1%}',
                                    '胜率': '{:.1%}', '最大单日赚': '{:+.2%}', '最大单日亏': '{:+.2%}', '贡献效率/天': '{:+.3%}'
                                }).apply(pnl_text_color, subset=['总收益贡献', '最大单日赚', '最大单日亏']).bar(subset=['总收益贡献'], color='#ffb3b3'),
                                use_container_width=True, height=430
                            )
                        with right_rank:
                            st.markdown("#### 📉 做空品种贡献排行")
                            short_rank = asset_side_summary[asset_side_summary['方向'] == '做空'].sort_values('总收益贡献', ascending=False)
                            st.dataframe(
                                short_rank.style.format({
                                    '总收益贡献': '{:+.2%}', '平均日贡献': '{:+.3%}', '平均仓位': '{:.1%}',
                                    '胜率': '{:.1%}', '最大单日赚': '{:+.2%}', '最大单日亏': '{:+.2%}', '贡献效率/天': '{:+.3%}'
                                }).apply(pnl_text_color, subset=['总收益贡献', '最大单日赚', '最大单日亏']).bar(subset=['总收益贡献'], color='#b7e4b7'),
                                use_container_width=True, height=430
                            )

                        st.divider()
                        st.markdown("#### 🗓️ 按时间查看多空收益")
                        c8a, c8b, c8c = st.columns([1, 1, 2])
                        with c8a:
                            side_filter = st.radio("方向", ['全部', '做多', '做空'], horizontal=True, key='t8_side_filter')
                        with c8b:
                            sort_mode = st.selectbox("排序", ['日期从新到旧', '贡献从高到低', '亏损从大到小'], key='t8_sort_mode')
                        with c8c:
                            asset_options_t8 = ['全部'] + sorted(pnl_df['asset'].unique().tolist())
                            asset_filter_t8 = st.selectbox("品种", asset_options_t8, key='t8_asset_filter')

                        detail_df = pnl_df.copy()
                        if side_filter != '全部':
                            detail_df = detail_df[detail_df['方向'] == side_filter]
                        if asset_filter_t8 != '全部':
                            detail_df = detail_df[detail_df['asset'] == asset_filter_t8]
                        if sort_mode == '日期从新到旧':
                            detail_df = detail_df.sort_values('date', ascending=False)
                        elif sort_mode == '贡献从高到低':
                            detail_df = detail_df.sort_values('收益贡献', ascending=False)
                        else:
                            detail_df = detail_df.sort_values('收益贡献', ascending=True)
                        detail_show = detail_df[['date', 'asset', '方向', '仓位', '品种涨跌', '收益贡献']].copy()
                        detail_show['date'] = detail_show['date'].dt.date
                        st.dataframe(
                            detail_show.style.format({'仓位': '{:.1%}', '品种涨跌': '{:+.2%}', '收益贡献': '{:+.2%}'}).apply(pnl_text_color, subset=['品种涨跌', '收益贡献']),
                            use_container_width=True, height=420
                        )

                        st.divider()
                        st.markdown("#### 🌊 多空持仓波段收益")
                        pnl_df_sorted = pnl_df.sort_values(['方向', 'asset', 'date']).copy()
                        pnl_df_sorted['is_new_wave'] = pnl_df_sorted.groupby(['方向', 'asset'])['date_idx'].diff() != 1
                        pnl_df_sorted['wave_id'] = pnl_df_sorted.groupby(['方向', 'asset'])['is_new_wave'].cumsum()
                        wave_side = pnl_df_sorted.groupby(['方向', 'asset', 'wave_id']).agg(
                            开始日期=('date', 'min'), 结束日期=('date', 'max'),
                            持仓天数=('date', 'count'), 平均仓位=('仓位', 'mean'),
                            区间收益贡献=('收益贡献', 'sum'),
                            区间品种累计涨跌=('品种涨跌', lambda s: np.prod(1 + s) - 1),
                        ).reset_index()
                        wave_side['开始日期'] = wave_side['开始日期'].dt.date
                        wave_side['结束日期'] = wave_side['结束日期'].dt.date
                        wave_side = wave_side.sort_values('区间收益贡献', ascending=False)

                        wcol1, wcol2 = st.columns([1, 1])
                        with wcol1:
                            st.markdown("##### 最赚钱波段 TOP 20")
                            st.dataframe(
                                wave_side.head(20).style.format({
                                    '平均仓位': '{:.1%}', '区间收益贡献': '{:+.2%}', '区间品种累计涨跌': '{:+.2%}'
                                }).apply(pnl_text_color, subset=['区间收益贡献', '区间品种累计涨跌']),
                                use_container_width=True, height=430
                            )
                        with wcol2:
                            st.markdown("##### 最大亏损波段 TOP 20")
                            st.dataframe(
                                wave_side.tail(20).sort_values('区间收益贡献', ascending=True).style.format({
                                    '平均仓位': '{:.1%}', '区间收益贡献': '{:+.2%}', '区间品种累计涨跌': '{:+.2%}'
                                }).apply(pnl_text_color, subset=['区间收益贡献', '区间品种累计涨跌']),
                                use_container_width=True, height=430
                            )

                        st.divider()
                        st.markdown("#### 🔍 单品种多空历史复盘")
                        focus_asset = st.selectbox("选择品种查看它在做多/做空两侧的完整贡献", sorted(pnl_df['asset'].unique().tolist()), key='t8_focus_asset')
                        focus_df = pnl_df[pnl_df['asset'] == focus_asset].copy().sort_values('date')
                        focus_daily = focus_df.pivot_table(index='date', columns='方向', values='收益贡献', aggfunc='sum').fillna(0.0)
                        for col in ['做多', '做空']:
                            if col not in focus_daily.columns:
                                focus_daily[col] = 0.0
                        focus_daily['合计'] = focus_daily['做多'] + focus_daily['做空']
                        focus_daily['累计贡献'] = focus_daily['合计'].cumsum()

                        f1, f2, f3, f4 = st.columns(4)
                        f1.metric(f"{focus_asset} 总贡献", f"{focus_daily['合计'].sum():+.2%}")
                        f2.metric("做多贡献", f"{focus_daily['做多'].sum():+.2%}")
                        f3.metric("做空贡献", f"{focus_daily['做空'].sum():+.2%}")
                        f4.metric("参与天数", f"{len(focus_df)} 天")

                        fig_focus, ax_focus = plt.subplots(figsize=(12, 3.8))
                        ax_focus.bar(focus_daily.index, focus_daily['做多'], color='#d62728', alpha=0.55, label='做多日贡献')
                        ax_focus.bar(focus_daily.index, focus_daily['做空'], color='#2ca02c', alpha=0.55, label='做空日贡献')
                        ax_focus.plot(focus_daily.index, focus_daily['累计贡献'], color='#1f77b4', lw=2, label='累计贡献')
                        ax_focus.axhline(y=0, color='gray', linestyle='--', alpha=0.45)
                        ax_focus.set_title(f"{focus_asset} 多空收益时间轴", fontproperties=my_font, fontsize=13)
                        ax_focus.grid(True, alpha=0.2)
                        ax_focus.legend(prop=my_font)
                        fig_focus.tight_layout()
                        st.pyplot(fig_focus)

                with t9:
                    st.markdown("### 🧩 策略框架诊断总控台")
                    st.caption("把策略拆成 数据 → 因子 → 过滤 → 选股 → 仓位 → 风控 → 收益 七个模块，哪里掉链子一眼看出来。")

                    detail_df5 = pd.DataFrame(res_cycle_details).copy()
                    if detail_df5.empty:
                        st.info("暂无可诊断的回测明细。")
                    else:
                        detail_df5['date'] = pd.to_datetime(detail_df5['date'])
                        detail_df5.set_index('date', inplace=True, drop=False)
                        detail_df5['drawdown'] = (res_nav['nav'] - res_nav['nav'].cummax()) / res_nav['nav'].cummax()
                        detail_df5['long_count'] = detail_df5['next_day_hold_long'].apply(lambda x: len([v for v in x.values() if v > 0]) if isinstance(x, dict) else 0)
                        detail_df5['short_count'] = detail_df5['next_day_hold_short'].apply(lambda x: len([v for v in x.values() if v > 0]) if isinstance(x, dict) else 0)
                        detail_df5['holding_count'] = detail_df5['long_count'] + detail_df5['short_count']
                        detail_df5['stop_count'] = detail_df5['stops'].apply(lambda x: len(x) if isinstance(x, list) else 0)
                        detail_df5['max_side_exposure'] = detail_df5[['long_exposure', 'short_exposure']].max(axis=1)
                        detail_df5['gross_limit_hit'] = detail_df5['gross_exposure'] >= (max_gross_exposure_ui * 0.995)
                        detail_df5['margin_limit_hit'] = detail_df5['margin_usage'] >= (max_margin_usage_ui * 0.995)
                        detail_df5['net_abs'] = detail_df5['net_exposure'].abs()
                        detail_df5['net_limit_hit'] = detail_df5['net_abs'] >= (net_exp_target * 0.995)

                        avg_valid_assets = df_p.notna().sum(axis=1).reindex(detail_df5.index).fillna(0)
                        avg_mom_pass = debug_data['momentum_filter'].reindex(detail_df5.index).sum(axis=1).fillna(0)
                        avg_ma_pass = debug_data['ma_filter'].reindex(detail_df5.index).sum(axis=1).fillna(0)
                        avg_valid_volume = (df_vol.reindex(detail_df5.index).fillna(0) >= MIN_ACTIVE_VOLUME).sum(axis=1)

                        latest = detail_df5.iloc[-1]
                        health_items = [
                            {
                                '模块': '数据池',
                                '状态': '✅ 正常' if avg_valid_assets.mean() >= hold_num * 2 else '⚠️ 偏少',
                                '核心指标': f"平均可用 {avg_valid_assets.mean():.0f} 个品种",
                                '诊断': '可交易品种足够' if avg_valid_assets.mean() >= hold_num * 2 else '可用样本偏少，选股稳定性可能下降'
                            },
                            {
                                '模块': '趋势/动量过滤',
                                '状态': '✅ 正常' if avg_mom_pass.mean() >= hold_num else '⚠️ 偏严',
                                '核心指标': f"平均动量通过 {avg_mom_pass.mean():.1f} 个",
                                '诊断': '多头过滤有足够候选' if avg_mom_pass.mean() >= hold_num else '过滤较严，可能导致多头不足'
                            },
                            {
                                '模块': '持仓覆盖',
                                '状态': '✅ 正常' if detail_df5['holding_count'].mean() >= hold_num else '⚠️ 偏低',
                                '核心指标': f"平均持仓 {detail_df5['holding_count'].mean():.1f} 个",
                                '诊断': '组合分散度尚可' if detail_df5['holding_count'].mean() >= hold_num else '持仓数量偏低，单品种风险可能上升'
                            },
                            {
                                '模块': '净敞口',
                                '状态': '✅ 正常' if detail_df5['net_limit_hit'].mean() < 0.35 else '⚠️ 频繁触发',
                                '核心指标': f"触发率 {detail_df5['net_limit_hit'].mean():.1%}",
                                '诊断': '多空偏置控制稳定' if detail_df5['net_limit_hit'].mean() < 0.35 else '净敞口约束经常卡仓，说明多空候选不平衡'
                            },
                            {
                                '模块': '杠杆',
                                '状态': '✅ 正常' if detail_df5['gross_limit_hit'].mean() < 0.5 else '⚠️ 频繁满仓',
                                '核心指标': f"最高 {detail_df5['gross_exposure'].max():.1%} / 上限 {max_gross_exposure_ui:.1%}",
                                '诊断': '名义杠杆有弹性空间' if detail_df5['gross_limit_hit'].mean() < 0.5 else '杠杆经常打满，收益和回撤高度依赖杠杆'
                            },
                            {
                                '模块': '保证金',
                                '状态': '✅ 正常' if detail_df5['margin_limit_hit'].mean() < 0.35 else '⚠️ 频繁占满',
                                '核心指标': f"最高 {detail_df5['margin_usage'].max():.1%} / 上限 {max_margin_usage_ui:.1%}",
                                '诊断': '保证金安全垫尚可' if detail_df5['margin_limit_hit'].mean() < 0.35 else '保证金约束经常触发，实盘容错下降'
                            },
                            {
                                '模块': '止损/异常',
                                '状态': '✅ 正常' if detail_df5['stop_count'].mean() < 1 else '⚠️ 止损偏多',
                                '核心指标': f"平均每日止损 {detail_df5['stop_count'].mean():.2f} 个",
                                '诊断': '止损频率可控' if detail_df5['stop_count'].mean() < 1 else '止损较频繁，需检查空头反弹或参数过紧'
                            },
                        ]
                        health_df = pd.DataFrame(health_items)

                        st.markdown("#### 🧠 策略流水线总览")
                        pipe_cols = st.columns(7)
                        pipe_steps = [
                            ('1 数据池', f"{avg_valid_assets.iloc[-1]:.0f} 可用"),
                            ('2 因子评分', '动量+流动性'),
                            ('3 过滤器', f"动量 {avg_mom_pass.iloc[-1]:.0f} / 趋势 {avg_ma_pass.iloc[-1]:.0f}"),
                            ('4 多空选股', f"多 {latest['long_count']} / 空 {latest['short_count']}"),
                            ('5 仓位分配', f"毛 {latest['gross_exposure']:.1%}"),
                            ('6 风控截断', f"保证金 {latest['margin_usage']:.1%}"),
                            ('7 收益输出', f"当日 {latest['ret']:+.2%}"),
                        ]
                        for col, (title, value) in zip(pipe_cols, pipe_steps):
                            col.markdown(
                                f"<div style='padding:14px; border-radius:12px; background:#f8f9fa; border:1px solid #e6e8eb; text-align:center;'>"
                                f"<div style='font-weight:700; font-size:15px;'>{title}</div>"
                                f"<div style='margin-top:8px; color:#1f77b4; font-size:16px; font-weight:700;'>{value}</div>"
                                f"</div>", unsafe_allow_html=True
                            )

                        st.markdown("#### 🚦 模块健康度雷达")
                        st.dataframe(health_df, use_container_width=True, height=285)

                        st.divider()
                        st.markdown("#### 📊 每日策略模块仪表盘")
                        diag_fig, diag_axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                        diag_axes[0].plot(detail_df5.index, res_nav['nav'], color='#1f77b4', lw=2, label='净值')
                        diag_axes[0].fill_between(detail_df5.index, 1, res_nav['nav'], color='#1f77b4', alpha=0.08)
                        diag_axes[0].set_title('净值与回撤', fontproperties=my_font)
                        ax_dd = diag_axes[0].twinx()
                        ax_dd.fill_between(detail_df5.index, detail_df5['drawdown'], 0, color='#d62728', alpha=0.18, label='回撤')
                        ax_dd.set_ylim(min(detail_df5['drawdown'].min() * 1.2, -0.01), 0.01)

                        diag_axes[1].plot(detail_df5.index, detail_df5['long_exposure'], color='#d62728', lw=1.7, label='多头仓位')
                        diag_axes[1].plot(detail_df5.index, detail_df5['short_exposure'], color='#2ca02c', lw=1.7, label='空头仓位')
                        diag_axes[1].plot(detail_df5.index, detail_df5['gross_exposure'], color='#9467bd', lw=1.8, label='杠杆')
                        diag_axes[1].axhline(max_gross_exposure_ui, color='#9467bd', ls='--', alpha=0.5)
                        diag_axes[1].legend(prop=my_font, ncol=3)
                        diag_axes[1].set_title('仓位与杠杆', fontproperties=my_font)

                        diag_axes[2].plot(detail_df5.index, detail_df5['net_exposure'], color='#ff7f0e', lw=1.6, label='净敞口')
                        diag_axes[2].axhline(net_exp_target, color='gray', ls='--', alpha=0.5)
                        diag_axes[2].axhline(-net_exp_target, color='gray', ls='--', alpha=0.5)
                        diag_axes[2].plot(detail_df5.index, detail_df5['margin_usage'], color='#8c564b', lw=1.6, label='保证金占用')
                        diag_axes[2].axhline(max_margin_usage_ui, color='#8c564b', ls='--', alpha=0.5)
                        diag_axes[2].legend(prop=my_font, ncol=2)
                        diag_axes[2].set_title('净敞口与保证金', fontproperties=my_font)

                        diag_axes[3].bar(detail_df5.index, detail_df5['long_count'], color='#ff9896', alpha=0.75, label='多头数量')
                        diag_axes[3].bar(detail_df5.index, detail_df5['short_count'], bottom=detail_df5['long_count'], color='#98df8a', alpha=0.75, label='空头数量')
                        diag_axes[3].plot(detail_df5.index, detail_df5['stop_count'], color='#d62728', lw=1.8, label='止损数量')
                        diag_axes[3].legend(prop=my_font, ncol=3)
                        diag_axes[3].set_title('持仓数量与止损压力', fontproperties=my_font)
                        for ax in diag_axes:
                            ax.grid(True, alpha=0.25)
                        diag_fig.tight_layout()
                        st.pyplot(diag_fig)

                        st.divider()
                        st.markdown("#### 🔬 单日框架拆解：选一天，看当天从候选池到最终仓位发生了什么")
                        diag_date = pd.to_datetime(st.date_input("选择诊断日期", value=detail_df5.index[-1], min_value=detail_df5.index[0], max_value=detail_df5.index[-1], key='t9_diag_date'))
                        if diag_date not in detail_df5.index:
                            st.warning("该日期无回测明细，请选择交易日。")
                        else:
                            drow = detail_df5.loc[diag_date]
                            candidate_df = pd.DataFrame({
                                '价格': df_p.loc[diag_date],
                                '成交量有效': df_vol.loc[diag_date] >= MIN_ACTIVE_VOLUME,
                                '动量分': debug_data['momentum_score'].loc[diag_date],
                                '流动性分': debug_data['liquidity_score'].loc[diag_date],
                                '多头分': debug_data['long_score'].loc[diag_date],
                                '空头分': debug_data['short_score'].loc[diag_date],
                                '趋势滤网': debug_data['ma_filter'].loc[diag_date],
                                '动量滤网': debug_data['momentum_filter'].loc[diag_date],
                            }).dropna(subset=['价格'])
                            candidate_df['状态'] = '观察'
                            candidate_df.loc[candidate_df.index.isin(drow.get('banned_list', [])), '状态'] = '黑名单'
                            candidate_df.loc[candidate_df.index.isin(drow.get('next_day_hold_long', {}).keys()), '状态'] = '做多入选'
                            candidate_df.loc[candidate_df.index.isin(drow.get('next_day_hold_short', {}).keys()), '状态'] = '做空入选'
                            candidate_df['最终仓位'] = 0.0
                            for a, w in drow.get('next_day_hold_long', {}).items(): candidate_df.loc[a, '最终仓位'] = w
                            for a, w in drow.get('next_day_hold_short', {}).items(): candidate_df.loc[a, '最终仓位'] = w
                            candidate_df['方向'] = np.where(candidate_df.index.isin(drow.get('next_day_hold_long', {}).keys()), '做多', np.where(candidate_df.index.isin(drow.get('next_day_hold_short', {}).keys()), '做空', ''))

                            flow_cols = st.columns(6)
                            flow_cols[0].metric('可用价格品种', f"{len(candidate_df)}")
                            flow_cols[1].metric('成交量有效', f"{candidate_df['成交量有效'].sum()}")
                            flow_cols[2].metric('趋势通过', f"{candidate_df['趋势滤网'].sum()}")
                            flow_cols[3].metric('动量通过', f"{candidate_df['动量滤网'].sum()}")
                            flow_cols[4].metric('最终做多/做空', f"{int(drow['long_count'])}/{int(drow['short_count'])}")
                            flow_cols[5].metric('当日收益', f"{drow['ret']:+.2%}")

                            show_mode_t9 = st.radio("查看视角", ['最终持仓', '多头候选排行', '空头候选排行', '全部品种诊断'], horizontal=True, key='t9_show_mode')
                            if show_mode_t9 == '最终持仓':
                                show_df = candidate_df[candidate_df['最终仓位'] > 0].sort_values('最终仓位', ascending=False)
                            elif show_mode_t9 == '多头候选排行':
                                show_df = candidate_df.sort_values('多头分', ascending=False).head(30)
                            elif show_mode_t9 == '空头候选排行':
                                show_df = candidate_df.sort_values('空头分', ascending=False).head(30)
                            else:
                                show_df = candidate_df.sort_values(['状态', '多头分'], ascending=[True, False])
                            st.dataframe(
                                show_df[['状态', '方向', '最终仓位', '价格', '成交量有效', '趋势滤网', '动量滤网', '动量分', '流动性分', '多头分', '空头分']]
                                .style.format({'最终仓位': '{:.1%}', '价格': '{:.2f}', '动量分': '{:.4f}', '流动性分': '{:.4f}', '多头分': '{:.4f}', '空头分': '{:.4f}'})
                                .bar(subset=['最终仓位'], color='#9ecae1')
                                .bar(subset=['多头分'], color='#ffb3b3')
                                .bar(subset=['空头分'], color='#b7e4b7'),
                                use_container_width=True, height=560
                            )

                        st.divider()
                        st.markdown("#### 🚨 问题日期定位器")
                        issue_df = detail_df5[['ret', 'drawdown', 'gross_exposure', 'net_exposure', 'margin_usage', 'long_count', 'short_count', 'stop_count']].copy()
                        issue_df['问题分数'] = (
                            issue_df['ret'].clip(upper=0).abs() * 4 +
                            issue_df['drawdown'].abs() * 2 +
                            issue_df['gross_exposure'] / max(max_gross_exposure_ui, 0.001) * 0.8 +
                            issue_df['margin_usage'] / max(max_margin_usage_ui, 0.001) * 0.8 +
                            issue_df['stop_count'] * 0.35
                        )
                        issue_show = issue_df.sort_values('问题分数', ascending=False).head(25).copy()
                        issue_show['日期'] = issue_show.index.date
                        st.dataframe(
                            issue_show[['日期', '问题分数', 'ret', 'drawdown', 'gross_exposure', 'net_exposure', 'margin_usage', 'long_count', 'short_count', 'stop_count']]
                            .style.format({'问题分数': '{:.3f}', 'ret': '{:+.2%}', 'drawdown': '{:.2%}', 'gross_exposure': '{:.1%}', 'net_exposure': '{:+.1%}', 'margin_usage': '{:.1%}'})
                            .bar(subset=['问题分数'], color='#fdae6b'),
                            use_container_width=True, height=420
                        )
