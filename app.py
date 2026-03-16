import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import unicodedata
from datetime import datetime, date
import re
import plotly.graph_objects as go

# ================= 1. 系统配置 =================
st.set_page_config(page_title="Enhanced Dual Momentum V9", layout="wide", page_icon="🚀")

# --- A. 字体适配 ---
FONT_FILE = "SimHei.ttf"
if os.path.exists(FONT_FILE):
    my_font = fm.FontProperties(fname=FONT_FILE)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
else:
    my_font = fm.FontProperties(family='SimHei')
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# --- B. 路径适配 ---
local_absolute_path = r"D:\SAR日频\全部品种日线"
relative_path = "data"

if os.path.exists(local_absolute_path):
    DEFAULT_DATA_FOLDER = local_absolute_path
elif os.path.exists(relative_path):
    DEFAULT_DATA_FOLDER = relative_path
else:
    DEFAULT_DATA_FOLDER = "."

# ================= 2. 数据处理 (保持代码1原有逻辑：含乘数修复) =================

# [新增] 全品种中文名映射表 (辅助乘数查找)
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
    '塑料': 'l', 'PVC': 'v', 'PP': 'pp', '苯乙烯': 'eb', '乙二醇': 'eg', 'LPG': 'pg',
    '甲醇': 'ma', 'PTA': 'ta', '短纤': 'pf', '纯碱': 'sa', '玻璃': 'fg',
    '尿素': 'ur', '烧碱': 'sh', '对二甲苯': 'px', '瓶片': 'pr',
    '豆一': 'a', '豆二': 'b', '豆粕': 'm', '豆油': 'y', '棕榈': 'p',
    '玉米': 'c', '淀粉': 'cs', '鸡蛋': 'jd', '生猪': 'lh',
    '白糖': 'sr', '棉花': 'cf', '棉纱': 'cy', '菜油': 'oi', '菜粕': 'rm', '花生': 'pk',
    '苹果': 'ap', '红枣': 'cj',
    '碳酸锂': 'lc', '工业硅': 'si',
    # [新增/修正以下内容]
    '二年债': 'TS', '五年债': 'TF', '十年债': 'T', '三十债': 'TL',
    '2年期国债': 'TS', '5年期国债': 'TF', '10年期国债': 'T', '30年期国债': 'TL',
    '1000股指': 'IM', '500股指': 'IC', '300股指': 'IF', '50股指': 'IH',
    '棕榈油': 'p', '菜籽油': 'oi', '菜籽粕': 'rm',
    '聚丙烯': 'pp', '丁二烯橡胶': 'br',
    '集运欧线': 'ec', '原木': 'lg', '多晶硅': 'si', '钯金': 'pd',
    '铂金': 'pt', # 假设多晶硅同工业硅或添加特定乘数
}

#  合约乘数表
CONTRACT_MULTIPLIERS = {
    'rb': 10, 'hc': 10, 'i': 100, 'j': 100, 'jm': 60, 'zc': 100, 'sf': 5, 'sm': 5, 'ss': 5, 'wr': 10,
    'cu': 5, 'al': 5, 'zn': 5, 'pb': 5, 'ni': 1, 'sn': 1, 'bc': 5, 'ao': 20,
    'au': 1000, 'ag': 15,
    'ru': 10, 'bu': 10, 'sp': 10, 'fu': 10, 'sc': 1000, 'pg': 20, 'l': 5, 'pp': 5, 'v': 5, 'eg': 10,
    'ta': 5, 'ma': 10, 'ur': 20, 'sa': 20, 'lu': 10, 'eb': 5, 'pf': 5, 'px': 5, 'nr': 10, 'sh': 10, 'br': 5,
    'c': 10, 'cs': 10, 'a': 10, 'b': 10, 'm': 10, 'y': 10, 'p': 10, 'oi': 10, 'rm': 10,
    'cf': 5, 'sr': 10, 'jd': 10, 'ap': 10, 'cj': 5, 'lh': 16, 'pk': 5, 'si': 5, 'lc': 1,
    'fg': 20,'T': 10000, 'TF': 10000, 'TS': 10000, 'TL': 10000,
    'ec': 50, 'lg': 90,'pt': 1000,
    'pd': 1000,
    'IF': 300, 'IH': 300, 'IC': 200, 'IM': 200, 'T': 10000, 'TF': 10000, 'TS': 10000, 'TL': 10000
}

# [新增] 板块归类字典 (使用英文代码进行归类)
SECTOR_DEF = {
    '贵金属': ['au', 'ag'],
    '有色': ['cu', 'al', 'zn', 'pb', 'ni', 'sn', 'bc', 'ao'],
    '黑色': ['rb', 'hc', 'i', 'j', 'jm', 'sm', 'sf', 'ss', 'wr'],
    '化工': ['sc', 'fu', 'lu', 'bu', 'ru', 'nr', 'br', 'l', 'v', 'pp', 'ta', 'ma', 'eg', 'eb', 'pg', 'sa', 'ur', 'fg',
             'sh', 'px', 'pr', 'pf', 'sp'],
    '农产品': ['m', 'y', 'p', 'oi', 'rm', 'c', 'cs', 'a', 'b', 'jd', 'lh', 'sr', 'cf', 'cy', 'ap', 'cj', 'pk'],
    '中金所': ['IF', 'IH', 'IC', 'IM', 'T', 'TF', 'TS', 'TL'],
    '广期所': ['lc', 'si', 'ps', 'pt', 'pd']
}

# 自动生成 代码 -> 板块 的反向映射表
CODE_TO_SECTOR = {code: sec for sec, codes in SECTOR_DEF.items() for code in codes}


def get_sector(asset_name):
    """辅助函数：根据品种名称获取所属板块 (已修复英文全称误判Bug)"""
    # 先把多余的后缀清洗干净
    clean_name = asset_name.replace("主连", "").replace("指数", "").replace("连续", "").replace("日线", "").replace(
        ".csv", "").strip()

    # 1. 优先查你的专属字典 (完美解决 PTA, LPG, PVC 的代码映射)
    if clean_name in CN_NAME_MAP:
        code = CN_NAME_MAP[clean_name]
    else:
        # 2. 如果字典里没登记，再尝试用正则提取开头的英文字母
        match = re.match(r"([a-zA-Z]+)", clean_name)
        if match:
            code = match.group(1).lower()
        else:
            code = clean_name.lower()

    # 最后去查代码归属哪个板块，找不到就丢进'其他'
    return CODE_TO_SECTOR.get(code.lower(), CODE_TO_SECTOR.get(code.upper(), '其他'))

def get_multiplier(asset_name):
    """辅助函数：获取合约乘数"""
    match = re.match(r"([a-zA-Z]+)", asset_name)
    if match:
        code = match.group(1).lower()
        return CONTRACT_MULTIPLIERS.get(code, 1)
    clean_name = asset_name.replace("主连", "").replace("指数", "") \
        .replace("连续", "").replace("日线", "").replace(".csv", "").strip()
    code = CN_NAME_MAP.get(clean_name)
    if code:
        return CONTRACT_MULTIPLIERS.get(code, 1)
    return 1


def read_robust_csv(f):
    """鲁棒的CSV读取函数"""
    for enc in ['gbk', 'utf-8', 'gb18030', 'cp936']:
        try:
            df = pd.read_csv(f, encoding=enc, engine='python')
            cols = [str(c).strip() for c in df.columns]
            rename_map = {}
            for c in df.columns:
                c_str = str(c).strip()
                if c_str in ['日期', '日期/时间', 'date', 'Date']: rename_map[c] = 'date'
                if c_str in ['收盘价', '收盘', 'close', 'price', 'Close']: rename_map[c] = 'close'
                if c_str in ['最高价', '最高', 'high', 'High']: rename_map[c] = 'high'
                if c_str in ['最低价', '最低', 'low', 'Low']: rename_map[c] = 'low'
                if c_str in ['开盘价', '开盘', 'open', 'Open']: rename_map[c] = 'open'
                if c_str in ['成交量', 'volume', 'Volume', 'vol']: rename_map[c] = 'volume'
                if c_str in ['成交额', 'amount', 'Amount']: rename_map[c] = 'amount'

            df.rename(columns=rename_map, inplace=True)
            if 'date' in df.columns and 'close' in df.columns:
                return df
        except:
            continue
    return None


@st.cache_data(ttl=3600)
def load_data_and_calc_metrics(folder, atr_window=20):
    """加载数据并计算基础指标"""
    if not os.path.exists(folder):
        return None, None, None, None, None, None, None, None, None, None, f"路径不存在: {folder}"

    try:
        files = sorted([f for f in os.listdir(folder) if f.endswith('.csv')])
    except:
        return None, None, None, None, None, None, None, None, None, None, "无法读取目录"

    if not files:
        return None, None, None, None, None, None, None, None, None, None, "无CSV文件"

    price_dict, low_dict, open_dict, high_dict = {}, {}, {}, {}
    atr_dict, atr_norm_dict = {}, {}
    vol_dict, vol_ma_dict = {}, {}
    amount_dict, liquidity_score_dict = {}, {}

    progress_bar = st.progress(0, text="正在加载数据...")

    for i, file in enumerate(files):
        file_norm = unicodedata.normalize('NFC', file)
        if any(x in file_norm for x in ["纤维板", "胶合板", "线材", "强麦", "早籼稻"]):
            continue

        name = file_norm.split('.')[0].replace("主连", "").replace("日线", "")
        multiplier = get_multiplier(name)  # 使用代码1的逻辑

        path = os.path.join(folder, file)
        df = read_robust_csv(path)
        if df is None: continue

        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            if 'volume' not in df.columns: df['volume'] = 0
            # 使用代码1的逻辑计算金额
            if 'amount' not in df.columns:
                df['amount'] = df['close'] * df['volume'] * multiplier

            df.dropna(subset=['date', 'close', 'high', 'low', 'open'], inplace=True)
            df['date'] = df['date'].dt.normalize()
            df.sort_values('date', inplace=True)
            df = df[~df.index.duplicated(keep='last')]
            df.set_index('date', inplace=True)

            prev_close = df['close'].shift(1)
            tr = pd.concat([df['high'] - df['low'],
                            (df['high'] - prev_close).abs(),
                            (df['low'] - prev_close).abs()], axis=1).max(axis=1)
            atr = tr.rolling(atr_window, min_periods=1).mean()
            natr = atr / df['close']

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

        except Exception as e:
            continue

        if i % 10 == 0:
            progress_bar.progress((i + 1) / len(files), text=f"加载: {name}")

    progress_bar.empty()

    if not price_dict:
        return None, None, None, None, None, None, None, None, None, None, "数据解析为空"

    return (pd.DataFrame(price_dict).ffill(),
            pd.DataFrame(atr_norm_dict).ffill(),
            pd.DataFrame(low_dict).ffill(),
            pd.DataFrame(open_dict).ffill(),
            pd.DataFrame(high_dict).ffill(),
            pd.DataFrame(atr_dict).ffill(),
            pd.DataFrame(vol_dict).fillna(0),
            pd.DataFrame(vol_ma_dict).fillna(0),
            pd.DataFrame(amount_dict).ffill(),
            pd.DataFrame(liquidity_score_dict).ffill(),
            None)


# ================= 3. 优化因子计算 =================

class EnhancedFactors:
    @staticmethod
    def calculate_multi_period_momentum(df_p, periods=[5, 10, 20, 60]):
        avg_roc = pd.DataFrame(0.0, index=df_p.index, columns=df_p.columns)
        mom_sign_matrix = pd.DataFrame(0.0, index=df_p.index, columns=df_p.columns)
        valid_count = 0
        for p in periods:
            roc = df_p.pct_change(p)
            avg_roc = avg_roc.add(roc.fillna(0), fill_value=0)
            mom_sign = (roc > 0).astype(int)
            mom_sign_matrix = mom_sign_matrix.add(mom_sign, fill_value=0)
            valid_count += 1
        raw_avg_momentum = avg_roc / valid_count
        momentum_score = raw_avg_momentum.rank(axis=1, pct=True)
        momentum_filter = (mom_sign_matrix >= 4)
        return momentum_score, momentum_filter


    @staticmethod
    def calculate_volatility_adjustment(df_p, target_vol=0.25):
        returns = df_p.pct_change()
        market_vol = returns.std(axis=1) * np.sqrt(252)
        market_vol_avg = market_vol.rolling(60).mean()
        vol_scaler = target_vol / (market_vol_avg + 1e-8)
        return pd.Series(vol_scaler.clip(0.3, 2.0), index=df_p.index)

    @staticmethod
    def calculate_liquidity_score(df_amount, window=20):
        liquidity = df_amount.rolling(window, min_periods=1).mean()
        liquidity_score = liquidity.rank(axis=1, pct=True)
        return liquidity_score


# ================= 4. 高级止损类 =================

class AdvancedStopLoss:
    def __init__(self):
        self.asset_records = {}

    def update_and_check(self, asset, entry_price, current_price, days_held,
                         atr_value, today_high, today_low):
        records = self.asset_records.get(asset, {'entry_price': entry_price, 'max_profit': 0, 'stop_reason': None})
        profit_ratio = (current_price - entry_price) / entry_price
        if profit_ratio > records['max_profit']: records['max_profit'] = profit_ratio

        stop_reason = None
        suggested_exit_price = current_price

        # ======= 1. 老样子：4%高点回撤止盈 =======
        if today_high > 0:
            drop_from_high = (today_high - current_price) / today_high
            # 【重要修改】：必须是大于4%，且目前是赚钱的(current_price > entry_price)，才叫止盈！
            if drop_from_high > 0.04 and current_price > entry_price:
                stop_reason = f"🎯动态止盈(高点回撤{drop_from_high:.1%})"
                execution_price = today_high * (1 - 0.04)
                suggested_exit_price = max(execution_price, current_price)

        # ======= 2. 老样子：持仓僵化止损 =======
        if not stop_reason:
            if days_held > 20 and profit_ratio < 0.03:
                stop_reason = "⏳时间止损(持仓僵化)"
                suggested_exit_price = current_price

        if stop_reason: records['stop_reason'] = stop_reason
        self.asset_records[asset] = records
        return (stop_reason, suggested_exit_price) if stop_reason else None


# ================= 5. 核心策略逻辑 =================

def run_enhanced_strategy_logic_fixed(df_p, df_atr_norm, df_l, df_o, df_h, df_atr_abs,
                                      df_vol, df_vol_ma, df_amount, df_liquidity, params):
    # 1. 参数
    lookback_periods = params['periods']
    hold_num = params['hold_num']
    filter_ma = params['ma']
    stop_loss_trail = params['stop_loss_trail']
    stop_loss_hard = params['stop_loss_hard']
    commission_rate = params.get('commission', 0.0)
    slippage_rate = params.get('slippage', 0.0)
    start_date = pd.to_datetime(params['start_date'])
    end_date = pd.to_datetime(params['end_date'])

    # 优化参数
    use_multi_period = params.get('use_multi_period', True)
    use_vol_scaling = params.get('use_vol_scaling', True)
    use_trend_filter = params.get('use_trend_filter', True)
    target_volatility = params.get('target_volatility', 0.15)

    # 2. 计算所有因子
    factors = EnhancedFactors()

    # (1) 动量
    if use_multi_period:
        momentum_score, momentum_filter = factors.calculate_multi_period_momentum(df_p, lookback_periods)
    else:
        mom_short = df_p.pct_change(lookback_periods[0])
        mom_long = df_p.pct_change(lookback_periods[-1])
        momentum_score = 0.4 * mom_short + 0.6 * mom_long
        momentum_filter = pd.DataFrame(True, index=momentum_score.index, columns=momentum_score.columns)

    # (2) 流动性
    liquidity_score = factors.calculate_liquidity_score(df_amount)


    # (4) 波动率调整
    if use_vol_scaling:
        vol_scaler = factors.calculate_volatility_adjustment(df_p, target_volatility)
    else:
        vol_scaler = pd.Series(1.0, index=df_p.index)

    # (5) 均线过滤
    ma_filter = df_p > df_p.rolling(filter_ma, min_periods=1).mean()

    # === 打包 Debug 数据 ===
    debug_factors = {
        'momentum_score': momentum_score,
        'momentum_filter': momentum_filter,
        'liquidity_score': liquidity_score,
        'ma_filter': ma_filter,
        'vol_scaler': vol_scaler
    }

    # 3. 起点
    dates = df_p.index
    try:
        start_idx = dates.get_indexer([start_date], method='bfill')[0]
    except:
        start_idx = 0
    if start_idx < 1: start_idx = 1

    # 4. 初始化
    capital = 1.0
    nav_record = []
    asset_contribution = {}
    logs = []

    current_holdings = {}
    entry_prices = {}
    entry_dates = {}
    banned_assets = set()
    stop_manager = AdvancedStopLoss()

    cycle_details = []
    all_daily_details = []
    last_iso_week = None
    cycle_count = 1

    def generate_weekly_log(details, count, current_nav, start_date_str, end_date_str):
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
            block_logs.append(f"📊 当天收益：{d['ret'] * 100:+.2f}%")
            if d.get('risk_event'): block_logs.append(f"{d['risk_event']}")
            if d.get('start_hold'):
                block_logs.append("📈 当天持仓：")
                for asset, weight in d['start_hold'].items():
                    ret = d['asset_rets'].get(asset, 0.0) * 100
                    contribution = weight * d['asset_rets'].get(asset, 0.0) * 100
                    block_logs.append(f"  {asset}：(仓位：{weight:.0%}, 涨幅：{ret:+.2f}%，贡献：{contribution:+.4f}%)")
            else:
                block_logs.append("📈 当天持仓：空仓")
            if d['stops']:
                stop_texts = []
                for s in d['stops']:
                    reason = s['reason']
                    if "高级" in reason:
                        reason = f"🎯{reason}"
                    elif "熔断" in reason:
                        reason = f"🔥{reason}"
                    else:
                        reason = f"⛔{reason}"
                    stop_texts.append(f"{s['asset']} {reason}")
                block_logs.append(f"⚠️ 当天止损：{'，'.join(stop_texts)} (已全部关入小黑屋🚫)")
            if d.get('next_day_hold'):
                block_logs.append("🔮 隔夜持仓：")
                for asset, weight in d['next_day_hold'].items():
                    block_logs.append(f"  {asset}：(仓位：{weight:.0%})")
            if d.get('unbanned'):
                block_logs.append(f"🔓 解禁品种：{', '.join(d['unbanned'])}")
            block_logs.append("")
        block_logs.append("=" * 60)
        return block_logs

    def portfolio_risk_control(nav_series, current_capital, lookback=60):
        if len(nav_series) < lookback: return 1.0, False
        peak = np.maximum.accumulate(nav_series)
        current_dd = (current_capital - peak[-1]) / peak[-1] if peak[-1] > 0 else 0
        returns = pd.Series(nav_series).pct_change().dropna()
        if len(returns) >= 20:
            recent_vol = returns.tail(20).std() * np.sqrt(252)
            historical_vol = returns.std() * np.sqrt(252)
            if historical_vol > 0 and recent_vol > historical_vol * 1.5:
                return 0.7, False
        return 1.0, False

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # --- 逐日回测 ---
    for i in range(start_idx, len(dates)):
        curr_date = dates[i]
        if curr_date > end_date: break
        prev_date = dates[i - 1]

        # 1. 周报逻辑：检测到新的一周时触发
        curr_iso = curr_date.isocalendar()[:2]
        if last_iso_week is not None and curr_iso != last_iso_week:
            week_logs = generate_weekly_log(cycle_details, cycle_count,
                                            cycle_details[-1]['nav'] if cycle_details else capital,
                                            start_date_str, end_date_str)
            logs.extend(week_logs)
            cycle_count += 1
            cycle_details = []

            # 如果你希望解禁信息打印在周报里，可以将下方的 unbanned_log 逻辑也放在这里
        last_iso_week = curr_iso

        # 2. 解禁逻辑：【核心修改】确保在循环内部，每天都检查
        assets_to_unban = []
        unbanned_log = []

        # 打印一次列名，确认df_h里到底有没有"沪金"
        # if i == start_idx:
        #     print("DEBUG: df_h 的列名有:", df_h.columns.tolist()[:10], "...")

        for banned_asset in list(banned_assets):
            try:
                # 1. 检查列名是否存在
                if banned_asset not in df_h.columns:
                    print(f"❌ 严重错误: {banned_asset} 不在价格表列名中！解禁无法执行。")
                    continue

                h_hist = df_h.loc[:prev_date, banned_asset]

                # 2. 检查数据长度
                if len(h_hist) < 5:
                    # 只有偶尔打印，避免刷屏
                    if banned_asset == "沪金":
                        print(f"⚠️ 沪金数据不足5天，当前长度: {len(h_hist)}")
                    continue

                h_prev = h_hist.iloc[-1]  # 昨日最高
                h_refs = h_hist.iloc[-4:-1]  # 前3天最高

                if h_prev > h_refs.min():
                    assets_to_unban.append(banned_asset)

            except Exception as e:
                print(f"❌ 代码报错 ({banned_asset}): {str(e)}")
                continue

        for asset in assets_to_unban:
            banned_assets.remove(asset)
            unbanned_log.append(asset)

        # 3. 记录解禁日志（可选，建议保留以便观察）
        if unbanned_log:
            logs.append(f"🔓 解禁品种：{', '.join(unbanned_log)}")

        # 4. 进入当天交易准备
        start_of_day_holdings = current_holdings.copy()

        # 风控
        # 仓位总乘数：只依赖波动率调整 (不再有回撤强制打折)
        position_multiplier = 1.0
        if use_vol_scaling:
            try:
                position_multiplier *= vol_scaler.loc[curr_date]
            except:
                pass

        risk_event_msg = None  # 占位符，防止后面 cycle_details 打包时报错

        # 止损
        daily_gross_pnl = 0.0
        stopped_assets_info = []
        daily_asset_rets = {}

        for asset, w in list(start_of_day_holdings.items()):
            if w == 0: continue
            prev_close = df_p.loc[prev_date, asset]
            today_open = df_o.loc[curr_date, asset]
            today_low = df_l.loc[curr_date, asset]
            today_high = df_h.loc[curr_date, asset]
            today_close = df_p.loc[curr_date, asset]
            today_vol = df_vol.loc[curr_date, asset]
            avg_vol_20 = df_vol_ma.loc[prev_date, asset]

            ref_trail = prev_close
            stop_price_trail = ref_trail * (1 - stop_loss_trail)

            ref_entry = entry_prices.get(asset, ref_trail)
            stop_price_hard = ref_entry * (1 - stop_loss_hard)

            atr_val = df_atr_abs.loc[prev_date, asset]
            atr_crash_price = prev_close - (3 * atr_val)

            is_high_vol = (today_vol > 2 * avg_vol_20) if avg_vol_20 > 0 else False
            effective_atr_stop = atr_crash_price if is_high_vol else -99999999.0

            effective_stop_price = max(stop_price_trail, stop_price_hard, effective_atr_stop)

            triggered = False
            exit_price = today_close
            stop_reason = ""

            if today_open < effective_stop_price:
                triggered = True
                exit_price = today_open
                stop_reason = "硬止损(跳空)" if today_open < stop_price_hard else "移动止损(跳空)"
            elif today_low < effective_stop_price:
                triggered = True
                if effective_stop_price == effective_atr_stop:
                    exit_price = atr_crash_price
                    stop_reason = "ATR熔断(盘中)"
                elif effective_stop_price == stop_price_hard:
                    exit_price = stop_price_hard
                    stop_reason = "硬止损(盘中)"
                else:
                    exit_price = stop_price_trail
                    stop_reason = "移动止损(盘中)"

            if not triggered:
                days_held = (curr_date - entry_dates.get(asset, curr_date)).days
                adv_result = stop_manager.update_and_check(
                    asset, entry_prices.get(asset, prev_close), today_close, days_held,
                    atr_val, today_high, today_low
                )
                if adv_result:
                    triggered = True
                    adv_reason, adv_exit_price = adv_result
                    exit_price = adv_exit_price
                    stop_reason = adv_reason  # 直接使用我们在上面定义的中文理由，如"🎯动态止盈"

            if triggered:
                actual_ret = (exit_price - prev_close) / prev_close
                current_holdings[asset] = 0
                if asset in entry_prices: del entry_prices[asset]
                if asset in entry_dates: del entry_dates[asset]

                # 记录离场日志
                stopped_assets_info.append({'asset': asset, 'ret': actual_ret, 'reason': stop_reason, 'weight': w})

                # 【核心实现】：只要离场（无论是止盈还是止损），立刻关进小黑屋！
                banned_assets.add(asset)
            else:
                actual_ret = (today_close - prev_close) / prev_close

            daily_gross_pnl += w * actual_ret
            asset_contribution[asset] = asset_contribution.get(asset, 0.0) + w * actual_ret
            daily_asset_rets[asset] = actual_ret

            # 👇 ！！注意这里！！上面的 for 循环已经结束了！！

            # =======================================================
            # 🚀 新增：全局全品种 ATR 熔断检测（每天只执行1次，不论是否持仓）
            # =======================================================
            # 提取当天的全市场横截面数据
        all_prev_closes = df_p.loc[prev_date]
        all_today_lows = df_l.loc[curr_date]
        all_today_vols = df_vol.loc[curr_date]
        all_avg_vols_20 = df_vol_ma.loc[prev_date]
        all_atr_vals = df_atr_abs.loc[prev_date]

        # 向量化计算：放量（成交量>2倍均量）且 暴跌（最低价击穿3倍ATR）
        all_atr_crash_prices = all_prev_closes - (3 * all_atr_vals)
        all_is_high_vols = all_today_vols > (2 * all_avg_vols_20)

        # 生成布尔掩码，找出今天触发熔断的所有品种
        global_crash_mask = (all_today_lows <= all_atr_crash_prices) & all_is_high_vols
        global_crashed_assets = global_crash_mask[global_crash_mask].index.tolist()

        # 将它们全部关进小黑屋
        for g_asset in global_crashed_assets:
            if g_asset not in banned_assets:
                banned_assets.add(g_asset)
                unbanned_log.append(f"🔥全局预警：{g_asset}放量暴跌，已拦截")
        # =======================================================

        # 选股与隔夜持仓
        next_day_holdings = {}
        daily_cost = 0.0

        try:
            filter_condition = (ma_filter.loc[curr_date] & momentum_filter.loc[curr_date])
            if filter_condition.any():
                # 计算总分：动量 0.7 + 流动性 0.3
                scores = momentum_score.loc[curr_date].dropna()
                if not liquidity_score.empty:  # 改为检查 liquidity_score
                    liquidity = liquidity_score.loc[curr_date].dropna()  # 改为 liquidity_score
                    # 重新对齐索引
                    common_idx = scores.index.intersection(liquidity.index)
                    scores = scores.loc[common_idx] * 0.7 + liquidity.loc[common_idx] * 0.3
                    #scores = scores.loc[common_idx] * 1 + liquidity.loc[common_idx] * 0.0

                valid_pool = [a for a in scores.index if filter_condition.get(a, False) and a not in banned_assets]
                ranked_pool = scores.loc[valid_pool].sort_values(ascending=False)

                keepers = []
                sector_counts = {}
                max_per_sector = params.get('max_per_sector', 2)

                ideal_assets = []

                # 【极简且暴力的纯粹排名过滤】
                # 从高到低遍历当天得分最高的品种
                for asset in ranked_pool.index:
                    sec = get_sector(asset)
                    # 只要该品种所在板块没满，直接入选！
                    if sector_counts.get(sec, 0) < max_per_sector:
                        ideal_assets.append(asset)
                        sector_counts[sec] = sector_counts.get(sec, 0) + 1

                    # 选满我们设定的目标持仓数量，立刻停止
                    if len(ideal_assets) == hold_num:
                        break

                ideal_weights = {}

                if ideal_assets:
                    vols = df_atr_norm.loc[curr_date, ideal_assets]

                    # 1. 波动率下限与平滑处理
                    vols = vols.clip(lower=0.005)
                    inv_vol = 1.0 / (vols ** 0.5)


                    raw_weights = inv_vol / inv_vol.sum()
                    raw_weights = raw_weights * position_multiplier

                    # 识别债券类品种（根据你的日志，国债包含“债”字，如“十年债”、“三十债”）

                    max_individual_weight = 0.30  # 单品种仓位上限 30%

                    # 2. 权重溢出再分配 (纯粹单品种上限)
                    while True:
                        capped = False

                        # A. 检查并限制单品种最高仓位
                        if (raw_weights > max_individual_weight + 1e-5).any():
                            raw_weights = raw_weights.clip(upper=max_individual_weight)
                            capped = True

                        # 如果没有触发任何截断，说明分配完毕，跳出循环
                        if not capped:
                            break

                        # B. 计算因为截断而溢出的总权重
                        current_total = raw_weights.sum()
                        shortfall = position_multiplier - current_total

                        # C. 把溢出的“水”倒给其他还没装满的品种
                        if shortfall > 1e-5:
                            # 找出还没装满 (没达到 30%) 的品种
                            receivers = [a for a in ideal_assets if raw_weights[a] < max_individual_weight - 1e-5]

                            if not receivers:
                                break  # 所有人全满了，分配结束

                            # 按比例将溢出的权重分给还能接收的品种
                            receiver_sum = raw_weights[receivers].sum()
                            if receiver_sum > 0:
                                raw_weights[receivers] += shortfall * (raw_weights[receivers] / receiver_sum)
                            else:
                                raw_weights[receivers] += shortfall / len(receivers)

                    ideal_weights = raw_weights.to_dict()

                    for asset, target_w in ideal_weights.items():
                        current_w = current_holdings.get(asset, 0.0)
                        if abs(target_w - current_w) > 0.02:  # 调仓阈值
                            daily_cost += abs(target_w - current_w) * commission_rate
                            if current_w == 0:
                                entry_prices[asset] = df_p.loc[curr_date, asset]
                                entry_dates[asset] = curr_date
                        next_day_holdings[asset] = target_w
                else:
                    pass

        except Exception as e:
            pass

        current_holdings = next_day_holdings.copy()
        daily_net_pnl = daily_gross_pnl - daily_cost
        capital *= (1 + daily_net_pnl)
        nav_record.append({'date': curr_date, 'nav': capital})

        # 记录每周期详情 (用于 UI 透视)
        cycle_details.append({
            'date': curr_date,
            'ret': daily_net_pnl,
            'risk_event': risk_event_msg,
            'start_hold': start_of_day_holdings,
            'asset_rets': daily_asset_rets,
            'stops': stopped_assets_info,
            'next_day_hold': next_day_holdings,
            'unbanned': unbanned_log,
            'banned_list': list(banned_assets),
            'entry_prices': entry_prices.copy(),  # 👈 新增：记录入场价用于计算次日止损
            'nav': capital
        })
        all_daily_details.append(cycle_details[-1])

    # 结果封装
    nav_df = pd.DataFrame(nav_record).set_index('date')
    trade_logs = logs
    if cycle_details:
        trade_logs.extend(generate_weekly_log(cycle_details, cycle_count, capital, start_date_str, end_date_str))

    # [修改] 增加返回值 all_daily_details 供 UI 调试使用
    return nav_df, trade_logs, debug_factors, asset_contribution, all_daily_details


# ================= 6. UI 主程序 (使用 Code 2 的布局) =================

with st.sidebar:
    st.header("Trend-Momentum")
    st.caption("Debug Mode: 全局因子透视 (UI Enhanced)")
    data_folder = st.text_input("数据路径", value=DEFAULT_DATA_FOLDER)

    st.divider()
    col1, col2 = st.columns(2)
    min_date = datetime(2000, 1, 1)
    max_date = datetime(2050, 12, 31)
    default_start = pd.to_datetime("2025-01-01")
    default_end = pd.to_datetime("2026-12-31")
    start_d = col1.date_input("开始日期", value=default_start, min_value=min_date, max_value=max_date)
    end_d = col2.date_input("结束日期", value=default_end, min_value=min_date, max_value=max_date)

    st.subheader("🎯 核心仓位参数")
    c1, c2 = st.columns(2)  # 把原来的两列改成三列
    hold_num = c1.number_input("目标持仓", 1, 20, 5)
    max_sector = c2.number_input("板块上限", 1, 10, 2)  # 新增参数

    st.write("🛑 **止损参数**")
    s1, s2 = st.columns(2)
    stop_trail = s1.number_input("移动止损(%)", 0.0, 20.0, 4.0, step=0.5)
    stop_hard = s2.number_input("硬止损(%)", 0.0, 20.0, 4.0, step=0.5)

    st.subheader("💸 成本设置")
    cc1, cc2 = st.columns(2)
    comm_bp = cc1.number_input("手续费(bp)", 0.0, 50.0, 0.0)  # 默认万3
    slip_bp = cc2.number_input("滑点(bp)", 0.0, 50.0, 0.0)


    with st.expander("🛠️ 因子与熔断配置"):
        periods_input = st.text_input("动量周期列表", value="5,10,20,60")
        periods = [int(p.strip()) for p in periods_input.split(",") if p.strip().isdigit()]
        if not periods: periods = [5, 10, 20, 60]
        ma_win = st.number_input("均线过滤", value=60)
        atr_win = st.number_input("ATR周期", value=20)
        target_volatility = st.slider("目标年化波动率%", 5, 50, 25) / 100

    run_btn = st.button("🚀 运行增强策略", type="primary", use_container_width=True)

st.title("Trend-Momentum")

if run_btn:
    with st.spinner("数据加载中..."):
        # 使用代码1的数据加载函数（包含乘数修正逻辑）
        (df_p, df_atr_norm, df_l, df_o, df_h, df_atr_abs,
         df_vol, df_vol_ma, df_amount, df_liquidity, err) = load_data_and_calc_metrics(data_folder, atr_win)

    if err:
        st.error(err)
    else:
        if start_d >= end_d:
            st.error("日期设置错误")
        else:
            # 参数转换适配
            params = {
                'periods': periods,
                'hold_num': hold_num,
                'max_per_sector': max_sector,
                'ma': ma_win,
                'stop_loss_trail': stop_trail / 100.0,  # 转换百分比
                'stop_loss_hard': stop_hard / 100.0,  # 转换百分比
                'start_date': start_d,
                'end_date': end_d,
                'commission': comm_bp / 10000,
                'slippage': slip_bp / 10000,
                'target_volatility': target_volatility,
                'use_multi_period': True,
                'use_vol_scaling': True
            }

            with st.spinner("策略回测中..."):
                # 解包返回值 (包含了新增加的 all_daily_details)
                res_nav, res_logs, debug_data, res_contrib, res_cycle_details = run_enhanced_strategy_logic_fixed(
                    df_p, df_atr_norm, df_l, df_o, df_h, df_atr_abs,
                    df_vol, df_vol_ma, df_amount, df_liquidity, params
                )

                # 转换贡献度格式以适配UI
                res_contrib_df = pd.DataFrame(list(res_contrib.items()), columns=['Asset', 'Contribution'])

            if res_nav.empty:
                st.warning("结果为空，请检查日期或数据")
            else:
                res_contrib_df.sort_values('Contribution', ascending=False, inplace=True)

                tot_ret = res_nav['nav'].iloc[-1] - 1
                days = (res_nav.index[-1] - res_nav.index[0]).days
                ann_ret = (1 + tot_ret) ** (365 / days) - 1 if days > 0 else 0
                peak = res_nav['nav'].cummax()
                dd = (res_nav['nav'] - peak) / peak
                max_dd = dd.min()
                d_rets = res_nav['nav'].pct_change().dropna()
                ann_vol = d_rets.std() * np.sqrt(252) if d_rets.std() != 0 else 0
                sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
                calmar = abs(ann_ret / max_dd) if max_dd != 0 else 0
                win_rate = (d_rets > 0).mean() * 100 if len(d_rets) > 0 else 0

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("总收益", f"{tot_ret * 100:.2f}%")
                col2.metric("年化收益", f"{ann_ret * 100:.2f}%")
                col3.metric("年化波动率", f"{ann_vol * 100:.2f}%")
                col4.metric("最大回撤", f"{max_dd * 100:.2f}%")

                col5, col6, col7, col8 = st.columns(4)
                col5.metric("夏普比率", f"{sharpe:.2f}")
                col6.metric("Calmar比率", f"{calmar:.2f}")
                col7.metric("胜率", f"{win_rate:.1f}%")
                col8.metric("交易天数", f"{len(res_nav)}")

                t1, t2, t3, t4 = st.tabs(["📈 净值曲线", "📊 盈亏分布", "📝 交易日志", "🔬 详细分析"])

                with t1:
                    # 1. 核心修改：把原来的 figsize=(12, 6) 改为 (12, 3.5) 或者更小的高度
                    fig, ax1 = plt.subplots(figsize=(12, 4.5))
                    x = res_nav.index
                    y = res_nav['nav']
                    ax1.plot(x, y, color='#1f77b4', lw=2, label=f'策略净值 (最终: {y.iloc[-1]:.2f})')
                    ax1.fill_between(x, y, 1, color='#1f77b4', alpha=0.1)
                    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
                    ax1.set_title(f"趋势动量策略", fontproperties=my_font, fontsize=14)
                    ax1.legend(prop=my_font)
                    ax1.grid(True, alpha=0.3)

                    # 2. 核心修改：增加紧凑布局，消除图片上下左右多余的白色留白
                    fig.tight_layout()

                    st.pyplot(fig)

                with t2:
                    if not res_contrib_df.empty:
                        res_contrib_df['Contribution_pct'] = res_contrib_df[
                                                                 'Contribution'] / tot_ret if tot_ret != 0 else 0
                        st.dataframe(
                            res_contrib_df.style.format({'Contribution': '{:.2%}', 'Contribution_pct': '{:.1%}'})
                            .background_gradient(cmap='RdYlGn', subset=['Contribution']),
                            use_container_width=True, height=600)

                with t3:
                    log_text = "\n".join(res_logs)
                    st.text_area("日志内容", log_text, height=600)
                    if log_text: st.download_button("📥 下载日志文件", log_text, "strategy_log.txt", "text/plain")

                with t4:
                    st.markdown("### 🔬 每日全品种得分透视")
                    st.caption("选择一个日期，查看当天所有品种的因子得分、排名以及过滤状态。这是排查'为什么没买它'的神器。")

                    # 1. 日期选择器
                    valid_dates = res_nav.index
                    default_date = valid_dates[-1] if len(valid_dates) > 0 else date.today()

                    c_sel1, c_sel2 = st.columns([1, 3])
                    with c_sel1:
                        target_date_input = st.date_input("选择查看日期", value=default_date,
                                                          min_value=valid_dates[0], max_value=valid_dates[-1])

                    target_date = pd.to_datetime(target_date_input)

                    if target_date not in valid_dates:
                        st.warning("该日期无交易数据（可能是周末或节假日），请选择临近日期。")
                    else:
                        # 2. 构建当日因子表
                        try:
                            # 寻找当日的 detail 记录
                            day_detail = next((d for d in res_cycle_details if d['date'] == target_date), None)
                            # 从 detail 中获取持仓、黑名单和入场价
                            held_assets = list(day_detail['next_day_hold'].keys()) if day_detail else []
                            banned_now = day_detail['banned_list'] if day_detail and 'banned_list' in day_detail else []
                            entry_prices_now = day_detail[
                                'entry_prices'] if day_detail and 'entry_prices' in day_detail else {}

                            # 提取因子
                            mom_s = debug_data['momentum_score'].loc[target_date]
                            liq_s = debug_data['liquidity_score'].loc[target_date]
                            ma_pass = debug_data['ma_filter'].loc[target_date]
                            mom_pass = debug_data['momentum_filter'].loc[target_date]
                            closes = df_p.loc[target_date]

                            # 👈 新增：计算次日止损价 (综合硬止损、移动止损、ATR熔断，取最高值)
                            stop_prices = pd.Series(np.nan, index=closes.index)
                            stop_trail_pct = stop_trail / 100.0
                            stop_hard_pct = stop_hard / 100.0

                            for asset in held_assets:
                                if asset in closes:
                                    prev_c = closes[asset]
                                    s_trail = prev_c * (1 - stop_trail_pct)
                                    s_hard = entry_prices_now.get(asset, prev_c) * (1 - stop_hard_pct)
                                    s_atr = prev_c - 3 * df_atr_abs.loc[
                                        target_date, asset] if asset in df_atr_abs.columns else 0
                                    stop_prices[asset] = max(s_trail, s_hard, s_atr)

                            # 合成总分 (0.7 * Mom + 0.3 * Liq)
                            final_score = (mom_s * 0.7 + liq_s * 0.3).fillna(-1)

                            # 构建 DataFrame
                            df_debug = pd.DataFrame({
                                '价格': closes,
                                '次日止损': stop_prices,  # 👈 新增列
                                '综合得分': final_score,
                                '动量分': mom_s,
                                '流动性分': liq_s,
                                'MA滤网': ma_pass,
                                '动量滤网': mom_pass,
                            })

                            # 标记状态
                            df_debug['状态'] = '观察'
                            df_debug.loc[df_debug.index.isin(banned_now), '状态'] = '🚫熔断黑名单'
                            df_debug.loc[df_debug.index.isin(held_assets), '状态'] = '✅ 持仓中'

                            # 过滤掉退市或无数据的品种
                            df_debug.dropna(subset=['价格'], inplace=True)

                            # 排序：持仓优先，然后按分数降序
                            df_debug.sort_values(by=['综合得分'], ascending=False, inplace=True)


                            # 3. 样式美化
                            def highlight_status(val):
                                if '持仓' in str(
                                    val): return 'background-color: #90ee90; color: black; font-weight: bold'
                                if '黑名单' in str(val): return 'background-color: #ffcccb; color: black'
                                return ''


                            def color_bool(val):
                                color = '#90ee90' if val else '#ffcccb'  # 绿/红
                                return f'background-color: {color}; color: black'


                            st.dataframe(
                                df_debug.style
                                .applymap(highlight_status, subset=['状态'])
                                .applymap(color_bool, subset=['MA滤网', '动量滤网'])
                                .format({
                                    '价格': '{:.2f}',
                                    '综合得分': '{:.4f}',
                                    '动量分': '{:.4f}',
                                    '流动性分': '{:.4f}'
                                })
                                .bar(subset=['综合得分'], color='#d65f5f', vmin=0, vmax=1),
                                use_container_width=True,
                                height=800
                            )

                            # 4. 显示当天的统计摘要
                            st.info(f"📅 **{target_date.date()} 统计**：共有 {len(df_debug)} 个有效品种，"
                                    f"其中 {df_debug['MA滤网'].sum()} 个站上均线，"
                                    f"最终选出 {len(held_assets)} 个持仓品种。")

                        except Exception as e:
                            st.error(f"无法生成当日透视表 (可能数据缺失): {str(e)}")

