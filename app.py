import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import unicodedata
from datetime import datetime, timedelta

# ================= 1. 系统配置 =================
st.set_page_config(page_title="Dual Momentum回测系统", layout="wide", page_icon="")

# --- 路径自动适配逻辑 ---
local_absolute_path = r"D:\SAR日频\全部品种日线"

if os.path.exists(local_absolute_path):
    DEFAULT_DATA_FOLDER = local_absolute_path
else:
    DEFAULT_DATA_FOLDER = "data"


# ================= 2. 数据处理 =================

def read_robust_csv(f):
    """
    通用CSV读取函数 (支持 gbk 和 utf-8)
    """
    for enc in ['gbk', 'utf-8', 'gb18030']:
        try:
            df = pd.read_csv(f, encoding=enc)
            cols = [str(c).strip() for c in df.columns]
            rename_map = {}

            # 模糊匹配列名
            for c in df.columns:
                c_str = str(c).strip()
                if c_str in ['日期', '日期/时间', 'date', 'Date']: rename_map[c] = 'date'
                if c_str in ['收盘价', '收盘', 'close', 'price', 'Close']: rename_map[c] = 'close'
                if c_str in ['最高价', '最高', 'high', 'High']: rename_map[c] = 'high'
                if c_str in ['最低价', '最低', 'low', 'Low']: rename_map[c] = 'low'

            df.rename(columns=rename_map, inplace=True)

            if 'date' in df.columns and 'close' in df.columns:
                return df
        except:
            continue
    return None


@st.cache_data(ttl=3600)
def load_data_and_calc_atr(folder, atr_window=20):
    """
    读取数据 (含 ATR 计算和 Low 价格读取)
    """
    if not os.path.exists(folder):
        return None, None, None, f"路径不存在: {folder}"

    # 【核心保留】必须排序，保证 Linux/Windows 读取顺序一致
    files = sorted([f for f in os.listdir(folder) if f.endswith('.csv')])
    
    if not files:
        return None, None, None, f"在 {folder} 中未找到CSV文件"

    price_dict = {}
    vol_dict = {}
    low_dict = {}
    
    progress_bar = st.progress(0, text="正在加载数据...")

    for i, file in enumerate(files):
        # 【核心保留】文件名标准化，防止跨平台编码问题
        file_norm = unicodedata.normalize('NFC', file)
        
        # 剔除逻辑
        if "纤维板" in file_norm or "胶合板" in file_norm or "线材" in file_norm:
            continue

        name = file_norm.split('.')[0].replace("主连", "").replace("日线", "")
        path = os.path.join(folder, file)

        df = read_robust_csv(path)
        if df is None: continue

        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date', 'close', 'high', 'low'], inplace=True)
            df['date'] = df['date'].dt.normalize()
            
            # 【核心保留】再次排序确保时间序列正确
            df.sort_values('date', inplace=True)
            
            # 去重
            df = df[~df.index.duplicated(keep='last')]
            df.set_index('date', inplace=True)

            # --- 计算 ATR/NATR ---
            prev_close = df['close'].shift(1)
            tr1 = df['high'] - df['low']
            tr2 = (df['high'] - prev_close).abs()
            tr3 = (df['low'] - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr = tr.rolling(atr_window).mean()
            natr = atr / df['close']

            price_dict[name] = df['close']
            vol_dict[name] = natr
            low_dict[name] = df['low']

        except Exception as e:
            continue

        if i % 10 == 0:
            progress_bar.progress((i + 1) / len(files), text=f"加载: {name}")

    progress_bar.empty()

    if not price_dict:
        return None, None, None, "未读取到有效数据，请检查CSV格式"

    # 合并为宽表
    df_prices = pd.DataFrame(price_dict).sort_index().ffill()
    df_vols = pd.DataFrame(vol_dict).sort_index().ffill()
    df_lows = pd.DataFrame(low_dict).sort_index().ffill()

    return df_prices, df_vols, df_lows, None


# ================= 3. 核心策略逻辑 =================

def run_strategy_logic(df_prices, df_vols, df_lows, params):
    """
    核心策略逻辑
    """
    lookback_short = params['short']
    lookback_long = params['long']
    hold_num = params['hold_num']
    filter_ma = params['ma']
    stop_loss_pct = params['stop_loss_pct']

    start_date = pd.to_datetime(params['start_date'])
    end_date = pd.to_datetime(params['end_date'])

    # --- A. 因子计算 ---
    mom_short = df_prices.pct_change(lookback_short)
    mom_long = df_prices.pct_change(lookback_long)
    momentum_score = 0.4 * mom_short + 0.6 * mom_long
    ma_filter = df_prices > df_prices.rolling(filter_ma).mean()
    asset_daily_rets = df_prices.pct_change().fillna(0)

    # --- B. 初始化 ---
    capital = 1.0
    nav_record = []
    asset_contribution = {}
    logs = []

    full_dates = df_prices.index
    try:
        start_idx_loc = full_dates.get_indexer([start_date], method='bfill')[0]
    except:
        start_idx_loc = 0

    min_idx = max(lookback_long, filter_ma, 20)
    start_idx_loc = max(start_idx_loc, min_idx)

    if start_idx_loc >= len(full_dates):
        return pd.DataFrame(), pd.DataFrame(), ["选定时间内数据不足"]

    weights = {}
    curr_holdings = {}
    entry_prices = {}
    log_buffer_pnl = []
    cycle_count = 1
    log_start_date = full_dates[start_idx_loc]

    # --- C. 按日循环 ---
    for i in range(start_idx_loc, len(full_dates)):
        curr_date = full_dates[i]
        if curr_date > end_date: break
        prev_date = full_dates[i - 1]

        # 1. 每日选股
        try:
            scores = momentum_score.loc[prev_date].dropna()
            vols = df_vols.loc[prev_date]

            if len(scores) < hold_num:
                weights = {}
            else:
                top = scores.sort_values(ascending=False).head(hold_num).index.tolist()
                valid = [a for a in top if ma_filter.loc[prev_date, a]]

                if not valid:
                    weights = {}
                else:
                    sub_vols = vols[valid]
                    inv = 1.0 / (sub_vols + 1e-6)
                    weights = (inv / inv.sum()).to_dict()

            entry_prices = {a: df_prices.loc[prev_date, a] for a in weights.keys()}
            curr_holdings = weights.copy()

        except KeyError:
            weights = {}
            curr_holdings = {}

        # 2. 结算与风控
        daily_pnl = 0.0
        stopped_assets = []

        for asset, w in list(curr_holdings.items()):
            if w == 0: continue

            today_low = df_lows.loc[curr_date, asset]
            ref_price = entry_prices.get(asset, df_prices.loc[curr_date, asset])

            if ref_price > 0 and (today_low / ref_price - 1) < -stop_loss_pct:
                actual_ret = -stop_loss_pct
                daily_pnl += w * actual_ret
                asset_contribution[asset] = asset_contribution.get(asset, 0.0) + w * actual_ret
                curr_holdings[asset] = 0
                stopped_assets.append(asset)
            else:
                ret = asset_daily_rets.loc[curr_date, asset]
                daily_pnl += w * ret
                asset_contribution[asset] = asset_contribution.get(asset, 0.0) + w * ret

        capital *= (1 + daily_pnl)
        nav_record.append({'date': curr_date, 'nav': capital})
        log_buffer_pnl.append(daily_pnl)

        if stopped_assets:
            logs.append(f"⚠️ [{curr_date.strftime('%Y-%m-%d')}] 触发止损: {', '.join(stopped_assets)}")

        if len(log_buffer_pnl) == 5 or i == len(full_dates) - 1 or curr_date == end_date:
            cycle_ret = (np.prod([1 + r for r in log_buffer_pnl]) - 1)
            hold_str = ", ".join([f"{a}({w:.1%})" for a, w in curr_holdings.items() if w > 0])
            if not hold_str: hold_str = "空仓"
            
            logs.append(f"Cycle {cycle_count:02d} | 收益: {cycle_ret * 100:>+5.1f}% | 净值: {capital:.4f} | 持仓: {hold_str}")
            logs.append("-" * 30)
            
            log_buffer_pnl = []
            cycle_count += 1
            if i < len(full_dates) - 1:
                log_start_date = full_dates[i + 1]

    return pd.DataFrame(nav_record), pd.DataFrame(list(asset_contribution.items()), columns=['Asset', 'Contribution']), logs


# ================= 4. UI 页面 =================

with st.sidebar:
    st.header("⚡ Dual Momentum")
    
    # 简单的路径显示，不再显示复杂的环境诊断
    st.caption(f"当前数据源: `{DEFAULT_DATA_FOLDER}`")
    data_folder = st.text_input("数据路径", value=DEFAULT_DATA_FOLDER)
    st.divider()

    st.subheader("🗓️ 核心参数")
    col_d1, col_d2 = st.columns(2)
    start_d_input = col_d1.date_input("开始日期", value=pd.to_datetime("2025-01-01"))
    end_d_input = col_d2.date_input("结束日期", value=pd.to_datetime("2025-12-31"))

    hold_num_input = st.number_input("持仓数量", 1, 20, 5)
    stop_loss_pct = st.number_input("止损 (%)", 0.0, 20.0, 4.0, step=0.5) / 100.0

    with st.expander("🛠️ 算法参数"):
        lookback_short = st.number_input("短期动量", value=5)
        lookback_long = st.number_input("长期动量", value=20)
        filter_ma = st.number_input("均线过滤", value=60)
        atr_window = st.number_input("ATR周期", value=20)

    run_btn = st.button("🚀 运行策略", type="primary", use_container_width=True)

# 主界面
st.title("Dual Momentum 策略回测")

if run_btn:
    with st.spinner('正在加载数据...'):
        # 调用时不再接收 debug_info
        df_prices, df_vols, df_lows, err = load_data_and_calc_atr(data_folder, atr_window)
    
    if err:
        st.error(err)
    else:
        params = {
            'short': lookback_short, 'long': lookback_long, 'ma': filter_ma,
            'hold_num': hold_num_input, 'stop_loss_pct': stop_loss_pct,
            'start_date': start_d_input, 'end_date': end_d_input
        }

        with st.spinner('正在计算策略...'):
            res_nav, res_contrib, res_logs = run_strategy_logic(df_prices, df_vols, df_lows, params)

        if res_nav.empty:
            st.warning("无交易数据。")
        else:
            res_nav.set_index('date', inplace=True)
            res_contrib.sort_values('Contribution', ascending=False, inplace=True)

            total_ret = res_nav['nav'].iloc[-1] - 1
            days = (res_nav.index[-1] - res_nav.index[0]).days
            annual_ret = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0
            max_dd = (res_nav['nav'] / res_nav['nav'].cummax() - 1).min()
            
            daily_rets = res_nav['nav'].pct_change().fillna(0)
            sharpe = (daily_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0

            st.success("回测完成！")
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("总收益率", f"{total_ret * 100:.2f}%", delta_color="normal")
            k2.metric("年化收益", f"{annual_ret * 100:.2f}%")
            k3.metric("最大回撤", f"{max_dd * 100:.2f}%", delta_color="inverse")
            k4.metric("夏普比率", f"{sharpe:.2f}")

            tab_chart, tab_attr, tab_log = st.tabs(["📈 曲线", "🏆 归因", "📝 日志"])

            with tab_chart:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=res_nav.index, y=(res_nav['nav'] - 1)*100,
                    mode='lines', name='收益率', line=dict(color='#ff7f0e', width=2)
                ))
                fig.update_layout(title='累计收益率 (%)', margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with tab_attr:
                res_contrib['Color'] = res_contrib['Contribution'].apply(lambda x: 'red' if x >= 0 else 'green')
                fig_bar = px.bar(res_contrib, x='Contribution', y='Asset', orientation='h',
                                 text_auto='.2%', color='Contribution',
                                 color_continuous_scale=['green', '#f0f2f6', 'red'])
                fig_bar.update_layout(height=max(400, len(res_contrib) * 20), yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)

            with tab_log:
                st.text_area("交易明细", "\n".join(res_logs), height=500)
else:
    st.info(f"👈 请点击【运行策略】")

