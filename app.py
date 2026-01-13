import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ================= 1. 系统配置 =================
st.set_page_config(page_title="Dual Momentum回测系统", layout="wide", page_icon="⚡")

# --- 路径自动适配逻辑 (修改部分) ---
# 1. 定义本地绝对路径 (你的电脑调试用)
local_absolute_path = r"D:\SAR日频\全部品种日线"

# 2. 自动判断环境
if os.path.exists(local_absolute_path):
    # 如果本地路径存在，说明在你的电脑上
    DEFAULT_DATA_FOLDER = local_absolute_path
else:
    # 否则说明在 Streamlit 云端，使用相对路径 'data'
    # 注意：你需要把 csv 文件放入项目根目录下的 data 文件夹中
    DEFAULT_DATA_FOLDER = "data"


# ================= 2. 数据处理 =================

def read_robust_csv(f):
    """
    通用CSV读取函数 (支持 gbk 和 utf-8)
    """
    for enc in ['gbk', 'utf-8']:
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
    # 路径检查
    if not os.path.exists(folder):
        return None, None, None, f"路径不存在: {folder} (请确保在GitHub上传了data文件夹)"

    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    if not files:
        return None, None, None, f"在 {folder} 中未找到CSV文件"

    price_dict = {}
    vol_dict = {}
    low_dict = {}  # 存储最低价用于止损

    progress_bar = st.progress(0, text="正在加载数据...")

    for i, file in enumerate(files):
        # 剔除逻辑
        if "纤维板" in file or "胶合板" in file or "线材" in file:
            continue

        name = file.split('.')[0].replace("主连", "").replace("日线", "")
        path = os.path.join(folder, file)

        df = read_robust_csv(path)
        if df is None: continue

        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date', 'close', 'high', 'low'], inplace=True)
            df['date'] = df['date'].dt.normalize()
            df.sort_values('date', inplace=True)
            df.set_index('date', inplace=True)

            # 去重
            df = df[~df.index.duplicated(keep='last')]

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
    核心策略逻辑：动量评分 + 均线过滤 + 波动率加权 + 日内止损
    """
    # 解包参数
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
    # 核心公式：0.4 * Short + 0.6 * Long
    momentum_score = 0.4 * mom_short + 0.6 * mom_long
    ma_filter = df_prices > df_prices.rolling(filter_ma).mean()
    asset_daily_rets = df_prices.pct_change().fillna(0)

    # --- B. 初始化回测变量 ---
    capital = 1.0
    nav_record = []
    asset_contribution = {}
    logs = []

    # 截取时间段
    full_dates = df_prices.index
    try:
        start_idx_loc = full_dates.get_indexer([start_date], method='bfill')[0]
    except:
        start_idx_loc = 0

    min_idx = max(lookback_long, filter_ma, 20)
    start_idx_loc = max(start_idx_loc, min_idx)

    if start_idx_loc >= len(full_dates):
        return pd.DataFrame(), pd.DataFrame(), ["选定时间内数据不足"]

    # 运行时状态变量
    weights = {}  # 目标持仓权重
    curr_holdings = {}  # 实际持仓权重
    entry_prices = {}  # 参考价

    # 日志缓存
    log_buffer_pnl = []
    cycle_count = 1
    log_start_date = full_dates[start_idx_loc]

    # --- C. 按日循环 ---
    for i in range(start_idx_loc, len(full_dates)):
        curr_date = full_dates[i]
        if curr_date > end_date: break
        prev_date = full_dates[i - 1]

        # 1. 每日选股 (Daily Rebalance)
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
                    # 风险平价 (Risk Parity based on Volatility)
                    sub_vols = vols[valid]
                    inv = 1.0 / (sub_vols + 1e-6)
                    weights = (inv / inv.sum()).to_dict()

            entry_prices = {a: df_prices.loc[prev_date, a] for a in weights.keys()}
            curr_holdings = weights.copy()

        except KeyError:
            weights = {}
            curr_holdings = {}

        # 2. 日内风控与收益结算
        daily_pnl = 0.0
        stopped_assets = []

        for asset, w in list(curr_holdings.items()):
            if w == 0: continue

            # 检查止损
            today_low = df_lows.loc[curr_date, asset]
            ref_price = entry_prices.get(asset, df_prices.loc[curr_date, asset])

            # 如果最低价触发止损线
            if ref_price > 0 and (today_low / ref_price - 1) < -stop_loss_pct:
                # 触发止损，按止损幅度结算
                actual_ret = -stop_loss_pct
                daily_pnl += w * actual_ret
                asset_contribution[asset] = asset_contribution.get(asset, 0.0) + w * actual_ret

                curr_holdings[asset] = 0  # 标记为平仓
                stopped_assets.append(asset)
            else:
                # 正常持有
                ret = asset_daily_rets.loc[curr_date, asset]
                daily_pnl += w * ret
                asset_contribution[asset] = asset_contribution.get(asset, 0.0) + w * ret

        capital *= (1 + daily_pnl)
        nav_record.append({'date': curr_date, 'nav': capital})

        # --- D. 生成日志 ---
        log_buffer_pnl.append(daily_pnl)

        if stopped_assets:
            logs.append(
                f"⚠️ [{curr_date.strftime('%Y-%m-%d')}] 触发止损: {', '.join(stopped_assets)} (按 {-stop_loss_pct * 100}% 离场)")

        # 每5天或最后一天聚合日志
        if len(log_buffer_pnl) == 5 or i == len(full_dates) - 1 or curr_date == end_date:
            cycle_ret = (np.prod([1 + r for r in log_buffer_pnl]) - 1)
            hold_str = ", ".join([f"{a}({w:.1%})" for a, w in curr_holdings.items() if w > 0])
            if not hold_str: hold_str = "空仓"

            end_d_str = curr_date.strftime('%Y-%m-%d')
            start_d_str = log_start_date.strftime('%Y-%m-%d')

            log_chunk = f"Cycle {cycle_count:02d} ({start_d_str} ~ {end_d_str}) | 收益: {cycle_ret * 100:>+5.1f}% | 净值: {capital:.4f}\n"
            log_chunk += f"   >> 持仓: {hold_str}\n"
            log_chunk += "-" * 60

            logs.append(log_chunk)
            log_buffer_pnl = []
            cycle_count += 1
            if i < len(full_dates) - 1:
                log_start_date = full_dates[i + 1]

    return pd.DataFrame(nav_record), pd.DataFrame(list(asset_contribution.items()),
                                                  columns=['Asset', 'Contribution']), logs


# ================= 4. UI 页面 =================

with st.sidebar:
    st.header("双重动量配置")

    # 显示当前使用的数据路径 (只读)
    st.info(f"当前数据源: `{DEFAULT_DATA_FOLDER}`")

    # 依然保留输入框，允许用户手动改 (可选)
    data_folder = st.text_input("数据文件夹路径", value=DEFAULT_DATA_FOLDER)
    st.divider()

    st.subheader("🗓️ 核心参数")
    col_d1, col_d2 = st.columns(2)
    start_d_input = col_d1.date_input("开始日期", value=pd.to_datetime("2024-01-01"))
    end_d_input = col_d2.date_input("结束日期", value=pd.to_datetime("2025-12-31"))

    hold_num_input = st.number_input("持仓数量", min_value=1, max_value=20, value=5)
    stop_loss_pct = st.number_input("单日个股止损 (%)", min_value=0.0, max_value=20.0, value=4.0, step=0.5) / 100.0

    with st.expander("🛠️ 算法参数 (5/20)"):
        lookback_short = st.number_input("短期动量 (Short)", value=5)
        lookback_long = st.number_input("长期动量 (Long)", value=20)
        filter_ma = st.number_input("均线过滤 (MA)", value=60)
        atr_window = st.number_input("ATR周期", value=20)

    run_btn = st.button(" 运行策略", type="primary", use_container_width=True)

# 主界面
st.title("Dual Momentum 策略回测")

if run_btn:
    with st.spinner('正在加载数据 (含最低价检查)...'):
        # 使用侧边栏最终确认的路径
        df_prices, df_vols, df_lows, err = load_data_and_calc_atr(data_folder, atr_window)

    if err:
        st.error(err)
        if "路径不存在" in err and "data" in err:
            st.warning("提示: 如果是在云端运行，请确保你已经将csv文件上传到了GitHub仓库的 'data' 文件夹中。")
    else:
        params = {
            'short': lookback_short,
            'long': lookback_long,
            'ma': filter_ma,
            'hold_num': hold_num_input,
            'stop_loss_pct': stop_loss_pct,
            'start_date': start_d_input,
            'end_date': end_d_input
        }

        with st.spinner('正在逐日模拟 (含日内止损逻辑)...'):
            res_nav, res_contrib, res_logs = run_strategy_logic(df_prices, df_vols, df_lows, params)

        if res_nav.empty:
            st.warning("该时间段内无交易数据或数据不足。")
        else:
            # 数据处理
            res_nav.set_index('date', inplace=True)
            res_contrib.sort_values('Contribution', ascending=False, inplace=True)

            # 指标计算
            total_ret = res_nav['nav'].iloc[-1] - 1
            days = (res_nav.index[-1] - res_nav.index[0]).days
            annual_ret = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0

            running_max = res_nav['nav'].cummax()
            dd = (res_nav['nav'] - running_max) / running_max
            max_dd = dd.min()

            daily_rets = res_nav['nav'].pct_change().fillna(0)
            sharpe = (daily_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0

            st.success("回测完成！")

            # 指标卡片
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("总收益率", f"{total_ret * 100:.2f}%", delta_color="normal")
            k2.metric("年化收益 (CAGR)", f"{annual_ret * 100:.2f}%")
            k3.metric("最大回撤", f"{max_dd * 100:.2f}%", delta_color="inverse")
            k4.metric("夏普比率", f"{sharpe:.2f}")

            # 图表 Tabs
            tab_chart, tab_attr, tab_log = st.tabs(["📈 资金曲线", "🏆 收益归因", "📝 交易日志"])

            with tab_chart:
                plot_data = res_nav.copy()
                plot_data['return_pct'] = (plot_data['nav'] - 1) * 100
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=plot_data.index, y=plot_data['return_pct'],
                    mode='lines', name='累计收益率',
                    line=dict(color='#ff7f0e', width=2.5),
                    fill='tozeroy', fillcolor='rgba(255, 127, 14, 0.1)'
                ))
                fig.update_layout(
                    title='<b>累计收益</b>', xaxis_title="日期", yaxis_title="累计收益率 (%)",
                    hovermode="x unified", margin=dict(l=20, r=20, t=60, b=20), plot_bgcolor='white'
                )
                fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
                fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.05)', ticksuffix="%")
                st.plotly_chart(fig, use_container_width=True)

            with tab_attr:
                st.markdown("#### 品种累计贡献度")
                res_contrib['Color'] = res_contrib['Contribution'].apply(lambda x: 'red' if x >= 0 else 'green')
                fig_bar = px.bar(res_contrib, x='Contribution', y='Asset', orientation='h',
                                 text_auto='.2%', color='Contribution',
                                 color_continuous_scale=['green', '#f0f2f6', 'red'])
                fig_bar.update_layout(height=max(400, len(res_contrib) * 20),
                                      yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)

                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.caption("🏆 盈利红榜")
                    st.dataframe(res_contrib.head(5).style.format({"Contribution": "{:.2%}"}), use_container_width=True)
                with col_t2:
                    st.caption("☠️ 亏损黑榜")
                    st.dataframe(
                        res_contrib.tail(5).sort_values("Contribution").style.format({"Contribution": "{:.2%}"}),
                        use_container_width=True)

            with tab_log:
                st.markdown("#### 聚合交易日志 (每5天 / 止损触发)")
                log_text = "\n".join(res_logs)
                st.text_area("Log Output", log_text, height=600)

else:
    st.info(f"👈 准备就绪，请点击【运行策略】\n\n当前检测路径: `{DEFAULT_DATA_FOLDER}`")
    if os.path.exists(data_folder):
        files_count = len([f for f in os.listdir(data_folder) if f.endswith('.csv')])
        st.write(f"📂 目录状态：找到 {files_count} 个CSV文件")
    else:
        st.write("⚠️ 目录状态：路径不存在 (请在本地创建或在GitHub上传data文件夹)")
