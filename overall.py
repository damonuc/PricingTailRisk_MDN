import wrds
import pandas as pd
import numpy as np
import sqlite3

def download_wrds_to_sqlite(db_path='tail_risk_data.db', start_date='1990-01-01', end_date='2023-12-31'):
    print(f"=== Starting WRDS Micro Data Pipeline ===")
    conn = sqlite3.connect(db_path)
    
    print("Connecting to WRDS... (This will take a while)")
    db = wrds.Connection()
    
    # 1. 提取 CRSP 月度量价
    crsp_query = f"""
        SELECT a.permno, a.date, a.ret, a.prc, a.shrout, a.vol, b.exchcd, b.shrcd
        FROM crsp.msf AS a
        LEFT JOIN crsp.msenames AS b
        ON a.permno = b.permno AND b.namedt <= a.date AND a.date <= b.nameendt
        WHERE a.date BETWEEN '{start_date}' AND '{end_date}'
        AND b.shrcd IN (10, 11) AND b.exchcd IN (1, 2, 3)
    """
    crsp = db.raw_sql(crsp_query)
    crsp['date'] = pd.to_datetime(crsp['date']) + pd.offsets.MonthEnd(0)
    crsp['me'] = np.abs(crsp['prc']) * crsp['shrout']
    
    # 2. 提取 Compustat 财务报表
    comp_query = f"""
        SELECT gvkey, datadate, at, ceq, ni, revt, cogs, xsga, xint
        FROM comp.funda
        WHERE datadate BETWEEN '{start_date}' AND '{end_date}'
        AND indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'
    """
    comp = db.raw_sql(comp_query)
    comp['datadate'] = pd.to_datetime(comp['datadate'])
    
    # 3. 提取 CCM 链接表
    ccm_query = """
        SELECT gvkey, lpermno AS permno, linkdt, linkenddt
        FROM crsp.ccmxpf_linktable
        WHERE linktype IN ('LU', 'LC') AND linkprim IN ('P', 'C')
    """
    ccm = db.raw_sql(ccm_query)
    ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
    ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt']).fillna(pd.to_datetime('today'))
    
    # 合并财务与链接表
    comp = pd.merge(comp, ccm, on='gvkey', how='inner')
    comp = comp[(comp['datadate'] >= comp['linkdt']) & (comp['datadate'] <= comp['linkenddt'])]
    
    # 强制财务数据滞后 6 个月，杜绝前视偏差
    comp['merge_date'] = comp['datadate'] + pd.DateOffset(months=6) + pd.offsets.MonthEnd(0)
    
    # 按照年月合并 CRSP 和 Compustat
    crsp['year_month'] = crsp['date'].dt.to_period('M')
    comp['year_month'] = comp['merge_date'].dt.to_period('M')
    
    # 完美的 Merge，并按照原本的 date 排序
    micro_df = pd.merge(crsp, comp, on=['permno', 'year_month'], how='left')
    micro_df = micro_df.sort_values(['permno', 'date']).reset_index(drop=True)
    
    # 清理无用列并存入 SQLite
    micro_df.drop(columns=['year_month', 'merge_date'], errors='ignore', inplace=True)
    print(f"Saving Micro Data to SQLite... (Shape: {micro_df.shape})")
    
    micro_df.to_sql('micro_data', conn, if_exists='replace', index=False)
    
    # 建立索引加速读取
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_micro_date ON micro_data (date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_micro_permno ON micro_data (permno);")
    
    conn.close()
    db.close()
    print("=== WRDS Pipeline Complete! ===")

if __name__ == "__main__":
    download_wrds_to_sqlite()


import pandas as pd
from pandas_datareader import data as pdr
import sqlite3

def download_fred_to_sqlite(db_path='tail_risk_data.db', start_date='1990-01-01', end_date='2023-12-31'):
    print(f"=== Starting FRED Macro Data Pipeline ===")
    conn = sqlite3.connect(db_path)
    
    # 为了计算同比或滞后，多拉取一年的历史数据
    start = pd.to_datetime(start_date) - pd.DateOffset(years=1)
    
    # 修复后的 Tickers：Moody's Baa 企业债为 'BAA'
    tickers = ['T10Y3M', 'BAA', 'AAA', 'VIXCLS']
    print(f"Fetching Tickers: {tickers}")
    macro_raw = pdr.get_data_fred(tickers, start, end_date)
    
    # 转换为月度末数据
    macro_df = macro_raw.resample('ME').last().reset_index() # Pandas 2.0+ 使用 'ME', 老版本使用 'M'
    macro_df.rename(columns={'DATE': 'date'}, inplace=True)
    macro_df['date'] = macro_df['date'] + pd.offsets.MonthEnd(0)
    
    # 计算宏观状态因子
    macro_df['TERM'] = macro_df['T10Y3M']
    macro_df['DEF'] = macro_df['BAA'] - macro_df['AAA']
    macro_df['VIX'] = macro_df['VIXCLS']
    
    # 剔除冗余列并处理缺失值
    macro_df = macro_df[['date', 'TERM', 'DEF', 'VIX']].ffill().dropna()
    
    print(f"Saving Macro Data to SQLite... (Shape: {macro_df.shape})")
    macro_df.to_sql('macro_data', conn, if_exists='replace', index=False)
    
    # 建立索引
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_macro_date ON macro_data (date);")
    
    conn.close()
    print("=== FRED Pipeline Complete! ===")

if __name__ == "__main__":
    download_fred_to_sqlite()

import pandas as pd
import numpy as np
import sqlite3
from scipy.stats.mstats import winsorize
import warnings
warnings.filterwarnings('ignore')

class AssetPricingPipeline:
    def __init__(self, db_path='tail_risk_data.db'):
        self.db_path = db_path
        self.df = None
        self.feature_cols = []
        
    def load_and_build_features(self):
        """
        从 SQLite 极速读取，并执行严格的特征工程和目标标准化
        """
        print("Loading data from SQLite database...")
        conn = sqlite3.connect(self.db_path)
        # SQLite 存储的时间是字符串，使用 parse_dates 直接转换为 datetime
        df_micro = pd.read_sql("SELECT * FROM micro_data", conn, parse_dates=['date', 'datadate', 'linkdt', 'linkenddt'])
        df_macro = pd.read_sql("SELECT * FROM macro_data", conn, parse_dates=['date'])
        conn.close()
        
        print("Executing Feature Engineering...")
        # 确保排序，这对时序计算至关重要
        df = df_micro.sort_values(['permno', 'date']).reset_index(drop=True)
        
        # --- 1. 底层截面特征 ---
        df['Size'] = np.log(df['me'])
        df['BM'] = df['ceq'] / df['me'] 
        df['BM'] = np.where(df['BM'] < 0, np.nan, df['BM'])
        df['OP'] = (df['revt'] - df['cogs'].fillna(0) - df['xsga'].fillna(0) - df['xint'].fillna(0)) / df['ceq']
        
        df['at_lag1'] = df.groupby('permno')['at'].shift(1) 
        df['INV'] = (df['at'] - df['at_lag1']) / df['at_lag1']
        df['MOM1m'] = df.groupby('permno')['ret'].shift(1)
        
        df['log_ret'] = np.log(1 + df['ret'])
        df['MOM12m'] = df.groupby('permno')['log_ret'].apply(
            lambda x: x.shift(2).rolling(window=11).sum()
        ).reset_index(level=0, drop=True)
        df['MOM12m'] = np.exp(df['MOM12m']) - 1
        
        # 与宏观数据合并
        df = pd.merge(df, df_macro, on='date', how='inner')
        
        # --- 2. 目标收益率构建 (Target Normalization) ---
        # 预测下一期收益率
        df['target_ret_raw'] = df.groupby('permno')['ret'].shift(-1)
        
        # 用 VIX 缩放目标收益率 (消除宏观异方差)
        df['target_ret_scaled'] = df['target_ret_raw'] / (df['VIX'] / 100 / np.sqrt(12))
        
        # 截面缩尾 (Winsorize) - 防止极值破坏 MDN 的 NLL 损失
        def cs_winsorize(group):
            if len(group.dropna()) > 50:
                return winsorize(group, limits=[0.01, 0.01])
            return group
        
        df['target_ret_final'] = df.groupby('date')['target_ret_scaled'].transform(cs_winsorize)
        
        # --- 3. 张量特征展开与截面秩标准化 ---
        micro_feats = ['Size', 'BM', 'OP', 'INV', 'MOM1m', 'MOM12m']
        macro_feats = ['TERM', 'DEF'] 
        
        tensor_cols = []
        for micro in micro_feats:
            norm_col = f'{micro}_norm'
            df[norm_col] = df.groupby('date')[micro].transform(
                lambda x: (x.rank() - 1) / (len(x.dropna()) - 1) * 2 - 1
            ).fillna(0) 
            tensor_cols.append(norm_col)
            
            # 张量交互: 允许 Beta 时变
            for macro in macro_feats:
                interact_name = f'{norm_col}_x_{macro}'
                df[interact_name] = df[norm_col] * df[macro]
                tensor_cols.append(interact_name)
                
        self.feature_cols = tensor_cols
        # 清除无法作为训练集的行（缺乏目标或特征）
        self.df = df.dropna(subset=['target_ret_final'] + self.feature_cols)
        print(f"Pipeline ready. Total Tensor Features: {len(self.feature_cols)}")

    def expanding_window_generator(self, initial_train_years=20, val_years=2, test_years=1):
        """
        纯样本外扩展窗口生成器
        """
        dates = np.sort(self.df['date'].unique())
        
        initial_train_months = initial_train_years * 12
        val_months = val_years * 12
        test_months = test_years * 12
        
        start_idx = 0
        current_split_idx = initial_train_months
        
        while current_split_idx + val_months + test_months <= len(dates):
            train_dates = dates[start_idx : current_split_idx]
            val_dates = dates[current_split_idx : current_split_idx + val_months]
            test_dates = dates[current_split_idx + val_months : current_split_idx + val_months + test_months]
            
            train_data = self.df[self.df['date'].isin(train_dates)]
            val_data = self.df[self.df['date'].isin(val_dates)]
            test_data = self.df[self.df['date'].isin(test_dates)]
            
            # 修复：使用 pd.Timestamp() 包装 numpy.datetime64 对象
            window_info = {
                'train': (pd.Timestamp(train_dates[0]).strftime('%Y-%m'), pd.Timestamp(train_dates[-1]).strftime('%Y-%m')),
                'val': (pd.Timestamp(val_dates[0]).strftime('%Y-%m'), pd.Timestamp(val_dates[-1]).strftime('%Y-%m')),
                'test': (pd.Timestamp(test_dates[0]).strftime('%Y-%m'), pd.Timestamp(test_dates[-1]).strftime('%Y-%m'))
            }
            
            yield window_info, train_data, val_data, test_data
            current_split_idx += test_months

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import sqlite3
import json
import copy
from tqdm import tqdm

# ==========================================
# 0. 自动设备检测 (支持 M1 GPU, Nvidia GPU 和 CPU)
# ==========================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ==========================================
# 1. 混合密度网络 (MDN) 核心架构
# ==========================================
class MDN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_components=3):
        super(MDN, self).__init__()
        self.num_components = num_components
        
        # 共享特征提取层 (Shared Representation)
        # 考虑到金融数据的低信噪比，加入 Dropout 和 BatchNorm 防止严重过拟合
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(), # ELU 比 ReLU 在处理金融特征时更能保留负向信号，且避免神经元死亡
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(0.2)
        )
        
        # 三个独立的多头输出层 (分别输出 权重、均值、方差)
        self.pi_head = nn.Linear(hidden_dim // 2, num_components)
        self.mu_head = nn.Linear(hidden_dim // 2, num_components)
        self.sigma_head = nn.Linear(hidden_dim // 2, num_components)

    def forward(self, x):
        h = self.feature_extractor(x)
        
        # 1. 权重 pi: 不在这里做 Softmax，我们在 loss 函数里用 log_softmax 保证数值稳定
        pi_logits = self.pi_head(h) 
        
        # 2. 均值 mu: 直接输出，无约束
        mu = self.mu_head(h)
        
        # 3. 标准差 sigma: 使用 ELU + 1 + epsilon 保证严格为正，且比 exp() 更不容易爆炸
        # EPSILON (1e-6) 是防爆破的关键底线
        sigma = nn.functional.elu(self.sigma_head(h)) + 1.0 + 1e-6 
        
        return pi_logits, mu, sigma

# ==========================================
# 2. 负对数似然损失函数 (NLL Loss)
# ==========================================
def mdn_nll_loss(pi_logits, mu, sigma, target):
    """
    使用 Log-Sum-Exp 技巧计算 NLL，绝对杜绝 NaN
    target: shape (batch_size, 1)
    """
    # 将 pi_logits 转化为 log(pi)
    log_pi = torch.log_softmax(pi_logits, dim=-1) # (batch, K)
    
    # 构造标准高斯分布对象 (利用 PyTorch 自带的 Normal 分布算 log_prob 更稳定)
    normal_dist = torch.distributions.Normal(mu, sigma)
    
    # 计算 log(N(y | mu, sigma))
    # target 扩展为 (batch, K) 以便与 mu, sigma 对齐计算
    log_normal = normal_dist.log_prob(target.expand_as(mu))
    
    # 混合概率的对数: log(pi * N) = log(pi) + log(N)
    log_mix = log_pi + log_normal
    
    # 核心技巧: log(sum(exp(log_mix))) 计算总体概率的对数
    loss = -torch.logsumexp(log_mix, dim=-1) 
    
    return loss.mean()
# ==========================================
# 3. 模型训练与 Early Stopping 闭环
# ==========================================
def train_mdn(X_train, y_train, X_val, y_val, k_components, epochs=1, batch_size=256, lr=1e-3, patience=1):
    """
    训练单个特定 K 值的 MDN 模型
    """
    input_dim = X_train.shape[1]
    # 模型移动到设备
    model = MDN(input_dim=input_dim, num_components=k_components).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) # AdamW 加入权重衰减防过拟合
    
    # 转换为 PyTorch Tensors
    X_tr, y_tr = torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1)
    X_va, y_va = torch.FloatTensor(X_val).to(device), torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            # 数据移动到设备
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pi_logits, mu, sigma = model(batch_x)
            loss = mdn_nll_loss(pi_logits, mu, sigma, batch_y)
            loss.backward()
            
            # 梯度裁剪 (Gradient Clipping)，防爆破的第二道防线
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        with torch.no_grad():
            pi_logits_val, mu_val, sigma_val = model(X_va)
            val_loss = mdn_nll_loss(pi_logits_val, mu_val, sigma_val, y_va).item()
            
        # Early Stopping 逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                # print(f"    [Early Stop] Epoch {epoch}: Val NLL = {best_val_loss:.4f}")
                break
                
    # 恢复最优权重
    model.load_state_dict(best_model_state)
    return model, best_val_loss

# ==========================================
# 4. 动态 K 值寻优 (Dynamic K Search)
# ==========================================
def find_best_mdn(X_train, y_train, X_val, y_val, k_candidates=[1, 3, 5, 7]):
    """
    在当前 Rolling Window 的验证集上，寻找表现最好的 K 个正态分布组合
    """
    best_k = None
    best_loss = float('inf')
    best_model = None
    
    print(f"    -> Searching best K in {k_candidates}...")
    for k in k_candidates:
        model, val_loss = train_mdn(X_train, y_train, X_val, y_val, k_components=k)
        print(f"       K={k} | Val NLL: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_k = k
            best_model = model
            
    print(f"    => Selected K={best_k} for the upcoming Test Window.")
    return best_model, best_k

# ==========================================
# 5. 主程序调用示例 (接续阶段一的流水线)
# ==========================================
if __name__ == "__main__":
    db_path = 'tail_risk_data.db'
    
    # 1. 实例化阶段一的数据管道，读取真实数据并完成特征工程
    print(">>> [1/3] Loading Real Data from SQLite and Building Features...")
    # 假设 AssetPricingPipeline 已经在前文定义
    pipeline = AssetPricingPipeline(db_path=db_path)
    pipeline.load_and_build_features()
    
    # 2. 初始化 Expanding Window 生成器
    generator = pipeline.expanding_window_generator(initial_train_years=15, val_years=2, test_years=1)
    
    # 用于收集所有纯样本外预测结果的容器
    all_out_of_sample_predictions = []
    
    print("\n>>> [2/3] Starting Expanding Window MDN Training on Real Data...")
    
    # 遍历每一个时间窗口
    for window_idx, (info, train_df, val_df, test_df) in enumerate(generator):
        print(f"\n[Window {window_idx + 1}] Train: {info['train'][0]}~{info['train'][1]} | Val: {info['val'][0]}~{info['val'][1]} | Test: {info['test'][0]}~{info['test'][1]}")
        
        # 提取当前窗口的 Numpy 矩阵
        X_train = train_df[pipeline.feature_cols].values
        Y_train = train_df['target_ret_final'].values
        X_val = val_df[pipeline.feature_cols].values
        Y_val = val_df['target_ret_final'].values
        X_test = test_df[pipeline.feature_cols].values
        
        print(f"   Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
        
        # 动态寻优寻找该窗口期下的最佳 K 值
        best_model, best_k = find_best_mdn(X_train, Y_train, X_val, Y_val, k_candidates=[1, 3, 5])
        print(f"   => Best K selected: {best_k}")
        
        # ----------------------------------------------------
        # 最关键的一步：对 Test 集合（样本外）进行预测并记录
        # ----------------------------------------------------
        best_model.eval()
        with torch.no_grad():
            # 移动 Test 数据到 GPU
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            pi_logits, mu, sigma = best_model(X_test_tensor)
            # 将 logits 转换为概率
            pi = torch.softmax(pi_logits, dim=-1)
            
            # 回传到 CPU 并转为 Numpy 以便后续存入数据库
            pi_np = pi.cpu().numpy()
            mu_np = mu.cpu().numpy()
            sigma_np = sigma.cpu().numpy()
            
        # 构建当前窗口的预测结果表
        # 为了应对动态 K 导致的数组长度不一，我们将参数数组 JSON 序列化为字符串
        preds_df = test_df[['permno', 'date', 'target_ret_final']].copy()
        preds_df['best_k'] = best_k
        
        # json.dumps 会把 [0.2, 0.8] 变成字符串 "[0.2, 0.8]" 存入数据库
        preds_df['pi_vec'] = [json.dumps(vec.tolist()) for vec in pi_np]
        preds_df['mu_vec'] = [json.dumps(vec.tolist()) for vec in mu_np]
        preds_df['sigma_vec'] = [json.dumps(vec.tolist()) for vec in sigma_np]
        
        all_out_of_sample_predictions.append(preds_df)
        
        # (为了节约演示时间，如果你只是测试 pipeline 是否跑通，可以加上 break)
        if window_idx == 2: break 

    # 3. 将所有样本外预测结果拼接，并永久保存至 SQLite3
    print("\n>>> [3/3] Saving Out-of-Sample Predictions to SQLite3...")
    final_predictions_df = pd.concat(all_out_of_sample_predictions, ignore_index=True)
    
    conn = sqlite3.connect(db_path)
    final_predictions_df.to_sql('mdn_predictions', conn, if_exists='replace', index=False)
    
    # 建立索引，方便阶段三极速读取
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_date ON mdn_predictions (date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_permno ON mdn_predictions (permno);")
    conn.close()
    
    print(f"=== Phase 2 Complete! Saved {len(final_predictions_df)} OOS predictions to 'mdn_predictions' table. ===")


import pandas as pd
import numpy as np
import sqlite3
import json
import torch
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# =================================================================
# 1. 核心数学工具：混合高斯分布的 CDF 和 ES 计算
# =================================================================

def mdn_cdf(y, pi, mu, sigma):
    """
    计算混合高斯分布在 y 处的累积概率 P(Y <= y)
    公式: F(y) = sum( pi_k * Phi((y - mu_k) / sigma_k) )
    """
    # norm.cdf 是标准正态分布的累积函数
    return np.sum(pi * norm.cdf((y - mu) / sigma))

def mdn_var_objective(y, pi, mu, sigma, alpha):
    """
    求解 VaR 的目标函数：F(y) - alpha = 0
    """
    return mdn_cdf(y, pi, mu, sigma) - alpha

def calculate_es_analytical(alpha, pi, mu, sigma):
    """
    混合高斯分布的 Expected Shortfall (ES) 解析解
    公式参考: ES_alpha = -(1/alpha) * sum( pi_k * mu_k * Phi(d1) - sigma_k * phi(d1) )
    其中 d1 是标准化后的 VaR 阈值
    """
    # 1. 首先通过数值求根找到 VaR (使得累积概率等于 alpha)
    # 搜索区间设定在均值的上下 10 倍标准差之间，确保覆盖尾部
    try:
        lower_bound = np.min(mu - 10 * sigma)
        upper_bound = np.max(mu + 10 * sigma)
        var_alpha = brentq(mdn_var_objective, lower_bound, upper_bound, args=(pi, mu, sigma, alpha))
    except:
        # 如果求根失败（极罕见），使用加权均值作为退路
        return np.sum(pi * mu), np.sum(pi * mu)

    # 2. 计算每个组件在 VaR 处的贡献
    # d = (VaR - mu) / sigma
    d = (var_alpha - mu) / sigma
    
    # phi 是标准正态概率密度函数，Phi 是标准正态累积分布函数
    term1 = mu * norm.cdf(d)
    term2 = sigma * norm.pdf(d)
    
    # ES = -(1/alpha) * Integral from -inf to VaR of (y * f(y)) dy
    # 对于混合高斯，积分具有解析形式
    es_val = -(1/alpha) * np.sum(pi * (term1 - term2))
    
    return var_alpha, es_val

def calculate_crps_numerical(y_true, pi, mu, sigma):
    """
    通过数值积分计算混合高斯分布的 CRPS
    """
    # 1. 构建积分网格 z
    # 因为我们的目标收益率已经用 VIX 缩放过，绝大多数值落在 [-5, 5] 之间。
    # 设置 [-10, 10] 的区间和 1000 个格点，足以保证极高的积分精度。
    z = np.linspace(-10, 10, 1000)
    
    # 2. 计算网格上每一个点的预测累积概率 F(z)
    cdf_z = np.zeros_like(z)
    for k in range(len(pi)):
        # 向量化计算：算出每一个网格点的正态 CDF 并加权
        cdf_z += pi[k] * norm.cdf((z - mu[k]) / sigma[k])
        
    # 3. 真实发生的阶跃函数 (Heaviside step function)
    # 当 z 小于真实收益率时为 0，大于等于时为 1
    step_z = (z >= y_true).astype(float)
    
    # 4. 计算平方差
    squared_diff = (cdf_z - step_z) ** 2
    
    # 5. 使用梯形法则 (Trapezoidal rule) 计算定积分
    crps_val = np.trapezoid(squared_diff, z)
    
    return crps_val

# =================================================================
# 2. 主程序：从 SQLite 读取数据并执行计算
# =================================================================

db_path = 'tail_risk_data.db'
conn = sqlite3.connect(db_path)

print(">>> Loading predictions from SQLite...")
# 读取阶段二存入的 mdn_predictions 表
df_preds = pd.read_sql("SELECT * FROM mdn_predictions", conn)
conn.close()

# 用于保存计算出的风险因子
results = []

print(">>> Extracting Tail Risk Factors (ES & VaR)...")
# 使用 tqdm 显示进度
for idx, row in tqdm(df_preds.iterrows(), total=len(df_preds)):
    # 1. 解码 JSON 字符串回 Numpy 数组
    pi = np.array(json.loads(row['pi_vec']))
    mu = np.array(json.loads(row['mu_vec']))
    sigma = np.array(json.loads(row['sigma_vec']))
    y_true = row['target_ret_final'] # 真实发生的（缩放后的）收益率
    
    # 2. 计算 5% 分位数的 VaR 和 ES
    alpha = 0.05
    var_5, es_5 = calculate_es_analytical(alpha, pi, mu, sigma)
    
    # 3. 计算 PIT (Probability Integral Transform)
    # 即：真实收益率 y_true 在预测分布中处于什么百分位
    pit_val = mdn_cdf(y_true, pi, mu, sigma)

    # 新增：计算 CRPS
    crps_val = calculate_crps_numerical(y_true, pi, mu, sigma)

    # 4. 计算分布的高阶矩 (用于控制变量)
    # 混合分布均值: sum(pi * mu)
    mean_pred = np.sum(pi * mu)
    # 混合分布方差: sum(pi * (mu^2 + sigma^2)) - mean^2
    var_pred = np.sum(pi * (mu**2 + sigma**2)) - mean_pred**2
    
    results.append({
        'permno': row['permno'],
        'date': row['date'],
        'es_5': es_5,       
        'var_5': var_5,     
        'pit': pit_val,     
        'crps': crps_val,   # <--- 把 CRPS 也存进最终的 DataFrame 里
        'mean_pred': mean_pred,
        'vol_pred': np.sqrt(var_pred),
        'realized_ret': y_true
    })

df_factors = pd.DataFrame(results)

# =================================================================
# 3. 分布校准检验 (PIT Test) - 顶刊必备图表
# =================================================================

plt.figure(figsize=(12, 5))

# 图 A: PIT 直方图 (理想状态下应接近均匀分布/水平线)
plt.subplot(1, 2, 1)
sns.histplot(df_factors['pit'], bins=20, kde=False, stat="density", color='skyblue')
plt.axhline(y=1, color='red', linestyle='--') # 理论均匀分布线
plt.title("PIT Histogram (Calibration Check)")
plt.xlabel("Probability Integral Transform (PIT)")

# 图 B: PIT QQ-Plot (理想状态下应在对角线上)
plt.subplot(1, 2, 2)
sorted_pit = np.sort(df_factors['pit'])
uniform_dist = np.linspace(0, 1, len(sorted_pit))
plt.plot(uniform_dist, sorted_pit, color='blue', label='MDN Model')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfect Calibration')
plt.title("PIT QQ-Plot")
plt.legend()

plt.tight_layout()
plt.show()

# =================================================================
# 4. 因子入库：保存最终的定价因子表
# =================================================================

print(">>> Saving final factor table to SQLite...")
conn = sqlite3.connect(db_path)
# 这一张表 df_factors 将是阶段四（组合回测和回归）的唯一输入
df_factors.to_sql('final_pricing_factors', conn, if_exists='replace', index=False)
conn.close()

print("=== Phase 3 Complete! Factors are ready for Backtesting. ===")

# =================================================================
# 5. CRPS 深度诊断与可视化 (引入传统历史基准线)
# =================================================================
import matplotlib.dates as mdates
from scipy.stats import norm

print("\n>>> Calculating Traditional Benchmark CRPS <<<")

# 1. 构建基准模型的均值和方差 (过去 12 个月的滚动历史)
# 为了严格防止前视偏差，我们必须先按股票和时间排序
df_factors = df_factors.sort_values(['permno', 'date'])

# realized_ret 其实是 t+1 的预测目标，所以我们把它滞后一期 (shift) 变成真正已知的历史观测值
df_factors['ret_lag'] = df_factors.groupby('permno')['realized_ret'].shift(1)

# 计算过去 12 个月（至少需要 3 个月数据）的滚动均值和标准差
df_factors['mu_bench'] = df_factors.groupby('permno')['ret_lag'].transform(lambda x: x.rolling(12, min_periods=3).mean())
df_factors['sigma_bench'] = df_factors.groupby('permno')['ret_lag'].transform(lambda x: x.rolling(12, min_periods=3).std())

# 对于刚上市或数据太少的截断点，我们用当月全市场的截面均值/波动率来兜底填充
df_factors['mu_bench'] = df_factors['mu_bench'].fillna(df_factors.groupby('date')['mu_bench'].transform('mean')).fillna(0)
df_factors['sigma_bench'] = df_factors['sigma_bench'].fillna(df_factors.groupby('date')['sigma_bench'].transform('mean')).fillna(0.1)
# 防止波动率出现 0 导致除以零崩溃
df_factors['sigma_bench'] = np.maximum(df_factors['sigma_bench'], 1e-6)

# 2. 使用单一正态分布的 CRPS 解析解公式极速计算
def crps_normal_analytical(y, mu, sigma):
    """单正态分布 CRPS 的封闭解析解 (Gneiting et al., 2005)"""
    z = (y - mu) / sigma
    # 公式: sigma * [ z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))

df_factors['crps_bench'] = crps_normal_analytical(
    df_factors['realized_ret'].values, 
    df_factors['mu_bench'].values, 
    df_factors['sigma_bench'].values
)

# 3. 打印统计输出 (Skill Score)
mean_crps_mdn = df_factors['crps'].mean()
mean_crps_bench = df_factors['crps_bench'].mean()
skill_score = (1 - mean_crps_mdn / mean_crps_bench) * 100

print("\n>>> CRPS Diagnostic Summary <<<")
print(f"Benchmark (12m Historical Normal) Mean CRPS : {mean_crps_bench:.4f}")
print(f"MDN (Dynamic Mixture Network) Mean CRPS     : {mean_crps_mdn:.4f}")
print(f"⭐ MDN Skill Score (Error Reduction)          : {skill_score:.2f}%")

# 4. 绘制对比走势图
monthly_crps = df_factors.groupby('date')[['crps', 'crps_bench']].mean().reset_index()

# 【修复关键点】：强制转为时间序列对象，让 Matplotlib 认识它
monthly_crps['date'] = pd.to_datetime(monthly_crps['date'])

plt.figure(figsize=(14, 6))

# 画 Benchmark 基准线
plt.plot(monthly_crps['date'], monthly_crps['crps_bench'], color='crimson', linewidth=1.5, linestyle='--', alpha=0.8, label='Benchmark (12m Historical Normal)')

# 画 MDN 模型线
plt.plot(monthly_crps['date'], monthly_crps['crps'], color='teal', linewidth=2, label='MDN (Dynamic Mixture Network)')

plt.title("Cross-Sectional Average CRPS: MDN vs. Traditional Benchmark", fontsize=15, fontweight='bold')
plt.ylabel("CRPS (Lower is Better)", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, loc='upper left')

# 标注著名的市场危机节点 (现在 X 轴是真正的时间轴了，axvline 可以完美运作)
plt.axvline(pd.to_datetime('2008-09-30'), color='red', alpha=0.5, linestyle=':', label='Lehman Bankruptcy')
if pd.to_datetime('2020-03-31') <= monthly_crps['date'].max(): # 确保你的数据里包含2020年
    plt.axvline(pd.to_datetime('2020-03-31'), color='purple', alpha=0.5, linestyle=':', label='COVID-19 Crash')

# 优化 X 轴日期显示
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.YearLocator()) # 每年标一个年份，更清爽
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.show()
