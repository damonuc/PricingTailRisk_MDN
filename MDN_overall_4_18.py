import pandas as pd
import numpy as np
import sqlite3
import os
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', 100)     # 显示前100行
# list the tables in the database to verify
wrds_path = '/Users/jianbinchen/NonSync/GitHub/Research/DataSet/WRDS.sqlite'
# set current working directory to the location of the database file
os.chdir('/Users/jianbinchen/NonSync/GitHub/Research/DCDE')

file_path='/Users/jianbinchen/NonSync/GitHub/Research/DCDE/data/gwz_Data2024.xlsx'
df = pd.read_excel(file_path, sheet_name='Monthly')

# 日期
df['yyyymm'] = df['yyyymm'].astype(str)
df['date'] = pd.to_datetime(df['yyyymm'], format='%Y%m') + pd.offsets.MonthEnd(0)

# 选择的宏观因子
'''
* **12 tbl:** 短期国库券利率 (T-bill)。
* **19 tms (Term Spread):** 期限利差 (lty - tbl)。长短期利率倒挂是衰退的最强前兆。
* **20 dfy (Default Yield Spread):** 违约收益率差 (BAA - AAA)。衡量市场对企业破产的恐慌度。
* **21 dfr (Default Return Spread):** 违约回报率差 (corpr - ltr)。衡量企业债相对于长期国债的风险溢价。
* **22 infl:** 通货膨胀率。
* **25 svar ($\sigma^2$):** 历史股票方差（经典变量）。
* **30 vp, 方差溢价 (Variance Premium)
* **31 impvar, 隐含波动率
* **32 vrp:** 方差风险溢价。期权市场定价和实际波动的差值，是预测崩盘的神器。
* **35 skew & 43 skvw:** 偏度 (Skewness) 和股票平均偏度。捕捉市场收益分布的非对称性。
* **44 tail (X-sect tail risk):** 横截面尾部风险！这简直是你 $ES_{5\%}$ 的宏观对标物。
* **49 rdsp (Stock return dispersion):** 股票收益率分散度。横截面上股票各自为战的程度。
* **50 rsvix:** 缩放后的风险中性 VIX 指数。
* **55 avgcor:** 股票收益率的平均相关性。危机时相关性通常会飙升（“所有东西都在跌”）。
* **14 d/p (Dividend Price Ratio):** 股息价格比 (d12/price)。
* **15 d/y (Dividend Yield):** 股息率 (d12 / 上期price)。
* **16 e/p (Earnings Price Ratio):** 盈利价格比 (e12/price)。
* **17 d/e (Dividend Payout Ratio):** 股息支付率 (d12/e12)。企业利润中有多少拿来分红。
* **18 b/m (Book-to-Market):** 宏观账面市值比（道琼斯工业平均指数的 B/M）。
* **23 eqis & 24 ntis:** 股权发行比例和净股权发行。管理层是“聪明钱”，疯狂发股票圈钱往往意味着大盘见顶。
* **7 AAA & 8 BAA:** 最高信用评级 (AAA) 和中等信用评级 (BAA) 的企业债收益率。
* **9 lty & 10 ltr:** 长期国债的收益率 (Yield) 和实际回报率 (Return)。
* **57 disag:** 分析师预测分歧度 (Analyst disagreement)。分歧越大，往往未来收益越差。
'''

required_cols=['AAA', 'BAA', 'lty', 'ltr', 'tbl', 'd/p', 'd/y', 'e/p', 'd/e', 'b/m', 'ntis', 'svar',  
               'disag', 'skvw', 'tail', 'rdsp','avgcor','tms', 'dfy', 'dfr', 'infl']

# 增加新的数据列
required_cols_1996 =['impvar',  'rsvix'] # 从1996年1月才有, 先不加
required_cols_2021 =['vrp','vp'] #数据只到2021年末的列, 可以不加, 看情况
# required_cols=required_cols+required_cols_1996+required_cols_2021

# 数值化
for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 提取
macro_df = df[['date'] + required_cols].copy()

# 排序
macro_df = macro_df.sort_values('date').reset_index(drop=True)

# lag（防未来信息）
lag_vars = ['infl', 'ntis', 'b/m']
for col in lag_vars:
    if col in macro_df.columns:
        macro_df[col] = macro_df[col].shift(1)

# 删除缺失
macro_df = macro_df.dropna().reset_index(drop=True)


# crsp_wrds_msfv2_query_1990=pd.read_parquet('./data/crsp_wrds_msfv2_query_1990.parquet', engine='pyarrow',columns=['permno',	'mthcaldt',	'mthret',	'mthprc',	'mthcap',	'mthvol'])
crsp_wrds_msfv2_query_1990=pd.read_parquet('./data/crsp_wrds_msfv2_query_1990.parquet', engine='pyarrow')


mdn_features_wrds = [
    # 破产与债务
    'short_debt', 'intcov_ratio', 'debt_at', 'debt_ebitda',
    # 流动性与造假预警
    'ocf_lct', 'cash_ratio', 'accrual', 
    # 估值泡沫
    'bm', 'pe_exi', 'evm', 'divyield',
    # 盈利与效率
    'roe', 'gpm', 'at_turn', 'rd_sale'  # rd_sale (研发占比) 对美股科技股防雷很有用
]
comp_wrds_ratios= pd.read_parquet('./data/comp_wrds_ratios.parquet', engine='pyarrow',columns=['gvkey', 'public_date']+mdn_features_wrds)


# merge crsp_wrds_msfv2_query_1990 and comp_wrds_ratios by ccm link
with sqlite3.connect(wrds_path) as db_conn:
    crsp_ccmxpf_linktable = pd.read_sql('SELECT * FROM crsp_ccmxpf_linktable', db_conn, parse_dates=['linkdt','linkenddt'])
crsp_ccmxpf_linktable['linkenddt'] = pd.to_datetime(crsp_ccmxpf_linktable['linkenddt']).fillna(pd.to_datetime('2099-12-31'))
crsp_ccmxpf_linktable['lpermno'] = crsp_ccmxpf_linktable['lpermno'].astype("Int64").astype(str).replace("<NA>", pd.NA).str.strip()
crsp_ccmxpf_linktable['gvkey'] = crsp_ccmxpf_linktable['gvkey'].astype(str).str.zfill(6)
crsp_wrds_msfv2_query_1990['permno'] = crsp_wrds_msfv2_query_1990['permno'].astype("Int64").astype(str).replace("<NA>", pd.NA).str.strip()
comp_wrds_ratios['gvkey'] = comp_wrds_ratios['gvkey'].astype(str).str.zfill(6)
ccm = crsp_ccmxpf_linktable[(crsp_ccmxpf_linktable['linktype'].isin(['LU', 'LC'])) & (crsp_ccmxpf_linktable['linkprim'].isin(['P', 'C']))]
ccm = ccm.rename(columns={'lpermno': 'permno'})
comp_ccm = pd.merge(comp_wrds_ratios, ccm[['gvkey', 'permno', 'linkdt', 'linkenddt']], on='gvkey', how='inner')
comp_ccm = comp_ccm[(comp_ccm['public_date'] >= comp_ccm['linkdt']) & 
                    (comp_ccm['public_date'] <= comp_ccm['linkenddt'])]

comp_ccm['public_date'] = comp_ccm['public_date'] + pd.offsets.MonthEnd(0) # 把日期推到月底，和CRSP日期对齐
crsp_wrds_msfv2_query_1990['mthcaldt_next'] = crsp_wrds_msfv2_query_1990['mthcaldt'] + pd.offsets.MonthEnd(0)# 把日期推到月底
merged_df = pd.merge(
    crsp_wrds_msfv2_query_1990,
    comp_ccm,
    left_on=['permno', 'mthcaldt_next'],
    right_on=['permno', 'public_date'],
    how='left'
)
merged_df = merged_df.sort_values(['permno', 'public_date']).reset_index(drop=True)
merged_df['mthret_lead1'] = merged_df.groupby('permno')['mthret'].shift(-1)
merged_df['turnover_1m'] = merged_df.groupby('permno')['mthvol'].shift(-1) / merged_df.groupby('permno')['shrout'].shift(-1).replace(0, np.nan)
merged_df['mthcap_log'] = np.log(merged_df['mthcap'].replace(0, np.nan))
# 特征 X4：MOM12m (过去12个月动量，通常跳过最近1个月，即 t-12 到 t-2)
merged_df['log_ret'] = np.log1p(merged_df['mthret'].fillna(0))
# 使用 shift(1) 跳过本月，然后 rolling 11 个月求和
merged_df['MOM12m'] = merged_df.groupby('permno')['log_ret'].transform(
    lambda x: x.shift(1).rolling(window=11, min_periods=8).sum()
)
merged_df['MOM12m'] = np.exp(merged_df['MOM12m']) - 1
merged_df = merged_df.drop(columns=['log_ret']) # 用完即弃
features_to_fill = ['short_debt', 'intcov_ratio', 'debt_at', 'debt_ebitda', 'ocf_lct', 'cash_ratio', 'accrual', 'bm', 'pe_exi', 'evm', 'divyield', 'roe', 'gpm', 'at_turn', 'rd_sale']
for feature in features_to_fill:
    merged_df[feature] = merged_df[feature].fillna(merged_df.groupby('mthcaldt')[feature].transform('mean'))


#--------------------------------------------------                         
#下面的代码请重点修改, 我不再需要生成新的feature, 并且使用结构化MDN, 宏观因子预测pi, 微观因子预测mu和sigma
#--------------------------------------------------


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
        with sqlite3.connect(self.db_path) as conn:
            # SQLite 存储的时间是字符串，使用 parse_dates 直接转换为 datetime
            df_micro = pd.read_sql("SELECT * FROM micro_data", conn, parse_dates=['date', 'datadate', 'linkdt', 'linkenddt'])
            df_macro = pd.read_sql("SELECT * FROM macro_data", conn, parse_dates=['date'])
        
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
        
        # 截面缩尾 (Winsorize) - 防止极值破坏 MDN 的 NLL 损失, Winsorize 是将
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
def train_mdn(X_train, y_train, X_val, y_val, k_components, epochs=999, batch_size=256, lr=1e-3, patience=20):
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
    wrds_path = '/Users/jianbinchen/NonSync/GitHub/Research/DataSet/WRDS.sqlite'
    
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
        
        # # 动态寻优寻找该窗口期下的最佳 K 值
        # best_model, best_k = find_best_mdn(X_train, Y_train, X_val, Y_val, k_candidates=[1, 3, 5])
        # print(f"   => Best K selected: {best_k}")

        # 全局固定 K=5，直接训练，极大节省算力
        best_k = 5
        print(f"   => Training MDN with fixed K={best_k}...")
        best_model, val_loss = train_mdn(X_train, Y_train, X_val, Y_val, k_components=best_k)
        
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