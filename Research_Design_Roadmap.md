# Structured MDN、股票尾部风险与预期收益：研究设计与实施路线图

> 当前版本：2026-07-31  
> 文档用途：固定研究问题、模型比较、训练协议、预测评价和资产定价检验。后续可以在此基础上扩展成论文正文。

## 一、研究设计摘要

本文研究一个核心问题：

> 一个区分共同宏观条件 mixture weight 和公司特定损失程度的结构化混合密度网络，能否更准确地预测股票的条件尾部风险？这种预测尾部风险是否获得未来收益补偿，风险补偿主要来自哪一个经济维度？

完整研究分成三个阶段：

1. 固定微观数据、宏观数据、行业分类、预测目标和样本外时间设计。
2. 比较 Rolling Normal、Linear Quantile、Fused Normal K1、Fused Normal K3 和 Structured Normal K3 的样本外分布与尾部预测表现。
3. 使用 Structured Normal K3 分解公司的预测 ES，检验总尾部风险及其不同经济组成部分是否获得未来收益补偿。

本文的主要贡献不应仅表述为“神经网络更准确地预测 ES”。更有经济意义的贡献是：

> Structured MDN 将公司层面的预测尾部风险分解为由宏观信息决定、同月所有公司共享的 adverse-component mixture weight，以及由公司特征决定的 adverse-component 条件损失程度，并检验这种公司层面的预测尾部风险是否获得横截面预期收益补偿，以及补偿是否随 adverse-component weight 变化。

这里的 adverse component 是所有公司采用同一标签的宏观条件情景，不等同于模型已经识别出一个由所有公司共同实现的真实 latent state。

样本外预测比较用于验证模型输出是否可靠；尾部风险分解及其资产定价结果才是论文的核心经济内容。

---

## 二、研究问题与可检验假设

### 2.1 研究问题

本文依次回答以下问题：

1. 单纯使用股票历史收益率，能否充分预测下一月的条件尾部风险？
2. 公司特征和宏观变量能否改善条件分位数与 ES 预测？
3. 神经网络的非线性是否比线性分位数回归更有效？
4. 混合分布是否比单一条件正态分布更有效？
5. 将宏观信息和公司信息映射到不同分布参数，是否比完全融合的网络更有效？
6. 预测 ES 是否与股票未来收益正相关？
7. 如果 ES 获得收益补偿，补偿来自共同的 adverse-component weight、公司在该 component 下的条件损失程度，还是两者的组合？

### 2.2 主要假设

#### H1：条件信息改善尾部预测

使用公司特征和宏观变量的条件模型，在样本外 Quantile Score、tail-CRPS、VaR coverage 和 joint VaR--ES score 上优于 Rolling Normal。

#### H2：mixture 改善尾部预测

Fused Normal K3 在下行尾部预测上优于 Fused Normal K1。这一比较识别 mixture 本身的增量作用。

#### H3：结构化参数映射改善尾部预测和经济解释

Structured Normal K3 在主要下行尾部评价指标上优于或至少不劣于 Fused Normal K3，并产生稳定、可解释的宏观条件 mixture components。

#### H4：预测 ES 获得未来收益补偿

在控制常见公司特征和风险变量后，预测 ES 较高的股票具有较高的下一月平均收益：

$$
E[R_{i,t+1}\mid ES_{i,t}\text{ high}]
>
E[R_{i,t+1}\mid ES_{i,t}\text{ low}].
$$

如果风险补偿假设成立，横截面回归中的 ES 系数应当为正。若结果为负，则需要将其解释为投资者偏好、安全需求、过度定价或其他机制，而不能继续称为正向风险补偿。

#### H5：ES 溢价主要来自公司的 adverse-component 损失暴露

公司在全局标记的 adverse component 下损失程度越大，未来平均收益越高。该收益差异在预测 adverse-component weight 较高的时期可能更强。

---

## 三、数据、信息时点与预测对象

### 3.1 观测单位

每条观测对应公司 $i$ 在月末 $t$ 的信息。定义：

$$
X_{i,t}=\text{公司层面微观变量},
$$

$$
M_t=\text{月度宏观变量},
$$

$$
J_{i,t}=\text{FF49 行业标签},
$$

$$
Y_{i,t+1}=R_{i,t+1}=\text{公司下一月原始收益率}.
$$

模型在月末 $t$ 使用 $X_{i,t}$、$M_t$ 和 $J_{i,t}$，预测 $R_{i,t+1}$ 的条件分布。

### 3.2 固定的数据范围

主分析暂时固定以下数据设计，不再根据模型表现或资产定价结果更换数据源：

- 34 个公司层面微观变量；
- 19 个宏观变量；
- FF49 行业分类；
- 下一月原始股票收益率作为预测目标；
- 当前已经确定的 CRSP/Compustat/FRED-MD 数据来源及清理规则；
- 当前已经确定的股票样本筛选规则。

数据源固定不代表数据质量问题可以忽略。正式运行前仍需确认：

- 每个特征在月末 $t$ 是否已经可用；
- `target_ret_final` 是否严格对应 $t+1$；
- 合并是否产生重复公司月份；
- 样本筛选是否造成幸存者偏差；
- 缺失值处理是否只使用当时可获得的信息；
- 宏观数据使用的 vintage 规则是否在论文中透明披露。

### 3.3 原始收益率和标准化

预测目标使用原始月收益率，不对 $Y_{i,t+1}$ 做横截面标准化。因此：

- 预测均值使用原始月收益率单位；
- VaR 使用原始收益率临界值；
- ES 使用原始月收益率损失单位；
- 不需要在预测后将结果反向转换回 raw return。

连续输入变量可以标准化，但均值和标准差只能由当前训练期估计：

$$
\widetilde X_{j,i,t}
=
\frac{X_{j,i,t}-\widehat\mu_{j,\mathrm{train}}}
{\widehat\sigma_{j,\mathrm{train}}}.
$$

同一组训练期均值和标准差应用于对应的 Validation 和 Test。

宏观变量在每个月对所有股票重复，因此宏观标准化统计量应当基于训练期的唯一月份计算，避免上市公司数量较多的月份被重复计权。

### 3.4 FF49 的模型特定编码

所有条件模型使用相同的行业信息，但编码方式可以根据模型性质不同：

- Linear Quantile：FF49 转换为行业 dummy，选一个行业作为参考组；
- Fused/Structured MDN：FF49 使用可训练 embedding；
- Rolling Normal：不使用行业信息。

不应把 `ffi49=1,\ldots,49` 直接当成连续数值，因为行业编号没有自然的大小和距离含义。

公平比较要求相同的信息集、信息截止时间、目标变量和测试样本，不要求所有模型采用相同的类别变量编码。

### 3.5 月份等权

不同月份的上市公司数量不同。若每个公司月份观测等权，公司数量较多的月份会对模型损失产生更大影响。

定义月份 $t$ 的公司数量为 $N_t$，公司 $i$ 在该月的基础权重为：

$$
w_{i,t}^{(0)}=\frac{1}{N_t}.
$$

因此每个月的总权重相同：

$$
\sum_{i=1}^{N_t}w_{i,t}^{(0)}=1.
$$

实现中可以将权重除以其全样本均值，使平均观测权重等于 1：

$$
w_{i,t}
=
\frac{w_{i,t}^{(0)}}
{\operatorname{mean}(w^{(0)})}.
$$

这一缩放不改变最优参数，只改善损失数值的可读性。所有条件模型原则上使用相同的 month-equal weighting。

---

## 四、样本外滚动设计

### 4.1 基本窗口

主分析采用固定长度 rolling window：

- Train：180 个月；
- Validation：12 个月；
- Test：随后 12 个月；
- 每次向前移动 12 个月；
- 最后不足 12 个月的 Test 可以保留，但需要在结果中注明。

例如：

| Window | Train | Validation | Test |
|---|---|---|---|
| 1 | 1991-01 至 2005-12 | 2006-01 至 2006-12 | 2007-01 至 2007-12 |
| 2 | 1992-01 至 2006-12 | 2007-01 至 2007-12 | 2008-01 至 2008-12 |
| 3 | 1993-01 至 2007-12 | 2008-01 至 2008-12 | 2009-01 至 2009-12 |

### 4.2 主分析不使用 warm start

主分析中，每一个 Fused 或 Structured 神经网络在每一个滚动窗口都从新的随机初始化开始训练，不继承上一个窗口的模型参数。

这样做的原因是：

1. 严格保持固定 180 个月训练窗口；
2. 防止模型参数保留已经离开当前滚动窗口的旧信息；
3. 使 Fused K1、Fused K3 和 Structured K3 的训练协议一致；
4. 使模型间比较更容易解释。

主分析使用固定随机种子保证可重复性。多个随机种子下的结果稳定性作为 robustness。

### 4.3 使用 Validation early stopping

每个神经网络窗口只训练一次：

1. 从头初始化模型；
2. 使用 180 个月 Train 更新模型参数；
3. 每个 epoch 后计算 12 个月 Validation loss；
4. 保存 Validation loss 最低时的模型权重；
5. 如果 Validation loss 连续若干个 epoch 没有改善，则停止训练；
6. 恢复最佳模型权重，并直接预测随后 12 个月 Test。

最佳 epoch 定义为：

$$
E_w^*
=
\arg\min_{e\in\{1,\ldots,E_{\max}\}}
L_{\mathrm{val},w}(e).
$$

最终用于 Test 预测的参数为最佳 checkpoint：

$$
\widehat\theta_w
=
\theta_{w,E_w^*}.
$$

Validation 不参与梯度更新，不与 Train 合并重新训练，也不用于报告最终样本外表现。它只决定何时停止训练以及保留哪个 checkpoint。这样可以避免第二次训练，并确保 Test 完全不参与模型选择。

Linear Quantile 采用相同的 early-stopping 逻辑`(为什么呢? 它不是closed form的吗? 到时候问问AI)`。Rolling Normal 没有需要 Validation 选择的训练轮数，因此直接使用 Test 之前最多 192 个月的历史信息。也就是说，Rolling Normal 可以直接使用这 192 个月估计均值和波动率，而神经网络只在 180 个月 Train 上更新权重、使用随后 12 个月选择 checkpoint。这一差异需要在论文中说明；它使 Rolling Normal 成为相对更强的基准。作为稳健性检验，可以另外将 Rolling Normal 的历史窗口限制为 180 个月。

### 4.4 Warm start robustness

Warm start 只作为稳健性检验：

- 新窗口从上一窗口最终模型权重开始；
- 使用新窗口的 Train/Validation 更新；
- 比较 warm start 是否改善预测分数、训练时间和 component 稳定性。

Warm start 模型具有超出固定 180 个月窗口的参数记忆，因此必须与主分析分开报告，不能将其结果混入主要 rolling comparison。

### 4.5 严禁使用 Test 和资产定价结果选择模型

以下信息不能用于模型选择：

- Test NLL、CRPS、VaR coverage 或 ES score；
- ES 排序组合的未来收益；
- Fama--MacBeth 系数；
- 因子调整 alpha；
- 根据最终结果挑选的子样本。

模型结构、主要超参数、ES 水平和主要评价指标应当在正式 Test 比较前固定。模型选择依赖 Validation，资产定价检验发生在预测模型冻结之后。

---

## 五、预测模型与识别逻辑

### 5.1 模型阶梯

主分析保留五个模型：

| 模型 | 条件信息 | 非线性 | Mixture | 结构化参数映射 |
|---|---:|---:|---:|---:|
| Rolling Normal | 否 | 否 | 否 | 否 |
| Linear Quantile | 是 | 否 | 不假定分布 | 否 |
| Fused Normal K1 | 是 | 是 | 否 | 否 |
| Fused Normal K3 | 是 | 是 | 是 | 否 |
| Structured Normal K3 | 是 | 是 | 是 | 是 |

逐步比较具有明确含义：

- Linear Quantile vs Rolling Normal：条件特征是否有用；
- Fused K1 vs Linear Quantile：非线性表达是否有用；
- Fused K3 vs Fused K1：mixture 是否有用；
- Structured K3 vs Fused K3：宏观—微观结构是否有用。

Quantile MLP 暂时不进入主分析。Student-t mixture 暂时只作为 robustness。

### 5.2 Rolling Normal

对公司 $i$，使用 Test 之前最多 $H=192$ 个月的历史收益估计：

$$
\widehat\mu_{i,t}
=
\frac{1}{n_{i,t}}
\sum_{s\in\mathcal H_{i,t}}R_{i,s},
$$

$$
\widehat\sigma_{i,t}
=
\sqrt{
\frac{1}{n_{i,t}-1}
\sum_{s\in\mathcal H_{i,t}}
(R_{i,s}-\widehat\mu_{i,t})^2
}.
$$

假设：

$$
R_{i,t+1}\mid\mathcal F_t
\sim
N(\widehat\mu_{i,t},\widehat\sigma_{i,t}^2).
$$

当公司历史不足以计算标准差时，可以使用当月横截面波动率作为预先固定的 fallback，并对波动率设置统一下限。Fallback 规则必须在看到 Test 结果之前确定。

对标准正态分布，令：

$$
z_\alpha=\Phi^{-1}(\alpha).
$$

收益率 VaR 临界值为：

$$
q_{i,t,\alpha}
=
\widehat\mu_{i,t}
+\widehat\sigma_{i,t}z_\alpha.
$$

左尾条件平均收益为：

$$
E[R_{i,t+1}\mid R_{i,t+1}\le q_{i,t,\alpha}]
=
\widehat\mu_{i,t}
-
\widehat\sigma_{i,t}
\frac{\phi(z_\alpha)}{\alpha}.
$$

若 ES 以正损失表示：

$$
ES_{i,t,\alpha}^{\mathrm{loss}}
=
-\widehat\mu_{i,t}
+
\widehat\sigma_{i,t}
\frac{\phi(z_\alpha)}{\alpha}.
$$

### 5.3 Linear Quantile

对每一个分位数水平 $\tau$，估计：

$$
Q_\tau(R_{i,t+1}\mid X_{i,t},M_t,J_{i,t})
=
\beta_{0,\tau}
+
\widetilde X_{i,t}'\beta_{\tau}^{X}
+
\widetilde M_t'\beta_{\tau}^{M}
+
D(J_{i,t})'\gamma_\tau,
$$

其中 $D(J_{i,t})$ 是 FF49 dummy。

不同分位数拥有不同系数。模型不假设收益率服从正态分布或其他参数分布。

定义预测误差：

$$
u_{i,t+1,\tau}
=
R_{i,t+1}
-
\widehat Q_\tau(X_{i,t},M_t,J_{i,t}).
$$

Pinball loss 为：

$$
\rho_\tau(u)
=
u\left[\tau-\mathbf 1(u<0)\right]
=
\begin{cases}
\tau u, & u\ge 0,\\
(\tau-1)u, & u<0.
\end{cases}
$$

月份等权目标函数为：

$$
L_{\mathrm{QR}}
=
\frac{
\sum_{i,t}w_{i,t}
\frac{1}{|\mathcal T|}
\sum_{\tau\in\mathcal T}
\rho_\tau(u_{i,t+1,\tau})
}{
\sum_{i,t}w_{i,t}
}.
$$

为了减少分位数交叉，可以加入：

$$
L_{\mathrm{cross}}
=
\frac{1}{|\mathcal T|-1}
\sum_{j=1}^{|\mathcal T|-1}
\max\left(
\widehat Q_{\tau_j}
-
\widehat Q_{\tau_{j+1}},
0
\right).
$$

最终训练目标为：

$$
L=L_{\mathrm{QR}}+\lambda_{\mathrm{cross}}L_{\mathrm{cross}}.
$$

预测后可以使用 monotone rearrangement 将每条观测的分位数从小到大排序。该处理保证最终分位数函数单调，但必须在方法部分披露。

Linear Quantile 没有天然概率密度，因此不能与密度模型比较精确 NLL。它主要参加 Quantile Score、VaR coverage、tail-CRPS 和 joint VaR--ES score 比较。

### 5.4 Fused Normal K1

将公司特征、宏观变量和行业 embedding 合并：

$$
Z_{i,t}
=
[\widetilde X_{i,t},\widetilde M_t,e(J_{i,t})].
$$

通过共同神经网络：

$$
h_{i,t}=g_{\mathrm{fused}}(Z_{i,t}),
$$

$$
\mu_{i,t}=g_\mu(h_{i,t}),
$$

$$
\sigma_{i,t}
=
\operatorname{softplus}(g_\sigma(h_{i,t}))
+\sigma_{\min}.
$$

条件分布为：

$$
R_{i,t+1}\mid\mathcal F_t
\sim
N(\mu_{i,t},\sigma_{i,t}^2).
$$

单条观测的负对数似然为：

$$
\ell_{i,t}^{K1}
=
\frac{1}{2}\log(2\pi)
+
\log\sigma_{i,t}
+
\frac{(R_{i,t+1}-\mu_{i,t})^2}{2\sigma_{i,t}^2}.
$$

月份等权 NLL 为：

$$
L_{\mathrm{NLL}}
=
\frac{\sum_{i,t}w_{i,t}\ell_{i,t}}
{\sum_{i,t}w_{i,t}}.
$$

Fused K1 控制神经网络非线性和全部条件信息，但不允许偏度、多个状态或其他 mixture 形状。

### 5.5 Fused Normal K3

Fused K3 使用同一个融合表示同时输出：

$$
\pi_{i,k,t},\quad
\mu_{i,k,t},\quad
\sigma_{i,k,t},
\qquad k=1,2,3.
$$

Mixture probability 满足：

$$
\pi_{i,k,t}
=
\frac{\exp(a_{i,k,t})}
{\sum_{j=1}^{3}\exp(a_{i,j,t})},
$$

$$
\sum_{k=1}^{3}\pi_{i,k,t}=1.
$$

条件密度为：

$$
f_{i,t}(r)
=
\sum_{k=1}^{3}
\pi_{i,k,t}
\frac{1}{\sigma_{i,k,t}}
\phi\left(
\frac{r-\mu_{i,k,t}}{\sigma_{i,k,t}}
\right).
$$

单条观测 NLL 为：

$$
\ell_{i,t}^{K3}
=
-
\log
\left[
\sum_{k=1}^{3}
\pi_{i,k,t}
\frac{1}{\sigma_{i,k,t}}
\phi\left(
\frac{R_{i,t+1}-\mu_{i,k,t}}
{\sigma_{i,k,t}}
\right)
\right].
$$

实际计算必须使用 log-sum-exp 保证数值稳定。

Fused K3 与 Fused K1 的比较识别 mixture 的预测增量，但 Fused K3 本身不提供清楚的“宏观概率—公司损失”分工。

### 5.6 Structured Normal K3

Structured K3 将宏观变量和公司变量映射到不同的分布参数。

宏观通道：

$$
h_t^M=g_M(\widetilde M_t),
$$

$$
\pi_{k,t}
=
\operatorname{softmax}_k(g_\pi(h_t^M)).
$$

由于 $M_t$ 在同一月份对所有公司相同，$\pi_{k,t}$ 也是同一个月所有公司共享的 macro-conditioned mixture probability。

公司通道：

$$
h_{i,t}^X
=
g_X([\widetilde X_{i,t},e(J_{i,t})]),
$$

$$
\mu_{i,k,t}=g_{\mu,k}(h_{i,t}^X),
$$

$$
\sigma_{i,k,t}
=
\operatorname{softplus}(g_{\sigma,k}(h_{i,t}^X))
+\sigma_{\min}.
$$

条件密度为：

$$
f_{i,t}(r)
=
\sum_{k=1}^{3}
\pi_{k,t}(M_t)
N\left(
r;\mu_{i,k,t}(X_{i,t},J_{i,t}),
\sigma_{i,k,t}^2(X_{i,t},J_{i,t})
\right).
$$

这一结构提供如下经济解释：

- 宏观变量决定不同 mixture components 的共同 weights；
- 公司特征和行业决定公司在各 component 中的收益位置和波动；
- 每个滚动窗口内使用一个对所有公司一致的 component 标签；
- 同一个宏观条件情景对不同公司产生不同的条件损失程度。

Structured K3 与 Fused K3 的差异不是单纯增加参数，而是施加经济结构限制。只有在预测结果、component 稳定性和经济验证共同支持时，才可以将 Structured K3 解释为 macro-conditioned scenario model。若没有共享 latent state 的联合 likelihood，不能进一步声称它识别了所有公司共同实现的真实宏观状态。

### 5.7 Normal 为主、Student-t 为稳健性检验

主分析使用 Normal components：

- 参数更少；
- 训练更稳定；
- mixture 本身可以通过多个正态成分产生厚尾、偏度和多峰；
- ES 分解具有清楚的解析形式。

Student-t components 需要额外估计自由度 $\nu$，可能增加识别和数值优化难度。因此 Student-t MDN 放入 robustness，检验主要结论是否依赖正态 component 假设。

---

## 六、VaR、ES 与 Structured Mixture 分解

### 6.1 Mixture CDF 和 VaR

Structured Normal K3 的条件 CDF 为：

$$
F_{i,t}(r)
=
\sum_{k=1}^{3}
\pi_{k,t}
\Phi\left(
\frac{r-\mu_{i,k,t}}{\sigma_{i,k,t}}
\right).
$$

收益率的左尾 $\alpha$ 分位数 $q_{i,t,\alpha}$ 满足：

$$
F_{i,t}(q_{i,t,\alpha})=\alpha.
$$

Mixture quantile 通常没有闭式解，可以使用稳定的一维求根方法计算。

本文将 `var_alpha` 定义为收益率分位数：

$$
VaR_{i,t,\alpha}^{\mathrm{return}}
=
q_{i,t,\alpha}.
$$

它通常是负数。若使用正损失 VaR，则定义：

$$
VaR_{i,t,\alpha}^{\mathrm{loss}}
=
-q_{i,t,\alpha}.
$$

所有 notebook 和表格必须固定一种符号约定，不能在不同模型之间混用。

### 6.2 Mixture ES

对正态 component：

$$
R\mid Z=k
\sim
N(\mu_k,\sigma_k^2).
$$

令：

$$
z_k(q)=\frac{q-\mu_k}{\sigma_k}.
$$

截断一阶矩为：

$$
E\left[
R\mathbf 1(R\le q)\mid Z=k
\right]
=
\mu_k\Phi(z_k)
-
\sigma_k\phi(z_k).
$$

因此 mixture 的左尾条件平均收益为：

$$
E[R_{i,t+1}\mid R_{i,t+1}\le q_{i,t,\alpha}]
=
\frac{1}{\alpha}
\sum_{k=1}^{3}
\pi_{k,t}
\left[
\mu_{i,k,t}\Phi(z_{i,k,t})
-
\sigma_{i,k,t}\phi(z_{i,k,t})
\right],
$$

其中：

$$
z_{i,k,t}
=
\frac{q_{i,t,\alpha}-\mu_{i,k,t}}
{\sigma_{i,k,t}}.
$$

本文资产定价部分建议使用正损失 ES：

$$
ES_{i,t,\alpha}^{\mathrm{loss}}
=
-
E[R_{i,t+1}\mid R_{i,t+1}\le q_{i,t,\alpha}],
$$

即：

$$
ES_{i,t,\alpha}^{\mathrm{loss}}
=
\frac{1}{\alpha}
\sum_{k=1}^{3}
\pi_{k,t}
\left[
\sigma_{i,k,t}\phi(z_{i,k,t})
-
\mu_{i,k,t}\Phi(z_{i,k,t})
\right].
$$

### 6.3 Component-level ES contribution

定义 component $k$ 对整体 ES 的贡献：

$$
C_{i,k,t,\alpha}
=
\frac{\pi_{k,t}}{\alpha}
\left[
\sigma_{i,k,t}\phi(z_{i,k,t})
-
\mu_{i,k,t}\Phi(z_{i,k,t})
\right].
$$

则有精确加总关系：

$$
ES_{i,t,\alpha}^{\mathrm{loss}}
=
\sum_{k=1}^{3}C_{i,k,t,\alpha}.
$$

这是论文中最直接、最可验证的尾部风险分解。代码必须逐行验证：

$$
\left|
ES_{i,t,\alpha}^{\mathrm{loss}}
-
\sum_kC_{i,k,t,\alpha}
\right|
<\epsilon.
$$

### 6.4 状态尾部权重与条件损失程度

Component $k$ 在整体 $\alpha$ 左尾事件中的后验权重为：

$$
\omega_{i,k,t,\alpha}
=
\frac{
\pi_{k,t}\Phi(z_{i,k,t})
}{\alpha}.
$$

由于整体 VaR 满足：

$$
\sum_k\pi_{k,t}\Phi(z_{i,k,t})=\alpha,
$$

因此：

$$
\sum_k\omega_{i,k,t,\alpha}=1.
$$

定义 component 内、低于整体 VaR 的正损失程度：

$$
S_{i,k,t,\alpha}
=
-
E[
R_{i,t+1}
\mid
R_{i,t+1}\le q_{i,t,\alpha},
Z=k
],
$$

$$
S_{i,k,t,\alpha}
=
-
\mu_{i,k,t}
+
\sigma_{i,k,t}
\frac{\phi(z_{i,k,t})}{\Phi(z_{i,k,t})}.
$$

于是：

$$
ES_{i,t,\alpha}^{\mathrm{loss}}
=
\sum_{k=1}^{3}
\omega_{i,k,t,\alpha}
S_{i,k,t,\alpha}.
$$

该表达式表明，ES 的精确构成包括：

1. 宏观条件 component weight $\pi_{k,t}$；
2. 该 component 落入整体左尾的概率 $\Phi(z_{i,k,t})$；
3. 落入左尾后的公司特定损失程度 $S_{i,k,t,\alpha}$。

导师所说的“共同不利状态概率 × 公司损失严重程度”是本文的经济直觉。当前模型将其具体化为：

> 共同的 macro-conditioned adverse-component weight × 公司特定的 adverse-component 条件损失程度。

这里的 mixture weight 是逐股票边际预测分布中的共同权重，而不是共享 latent state 联合模型估计出的 realized-state probability。正式论文应当使用 $C_{i,k,t,\alpha}$、$\omega_{i,k,t,\alpha}$ 和 $S_{i,k,t,\alpha}$ 给出精确分解。

### 6.5 Adverse component 的识别

Mixture component 编号本身没有经济含义，并存在 label switching。不能未经验证就把固定的 component 1 称为 adverse state。

主分析需要预先固定跨窗口识别规则。一个可执行的基本规则是：

1. 在每个滚动窗口的 Train + Validation 样本上计算每个 component 的 month-equal、cross-sectional average component mean；
2. 平均 $\mu_{i,k,t}$ 最低的 component 标记为 `adverse`；
3. 平均 $\mu_{i,k,t}$ 最高的 component 标记为 `favorable`；
4. 其余 component 标记为 `normal`；
5. 该标签映射固定后，再应用于对应 Test，不使用 Test realized return 决定标签。

设窗口 $w$ 内用 Train + Validation 确定的 adverse component 为：

$$
b_w
=
\arg\min_k
\frac{1}{|\mathcal T_w|}
\sum_{t\in\mathcal T_w}
\frac{1}{N_t}
\sum_{i\in\mathcal I_t}
\mu_{i,k,t},
$$

其中 $\mathcal T_w$ 是窗口 $w$ 的 Train + Validation 月份，$\mathcal I_t$ 是月份 $t$ 的股票集合，$N_t=|\mathcal I_t|$。该定义先在每个月内横截面平均，再对月份平均，避免公司数量较多的月份主导标签。

在该窗口的 Test 中，所有公司都使用同一个标签 $b_w$：

$$
\pi_{\mathrm{adv},t}=\pi_{b_w,t},
\qquad
f^{\mathrm{adv}}_{i,t}(r)=f_{i,b_w,t}(r).
$$

因此，不同公司的 adverse-component distribution 可以具有不同的 $\mu_{i,b_w,t}$ 和 $\sigma_{i,b_w,t}$，但它们对应同一个全局 component 标签。不能再对每只公司单独定义

$$
b_{i,t}=\arg\min_k\mu_{i,k,t},
$$

因为逐公司选取均值最低的 component 会破坏“共同 adverse scenario”的含义。全局标签解决的是 component 对齐问题，使“公司 $i$ 在共同 adverse scenario 下的条件分布”成为可计算对象；它不会把逐股票 likelihood 自动变成共享 realized state 的联合 likelihood。

如果 component 均值排序不稳定，需要增加跨窗口 matching，或者在模型中加入 ordered means 参数化。无论使用哪种方案，都必须在运行资产定价测试前固定。

### 6.6 “宏观状态”的解释边界

当前逐股票 MDN likelihood 通常是：

$$
\prod_i
\left[
\sum_k\pi_{k,t}f_{i,k,t}(R_{i,t+1})
\right].
$$

它与一个所有公司共享同一 realized latent state 的联合 likelihood：

$$
\sum_k\pi_{k,t}
\prod_i f_{i,k,t}(R_{i,t+1})
$$

并不相同。

两者的差别是概率模型本身不同。当前模型允许同一个月中，每只股票在其边际 mixture 中具有自己的未观测 component realization；联合 likelihood 则要求该月所有股票共享同一个 $S_{t+1}$。

因此，本文主分析可以主张：

1. 同月所有股票共享由宏观变量决定的 mixture weights；
2. 每个滚动窗口内，所有股票共享同一个 adverse-component 标签；
3. 每只股票具有该共同 adverse component 下的公司特定条件收益分布；
4. 可以研究 adverse-component weight 与公司损失程度如何共同决定预测 ES。

本文主分析不能直接主张：

1. 某个月所有股票共同实现了同一个 latent state；
2. $\pi_{\mathrm{adverse},t}$ 已经是共享 realized macro state 的后验概率；
3. 当前逐股票 likelihood 识别了真实经济衰退或市场崩盘状态。

最安全的表述是：

> macro-conditioned adverse-return component

或者：

> a globally aligned adverse component with macro-conditioned common weights and firm-specific conditional return distributions

而不是直接声称模型识别了所有公司共同实现的真实宏观隐状态。如果未来要作出后一种主张，需要另行建立并估计共享 $S_{t+1}$ 的联合 likelihood。

为了支持经济解释，需要验证：

- $\pi_{\mathrm{adverse},t}$ 是否预测未来市场下跌；
- 是否在高 VIX、高信用利差等时期更高；
- adverse component 是否贡献了大部分公司左尾 ES；
- component 标签是否跨窗口稳定；
- 是否存在 component collapse。

此外，$\pi_{\mathrm{adverse},t}$ 是逐股票边际预测分布下的 component probability，而不是共享 realized macro state 的概率，也不是随机贴现因子意义上的风险价格。

---

## 七、样本外预测评价

### 7.1 评价原则

预测评价必须同时考虑：

- Accuracy：预测值与 realized return 的接近程度；
- Calibration：预测概率是否与实际发生频率一致；
- Sharpness：在保持校准的前提下，预测分布是否足够集中；
- Tail performance：模型是否准确刻画本文关心的下行尾部。

Structured K3 不需要在所有指标上都第一。主假设是它在预先指定的下行尾部指标上具有更好的样本外表现，并且 component 具有稳定经济意义。

### 7.2 NLL

对具有密度 $f_{i,t}$ 的模型：

$$
NLL_{i,t}
=
-
\log f_{i,t}(R_{i,t+1}).
$$

月度平均 NLL：

$$
\overline{NLL}_t
=
\frac{1}{N_t}
\sum_{i=1}^{N_t}NLL_{i,t}.
$$

NLL 越低越好。它对在 realized return 附近分配极低密度的模型惩罚很大。

适用模型：

- Rolling Normal；
- Fused Normal K1；
- Fused Normal K3；
- Structured Normal K3。

Linear Quantile 没有天然密度，不参加精确 NLL 比较。

### 7.3 CRPS

连续排序概率分数定义为：

$$
CRPS(F,y)
=
\int_{-\infty}^{\infty}
\left[
F(z)-\mathbf 1(y\le z)
\right]^2dz.
$$

CRPS 同时评价整个预测分布的 calibration 和 sharpness，越低越好。

CRPS 也可以写成分位数损失积分：

$$
CRPS(F,y)
=
2\int_0^1
\rho_\tau(y-Q_\tau)d\tau.
$$

这一关系允许所有模型通过共同分位数网格计算可比的离散 CRPS approximation。

### 7.4 Tail-CRPS

本文关心下行尾部，因此定义下行加权分数：

$$
twCRPS_{\bar\alpha}(F,y)
=
\frac{2}{\bar\alpha}
\int_0^{\bar\alpha}
\rho_\tau(y-Q_\tau)d\tau.
$$

主分析建议预先固定：

$$
\bar\alpha=0.10.
$$

即只评价最差 10% 分布区域，并通过 $1/\bar\alpha$ 标准化。离散实现使用同一组低分位数网格和梯形积分。分数越低越好。

### 7.5 Quantile Score

分位数 $\alpha$ 的 Quantile Score 可以定义为：

$$
QS_{\alpha,i,t}
=
2\rho_\alpha
\left(
R_{i,t+1}-q_{i,t,\alpha}
\right).
$$

模型预测的 $q_{i,t,\alpha}$ 越接近真实条件分位数，平均 Quantile Score 越低。

所有模型都可以比较 1%、5% 和 10% Quantile Score。

### 7.6 PIT

Probability Integral Transform 为：

$$
PIT_{i,t}
=
F_{i,t}(R_{i,t+1}).
$$

若连续预测分布正确，则：

$$
PIT_{i,t}\sim U(0,1).
$$

需要检查：

- PIT histogram 是否接近均匀；
- 是否 U-shaped，表示预测分布过窄；
- 是否 hump-shaped，表示预测分布过宽；
- 是否偏向 0 或 1，表示位置或尾部失准；
- 月度 PIT 是否存在时间依赖。

PIT 适用于具有完整 CDF 的密度模型。Linear Quantile 可以通过分位数插值构造近似 PIT，但不应与精确 PIT 混为同一主要指标。

### 7.7 VaR coverage

定义 exceedance indicator：

$$
I_{i,t,\alpha}
=
\mathbf 1
\left(
R_{i,t+1}<q_{i,t,\alpha}
\right).
$$

正确的 VaR 应满足：

$$
E[I_{i,t,\alpha}]=\alpha.
$$

经验 exceedance rate 为：

$$
\widehat p_\alpha
=
\frac{1}{N}
\sum_{i,t}I_{i,t,\alpha}.
$$

Coverage error 为：

$$
CE_\alpha=\widehat p_\alpha-\alpha.
$$

由于同月股票相关，不能把所有公司月份简单视为独立 Bernoulli trials。主结果应报告月度 exceedance rate 序列，并使用按月或按年度的时间序列推断。

作为补充，可以报告 Kupiec unconditional coverage test。若共有 $n_1$ 次 exceedance、$n_0$ 次 non-exceedance，$\widehat p=n_1/(n_0+n_1)$，则：

$$
LR_{\mathrm{uc}}
=
-
2\log
\left[
\frac{
(1-\alpha)^{n_0}\alpha^{n_1}
}{
(1-\widehat p)^{n_0}\widehat p^{n_1}
}
\right].
$$

在独立 Bernoulli 假设下：

$$
LR_{\mathrm{uc}}\sim\chi^2(1).
$$

由于股票截面相关，传统检验的 $p$-value 只能作为辅助诊断。

### 7.8 Joint VaR--ES score

VaR 和 ES 可以使用严格一致的联合评分函数评价。令：

- $q<0$：收益率 VaR；
- $e<0$：左尾平均收益；
- 代码中若保存正损失 ES，则 $e=-ES^{\mathrm{loss}}$。

FZ0 loss 为：

$$
L_{\mathrm{FZ0}}(y,q,e;\alpha)
=
-
\frac{1}{\alpha e}
\mathbf 1(y\le q)(q-y)
+
\frac{q}{e}
+
\log(-e)
-1,
$$

其中 $e<0$。平均 FZ0 loss 越低，VaR 和 ES 联合预测越好。

实现时必须检查：

- 传入公式的是负的左尾平均收益 $e$，不是正损失 ES；
- $e<q$ 通常成立；
- 对接近零的 $e$ 设置数值保护；
- 所有模型采用完全相同的符号约定。

### 7.9 主要与次要预测指标

为了避免事后选择指标，建议预先确定：

**主要指标**

1. 5% FZ0 joint VaR--ES score；
2. 0%--10% 下行 tail-CRPS。

**次要指标**

- 1% 和 10% FZ0 score；
- 1%、5%、10% Quantile Score；
- VaR coverage；
- full-distribution CRPS；
- NLL；
- PIT diagnostics。

1% 尾部最接近研究问题，但有效 exceedance 数量较少，因此 5% 适合作为统计上更稳定的主结果，1% 作为更极端的补充结果。

### 7.10 月度聚合和模型显著性比较

不能将所有股票月份损失当作独立观测。对任意 loss，先计算月度平均：

$$
\overline L_{m,t}
=
\frac{1}{N_t}
\sum_{i=1}^{N_t}L_{m,i,t}.
$$

总体模型分数为：

$$
\overline L_m
=
\frac{1}{T}
\sum_{t=1}^{T}\overline L_{m,t}.
$$

比较模型 $A$ 和 $B$：

$$
d_t
=
\overline L_{A,t}
-
\overline L_{B,t}.
$$

检验：

$$
H_0:E[d_t]=0.
$$

使用月度 loss differential 的 Newey--West 标准误或时间 block bootstrap。若：

$$
E[d_t]<0,
$$

则模型 $A$ 的平均预测损失低于模型 $B$。

---

## 八、Structured Component 的经济验证

在资产定价测试之前，必须证明 Structured K3 的 component 输出具有最低限度的稳定性和经济含义。

### 8.1 Component stability

每个滚动窗口报告：

- 各 component 的平均 $\pi_k$；
- 各 component 的横截面平均 $\mu_k$；
- 各 component 的平均 $\sigma_k$；
- adverse/normal/favorable 标签映射；
- component 排序是否在 Test 内大量交叉。

### 8.2 Component collapse

若某个 component 长期满足：

$$
\pi_{k,t}\approx 0,
$$

或两个 components 的 $\mu$ 和 $\sigma$ 几乎相同，则 K3 可能退化为更少的有效 components。

可以报告有效 component 数：

$$
K_{\mathrm{eff},t}
=
\frac{1}{\sum_{k=1}^{K}\pi_{k,t}^2}.
$$

当一个 component 完全主导时，$K_{\mathrm{eff}}\approx1$；当三个 component 等权时，$K_{\mathrm{eff}}=3$。

### 8.3 Adverse probability 的宏观验证

检验：

$$
\pi_{\mathrm{adverse},t}
$$

与以下变量或事件之间的关系：

- 下一月市场收益；
- 下一月市场左尾事件；
- VIX；
- 信用利差；
- 期限利差；
- 衰退或金融压力时期。

由于部分宏观变量本身是 $\pi_t$ 的输入，输入变量与 $\pi_t$ 的同期相关性只能解释模型映射，不能单独证明 component 是真实宏观状态。更有说服力的是 $\pi_t$ 对未来市场不利结果的样本外关联。

### 8.4 ES decomposition diagnostics

报告：

- adverse component 对总 ES 的平均贡献比例；
- 不同公司和行业的 adverse contribution 分布；
- 高 $\pi_{\mathrm{adverse}}$ 和低 $\pi_{\mathrm{adverse}}$ 月份的 contribution；
- 总 ES 与 component contributions 的逐行加总误差。

只有当 adverse component 确实对下行尾部具有显著贡献时，才能将其用于主要经济解释。

---

## 九、资产定价检验

### 9.1 基本原则

资产定价部分只使用真正的样本外预测：

$$
\widehat{ES}_{i,t}
\longrightarrow
R_{i,t+1}.
$$

不得使用 $R_{i,t+1}$ 反向选择模型、调整 ES 定义或确定 adverse component 标签。

资产定价和预测评价回答不同问题：

- 预测评价：模型是否准确描述条件分布和尾部？
- 资产定价：预测尾部风险是否与未来平均收益相关？

一个模型预测更准确，不保证它的 ES 横截面排序具有更强收益关系；反之亦然。

### 9.2 单变量组合排序

每月按照预测 ES 将股票分成 $G=5$ 或 $G=10$ 组。定义：

$$
P_{g,t+1}
=
\sum_{i\in g_t}
\omega_{i,t}R_{i,t+1}.
$$

报告：

- Equal-weighted portfolio return；
- Value-weighted portfolio return；
- High ES minus Low ES：

$$
HML^{ES}_{t+1}
=
R^{High\,ES}_{t+1}
-
R^{Low\,ES}_{t+1}.
$$

风险补偿假设预测：

$$
E[HML^{ES}]>0.
$$

时间序列均值使用 Newey--West $t$-statistics。主结果应同时报告组合单调性，而不仅仅报告最高组减最低组。

如数据允许，使用 NYSE breakpoints 减少小市值股票对分组点的支配；否则使用全样本 breakpoints，并在 robustness 中处理 microcaps。

### 9.3 因子调整 alpha

对 High-minus-Low 组合：

$$
HML^{ES}_{t}
=
\alpha
+
\beta'F_t
+
\varepsilon_t,
$$

其中 $F_t$ 为标准资产定价因子。

若：

$$
\alpha>0,
$$

说明 ES 排序收益不能完全被已有因子解释。需要报告 Newey--West 标准误。

因子 alpha 不能自动证明 ES 是新的系统性风险因子，但可以排除部分已有因子暴露解释。

### 9.4 Fama--MacBeth 横截面回归

每个月估计：

$$
R_{i,t+1}
=
a_t
+
b_t ES_{i,t,\alpha}^{\mathrm{loss}}
+
\Gamma_t'Controls_{i,t}
+
\varepsilon_{i,t+1}.
$$

然后计算：

$$
\overline b
=
\frac{1}{T}\sum_{t=1}^{T}b_t.
$$

使用 $b_t$ 时间序列的 Newey--West 标准误检验：

$$
H_0:E[b_t]=0.
$$

控制变量应覆盖最主要的替代解释，例如：

- size；
- book-to-market；
- momentum；
- market beta；
- idiosyncratic volatility；
- liquidity；
- 过去偏度或其他可获得的尾部风险 proxy。

控制变量数量应有限且具有明确经济理由。避免根据最终显著性反复增删控制变量。

### 9.5 不同模型 ES 的资产定价比较(只有前面的部分都做的很好了, 这一部分才有意义)

对以下 ES 使用相同的组合排序和基础横截面回归：

- Rolling Normal ES；
- Linear Quantile ES；
- Fused Normal K1 ES；
- Fused Normal K3 ES；
- Structured Normal K3 ES。

这一比较用于回答：

- ES 溢价是否只存在于 Structured 模型？
- 更准确的尾部预测是否产生更清楚的横截面收益关系？
- Structured 分解是否提供其他模型没有的经济解释？

不能根据哪个模型的组合收益最高，反向宣布它是预测最好的模型。预测优劣由第七部分的 scoring rules 决定。

### 9.6 Adverse component contribution 的定价

使用：

$$
C_{i,\mathrm{adverse},t,\alpha}
$$

进行组合排序和 Fama--MacBeth 回归：

$$
R_{i,t+1}
=
a_t
+
b_t^C
C_{i,\mathrm{adverse},t,\alpha}
+
\Gamma_t'Controls_{i,t}
+
\varepsilon_{i,t+1}.
$$

若 $b_t^C$ 的时间序列均值为正，说明股票在 adverse component 中承担的尾部损失贡献获得未来收益补偿。

还可以同时加入其他 component contributions，但需要检查它们与总 ES 之间的机械加总关系和共线性。

### 9.7 Adverse-component severity 的定价

使用：

$$
S_{i,\mathrm{adverse},t,\alpha}
$$

检验公司在全局 adverse component 下的条件损失程度：

$$
R_{i,t+1}
=
a_t
+
b_t^S
S_{i,\mathrm{adverse},t,\alpha}
+
\Gamma_t'Controls_{i,t}
+
\varepsilon_{i,t+1}.
$$

这一检验回答：

> 在同一个宏观条件 mixture-weight 环境下，adverse component 中损失更严重的公司是否获得更高未来收益？

### 9.8 共同 adverse-component weight 不能直接用于月内股票排序

$\pi_{\mathrm{adverse},t}$ 在同一个月对所有公司相同，因此：

- 不能按 $\pi_{\mathrm{adverse},t}$ 在月内对股票排序；
- 在单个月 Fama--MacBeth 横截面回归中，它被截距吸收；
- $\pi_{\mathrm{adverse},t}S_{i,t}$ 在单个月内只是 $S_{i,t}$ 的常数倍，不能在同一个月同时识别两者的系数。

因此，不能声称普通横截面回归直接估计了“adverse-component weight 的风险价格”。

### 9.9 检验 severity premium 是否随 adverse-component weight 变化

可以使用以下三种方法。

#### 方法一：高 weight 与低 weight 月份分组

1. 根据 $\pi_{\mathrm{adverse},t}$ 将月份分为 High-adverse-weight 和 Low-adverse-weight；
2. 在每类月份中计算 severity 或 adverse contribution 的 High-minus-Low 股票组合；
3. 比较两类月份的平均组合收益。

核心检验：

$$
E[HML^S_{t+1}\mid \pi_{\mathrm{adverse},t}\text{ high}]
-
E[HML^S_{t+1}\mid \pi_{\mathrm{adverse},t}\text{ low}].
$$

#### 方法二：月度横截面 slope 的时间序列回归

先在每个月估计：

$$
R_{i,t+1}
=
a_t+b_t^SS_{i,t}+\Gamma_t'Controls_{i,t}+\varepsilon_{i,t+1}.
$$

然后估计：

$$
b_t^S
=
c_0
+
c_1\pi_{\mathrm{adverse},t}
+
u_t.
$$

若 $c_1>0$，说明 adverse-component weight 较高时，市场对公司 adverse-component 条件损失暴露给予更高补偿。

#### 方法三：Pooled panel interaction

估计：

$$
R_{i,t+1}
=
a_t
+
bS_{i,t}
+
\delta
\left[
\pi_{\mathrm{adverse},t}S_{i,t}
\right]
+
\Gamma'Controls_{i,t}
+
\varepsilon_{i,t+1},
$$

其中 $a_t$ 为月份固定效应。标准误至少按月份聚类；如可行，可以考虑公司和月份双向聚类。

若 $\delta>0$，说明 severity premium 随 adverse-component weight 增加。

### 9.10 风险价格与物理概率的区别

在逐股票边际 mixture 的 latent-component 表示下，Structured MDN 估计：

$$
\pi_{\mathrm{adverse},t}
=
P(Z_{i,t+1}=b_w\mid M_t).
$$

这个权重对同月所有股票相同，但 $Z_{i,t+1}$ 仍是股票层面的 latent component indicator。因此，它是物理预测分布中的 component probability，不是所有公司共同实现的宏观状态概率，也不是随机贴现因子中的状态价格。

本文可以检验：

- 公司在不利 component 中的损失暴露是否获得收益补偿；
- 该补偿是否随预测 adverse-component weight 变化。

本文不能仅凭 $\pi_{\mathrm{adverse},t}$ 宣称已经估计出投资者的 marginal utility 或 tail-risk price。

---

## 十、主要结果的可能组合与解释

### 10.1 Structured 预测更好，ES 和 adverse severity 都获得补偿

这是最理想结果。论文可以同时贡献：

- 更可靠的尾部预测；
- 具有经济结构的分布分解；
- 公司不利状态损失暴露的资产定价证据。

### 10.2 Structured 预测更好，但 ES 不获得补偿

模型具有预测贡献，但主要资产定价假设不成立。论文更接近条件分布预测研究，不能声称发现尾部风险溢价。

### 10.3 Structured 预测没有显著领先，但 adverse severity 获得补偿

经济机制可能仍有价值，但必须说明结构化模型的优势主要是可解释分解，而不是全面预测优势。需要证明结果不是弱预测模型产生的噪声。

### 10.4 所有模型的 ES 都获得类似补偿

这说明 ES 本身可能具有资产定价意义，但 Structured 模型并不是发现该关系的必要条件。Structured 的增量贡献需要来自更可靠的 calibration 或更细致的 component decomposition。

### 10.5 只有 Structured ES 获得补偿

这是有吸引力的结果，但也最容易受到 data mining 质疑。必须确保：

- 模型在资产定价测试前已经冻结；
- 相同样本和分组规则用于所有模型；
- 预测指标独立支持 Structured 输出；
- 结果在多个 ES 水平和组合权重下稳定。

### 10.6 结果与预期方向相反

负的 ES--return 关系不能被隐藏。可能解释包括：

- 投资者偏好高尾部风险股票；
- 高 ES 股票被过度定价；
- 预测 ES 捕捉了 distress、lottery demand 或其他特征；
- ES 符号、时间对齐或样本构建存在问题。

首先应排除代码和时间对齐错误，再讨论经济解释。

---

## 十一、稳健性检验

稳健性检验在主分析完成后进行，不用于反向选择主要模型。

### 11.1 模型稳健性

- Warm start；
- Student-t components；
- $K=2$ 或 $K=4$；
- 不同隐藏层宽度；
- 不同随机种子；
- 不同训练窗口长度；
- 不同 $\sigma_{\min}$；
- ordered means 或其他 component identification。

### 11.2 特征与行业稳健性

- FF49；
- FF12；
- 不使用行业；
- 删除某些宏观变量组；
- 仅微观变量；
- 仅宏观概率通道。

### 11.3 预测评价稳健性

- 1%、5%、10% ES；
- 不同 tail-CRPS cutoff；
- 不同月度 loss 聚合方式；
- Newey--West 与时间 block bootstrap；
- 危机期和非危机期；
- 排除极端收益观测。

### 11.4 资产定价稳健性

- 五组和十组组合；
- Equal-weighted 和 Value-weighted；
- NYSE 和全样本 breakpoints；
- 排除 microcaps；
- 不同因子模型；
- 不同控制变量组合；
- 高低 adverse-component weight 子样本；
- 危机期与正常期；
- 不同 component 标签规则。

---

## 十二、Notebook 实施顺序

### 12.1 `1DataDownload_Clean.ipynb`

目标：

- 固定微观表和宏观表；
- 固定变量定义、日期、单位和缺失处理；
- 输出唯一公司月份观测；
- 保存数据版本和时间范围。

完成标准：

- `permno`--`mthcaldt` 唯一；
- `target_ret_final` 与下一月收益严格对齐；
- 宏观数据每月唯一；
- 特征和标签合同固定。

### 12.2 `2Baseline_model.ipynb`

实现：

- Rolling Normal；
- Linear Quantile with FF49 dummies。

输出至少包括：

- `permno`；
- `forecast_date`；
- `target_date`；
- `realized_ret`；
- `model_id`；
- 预测分位数；
- 1%、5%、10% VaR；
- 1%、5%、10% ES；
- 计算共同评价指标所需的分布或 quantile grid。

### 12.3 `3MDN_model.ipynb`

实现：

- Fused Normal K1；
- Fused Normal K3；
- Structured Normal K3；
- 每个窗口从头训练；
- 使用 180 个月 Train 更新参数；
- 使用 12 个月 Validation early stopping；
- 恢复最佳 Validation checkpoint 后直接预测 Test；
- 保存每个窗口最终模型预测参数。

所有模型使用相同：

- 滚动日期；
- Test 样本；
- 连续特征；
- FF49 信息；
- month-equal weights；
- 训练和评估符号约定。

### 12.4 `4MDN_ES_Calculation.ipynb`

只负责从 MDN 参数计算：

- mixture CDF；
- quantiles/VaR；
- ES；
- component ES contributions；
- $\omega_{i,k,t,\alpha}$；
- severity $S_{i,k,t,\alpha}$；
- adverse component 标签和相关输出。

不在该 notebook 中重复计算预测比较指标。

### 12.5 `5Compare_model.ipynb`

比较：

- 5% FZ0；
- 0%--10% tail-CRPS；
- 1%、5%、10% Quantile Score；
- VaR coverage；
- NLL；
- CRPS；
- PIT；
- component diagnostics。

所有 observation-level loss 先按月聚合，再进行模型平均和显著性比较。

### 12.6 `6Pricing_Test.ipynb`

依次执行：

1. 各模型总 ES 的组合排序；
2. 各模型总 ES 的 Fama--MacBeth；
3. Structured adverse contribution 排序与回归；
4. Structured adverse severity 排序与回归；
5. 高低 adverse-component weight 时期比较；
6. 月度 severity slope 与 adverse-component weight 的时间序列关系；
7. 因子 alpha 和控制变量稳健性。

---

## 十三、论文结构建议

### 1. Introduction

- 为什么公司层面条件尾部风险重要；
- 现有方法为什么难以区分共同 adverse-scenario weight 和公司条件损失程度；
- Structured MDN 的经济结构；
- 主要预测和资产定价发现；
- 论文贡献。

### 2. Related Literature

- 公司尾部风险与预期收益；
- VaR/ES 与资产定价；
- 条件分布和 quantile forecasting；
- mixture density networks；
- 宏观条件 mixture components、latent-state 模型和横截面风险补偿。

### 3. Data and Predictive Design

- 数据来源与变量；
- 信息时点；
- Train/Validation/Test；
- OOS rolling protocol；
- month-equal weighting；
- 目标收益率。

### 4. Models

- Rolling Normal；
- Linear Quantile；
- Fused K1；
- Fused K3；
- Structured K3；
- NLL 和训练过程。

### 5. Out-of-Sample Distribution Forecasts

- NLL/CRPS；
- tail-CRPS；
- Quantile Score；
- PIT；
- VaR coverage；
- joint VaR--ES score；
- 模型 loss differential。

### 6. Economic Interpretation of Structured Components

- component identification；
- adverse-component weight；
- ES decomposition；
- adverse component 的宏观含义验证；
- 公司和行业异质性。

### 7. Expected Shortfall and the Cross-Section of Returns

- 总 ES 排序；
- Fama--MacBeth；
- 因子 alpha；
- adverse contribution；
- severity；
- state-dependent premium。

### 8. Robustness

- Warm start；
- Student-t；
- alternative $K$；
- random seeds；
- alternative industries；
- alternative ES levels；
- alternative portfolio designs。

### 9. Conclusion

- 预测结论；
- 经济机制；
- 资产定价含义；
- 解释边界。

---

## 十四、执行前必须锁定的决定

以下项目在正式运行完整 OOS 测试前固定：

- [x] 数据源暂时固定；
- [x] 目标变量为下一月原始收益率；
- [x] 主模型使用 Normal components；
- [x] 主分析不使用 warm start；
- [x] Warm start 只作为 robustness；
- [x] Rolling window 为 180/12/12；
- [x] Test 每 12 个月向前滚动；
- [x] FF49 在 Linear Quantile 中使用 dummies；
- [x] FF49 在 MDN 中使用 embedding；
- [x] 主要模型包括 Rolling Normal、Linear Quantile、Fused K1、Fused K3、Structured K3；
- [x] Quantile MLP 不进入当前主分析；
- [x] Student-t 不进入当前主分析；
- [ ] 固定 Linear Quantile 的 quantile grid；
- [ ] 固定 FZ0 的 ES 符号与实现；
- [ ] 固定 tail-CRPS 权重和积分网格；
- [ ] 固定每个窗口的随机种子规则；
- [ ] 固定 adverse component 跨窗口识别规则；
- [ ] 固定 severity 的主要定义；
- [ ] 固定主要资产定价控制变量；
- [ ] 固定组合数量和 breakpoint 规则；
- [ ] 固定主要因子模型；
- [ ] 冻结 Test 结果前的超参数。

---

## 十五、核心理论与方法参考

后续扩展论文时，至少需要系统整理以下方法文献：

- Koenker and Bassett (1978)：Quantile Regression；
- Bishop (1994)：Mixture Density Networks；
- Fama and MacBeth (1973)：横截面资产定价回归；
- Kupiec (1995)：VaR unconditional coverage；
- Christoffersen (1998)：VaR conditional coverage；
- Diebold, Gunther, and Tay (1998)：Density Forecast Evaluation 和 PIT；
- Gneiting and Raftery (2007)：Proper Scoring Rules 和 CRPS；
- Fissler and Ziegel (2016)：VaR 和 ES 的联合可识别与一致评分；
- 预测 tail risk、公司 expected shortfall 和横截面预期收益的相关资产定价文献。

正式论文需要进一步补充与“aggregate adverse states”“downside beta”“tail risk premium”“disaster risk”和“firm-level expected shortfall”直接相关的理论和实证研究。

---

## 十六、当前最重要的下一步

正式继续计算前，按以下顺序完成：

1. 冻结统一的滚动窗口生成器和单次训练 early-stopping 协议；
2. 完成 Rolling Normal 和 Linear Quantile；
3. 让 Fused K1、Fused K3、Structured K3 使用完全一致的无 warm-start 训练协议；
4. 在查看完整 Test 结果前固定 5% FZ0 和 0%--10% tail-CRPS 实现；
5. 检查 Structured components 是否稳定，再确定 adverse component matching；
6. 完成预测比较后冻结模型；
7. 最后运行总 ES、component contribution 和 severity 的资产定价检验。

这一路线将“数据—预测—结构解释—资产定价”连成一条可以审计的研究链条，并将预测模型选择与资产定价结果严格分离。
