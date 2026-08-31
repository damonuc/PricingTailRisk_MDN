# Machine Learning in Asset Pricing：毕业论文研究路线图

> 暂定总题目：**Machine Learning Conditional Return Distributions for Asset Pricing**  
> 中文题目：**资产定价中的机器学习：条件收益分布、经济结构与模型可靠性**  
> 当前版本：2026-08-10

---

## 一、重新定位毕业论文

毕业论文的中心不应当只是：

> 使用机器学习计算 Expected Shortfall，然后检验 ES 是否被定价。

这个表述中，机器学习只是一个计算工具，论文的中心仍然是尾部风险。

更合适的中心问题是：

> **传统资产定价中的机器学习主要预测条件平均收益，但投资者面对的是完整的未来收益分布。机器学习能否更准确地学习这一分布？经济结构能否提高模型的样本外泛化与可解释性？在市场分布发生变化时，我们什么时候可以相信这些预测？**

整篇毕业论文围绕三个机器学习问题展开：

1. **What to learn：**机器学习是否应当预测完整条件分布，而不只是条件均值？
2. **How to structure：**经济结构能否作为 neural network 的有效 inductive bias？
3. **When to trust：**面对 distribution shift 和 estimation uncertainty，机器学习预测什么时候可靠？

对应三篇文章：

| 章节 | 机器学习问题 | 资产定价问题 | 核心方法 |
|---|---|---|---|
| Chapter 1 | 如何学习完整条件收益分布？ | 机器学习预测的尾部风险是否获得收益补偿？ | MDN、Quantile ML、proper scoring rules |
| Chapter 2 | 经济结构能否改善学习与解释？ | 共同不利状态概率和公司损失程度中，哪一部分被定价？ | Structured MDN、shared latent state、architecture ablation |
| Chapter 3 | 非平稳环境下何时可以相信 ML？ | 尾部风险信号是否混合了经济风险与模型不确定性？ | Deep ensemble、distribution shift、adaptive learning |

因此，论文的统一主线是：

$$
\boxed{
\text{Prediction}
\rightarrow
\text{Economic Structure}
\rightarrow
\text{Reliability}
}
$$

尾部风险是贯穿三章的经济应用，但不是整篇毕业论文唯一的学术身份。

---

## 二、统一统计框架

### 2.1 传统机器学习资产定价

许多机器学习资产定价研究估计：

$$
\widehat\mu_{i,t}
=
E[R_{i,t+1}\mid X_{i,t},M_t],
$$

其中：

- $R_{i,t+1}$ 是股票 $i$ 的下一期收益；
- $X_{i,t}$ 是公司特征；
- $M_t$ 是宏观和市场状态变量。

这种方法集中在条件均值，因此无法完整描述波动率、偏度、多峰性以及尾部损失。

### 2.2 本毕业论文的研究对象

本论文直接学习条件分布：

$$
F_{i,t}(r)
=
P(R_{i,t+1}\le r\mid X_{i,t},M_t).
$$

一旦得到 $F_{i,t}$，可以统一计算：

$$
\mu_{i,t}=E[R_{i,t+1}\mid X_{i,t},M_t],
$$

$$
VaR_{i,t}(\alpha)=F_{i,t}^{-1}(\alpha),
$$

$$
ES_{i,t}^{\mathrm{return}}(\alpha)
=
E[R_{i,t+1}\mid R_{i,t+1}\le VaR_{i,t}(\alpha),X_{i,t},M_t].
$$

如果论文使用正数表示损失，则定义：

$$
ES_{i,t}^{\mathrm{loss}}(\alpha)
=
-ES_{i,t}^{\mathrm{return}}(\alpha).
$$

这里不需要假定 $ES^{\mathrm{loss}}$ 必须为正。它只是 return-tail ES 的符号变换。实证表格和图中必须始终明确使用的是 return ES 还是 loss ES。

### 2.3 统一的模型评估原则

模型选择只能使用 Train 和 Validation 中的预测损失，不得使用 Test portfolio returns。

主要预测指标包括：

#### Negative log-likelihood

$$
NLL
=
-\frac{1}{N}\sum_{i,t}\log \widehat f_{i,t}(R_{i,t+1}).
$$

NLL 越低越好，重点评价实现值附近的预测密度。

#### Continuous ranked probability score

$$
CRPS(F,y)
=
\int_{-\infty}^{\infty}
\left(F(z)-\mathbf 1\{y\le z\}\right)^2dz.
$$

CRPS 越低越好，同时评价 calibration 和 sharpness。

#### Quantile score

$$
QS_\alpha(q,y)
=
(\alpha-\mathbf 1\{y<q\})(y-q).
$$

用于评价特定分位数，越低越好。

#### VaR coverage

正确校准的模型应满足：

$$
P(R_{i,t+1}\le \widehat{VaR}_{i,t}(\alpha))=\alpha.
$$

#### PIT calibration

$$
PIT_{i,t}
=
\widehat F_{i,t}(R_{i,t+1}).
$$

若分布预测正确，PIT 应接近 $U(0,1)$。

#### Joint VaR--ES score

VaR 和 ES 应当联合评价，使用 strictly consistent joint score，例如 Fissler--Ziegel score。具体公式和符号必须与代码实现完全一致。

所有 panel loss 首先按月聚合，再对月份等权，防止股票数量多的月份支配比较。

---

## 三、Chapter 1：Machine Learning the Conditional Distribution of Stock Returns

### 3.1 暂定题目

**Machine Learning the Conditional Distribution of Individual Stock Returns: Forecasting and Pricing Tail Risk**

### 3.2 核心机器学习问题

现有机器学习资产定价主要预测条件均值。本章研究：

1. ML 能否有效预测 individual stock return 的完整条件分布？
2. Mixture model 是否优于单一正态分布和分位数模型？
3. 把宏观信息与公司信息放入不同网络通道，能否在保持预测能力的同时提供经济解释？
4. 从预测分布提取的尾部风险是否与下一期股票收益相关？

### 3.3 当前模型体系

#### Rolling Normal

使用预测月份之前最多 192 个日历月的公司历史收益：

$$
R_{i,t+1}\sim N(\widehat\mu_{i,t},\widehat\sigma_{i,t}^2).
$$

它是不使用公司特征和宏观特征的传统基准。

#### Linear Quantile

对每个 $\alpha$ 分别估计：

$$
Q_\alpha(R_{i,t+1}\mid X_{i,t},M_t)
=
\beta_{0,\alpha}+X_{i,t}'\beta_{x,\alpha}+M_t'\beta_{m,\alpha}+Industry_{i,t}'\gamma_\alpha.
$$

这是可以使用全部输入信息但只允许线性关系的基准。

#### Fused Normal K=1

所有特征进入同一个神经网络：

$$
h_{i,t}=g_\theta(X_{i,t},M_t,Industry_{i,t}),
$$

$$
R_{i,t+1}\mid X_{i,t},M_t\sim
N(\mu_\theta(h_{i,t}),\sigma_\theta^2(h_{i,t})).
$$

它识别非线性神经网络相对于线性模型的增量价值。

#### Fused Normal K=3

$$
f_{i,t}(r)
=
\sum_{k=1}^{3}
\pi_{i,k,t}
\phi\left(
\frac{r-\mu_{i,k,t}}{\sigma_{i,k,t}}
\right)\frac{1}{\sigma_{i,k,t}}.
$$

所有参数均由 fused features 决定。它识别 mixture flexibility 的价值。

#### Structured Normal K=3

当前设定为：

$$
\pi_{k,t}=g_{\pi,k}(M_t),
$$

$$
(\mu_{i,k,t},\sigma_{i,k,t})
=
g_{\theta,k}(X_{i,t},Industry_{i,t}).
$$

宏观变量决定共同 mixture weights，公司特征决定 component-specific return distribution。

### 3.4 为了强化 ML 论文身份，需要增加什么

#### 必做：模型比较形成明确的 learning ladder

模型顺序必须清楚地对应不同能力：

$$
\text{Historical}
\rightarrow
\text{Linear}
\rightarrow
\text{Nonlinear K1}
\rightarrow
\text{Nonlinear Mixture}
\rightarrow
\text{Economically Structured Mixture}.
$$

每次比较只改变一个关键维度：

| 比较 | 识别的 ML 问题 |
|---|---|
| Linear Quantile vs Fused K1 | 非线性及变量交互是否有价值？ |
| Fused K1 vs Fused K3 | 学习 mixture 是否有价值？ |
| Fused K3 vs Structured K3 | 经济结构是否有价值？ |
| Rolling Normal vs conditional models | 使用条件信息是否有价值？ |

#### 建议：增加一个 tree-based benchmark

如果计算允许，增加一个基于树的 conditional quantile 模型，例如 gradient-boosted quantile trees。原因不是为了堆模型，而是防止文章只能回答“神经网络是否优于线性模型”。

这个 benchmark 可作为投稿前增强项，不需要立即阻碍现有论文完成。

#### 必做：architecture ablation

至少报告：

1. 去掉 macro inputs；
2. 去掉 micro inputs；
3. 去掉 industry embedding；
4. $K=1$ 与 $K=3$；
5. fused 与 structured。

不需要对每个 ablation 做完整资产定价检验。其任务是解释预测表现来自哪里。

#### 必做：多个随机种子

神经网络的预测结果不能只依赖一个 seed。对最终候选模型至少运行 3 个固定 seed；理想情况为 5 个。

报告：

$$
\overline L_m
=
\frac{1}{S}\sum_{s=1}^{S}L_{m,s},
$$

以及 seed 间标准差。

这一步既是 reproducibility，也是 Chapter 3 的数据基础。

### 3.5 本章资产定价部分

预测评价和资产定价必须严格分开。

#### 预测问题

哪个模型的 density、quantile 和 tail forecast 更好？

#### 资产定价问题

给定事前预测的 tail risk：

$$
Tail_{i,t}=ES_{i,t}^{\mathrm{loss}}(\alpha),
$$

高 tail-risk 公司是否具有更高或更低的下一期平均收益？

主要检验：

1. 月度十分位组合；
2. equal-weighted 和 value-weighted returns；
3. H--L factor alpha；
4. Fama--MacBeth regressions；
5. size、book-to-market、momentum、historical volatility 和 industry controls；
6. 不同市场状态下的 conditional pricing。

资产定价收益不能用于选择 $K$、网络宽度、seed 或 stopping epoch。

### 3.6 本章可以成立的贡献

最合理的贡献不是强行声称 Structured K3 在所有 tail metrics 上都最好，而是：

1. 把 ML asset pricing 从 conditional mean 扩展到 conditional distribution；
2. 统一比较 parametric、quantile、fused neural density 和 structured neural density；
3. 展示 mixture flexibility 对整体 density forecast 的价值；
4. 从严格样本外预测分布中构造可用于资产定价的 tail-risk measure；
5. 为后续研究 economic structure 和 model reliability 建立统一实验平台。

### 3.7 Chapter 1 完成标准

- [ ] 所有模型使用完全相同的 Test observations；
- [ ] 所有模型使用相同的 rolling calendar windows；
- [ ] 模型选择只使用 Validation prediction loss；
- [ ] 输出 NLL、CRPS、tail-CRPS、PIT、VaR coverage 和 joint VaR--ES score；
- [ ] loss difference 按月计算，并使用 Newey--West inference；
- [ ] 最终神经网络至少报告 3 个 seed；
- [ ] 完成主要 architecture ablation；
- [ ] 资产定价不参与预测模型选择；
- [ ] 论文结论与实际预测结果一致，不要求 Structured K3 赢得所有指标。

---

## 四、Chapter 2：Economic Structure as Inductive Bias

### 4.1 暂定题目

**Economic Structure as Inductive Bias in Machine-Learning Asset Pricing**

或者更具体：

**Learning Common Adverse States and Firm-Specific Loss Distributions**

### 4.2 核心机器学习问题

一个完全 fused neural network 具有较高灵活性，但可能：

1. 在有限月份上过拟合；
2. 无法稳定识别共同经济状态；
3. 将宏观状态和公司特征的作用混合在一起；
4. 在新的经济状态下泛化较差。

本章的问题是：

> **将经济结构写入网络架构，能否作为有效的 inductive bias，提高样本效率、状态稳定性、样本外泛化和经济可解释性？**

### 4.3 当前 Structured K3 的限制

当前逐股票 likelihood 是：

$$
\mathcal L_{\mathrm{marginal},t}
=
\prod_i
\left[
\sum_{k=1}^{K}
\pi_{k,t}f_{i,k,t}(R_{i,t+1})
\right].
$$

虽然同月股票共享 $\pi_{k,t}$，但模型相当于允许每只股票独立抽取 component。因此，$\pi_{k,t}$ 是共同的 marginal mixture weight，不是“全市场当月共同 realized state”的概率。

真正的共同状态模型为：

$$
Z_{t+1}\sim \operatorname{Categorical}(\pi_t),
$$

$$
R_{i,t+1}\mid Z_{t+1}=k,X_{i,t}
\sim f_{i,k,t},
$$

对应联合 likelihood：

$$
\mathcal L_{\mathrm{joint},t}
=
\sum_{k=1}^{K}
\pi_{k,t}
\prod_i f_{i,k,t}(R_{i,t+1}).
$$

这里一个月份只有一个共同 realized state，但每只股票在该状态下仍有自己的条件均值和波动率。

### 4.4 本章的模型阶梯

#### Model A：Fused K3

完全数据驱动的 black-box benchmark。

#### Model B：Structured marginal K3

当前模型：macro 决定 $\pi$，micro 决定 $\mu$ 和 $\sigma$。

#### Model C：Structured K3 with macro modulation

允许 macro 通过 FiLM 或低维 interaction 影响公司损失程度：

$$
h_{i,t}^{*}
=
\gamma(M_t)\odot h_i(X_{i,t})+\beta(M_t).
$$

它检验严格通道分离是否过强。

#### Model D：Shared-state K2

以两个共同状态为主模型：Normal 与 Bad。

$$
p_t=P(Z_{t+1}=Bad\mid M_t),
$$

$$
R_{i,t+1}\mid Z_{t+1}=Bad
\sim N(\mu_{i,B,t},\sigma_{i,B,t}^2).
$$

先使用 K=2 是为了减少 label switching，提高状态识别的稳定性。

### 4.5 可执行的两阶段开发方案

#### Phase A：Observed-state MVP

1. 在每个 Train + Validation window 中构造 aggregate market outcomes；
2. 使用预先规定的规则识别 Normal/Bad state；
3. 用 macro variables 预测下一期 state probability $p_t$；
4. 分状态估计 firm conditional distributions；
5. 在 Test 中固定规则并严格样本外预测。

这个版本不是最终模型，但能快速检验研究问题是否有经济意义。

#### Phase B：Latent shared-state model

直接最大化共同状态 joint likelihood，或使用 EM/variational inference：

$$
q_{k,t}
=
P(Z_{t+1}=k\mid R_{1:N_t,t+1},X_t,M_t).
$$

E-step 估计 $q_{k,t}$；M-step 更新 state-probability network 和 firm-distribution network。

### 4.6 必须解决的 ML 问题

#### Label switching

不能每个窗口事后把均值最低的 component 随意叫作 Bad state。需要一个预先固定且跨窗口一致的识别规则，例如：

$$
E[R_{m,t+1}\mid Z=Bad]
<
E[R_{m,t+1}\mid Z=Normal].
$$

也可以通过有序参数化或 anchoring loss 约束状态含义。

#### Month-level effective sample size

宏观网络真正拥有的独立时间观测约为几百个月，而不是数百万 firm-month observations。因此需要：

1. 小型 macro network；
2. 强 weight decay；
3. dropout 或平滑约束；
4. month-level validation；
5. 限制 macro interaction 的维度。

#### Cross-sectional dependence

同月股票不能被当成完全独立的宏观信息。训练 loss、标准误和评估都应尊重 month clustering。

#### Matched capacity

比较 fused 和 structured architectures 时，应尽量匹配参数量和训练预算，避免把“结构优势”与“模型大小差异”混为一谈。

### 4.7 预测层面的评价

除 Chapter 1 的 density metrics 外，本章重点评价：

#### Sample efficiency

分别使用 60、120 和 180 个月训练历史，检验结构化模型是否在有限样本中表现更稳定。

#### State stability

1. 相邻窗口的状态含义是否一致；
2. predicted bad-state probability 是否连续；
3. 不同 seed 下状态概率相关性；
4. 状态是否与 recession、market drawdown、VIX 或 aggregate volatility 有一致关系。

#### Stress generalization

分别报告 normal periods 与 stress periods 的 NLL、CRPS、tail score 和 coverage。

#### Architecture ablation

逐步改变 macro-to-$\pi$、macro-to-severity、shared state 等结构，确定增益来自哪里。

### 4.8 资产定价分解

共同状态模型可以生成：

$$
Probability_t=p_t,
$$

$$
Severity_{i,t}
=
-E[R_{i,t+1}\mid Z_{t+1}=Bad,X_{i,t},M_t].
$$

总尾部暴露可以表示为：

$$
TailExposure_{i,t}=p_t\times Severity_{i,t}.
$$

注意：同一个月内 $p_t$ 对所有股票相同，因此它不能单独解释当月横截面排序；它主要解释 severity price 随时间为什么变化。

核心回归：

$$
R_{i,t+1}
=
a_t+\gamma_t Severity_{i,t}
+\delta_t'Controls_{i,t}+\varepsilon_{i,t+1},
$$

以及：

$$
\widehat\gamma_t
=
a+b\,p_t+u_t.
$$

### 4.9 本章的真正贡献

本章不能只说“多设计了一个神经网络”。目标应当是：

1. 证明 economic structure 可以作为有效 inductive bias；
2. 区分 marginal mixture weights 与 shared realized state；
3. 量化结构对 sample efficiency、stability 和 stress generalization 的影响；
4. 将 black-box density forecast 分解为共同状态概率和公司损失程度；
5. 展示预测结构如何转化为新的资产定价解释。

### 4.10 Chapter 2 go/no-go 标准

继续发展完整模型，需要至少满足以下两项：

- [ ] shared state 在不同 seed 和窗口中具有稳定经济含义；
- [ ] structured/shared model 的预测表现不显著差于 fused K3；
- [ ] stress-period calibration 明显改善；
- [ ] probability 或 severity decomposition 提供当前 Chapter 1 没有的经济结果；
- [ ] matched-capacity ablation 支持 inductive-bias 解释。

如果以上均不成立，本章可缩减为 Chapter 1 的 architecture section，而不应强行独立成篇。

---

## 五、Chapter 3：Reliable Machine Learning under Distribution Shift

### 5.1 暂定题目

**When Should Investors Trust Machine Learning? Distribution Shift and Forecast Uncertainty in Asset Pricing**

### 5.2 核心机器学习问题

金融数据具有明显的非平稳性：

$$
P_{t}(R,X,M)
\neq
P_{t+h}(R,X,M).
$$

即使一个模型在平均样本外指标上表现良好，也可能在新经济状态下严重失准。

本章研究：

1. 能否在实现收益之前识别模型可能失效的时期和股票？
2. seed、model 和 training-window 之间的预测分歧，能否衡量 epistemic uncertainty？
3. distribution shift 是否解释 NLL、tail calibration 和资产定价信号的变化？
4. adaptive ensemble 或 state-adaptive training 能否改善可靠性？

### 5.3 区分两类不确定性

#### Aleatoric uncertainty

来自下一期收益本身的随机性，由预测分布描述：

$$
Var(R_{i,t+1}\mid X_{i,t},M_t,\widehat\theta).
$$

#### Epistemic uncertainty

来自有限样本、模型结构、初始化和非平稳性的参数不确定性。

训练 $S$ 个模型后，可以定义 ES disagreement：

$$
U_{i,t}^{ES}
=
\sqrt{
\frac{1}{S-1}
\sum_{s=1}^{S}
\left(
ES_{i,t}^{(s)}-\overline{ES}_{i,t}
\right)^2
}.
$$

其中：

$$
\overline{ES}_{i,t}
=
\frac{1}{S}\sum_{s=1}^{S}ES_{i,t}^{(s)}.
$$

本章的重要经济区分是：

> 高预测尾部风险是经济风险；高模型分歧是研究者对该风险估计缺乏信心。两者不能混为一谈。

### 5.4 最小可行版本

不需要一开始开发复杂的新模型。首先使用 Chapter 1 已有架构：

1. 对 Fused K3 和 Structured K3 各运行 5 个固定 seed；
2. 保存每个 seed 的完整 OOS density parameters、VaR 和 ES；
3. 构造 ensemble predictive density；
4. 计算 stock-month model disagreement；
5. 检验 disagreement 是否预测下一期 forecast loss。

Ensemble density 为：

$$
\widehat f_{i,t}^{ens}(r)
=
\frac{1}{S}\sum_{s=1}^{S}\widehat f_{i,t}^{(s)}(r).
$$

必须先平均 density，再从 ensemble density 计算 VaR 和 ES；不能简单平均各模型的 VaR。

### 5.5 Distribution shift 的测量

构造月度 shift 指标：

#### Macro shift

当前 macro vector 与训练分布中心之间的距离：

$$
D_t^{macro}
=
(M_t-\widehat\mu_M)'
\widehat\Sigma_M^{-1}
(M_t-\widehat\mu_M).
$$

高维情况下使用 shrinkage covariance 或低维 PCA scores。

#### Characteristic shift

比较当前月公司特征分布与训练期分布，可使用 Wasserstein distance、maximum mean discrepancy 或分位数差异。

#### Predictive calibration shift

在滚动窗口内观察 PIT、coverage error 和 monthly NLL 是否恶化。该指标只能用于事后评价；若用于实时决策，只能使用当时已经实现的历史误差。

### 5.6 核心假设

#### H1：不确定性预测误差

$$
Loss_{i,t+1}
=
a+bU_{i,t}+Controls_{i,t}'\delta+\varepsilon_{i,t+1},
$$

预期 $b>0$。

#### H2：distribution shift 预测模型失准

$$
MonthlyLoss_{t+1}
=
a+bD_t+u_{t+1},
$$

预期 $b>0$。

#### H3：structured model 在 shift 下更加稳定

比较不同 shift quintile 中：

$$
L_{Structured,t}-L_{Fused,t}.
$$

如果经济结构是有效 inductive bias，高 shift 状态下 structured model 的相对表现应当更稳定。

#### H4：ensemble 改善 calibration

比较单一 seed 与 ensemble 的 NLL、CRPS、PIT 和 VaR coverage。

### 5.7 可进一步开发的 adaptive learning

在 MVP 成立后，再开发以下一种方法，不需要全部都做。

#### State-adaptive training weights

对历史月份 $s$ 的训练权重设为：

$$
w_{s\mid t}
\propto
\exp(-\lambda\,d(M_s,M_t))
\exp(-\rho(t-s)).
$$

第一项提高与当前宏观状态相似月份的权重；第二项允许 recency weighting。

$\lambda$ 和 $\rho$ 只能根据 Validation NLL/CRPS 选择。

#### Adaptive ensemble

根据最近可用 validation performance 对模型加权：

$$
a_{m,t}
=
\frac{\exp(-\eta L_{m,t}^{past})}
{\sum_j\exp(-\eta L_{j,t}^{past})}.
$$

$$
f_{i,t}^{adaptive}(r)
=
\sum_m a_{m,t}f_{i,t}^{(m)}(r).
$$

所有权重只能使用预测时点已经知道的信息。

### 5.8 资产定价应用

本章不应简单重复 Chapter 1 的 ES portfolio sorts。新的经济问题是：

> 当 ML 模型本身不确定时，tail-risk signal 的收益关系是否减弱、增强或反转？

可以估计：

$$
R_{i,t+1}
=
a_t+\gamma_{1,t}ES_{i,t}
+\gamma_{2,t}U_{i,t}
+\gamma_{3,t}ES_{i,t}U_{i,t}
+Controls_{i,t}'\delta_t+\varepsilon_{i,t+1}.
$$

需要谨慎解释：$U_{i,t}$ 的收益关系未必是风险补偿，也可能反映 limits to learning、illiquidity 或数据质量。论文的主要贡献仍应是 forecast reliability，而不是把所有结果都称为新的 risk premium。

### 5.9 Chapter 3 成功标准

- [ ] disagreement 在实现结果之前可计算；
- [ ] disagreement 显著预测 OOS forecast loss 或 calibration failure；
- [ ] distribution shift 与模型失准具有稳定关系；
- [ ] ensemble 或 adaptive method 在严格 OOS 中改善至少一组主要 proper scores，同时不恶化 tail calibration；
- [ ] 结果对 model class、seed 和危机样本不是完全依赖；
- [ ] 明确区分 economic tail risk 与 model uncertainty。

如果 disagreement 不能预测 forecast failure，本章不应继续加入复杂方法；可以改为研究训练窗口、recency weights 和 regime-specific generalization。

---

## 六、三章之间如何避免重复

### Chapter 1 回答“预测对象”

比较 conditional mean、quantiles 和 full density，建立预测与资产定价事实。

### Chapter 2 回答“模型结构”

研究经济限制如何改变模型学习、样本效率、状态稳定性和经济解释。

### Chapter 3 回答“预测可靠性”

研究模型不确定性和时间分布变化，说明什么时候 ML forecast 值得信任。

三章不能仅仅是同一模型换参数：

| 章节 | 主要因变量/评价对象 | 新方法 | 主要经济输出 |
|---|---|---|---|
| Chapter 1 | OOS density and tail forecast | MDN distribution learning | Predicted ES pricing |
| Chapter 2 | Structure、state stability、generalization | Shared-state structured network | Probability vs severity |
| Chapter 3 | Forecast failure and uncertainty | Ensemble/adaptive learning | Economic risk vs model uncertainty |

---

## 七、机器学习论文必须遵守的实验设计

### 7.1 时间顺序

所有模型采用相同的滚动窗口：

- Train：180 个月；
- Validation：12 个月；
- Test：随后 12 个月；
- 每次向前移动 12 个月。

主结果中每个窗口独立初始化，不使用 warm start。Warm start 可以作为计算效率或稳定性的 robustness。

### 7.2 Hyperparameter selection

超参数只能根据 Validation proper score 选择。

推荐优先级：

1. Validation NLL；
2. Validation CRPS；
3. 如果研究重点为左尾，预先规定 NLL 与 tail score 的组合；
4. 不允许根据 Test returns 选择模型。

### 7.3 Seed protocol

在实验开始前固定 seed list，例如：

```text
[42, 123, 2026, 3407, 8888]
```

不能因为某个 seed 的资产定价结果更好而只报告该 seed。

### 7.4 比较公平性

模型之间尽量保持：

1. 相同输入；
2. 相同 Train/Validation/Test observations；
3. 相同 preprocessing information set；
4. 相似参数预算；
5. 相同 early-stopping rule；
6. 相同计算预算或明确报告差异。

### 7.5 统计推断

预测 loss 先聚合至月份：

$$
\overline L_{m,t}
=
\frac{1}{N_t}\sum_{i=1}^{N_t}L_{m,i,t}.
$$

模型 $A$ 与 $B$ 的月度差异：

$$
d_t=\overline L_{A,t}-\overline L_{B,t}.
$$

对 $E[d_t]$ 使用 Newey--West 标准误。负值表示模型 $A$ 的 loss 更低。

### 7.6 Reproducibility

每次完整运行必须保存：

1. timestamp 与 experiment ID；
2. git commit hash；
3. data version；
4. feature list；
5. hyperparameters；
6. seed；
7. window dates；
8. epoch history；
9. OOS forecasts；
10. prediction metrics；
11. runtime 和 hardware。

---

## 八、执行顺序

### Stage 1：冻结 Chapter 1 主结果

预计时间：1--2 个月。

1. 修正文字与代码定义不一致；
2. 固定最终数据版本和 sample construction；
3. 完成模型 prediction comparison；
4. 至少补充 3 个 seed；
5. 完成必要 architecture ablation；
6. 完成资产定价主表与稳健性；
7. 将当前论文改写为“distributional ML in asset pricing”。

### Stage 2：Chapter 2 MVP

预计时间：2--3 个月。

1. 先实现 K=2 observed-state model；
2. 生成严格 OOS $p_t$ 和 firm severity；
3. 比较 marginal structured model 和 shared-state MVP；
4. 检验状态稳定性与 stress generalization；
5. 决定是否开发 joint latent-state likelihood。

### Stage 3：Chapter 2 full model

预计时间：3--5 个月。

1. 开发 shared-state likelihood；
2. 解决 label switching；
3. 完成 matched-capacity ablation；
4. 完成 probability/severity pricing decomposition；
5. 写成独立 working paper。

### Stage 4：Chapter 3 MVP

预计时间：2--3 个月。

1. 对最终模型运行 5 个 seed；
2. 保存 seed-level density predictions；
3. 构造 ensemble density 和 disagreement；
4. 构造 macro/characteristic shift measures；
5. 检验 uncertainty 是否预测 forecast failure。

### Stage 5：Chapter 3 adaptive model

预计时间：3--4 个月。

仅当 MVP 结果支持研究问题时：

1. 选择 state-adaptive weighting 或 adaptive ensemble 中的一种；
2. 使用 validation 选择超参数；
3. 完成严格 OOS comparison；
4. 写作并形成第三篇文章。

### Stage 6：毕业论文整合

预计时间：1--2 个月。

1. 写统一 introduction；
2. 写统一 literature review；
3. 解释三章的共同数据与不同识别问题；
4. 统一符号和预测时间线；
5. 写 conclusion：prediction、structure、reliability；
6. 删除三章之间重复的数据说明和模型背景。

---

## 九、未来六周的具体任务

### Week 1：重写 Chapter 1 定位

- [ ] 将论文标题和 introduction 改成 conditional distribution learning；
- [ ] 明确每个 baseline 识别什么；
- [ ] 将 tail-risk pricing 写成 economic application；
- [ ] 不声称 Structured K3 必须在所有指标上最好。

### Week 2：锁定实验协议

- [ ] 固定 data version；
- [ ] 固定 seed list；
- [ ] 固定 Train/Validation/Test windows；
- [ ] 固定 model-selection score；
- [ ] 固定最终 model configurations。

### Week 3--4：补充 ML 证据

- [ ] 运行必要 seeds；
- [ ] 完成 K1/K3、fused/structured 比较；
- [ ] 完成 macro/micro/industry ablation；
- [ ] 汇总 seed stability 和 runtime。

### Week 5：完成 Chapter 1 结果结构

- [ ] Table 1：data and sample；
- [ ] Table 2：model architecture and parameter count；
- [ ] Table 3：average OOS forecast scores；
- [ ] Table 4：pairwise monthly loss tests；
- [ ] Figure 1：model architecture；
- [ ] Figure 2：PIT and coverage；
- [ ] Figure 3：cumulative loss differences；
- [ ] Table 5--7：asset-pricing results。

### Week 6：启动 Chapter 2

- [ ] 构造 aggregate monthly outcomes；
- [ ] 定义不依赖未来信息的 Bad-state rule；
- [ ] 运行 observed-state K2 MVP；
- [ ] 画出 OOS $p_t$ 时间序列；
- [ ] 检查状态与市场结果是否一致。

---

## 十、投稿层面的最低要求

### Chapter 1

不能只展示神经网络比 Rolling Normal 好。至少需要：

1. 与有相同输入信息的 linear/quantile benchmark 比较；
2. 分离 nonlinearity、mixture 和 structure 的价值；
3. 严格 OOS probabilistic evaluation；
4. 资产定价结果不参与模型选择；
5. 有清晰的经济应用和稳健性。

### Chapter 2

必须是实质性的新 architecture 或 learning framework，而不是把 FiLM 打开再运行一次。共同状态、样本效率、状态稳定性和 stress generalization 至少要形成一条完整贡献。

### Chapter 3

必须在实现收益前构造 uncertainty/shift measure，并证明它能预测模型失准。仅仅报告不同 seed 的结果不够构成独立文章。

---

## 十一、风险与应对

| 风险 | 含义 | 应对 |
|---|---|---|
| Structured K3 尾部指标不领先 | 结构提高解释力但未提高所有预测分数 | 将“competitive forecast + interpretable structure”作为准确表述 |
| Chapter 1 与 Chapter 2 太相似 | 可能被认为拆分同一篇文章 | Chapter 2 必须实现 shared state，并以 inductive bias/generalization 为中心 |
| Chapter 3 计算成本过高 | 多 seed、多个窗口耗时 | 先用 3 seed 和部分窗口做 MVP，成立后扩展到 5 seed |
| 状态 label 不稳定 | 经济解释无法跨窗口比较 | 使用 K=2、ordered parameterization 和 anchor rule |
| Distribution shift 方法过多 | 容易变成方法堆积 | MVP 后只选择一种 adaptive method |
| 资产定价结果不显著 | 容易为了收益反复调模型 | 将 prediction 与 pricing 分离，不用 returns 调参 |
| ML 贡献不清楚 | 被认为只是把 NN 用在 ES | 每章分别强调 prediction target、inductive bias 和 reliability |

---

## 十二、需要避免的方向

1. 不把“尝试更多神经网络”本身当成贡献；
2. 不把每种 $K$、分布或 training trick 拆成一篇文章；
3. 不根据 portfolio return 决定模型、seed 或 hyperparameters；
4. 不为了让 Structured K3 获胜而删除不利 benchmark；
5. 不将 marginal mixture weight 直接称为共同 realized-state probability；
6. 不同时开发 HMM、Transformer、VAE、GAN 和 diffusion model；
7. 不在 Chapter 1 尚未冻结时大规模重写所有训练代码；
8. 不把 statistical significance 当成唯一成功标准；
9. 不忽视计算成本、seed sensitivity 和 reproducibility；
10. 不把风险预测、风险补偿和风险中性概率混为一谈。

---

## 十三、核心文献路线

### Machine learning and expected returns

- Gu, Kelly, and Xiu (2020), [Empirical Asset Pricing via Machine Learning](https://doi.org/10.1093/rfs/hhaa009), *Review of Financial Studies*.
- Gu, Kelly, and Xiu (2021), [Autoencoder Asset Pricing Models](https://doi.org/10.1016/j.jeconom.2020.07.009), *Journal of Econometrics*.
- Kelly, Pruitt, and Su (2019), [Characteristics Are Covariances: A Unified Model of Risk and Return](https://doi.org/10.1016/j.jfineco.2019.05.001), *Journal of Financial Economics*.
- Chen, Pelger, and Zhu (2023), [Deep Learning in Asset Pricing](https://doi.org/10.1287/mnsc.2023.4695), *Management Science*.

### Probabilistic forecasting

- Gneiting and Raftery (2007), [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437), *Journal of the American Statistical Association*.
- Gneiting, Balabdaoui, and Raftery (2007), [Probabilistic Forecasts, Calibration and Sharpness](https://doi.org/10.1111/j.1467-9868.2007.00587.x), *Journal of the Royal Statistical Society: Series B*.

### Tail risk forecasting and evaluation

- Fissler and Ziegel (2016), Higher order elicitability and Osband's principle, *Annals of Statistics*.
- Qiu, Lazar, and Nakata (2024), [VaR and ES Forecasting via Recurrent Neural Network-Based Stateful Models](https://doi.org/10.1016/j.irfa.2024.103102), *International Review of Financial Analysis*.

### ML uncertainty and distribution shift

- Kumar, Ma, Liang, and Raghunathan (2022), [Calibrated Ensembles Can Mitigate Accuracy Tradeoffs under Distribution Shift](https://proceedings.mlr.press/v180/kumar22a.html), *UAI*.
- Liao, Ma, Neuhierl, and Schilling (2025), [The Uncertainty of Machine Learning Predictions in Asset Pricing](https://doi.org/10.2139/ssrn.5160731), working paper.

使用这些文献时，需要明确本文与 conditional-mean ML、ML-SDF、time-series VaR/ES 和 generic uncertainty estimation 的区别。

---

## 十四、最终论文陈述

整篇毕业论文最简洁的陈述可以是：

> This dissertation studies how machine learning can move empirical asset pricing beyond point forecasts of expected returns. The first chapter learns the full conditional distribution of individual stock returns and uses it to measure priced downside risk. The second chapter studies whether economic structure serves as a useful inductive bias for learning common market states and firm-specific loss distributions. The third chapter examines when these forecasts can be trusted by separating economic risk from model uncertainty under distribution shift.

对应中文：

> 本毕业论文研究机器学习如何使实证资产定价超越条件平均收益的点预测。第一章学习个股的完整条件收益分布，并利用该分布度量被定价的下行风险；第二章研究经济结构能否作为有效的归纳偏置，帮助模型学习共同市场状态和公司特定损失分布；第三章研究在分布变化的市场环境中何时可以相信这些预测，并区分经济风险与模型不确定性。

最终主线为：

$$
\boxed{
\text{Machine Learning in Asset Pricing}
=
\text{Distributional Prediction}
+
\text{Economic Inductive Bias}
+
\text{Forecast Reliability}
}
$$

---

## 十五、当前最高优先级

### 现在开始做

1. 保留现有 MDN 项目作为 Chapter 1；
2. 将 Chapter 1 的 framing 从“ES paper”改为“distributional ML paper”；
3. 固定最终实验协议并补充多 seed 和关键 ablation；
4. 完成 Chapter 1 后，先开发 Chapter 2 的 K=2 observed-state MVP；
5. 从现在开始保存 seed-level forecasts，为 Chapter 3 准备数据。

### 现在不要做

1. 不推倒重做当前全部模型；
2. 不立刻加入大量新的深度学习架构；
3. 不立刻购买或合并新的大型数据库；
4. 不把 OptionMetrics 或 intermediary constraints 设为主线；
5. 不在 Chapter 2 MVP 尚未成立前开发复杂 joint neural HMM；
6. 不在 Chapter 3 uncertainty hypothesis 尚未成立前开发 adaptive network。

这条路线既保留你已经完成的大量工作，也使三章都具有清楚、独立而统一的 machine-learning contribution。
