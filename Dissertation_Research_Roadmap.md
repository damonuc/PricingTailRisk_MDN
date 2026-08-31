# 条件尾部风险、共同不利状态与股票预期收益：毕业论文研究路线图

> 当前版本：2026-08-10  
> 文档用途：固定毕业论文的统一研究问题、三章结构、执行顺序、完成标准和备选方案。  
> 第一章的详细模型与实证协议见 [`Research_Design_Roadmap.md`](Research_Design_Roadmap.md)。

---

## 一、总体判断与研究定位

当前 Structured MDN 研究适合作为毕业论文的第一篇实证章节，但不建议通过不断增加网络架构、分布类型和 robustness，把同一篇文章机械拉长为整篇毕业论文。

更好的毕业论文结构是围绕一个统一经济问题形成三篇相对独立、但逻辑连续的研究：

> 投资者究竟在为什么样的尾部风险支付补偿？这种补偿来自共同不利状态的发生概率、公司在不利状态中的损失程度，还是不利状态本身的风险价格？

建议毕业论文总题目为：

> **Conditional Tail Risk, Aggregate States, and Expected Stock Returns**  
> **条件尾部风险、共同状态与股票预期收益**

建议的三章主线为：

1. **预测损失：**估计每只股票的条件收益分布，计算 ES，并检验其是否以及何时被定价；
2. **识别状态：**建立真正由所有公司共享的 aggregate adverse state，分离 state probability 和 firm-specific severity；
3. **解释价格：**研究不利状态损失为何被定价，区分 physical risk 与 risk-neutral compensation，或检验金融中介约束机制。

三章的关系可以概括为：

$$
\boxed{
\text{Forecast the loss}
\;\longrightarrow\;
\text{Identify the common state}
\;\longrightarrow\;
\text{Explain the price of the loss}
}
$$

---

## 二、统一经济框架

### 2.1 两状态表示

设下一期存在 Good 和 Bad 两种共同状态：

$$
Z_{t+1}\in\{G,B\}.
$$

宏观信息决定下一期 Bad state 的 physical probability：

$$
p_t
=
P_t(Z_{t+1}=B).
$$

公司 $i$ 在 Good 和 Bad 状态下具有不同的条件收益分布：

$$
R_{i,t+1}\mid Z_{t+1}=s,X_{i,t}
\sim
f_{i,s,t},
\qquad s\in\{G,B\}.
$$

一种直观的公司特定 bad-state severity 定义是：

$$
Severity_{i,t}
=
\mu_{i,G,t}-\mu_{i,B,t}.
$$

该变量衡量同一家公司从 Good state 转入 Bad state 时，条件平均收益下降多少。也可以使用 Bad-state conditional loss ES 作为 severity 的替代定义。

### 2.2 简单资产定价直觉

在一个两状态定价模型中，个股风险补偿大致取决于：

$$
E_t[R_{i,t+1}^{e}]
\propto
p_t(1-p_t)
\left(M_{B,t}-M_{G,t}\right)
\left(\mu_{i,G,t}-\mu_{i,B,t}\right),
$$

其中：

- $p_t$ 是共同 Bad state 的发生概率；
- $M_{B,t}-M_{G,t}$ 是 Bad state 相对于 Good state 的边际价值差，可理解为 bad-state price of risk；
- $\mu_{i,G,t}-\mu_{i,B,t}$ 是公司在 Bad state 中的相对损失程度。

该框架说明，仅仅发现一家公司具有较大的 physical ES，并不自动意味着其未来收益一定较高。收益补偿还取决于：

1. 该损失是否发生在真正的共同不利状态；
2. 不利状态发生概率是否足够高；
3. 投资者是否对该状态赋予较高价格；
4. 该尾部指标是否同时包含 distress persistence、mispricing 或 volatility 等成分。

### 2.3 三章分别识别什么

| 章节 | 主要识别对象 | 核心输出 |
|---|---|---|
| Chapter 1 | Firm-level conditional tail loss | Total ES、component contribution、severity 及其状态依赖定价 |
| Chapter 2 | Common state probability 与 state-conditional firm loss | $p_t$、$f_{i,B,t}$、relative severity、expected bad-state loss |
| Chapter 3 | Bad-state price of risk 或其经济来源 | $ES^Q-ES^P$，或 intermediary constraint 对 severity price 的影响 |

---

## 三、Chapter 1：条件分布预测与状态依赖尾部风险定价

### 3.1 暂定题目

> **When Is Firm-Level Tail Risk Priced? Structured Density Forecasts and Expected Stock Returns**

### 3.2 核心研究问题

1. 使用公司、行业和宏观信息能否改善股票下一月收益分布的样本外预测？
2. Structured Normal K3 是否在主要尾部评分上优于或至少不劣于可比模型？
3. 预测 total ES 是否获得稳定的无条件收益补偿？
4. ES 中的 adverse contribution 和 adverse severity 是否具有不同的资产定价含义？
5. Tail-risk price 是否随可观察金融压力变化？

### 3.3 当前主要结果应当如何表述

当前最稳妥的主要结论是：

> **Firm-level tail-loss severity is not priced at a constant rate. Its cross-sectional price becomes more positive when financial conditions are weak.**

当前结果不支持以下过强表述：

- Structured MDN 在所有预测指标上都优于其他模型；
- 高 ES 股票始终获得正向收益补偿；
- 当前模型的 common mixture weight 已经是严格的 aggregate bad-state probability；
- 状态依赖尾部风险定价只存在于 Structured K3。

### 3.4 第一章需要完成的主要工作

#### A. 固定组合形成方法

主结果使用：

- 5% positive loss ES；
- NYSE 10%、20%、……、90% breakpoints；
- 所有合格股票参与组合；
- 十分组；
- Portfolio 10 minus Portfolio 1；
- equal-weighted 和 value-weighted returns；
- 1% 和 10% ES 作为 robustness。

需要明确：NYSE breakpoints 不保证全部股票在每组数量相同。全样本 breakpoints 或严格等量分组作为 robustness，而不是与主结果混在一起。

#### B. Severity 与 volatility 的直接 horse race

在同一个模型中估计：

$$
R_{i,t+1}
=
a_t
+\beta_1 Severity_{i,t}
+\beta_2 Severity_{i,t}\times Stress_t
+\beta_3 Volatility_{i,t}
+\beta_4 Volatility_{i,t}\times Stress_t
+\Gamma'Controls_{i,t}
+\varepsilon_{i,t+1}.
$$

目标是确认 severity 的状态依赖关系不是历史波动率状态关系的简单替代。

#### C. 特殊危机排除

至少重新估计：

- 排除 2008--2009；
- 排除 2020；
- 同时排除 2008--2009 和 2020；
- 保留现有 leave-one-window-out 结果。

#### D. 分组与样本稳健性

- NYSE breakpoints；
- full-sample breakpoints；
- 五组与十组；
- 排除 NYSE size bottom 20%；
- EW 与 VW；
- 1%、5%、10% ES；
- total ES、adverse contribution、adverse severity、residual severity。

#### E. 补充最终 summary statistics

至少报告：

- next-month realized return；
- total loss ES；
- adverse contribution；
- adverse severity；
- residual severity；
- historical volatility；
- market capitalization；
- book-to-market；
- momentum；
- 每月股票数量。

#### F. 明确 Fama--MacBeth 表达

每一个 tail signal 单独运行回归：

$$
R_{i,t+1}
=
a_t
+\gamma_{s,t}Z_t(Signal^{(s)}_{i,t})
+\delta_t'Z_t(Controls_{i,t})
+\varepsilon_{i,t+1}.
$$

其中 $s$ 分别表示：

- 五个模型各自的 total 5% ES；
- Structured adverse contribution；
- Structured adverse severity；
- residual adverse severity。

报告 $\gamma_{s,t}$ 的时间序列平均值，并使用 12 阶 Newey--West 标准误。Newey--West 不负责计算均值，只负责标准误和统计推断。

### 3.5 第一章完成标准

- [x] 冻结主要预测模型和模型时间戳；
- [x] 完成完整样本外预测比较；
- [x] 完成 total ES、contribution 和 severity 计算；
- [x] 完成基础 portfolio、factor alpha 和 Fama--MacBeth 检验；
- [x] 完成 observable stress 状态检验；
- [x] 完成主要论文草稿；
- [ ] 完成 severity 与 volatility 的直接 interaction horse race；
- [ ] 完成危机排除；
- [ ] 完成 full-sample breakpoint robustness；
- [ ] 补充 summary statistics；
- [ ] 统一表格、图、变量名称和论文数字；
- [ ] 导师完整审阅并冻结投稿版本。

### 3.6 不应继续无限扩张的内容

以下内容只作为 appendix robustness，不应成为第一章继续拖延的原因：

- Normal 改 Student-$t$；
- $K=2,3,4$；
- FiLM 与 no-FiLM；
- warm start 与 no warm start；
- 不同 hidden dimensions；
- 更多 quantile levels；
- 少量预测评分的反复调参。

---

## 四、Chapter 2：共同不利状态与公司特定损失程度

### 4.1 暂定题目

首选：

> **Common Adverse States and Firm-Specific Loss Severity**

备选：

> **The Probability and Severity of Firm-Level Losses in Aggregate Bad States**

### 4.2 研究动机

当前 Structured K3 的逐股票边际 likelihood 是：

$$
\prod_i
\left[
\sum_k\pi_{k,t}f_{i,k,t}(R_{i,t+1})
\right].
$$

它使同月股票共享相同的 mixture weights，但不要求所有股票在该月共享同一个 realized state。

真正的共同状态模型应当围绕：

$$
Z_{t+1}\in\{Normal,Bad\},
$$

并要求：

$$
R_{i,t+1}\mid Z_{t+1}=Bad
\sim f_{i,B,t}.
$$

此时，每只股票的 Bad-state distribution 才能解释为：

> 该股票在同一个全市场共同不利状态下的条件收益分布。

### 4.3 为什么从两个状态开始

- 月度样本只有约 408 个月；
- 每个 rolling Train + Validation 只有约 192 个独立月份；
- 真正严重的 Bad months 数量较少；
- 两状态更容易识别、解释和保持跨窗口标签稳定；
- 三状态可以在两状态模型成功后作为扩展，而不是起点。

### 4.4 第一阶段：Minimum Viable Model

#### Step 1：构造 aggregate outcome vector

每个月构造：

$$
A_{t+1}
=
\left[
R_{m,t+1},
Q_{0.05,t+1}^{CS},
Vol_{t+1}^{CS},
CrashShare_{t+1}
\right],
$$

其中：

- $R_{m,t+1}$：value-weighted market return；
- $Q_{0.05,t+1}^{CS}$：个股收益横截面 5% 分位数；
- $Vol_{t+1}^{CS}$：横截面股票收益波动率；
- $CrashShare_{t+1}$：收益低于预先固定阈值的股票比例。

需要先生成一个只有月份层面的 `aggregate_state_panel`，每个月只能有一行。

#### Step 2：在 Train + Validation 中识别 realized state

MVP 可以先使用透明的 observed-state 定义。例如，仅使用 Train + Validation 分位点，将以下月份定义为 Bad：

$$
Bad_{t+1}=1
$$

如果 market return 或 cross-sectional tail outcome 低于训练样本中的固定不利阈值。

候选定义应当预先固定，不能根据 Test 资产定价结果选择。建议只比较少量经济上合理的定义：

1. market return bottom 10%；
2. cross-sectional 5% return quantile bottom 10%；
3. 两者的标准化平均指标 bottom 10% 或 15%。

标签只能在对应 rolling window 的 Train + Validation 中确定，不能使用 Test outcome 调整。

#### Step 3：预测下一期 Bad-state probability

使用月末 $t$ 的宏观变量预测：

$$
p_t
=
P(Z_{t+1}=Bad\mid M_t).
$$

第一版依次比较：

1. unconditional historical probability；
2. regularized logistic regression；
3. 小型 MLP。

由于宏观通道只有约 192 个训练月份，小型模型应当是主模型。不要直接使用深而宽的宏观网络。

#### Step 4：估计状态条件的公司收益分布

设：

$$
R_{i,t+1}\mid Z_{t+1}=s,X_{i,t},J_{i,t}
\sim
\mathcal N(\mu_{i,s,t},\sigma_{i,s,t}^{2}),
$$

其中公司特征和行业信息决定：

$$
\mu_{i,G,t},\;\sigma_{i,G,t},\;
\mu_{i,B,t},\;\sigma_{i,B,t}.
$$

样本外边际预测分布为：

$$
f_{i,t}(r)
=
p_t f_{i,B,t}(r)
+
(1-p_t)f_{i,G,t}(r).
$$

#### Step 5：生成核心经济变量

共同状态概率：

$$
p_t=P_t(Z_{t+1}=Bad).
$$

Bad-state conditional loss：

$$
BadLoss_{i,t}
=
-E_t[R_{i,t+1}\mid Z_{t+1}=Bad].
$$

Good-to-Bad relative severity：

$$
RelativeSeverity_{i,t}
=
\mu_{i,G,t}-\mu_{i,B,t}.
$$

Expected bad-state loss：

$$
ExpectedBadLoss_{i,t}
=
p_t\times BadLoss_{i,t}.
$$

### 4.5 第二阶段：完整 latent-state 模型

只有 MVP 成功后，才考虑 latent mixture 或 HMM。

候选方法：

1. 两状态 Gaussian mixture based on aggregate outcomes；
2. 两状态 HMM，使状态具有持续性；
3. EM 估计 common state posterior；
4. 使用行业组合或 aggregate summaries 构造 joint/composite likelihood。

不要直接对数千只股票使用完全条件独立的：

$$
\sum_k\pi_{k,t}\prod_i f_{i,k,t}(R_{i,t+1}),
$$

因为股票横截面具有强相关性，简单乘积会让状态 posterior 过度确定。较安全的做法是先使用：

- 市场收益；
- FF49 行业组合收益；
- 横截面尾部统计量；
- 少量 aggregate principal components。

识别共同状态，然后估计公司层面的 state-conditional distributions。

### 4.6 预测验证

#### State probability calibration

至少报告：

- Brier score；
- state log loss；
- ROC/AUC；
- calibration curve；
- predicted probability deciles 下的 realized Bad-state frequency；
- 相对于 unconditional state probability 的提升。

#### State economic validation

Bad state 应当对应：

- 更低 market return；
- 更差 cross-sectional 5% tail return；
- 更高 crash fraction；
- 更高 VIX；
- 更高 credit spread；
- 更大 market drawdown；
- 更高 NBER recession frequency，若样本数量允许。

如果这些关系不存在，就不能将该状态称为 aggregate adverse state。

#### Firm distribution validation

分别在 predicted Bad 和 Good periods 中比较：

- NLL；
- CRPS；
- tail-CRPS；
- VaR coverage；
- FZ0；
- state-conditional mean 和 volatility calibration。

### 4.7 资产定价检验

#### Severity 横截面价格

$$
R_{i,t+1}
=
a_t
+\gamma_t Z_t(RelativeSeverity_{i,t})
+\delta_t'Z_t(Controls_{i,t})
+\varepsilon_{i,t+1}.
$$

#### Probability 解释 severity price

$$
\gamma_t
=
a+b\,p_t+u_t.
$$

检验：

$$
b>0.
$$

#### Firm-level interaction

$$
R_{i,t+1}
=
a_t
+\beta_1 Severity_{i,t}
+\beta_2 Severity_{i,t}\times p_t
+\Gamma'Controls_{i,t}
+\varepsilon_{i,t+1}.
$$

#### 与当前 Chapter 1 的比较

比较：

- 当前 marginal Structured K3 severity；
- common-state BadLoss；
- common-state RelativeSeverity；
- expected bad-state loss；
- Linear Quantile total ES；
- historical volatility 和 downside beta。

### 4.8 第二章成功标准

- [ ] $p_t$ 在样本外优于 unconditional probability；
- [ ] calibration curve 具有合理单调性；
- [ ] Bad-state label 在不同窗口具有稳定经济含义；
- [ ] Bad-state label 不需要按公司重新选择；
- [ ] firm-specific severity 不等同于 historical volatility；
- [ ] severity price 随 $p_t$ 或 external stress 上升；
- [ ] 结果不完全由 2008--2009 或 2020 驱动；
- [ ] common-state 模型相对于 Chapter 1 提供更严格的经济解释；
- [ ] 完成一篇可独立阅读的论文草稿。

### 4.9 第二章停止或转向规则

如果发生以下情况，应当停止继续增加模型复杂度：

- $p_t$ 长期无法战胜 unconditional probability；
- predicted probability 与 realized state 没有单调校准关系；
- 状态主要由一个异常月份决定；
- latent model 在不同种子和窗口中无法稳定识别 Bad state；
- RelativeSeverity 与 volatility 几乎完全相同且没有额外解释力。

遇到这些情况时，优先转向透明的 observed aggregate state，而不是继续增加 hidden layers。

---

## 五、Chapter 3A：Physical 与 Risk-Neutral Tail Risk

### 5.1 适用条件

如果可以获得 OptionMetrics，并且能够构造足够稳定的个股 30-day risk-neutral distributions，优先选择本章。

### 5.2 暂定题目

> **Physical Tail Risk, Option-Implied Tail Risk, and Expected Stock Returns**

### 5.3 核心研究问题

Chapter 1 和 Chapter 2 使用 realized future returns 训练模型，因此估计 physical distribution：

$$
P_t(R_{i,t+1}).
$$

期权价格反映 risk-neutral distribution：

$$
Q_t(R_{i,t+1}).
$$

分别计算：

$$
ES_{i,t}^{P,L}
\quad\text{和}\quad
ES_{i,t}^{Q,L}.
$$

定义个股 tail-risk premium：

$$
TRP_{i,t}
=
ES_{i,t}^{Q,L}
-
ES_{i,t}^{P,L}.
$$

核心问题是：

1. Future return 与 physical loss risk 还是 risk-neutral loss risk 关系更强？
2. 当前的负向 unconditional ES relation 是否来自 distress persistence？
3. 真正的正向风险补偿是否体现在 $ES^Q-ES^P$？
4. 压力时期的 severity price 上升，是 physical risk 上升还是 price of risk 上升？

### 5.4 数据需求

- OptionMetrics IvyDB；
- CRSP returns 和 market capitalization；
- 当前 physical distribution forecasts；
- risk-free rate；
- dividend information；
- option volume、open interest、bid--ask spread；
- 最好具有 shorting fee、securities lending 或 short-interest proxy。

### 5.5 执行步骤

#### Step 1：构造 30-day risk-neutral distribution

- 清理个股期权；
- 对齐约 30-day maturity；
- 处理 moneyness、股息和无风险利率；
- 平滑 implied volatility surface；
- 检查基本 no-arbitrage 条件；
- 从期权曲面恢复 risk-neutral CDF 或直接计算 left-tail expected payoff。

#### Step 2：计算 risk-neutral tail measures

$$
VaR_{i,t}^{Q},
\qquad
ES_{i,t}^{Q,L}.
$$

#### Step 3：与 physical forecasts 对齐

确保：

- 相同股票；
- 相同信号日期；
- 相同约一个月预测期限；
- 相同 return definition；
- 相同 5% 左尾；
- 一致的正损失符号。

#### Step 4：形成信号

- $ES^P$；
- $ES^Q$；
- $ES^Q-ES^P$；
- $VaR^Q-VaR^P$；
- physical severity；
- option-implied tail premium。

#### Step 5：联合资产定价回归

$$
R_{i,t+1}
=
a_t
+\beta_1 ES_{i,t}^{P,L}
+\beta_2\left(ES_{i,t}^{Q,L}-ES_{i,t}^{P,L}\right)
+\Gamma'Controls_{i,t}
+\varepsilon_{i,t+1}.
$$

如果出现：

- $ES^P$ 系数为负；
- $ES^Q-ES^P$ 系数为正；

则可以把 distress persistence 与真正的 tail-risk compensation 分开。

### 5.6 必须处理的问题

- 个股期权样本偏向大型、流动性较好的公司；
- risk-neutral density 对极端 strike 很敏感；
- 个股期权多为 American options；
- 深度虚值期权可能缺乏可靠价格；
- option-implied signal 可能反映股票借贷费用和 short-sale constraints；
- 需要报告 OptionMetrics 子样本和全 CRSP 样本的差异。

### 5.7 第三章 A 成功标准

- [ ] 每月具有足够数量的有效个股 option surfaces；
- [ ] $ES^Q$ 对合理的清理和外推规则稳定；
- [ ] $ES^Q$ 与 $ES^P$ 有相关性但并不完全相同；
- [ ] $ES^Q-ES^P$ 具有可解释的时间序列和横截面变化；
- [ ] 结果控制 implied volatility、skew、liquidity 和 short-sale proxies 后仍然存在；
- [ ] 主要结果不只来自少量大型科技股票或危机月份；
- [ ] 完成一篇可独立阅读的论文草稿。

---

## 六、Chapter 3B：金融中介约束与尾部损失补偿

### 6.1 适用条件

如果无法取得 OptionMetrics，或者个股 option surface 质量不足，则选择本章。

### 6.2 暂定题目

> **Intermediary Constraints and the Price of Firm-Level Tail Severity**

### 6.3 核心研究问题

Chapter 1 已经发现 severity price 在 VIX、credit spread 和 drawdown 较高时变得更加正向。本章进一步问：

> 金融压力为什么会提高 firm-level bad-state loss 的价格？金融中介资本和融资约束是否是这一关系的经济机制？

### 6.4 数据需求

宏观或中介变量：

- intermediary capital ratio；
- broker-dealer leverage shock；
- primary dealer balance-sheet measures；
- credit spread；
- funding liquidity；
- TED spread；
- VIX 和 market drawdown。

公司异质性变量：

- leverage；
- cash holdings；
- interest coverage；
- short-term debt；
- debt maturity；
- external financing dependence；
- credit rating；
- liquidity；
- institutional ownership；
- analyst coverage；
- short interest 或 arbitrage-cost proxies。

### 6.5 主要检验

#### Severity price 与 intermediary constraint

首先每个月估计 severity 的横截面价格 $\gamma_t$，然后估计：

$$
\gamma_t
=
a+b\,IntermediaryConstraint_t+u_t.
$$

如果 constraint 数值越高代表中介约束越强，预期：

$$
b>0.
$$

如果变量定义为 intermediary capital，则预期符号可能相反，必须根据具体定义预先确定。

#### 公司融资依赖的三重交互

$$
\begin{aligned}
R_{i,t+1}
=\;&a_t
+\beta_1 Severity_{i,t}
+\beta_2 Severity_{i,t}\times Constraint_t\\
&+\beta_3 Severity_{i,t}\times Constraint_t\times FinancingNeed_{i,t}
+\Gamma'Controls_{i,t}
+\varepsilon_{i,t+1}.
\end{aligned}
$$

如果中介机制成立，severity compensation 应该更集中于：

- leverage 较高；
- cash 较低；
- 短期债务较高；
- interest coverage 较低；
- 更依赖外部融资；
- 流动性较差；
- 套利成本较高的公司。

#### 真实经营结果

为了使本章不仅是 Chapter 1 的状态变量替换，还应检验 high-severity firms 在中介约束时期是否随后出现：

- investment cuts；
- employment decline；
- debt issuance decline；
- equity issuance；
- profitability decline；
- default、delisting 或 credit downgrade frequency 上升。

### 6.6 独立成章的最低标准

只把 VIX 换成 intermediary factor 不足以成为独立章节。至少需要同时建立：

1. 中介约束与 severity price 的时间序列关系；
2. 公司融资依赖的横截面异质性；
3. 后续融资或真实经营结果；
4. 相对于 volatility、recession、VIX 和 credit spread 的增量解释力。

如果无法满足这些条件，本方向应当并回 Chapter 1 作为 mechanism extension，而不单独成章。

### 6.7 第三章 B 成功标准

- [ ] intermediary variable 的时点和定义清楚；
- [ ] severity price 对 intermediary constraint 有稳定关系；
- [ ] 关系在控制 VIX、credit spread 和 drawdown 后仍然存在；
- [ ] 关系集中在更依赖外部融资的公司；
- [ ] 至少一种后续融资或真实经营结果支持机制；
- [ ] 排除危机月份后结论仍有方向一致性；
- [ ] 完成一篇可独立阅读的论文草稿。

---

## 七、第三章选择规则

### 7.1 立即确认的数据问题

在开始 Chapter 2 的同时，尽快确认：

- [ ] 是否拥有 OptionMetrics 权限；
- [ ] OptionMetrics 是否能够稳定链接 CRSP PERMNO；
- [ ] 是否能获得个股 implied volatility surface；
- [ ] 是否有 short-interest、borrow fee 或可接受的代理变量；
- [ ] 是否能获得 intermediary capital factor 和 broker-dealer data；
- [ ] Compustat 中是否具备债务期限和融资需求变量。

### 7.2 决策树

1. **有 OptionMetrics 且个股 option coverage 足够：**选择 Chapter 3A；
2. **有 OptionMetrics 但 surface 质量不足：**先做小样本 feasibility test，再决定；
3. **没有 OptionMetrics，但中介与公司融资数据可用：**选择 Chapter 3B；
4. **两者都不可用：**将 intermediary mechanism 缩小并入 Chapter 1，同时寻找一个使用现有 CRSP/Compustat 数据的第三题，例如 tail severity 与企业真实投资或 distress outcomes。

---

## 八、研究项目与文件结构建议

建议逐步把三章拆成相互独立但可共享数据的目录：

```text
PricingTailRisk_MDN/
├── github/
│   ├── chapter1_distribution_pricing/
│   ├── chapter2_common_state/
│   ├── chapter3_option_tail_premium/      # 若选择 3A
│   ├── chapter3_intermediary_constraints/ # 若选择 3B
│   ├── Research_Design_Roadmap.md
│   └── Dissertation_Research_Roadmap.md
├── data/
│   ├── shared_clean_data/
│   ├── common_state_data/
│   └── option_or_intermediary_data/
├── output/
│   ├── chapter1/
│   ├── chapter2/
│   └── chapter3/
└── overleaf/
    ├── chapter1/
    ├── chapter2/
    ├── chapter3/
    └── dissertation/
```

不需要现在立刻重构全部目录。应当在 Chapter 1 正式冻结后再拆分，避免破坏当前可以运行的 notebook。

### 8.1 共享层应当包含

- 原始数据读取；
- 基础样本筛选；
- 日期和标签定义；
- CRSP/Compustat 链接；
- 宏观数据；
- FF49；
- 通用 Newey--West、portfolio sort 和 Fama--MacBeth 函数。

### 8.2 每章必须独立保存

- model version；
- prediction timestamp；
- pricing timestamp；
- input data version；
- sample filters；
- main tables；
- appendix tables；
- figures；
- frozen paper draft。

---

## 九、十二个月执行计划

### Month 1--2：冻结 Chapter 1

- 完成 severity--volatility horse race；
- 完成危机排除；
- 完成 breakpoints robustness；
- 补 summary statistics；
- 修订论文；
- 与导师确认主要结论和投稿定位；
- 冻结 Chapter 1 的模型与数据版本。

### Month 2：确认 Chapter 3 数据权限

- 联系学校或导师确认 OptionMetrics；
- 检查 intermediary factor 数据；
- 小规模读取 option 或 intermediary sample；
- 在 Month 2 结束前决定 Chapter 3A 或 3B 的优先级。

### Month 2--3：Chapter 2 MVP

- 构造 aggregate monthly outcome panel；
- 固定两个 observed-state definitions；
- logistic state probability benchmark；
- 生成 walk-forward state forecasts；
- 完成 Brier、log loss、AUC 和 calibration plot。

### Month 3--4：Chapter 2 firm-state model

- 估计 Good/Bad state conditional Normal models；
- 生成 $BadLoss$、$RelativeSeverity$ 和 $ExpectedBadLoss$；
- 与当前 Structured K3 severity 对比；
- 完成基本预测评分。

### Month 4--6：Chapter 2 资产定价与论文

- 完成 portfolio sorts；
- 完成 Fama--MacBeth；
- 完成 $severity\times p_t$；
- 完成 volatility horse race；
- 完成 crisis exclusion；
- 决定是否升级 latent/HMM；
- 写出 Chapter 2 第一版完整论文。

### Month 6--10：Chapter 3

若选择 3A：

- option cleaning；
- risk-neutral distribution；
- $ES^Q$；
- $ES^P$ 对齐；
- tail-risk premium tests；
- liquidity 和 borrow-cost controls。

若选择 3B：

- intermediary variables；
- severity price time-series tests；
- financing-dependence heterogeneity；
- real outcome tests；
- mechanism horse races。

### Month 10--12：整合毕业论文

- 撰写统一 introduction；
- 撰写统一 economic framework；
- 统一符号、样本和变量定义；
- 解释三章差异；
- 补充 general conclusion；
- 准备答辩 slides；
- 冻结 replication files。

---

## 十、Chapter 2 开始后的前六周任务

### Week 1：Aggregate panel

- [ ] 生成月度 market return；
- [ ] 生成 cross-sectional 1%、5%、10% return quantiles；
- [ ] 生成 cross-sectional volatility；
- [ ] 生成 crash share；
- [ ] 合并 VIX、credit spread、drawdown 和 recession indicator；
- [ ] 输出月度 summary figure。

### Week 2：Observed Bad state

- [ ] 固定两个候选 Bad-state definitions；
- [ ] 检查 Bad months 数量；
- [ ] 检查危机和非危机覆盖；
- [ ] 只用 Train + Validation 形成阈值；
- [ ] 验证 rolling label 逻辑没有使用 Test。

### Week 3：State probability forecast

- [ ] unconditional benchmark；
- [ ] logistic regression；
- [ ] L1/L2 regularization；
- [ ] 小型 MLP；
- [ ] walk-forward probability forecasts；
- [ ] Brier/log loss/AUC/calibration。

### Week 4：Firm conditional distributions

- [ ] Good-state conditional Normal；
- [ ] Bad-state conditional Normal；
- [ ] month-equal weighting；
- [ ] state imbalance weights；
- [ ] out-of-sample distribution forecasts。

### Week 5：Economic measures

- [ ] $BadLoss$；
- [ ] $RelativeSeverity$；
- [ ] $ExpectedBadLoss$；
- [ ] correlation with total ES and volatility；
- [ ] state-conditional calibration figures。

### Week 6：First pricing results

- [ ] NYSE-decile portfolios；
- [ ] EW/VW H--L；
- [ ] Fama--MacBeth；
- [ ] $severity\times p_t$；
- [ ] comparison with Chapter 1 severity；
- [ ] 一页结果摘要交给导师。

---

## 十一、研究决策原则

### 11.1 不使用资产定价结果选择预测模型

以下内容不能根据 portfolio return 或 Fama--MacBeth significance 选择：

- 模型架构；
- number of components/states；
- random seed；
- tail probability；
- Bad-state definition；
- early-stopping epoch；
- predictor set。

这些选择必须依据：

- 经济理论；
- Train/Validation；
- forecast score；
- state classification score；
- 预先固定的研究设计。

### 11.2 预测成功与资产定价成功分开

一个模型可能：

- 预测更准确，但没有发现正收益补偿；
- 预测分数与 benchmark 接近，但提供更强经济分解；
- 在 full-distribution score 上一般，但在 tail score 上更好；
- 发现负向 relation，支持 distress persistence 而不是 risk compensation。

所有这些结果都可以有研究价值，不能只保留“显著且符合预期”的结果。

### 11.3 简单模型优先

每个新问题先建立最简单可运行版本：

- two states before three states；
- logistic before deep macro network；
- observed state before latent HMM；
- conditional Normal before Student-$t$；
- existing data before expensive new data；
- one clear mechanism before many weak interactions。

### 11.4 章节必须回答不同问题

- Chapter 1 不能仅仅变成模型竞赛；
- Chapter 2 不能只是把 $K=3$ 改成 $K=2$；
- Chapter 3 不能只把 VIX 换成另一个 stress variable；
- 每章必须有独立研究问题、独立主要表格、独立结论和独立失败标准。

---

## 十二、主要研究风险与应对方案

| 风险 | 可能后果 | 应对方案 |
|---|---|---|
| Structured K3 预测优势不稳定 | Chapter 1 方法贡献变弱 | 将贡献放在经济分解和状态依赖定价，透明报告预测边界 |
| Tail calibration 仍然不足 | ES 解释受到质疑 | 报告 coverage、FZ0、recalibration robustness，不宣称 perfect calibration |
| 当前 $\pi$ 无法验证为 macro state | 共同状态解释过强 | Chapter 1 限定为 common mixture weight；Chapter 2 建立严格共同状态 |
| Chapter 2 state probability 难预测 | $p_t$ 缺乏经济价值 | 使用透明 observed state、小模型、外部 state validation；必要时停止 latent expansion |
| Bad months 太少 | 参数不稳定 | 先用两状态；控制模型维度；报告事件数量和 LOO stability |
| Severity 与 volatility 高度相关 | 经济贡献不清楚 | direct interaction horse race、residual severity、state-conditional difference |
| OptionMetrics 样本过小 | Chapter 3A 无法完成 | Month 2 前完成 feasibility test；转向 Chapter 3B |
| Option signal 反映 shorting cost | $Q-P$ 解释错误 | 控制 borrow/short-interest/liquidity；排除高费用股票 |
| Intermediary mechanism 不独立 | Chapter 3B 只是 Chapter 1 extension | 加入 financing heterogeneity 和 real outcomes，否则并回 Chapter 1 |
| 计算成本过高 | 项目拖延 | 简单模型先行；冻结主模型；只在必要步骤用 GPU |
| 不断调参追求显著性 | 数据挖掘风险 | 预先固定设计，保存版本，资产定价结果不参与模型选择 |

---

## 十三、导师会议更新模板

每次导师会议前，用一页内容回答以下问题：

### 1. 本阶段研究问题

> 本周要识别或检验的唯一主要问题是什么？

### 2. 新增结果

- 新增了哪一个表或图？
- 样本、模型和变量版本是什么？
- 结果方向和统计强度如何？

### 3. 结果解释

- 支持哪一个假设？
- 不支持哪一个假设？
- 是否可能由 volatility、size、crisis period 或数据时点解释？

### 4. 当前阻碍

- 数据权限；
- 模型识别；
- 样本数量；
- 计算成本；
- 经济解释。

### 5. 下一步决策

列出最多三个下一步，并明确需要导师决定的问题。避免一次汇报十几个零散 robustness。

---

## 十四、核心文献起点

### Tail risk 与股票收益

- Kelly, B., and Jiang, H. (2014), [Tail Risk and Asset Prices](https://doi.org/10.1093/rfs/hhu039), *Review of Financial Studies*.
- Huang, W., Liu, Q., Rhee, S. G., and Wu, F. (2012), [Extreme Downside Risk and Expected Stock Returns](https://doi.org/10.1016/j.jbankfin.2011.12.014), *Journal of Banking & Finance*.
- Atilgan, Y., Bali, T. G., Demirtas, K. O., and Gunaydin, A. D. (2020), [Left-Tail Momentum](https://doi.org/10.1016/j.jfineco.2019.07.006), *Journal of Financial Economics*.

### 完整收益分布预测

- Bishop, C. M. (1994), *Mixture Density Networks*.
- Baruník, J., Hronec, M., and Tobek, O. (2026), [Forecasting Stock Return Distributions Around the Globe with Quantile Neural Networks](https://doi.org/10.1016/j.ijforecast.2026.04.004), *International Journal of Forecasting*.
- Patton, A. J., Ziegel, J. F., and Chen, R. (2019), [Dynamic Semiparametric Models for Expected Shortfall](https://doi.org/10.1016/j.jeconom.2018.10.008), *Journal of Econometrics*.

### Rare disasters 与状态依赖风险价格

- Gabaix, X. (2012), [Variable Rare Disasters](https://doi.org/10.1093/qje/qjs001), *Quarterly Journal of Economics*.
- Barro, R. J. (2009), [Rare Disasters, Asset Prices, and Welfare Costs](https://doi.org/10.1257/aer.99.1.243), *American Economic Review*.

### Risk-neutral tail risk

- Bollerslev, T., Todorov, V., and Xu, L. (2015), [Tail Risk Premia and Return Predictability](https://doi.org/10.1016/j.jfineco.2015.02.010), *Journal of Financial Economics*.
- Bollerslev, T., Tauchen, G., and Zhou, H. (2009), [Expected Stock Returns and Variance Risk Premia](https://doi.org/10.1093/rfs/hhp008), *Review of Financial Studies*.

### 金融中介资产定价

- Adrian, T., Etula, E., and Muir, T. (2014), [Financial Intermediaries and the Cross-Section of Asset Returns](https://doi.org/10.1111/jofi.12189), *Journal of Finance*.
- He, Z., Kelly, B., and Manela, A. (2017), [Intermediary Asset Pricing: New Evidence from Many Asset Classes](https://doi.org/10.1016/j.jfineco.2017.08.002), *Journal of Financial Economics*.

---

## 十五、当前最高优先级

### 立即执行

1. 完成并冻结 Chapter 1 的少量高优先级 robustness；
2. 确认 OptionMetrics 和 intermediary data 权限；
3. 开始构造 Chapter 2 的 `aggregate_state_panel`；
4. 从 two-state observed-state MVP 开始；
5. 在六周内得到第一版 $p_t$、severity 和 pricing results。

### 暂时不要执行

1. 不重新大范围搜索 Structured K3 超参数；
2. 不因为 portfolio return 不显著而更换主模型；
3. 不一开始估计三状态 joint neural HMM；
4. 不把 Normal、Student-$t$、FiLM、warm start 分别做成新章节；
5. 不在未确认数据权限前投入大量时间设计 OptionMetrics 细节。

整篇毕业论文最有价值的下一步是：

$$
\boxed{
\text{建立一个真正共享的两状态模型，分别估计 }
p_t
\text{ 与 }
Severity_{i,t}
}
$$

这一步能够把 Chapter 1 最大的解释限制，转化成毕业论文最重要的新贡献。
