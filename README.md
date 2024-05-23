# 开源金融衍生品定价引擎库PriceLib

## 1. 项目简介

本项目是一个使用Python语言编写的开源金融衍生品定价引擎库，包含多种定价方法，涵盖了国内场外衍生品市场主流的权益类衍生品结构。定价方法方面，本定价库支持解析解、Monte Carlo模拟法、PDE有限差分法、数值积分法以及树方法；波动率模型方面，本定价库支持常数波动率模型、局部波动率模型、Heston随机波动率模型。

基于国内场外衍生品市场发展的现状，市场对自动赎回结构(Autocallable)，尤其是各种类型的雪球、凤凰结构的定价有更大的需求。本定价库立足于国内市场需求，支持多种雪球变种和凤凰结构的定价，具体如下：

| **产品分类**     | **产品子类**                            | **模型**                                                                               |
|--------------|-------------------------------------|--------------------------------------------------------------------------------------|
| **1 香草及其组合** | 欧式，价差，跨式，宽跨式，折价，领口，风险逆转，蝶式，鹰式       | Black Scholes Merton解析解，蒙特卡洛模拟，PDE，积分法，二叉树                                           |
|              | 美式期权                                | BAW(1987)近似解、Bjerksund Stensland(2002)近似解、LSMC(最小二乘蒙特卡洛)、PDE、积分法、二叉树                 |
| **2 亚式期权**   | 亚式期权-几何平均/算术平均替代收盘价                 | Kemma Vorst(1990)几何平均收盘价亚式解析解，Turnbull Wakeman (1991)算术平均收盘价亚式近似估计，算术平均收盘价二叉树，蒙特卡洛模拟 |
|              | 亚式期权-几何平均/算术平均替代执行价，增强亚式            | 蒙特卡洛模拟                                                                               |
| **3 二元及其组合** | 单边二元：欧式二元，美式二元，一触即付                 | Reiner Rubinstein(1991b)解析解，蒙特卡洛模拟，PDE，二叉树，积分法                                       |
|              | 双边二元：(欧式)二元凸式/二元凹式；(美式)双接触/双不接触     | Hui(1996)美式双接触/双不接触近似解，蒙特卡洛模拟，PDE                                                    |
| **4 障碍期权**   | 单边障碍：敲出期权(单鲨)，敲入期权                  | Reiner Rubinstein(1991a)单边障碍解析解，蒙特卡洛模拟，PDE，二叉树，积分法                                   |
|              | 双边障碍，双鲨                             | Ikeda Kunitomo(1992)双边障碍级数解，Haug(1998)双边障碍近似解，蒙特卡洛模拟，PDE                             |
|              | 安全气囊                                | 解析解(障碍与二元组合)，蒙特卡洛模拟，PDE                                                              |
| **5 自动赎回结构** | 无敲入的自动赎回：二元小雪球                      | 蒙特卡洛模拟，PDE，积分法                                                                       |
|              | 定期派息票据：FCN、DCN、凤凰                   | 蒙特卡洛模拟，PDE，积分法                                                                       |
|              | 雪球结构：平敲、降敲、早利、蝶变、双降、限损、OTM、降落伞、看涨雪球 | 蒙特卡洛模拟，PDE，积分法                                                                       |
|              | 雪球变种：巴黎雪球                           | 蒙特卡洛模拟                                                                               |
| **6 累计期权**   | 标准累购，标准累沽，区间累计                      | 蒙特卡洛模拟                                                                               |
>注：二元期权和障碍期权的解析解，都可以选择Broadie Glasserman Kou(1995)离散观察修正，既支持连续观察，也支持离散观察。

具体的产品及其参数的介绍，在"场外百科全书"微信小程序中，已经整理成了结构百科，可以方便地查阅。
<div align=center>
<img src="./docs/otc_wiki_0.png" alt="场外百科全书-微信小程序" height="150">
<br>
<img src="./docs/otc_wiki_1.png" alt="场外百科全书-结构列表" height="380">
<img src="./docs/otc_wiki_2.png" alt="场外百科全书-结构介绍" height="380">
<img src="./docs/otc_wiki_3.png" alt="场外百科全书-情景分析图" height="380">
</div>

我们会持续关注市场上的最新动态，迅速地在PriceLib中更新流行产品的定价模型。如果您有特别的产品定价需求，也可以与我们联系，我们通常可以在两至三天更新至PriceLib中。您可以在Gitee提交Issue，或通过下方电话、邮件联系我们，也欢迎大家扫描下方二维码关注"凌瓴科技"微信公众号。

- 电话：021-50186069
- 邮箱：marx@galatech.com.cn
- 微信公众号：凌瓴科技
<div align=center>
<img src="./docs/official_accounts.jpg" alt="微信公众号" height="150">
</div>

## 2. 安装方法
1. 如果您只想调用PriceLib实现定价，在您的终端中使用命令`pip install pricelib`即可将PriceLib安装到Python解释器的`Lib/site-packages`中。

2. 如果您是开发者并且想要在本地安装这个Python库，您需要在 https://gitee.com/lltech/pricelib 下载项目的源码zip压缩包，然后在项目的根目录下，用终端运行`pip install -e .`命令，这个目录应该包含`setup.py`文件。这个命令将以"editable"模式安装这个库，这意味着您对源代码的任何修改都会立即反映到您的环境中，无需重新安装。

3. 如果您还想安装绘图依赖项和开发依赖项（在`setup.py`文件的`extras_require`中定义，绘图依赖项包括`matplotlib`和`plotly`，开发依赖项包括静态分析工具`flake8`、`pylint`和测试工具`pytest`），您应该在终端中使用`pip install -e .[dev,plot]`命令， 这将安装PriceLib库、绘图依赖项和开发依赖项。   
如果您只想安装PriceLib库和开发依赖项，则您应该使用`pip install -e .[dev]`命令。

## 3. 使用教程

> * 如果您使用`pip install pricelib`安装，用jupyter notebook运行代码，当您想要查看类和函数的说明文档时，有两种方式（以`StandardSnowball`为例）：
>   * 方法1：在jupyter notebook的一个空白代码块中，输入`?StandardSnowball`，然后运行。
>   * 方法2：将光标移动到`StandardSnowball`上面，按`Shift + Tab`，弹出文档，右上角的`^`和`+`会展开更多文档。
> * 如果您使用`pip install -e .`本地安装，用Pycharm运行代码，当您想要查看类和函数的说明文档时，有两种方式：
>   * 方法1：将鼠标悬停在`StandardSnowball`上面，Pycharm会自动弹出文档。
>   * 方法2：可以按住`Ctrl`然后鼠标左键点击`StandardSnowball`，即可自动跳转，查看源码和注释。
> * 教程篇幅有限，您可以在`examples`文件夹浏览详细的示例代码。如果您想了解产品类详情，可以在`pricelib/products`文件夹中查看对应产品的说明，每个产品的说明都详细介绍了所有输入参数，以及支持的定价引擎。

### 3.1 简易接口: 使用默认的定价引擎和BSM模型，快速实现定价

只需两行代码，创建`产品结构`，然后调用`price`方法，即可完成定价。 以美式期权为例：
```python
from pricelib import *
# 1. 创建产品结构，输入s、r、q、vol参数，此时会直接使用默认的定价引擎和BSM模型
option = VanillaOption(strike=100, maturity=1, callput=CallPut.Call,
                       exercise_type=ExerciseType.American, 
                       s=100, r=0.02, q=0.01, vol=0.2)
# 2. 完成定价
print(option.price())  
```
在上面的例子中，美式期权会自动使用默认的BAW近似解定价引擎进行定价。

同一个产品对象可以切换不同的定价引擎，将上面的美式期权改为PDE有限差分法进行定价：
```python
# 创建PDE有限差分法定价引擎，输入s、r、q、vol参数，此时会直接使用默认的BSM模型
fdm_engine = FdmVanillaEngine(s=100, r=0.02, q=0.01, vol=0.2)
# 为产品对象切换定价引擎
option.set_pricing_engine(fdm_engine)
print(option.price())  
```
调用产品对象的`delta, gamma, vega, theta, rho`方法，即可计算常见的5种期权希腊值：
```python
print({"delta": option.delta(), "gamma": option.gamma(), "vega": option.vega(),
       "theta": option.theta(), "rho": option.rho()})
```
如果您想一次性计算期权估值和5种希腊值，可以调用产品对象的`pv_and_greeks`方法：
```python
result = option.pv_and_greeks()
# result == {'pv': pv, 'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}
```

再比如较为复杂的雪球结构，"敲出票息10%的一年期锁三103-80平敲雪球"，一样只需两行代码即可完成定价：
```python
from pricelib import *
option = StandardSnowball(maturity=1, lock_term=3, s0=100, barrier_out=103,
                          barrier_in=80, coupon_out=0.10, s=100, r=0.02,
                          q=0.01, vol=0.16)
print(option.price())  
```
这里的经典雪球会使用默认的PDE有限差分定价引擎。输入参数`s0=100, barrier_out=103, barrier_in=80`的价格是百分比形式，您也可以输入价格的绝对值，例如`s0=5535.40, barrier_out=5701.46, barrier_in=4428.32`，只要所有价格参数保持一致即可。

> #### 常见问题QA
> 1. Q: 我使用上述简易定价接口设置了r和vol，没有定价成功，为什么呢？
> * A: 目前实例化产品的时候，您需要输入s、r、q、vol全部4个参数，才能自动创建默认的定价引擎。如果不需要分红率，您可以输入q=0。参数s指的是估值时的标的价格，与期初价格s0不同，因此需要指定s参数。若您想获得期初估值，可以将s与s0设为相同的值。
> 2. Q: 雪球的定价结果为什么与examples中的demo有微小差异呢？
> * A: 日期设置在后文有详细介绍，雪球敲出观察日缺省时，会按照起始日和交易日历自动生成敲出观察日序列，估值日和起始日会默认使用今天的日期，这会对定价结果产生微小的影响。您需要用`set_evaluation_date(datetime.date(2022, 1, 5))`设置估值日、在实例化`StandardSnowball`时加上`start_date=datetime.date(2022, 1, 5)`设置起始日，计算结果才能完全一致。

对于雪球结构的PDE定价引擎，您可以获取有限差分的price、delta和gamma网格，进行绘图：
```python
from pricelib.common.utilities import pde_plot
# 这里的option是上面的StandardSnowball对象，其默认定价引擎是FdmSnowBallEngine
pde_engine = option.engine
# 绘制delta和gamma曲面图
delta_matrix, s_vec = pde_engine.delta_matrix(status=StatusType.NoTouch)
fig1 = pde_plot.draw_greeks_surface("delta", delta_matrix, 
                                    spot_range=(65, 115), show_plot=True)
gamma_matrix, s_vec = pde_engine.gamma_matrix(status=StatusType.NoTouch)
fig2 = pde_plot.draw_greeks_surface("gamma", gamma_matrix, show_plot=True)
# 绘制t=0时的delta和gamma曲线
fig3 = pde_plot.draw_greeks_curve(delta_matrix, gamma_matrix, t=0, 
                                  spot_range=(65, 115), show_plot=True)
```

### 3.2 进阶使用: 为产品结构自行配置随机过程和定价引擎
 
除了使用默认的定价引擎之外，您也可以自行为产品结构配置所需的随机过程和定价引擎。

本定价库搭建了一个统一的框架：定价引擎、随机过程和波动率模型使用策略模式:
* 产品对象可以随时切换不同的定价引擎
* 定价引擎可以设置不同种类的随机过程
* 随机过程可以对应不同的波动率模型

本框架不仅易于使用，而且方便扩展新的产品和定价方法。

以香草期权为例，首先创建标的价格spot、无风险利率r、分红率q、波动率vol等估值参数对象。
```python
from pricelib import *
# 1. 市场数据，Observable被观察者
riskfree = SimpleQuote(value=0.02, name="无风险利率")
dividend = SimpleQuote(value=0.05, name="中证1000贴水率")
volatility = BlackConstVol(0.16, name="中证1000波动率")
spot_price = SimpleQuote(value=100, name="中证1000指数")
# 常数波动率
process = GeneralizedBSMProcess(spot=spot_price, interest=riskfree, 
                                div=dividend, vol=volatility)
```
其中r、q、vol可以设为常数，也可以设为期限结构、局部波动率。
```python
import pandas as pd
# 从csv文件中读取数据
div = pd.read_csv("./tests/div.csv")
loc_vol_df = pd.read_csv("./tests/loc_vol.csv", index_col=0)
expirations = loc_vol_df.index.values
strikes = loc_vol_df.columns.values.astype(float)
volval = loc_vol_df.values
# q期限结构
dividend = RateTermStructure.from_array(div['maturity'].values, div['q'].values)
# 局部波动率
volatility = LocalVolSurface(expirations=expirations, strikes=strikes,
                             volval=volval)
```
然后，实例化一个随机过程，可以选择广义BSM过程或Heston过程，其属性包括上述spot、r、q、vol等估值参数。  
随机过程是Observer观察者，在估值参数改变时会收到通知。  
根据不同的r和q值，广义BSM过程可以变为: 
* 1973年Black-Scholes无股利股票期权模型
* 1973年Merton连续股利股票期权模型
* 1976年Black期货期权模型
* 1983年Garman-Kohlhagen外汇期权定价模型。
```python
# 2. 随机过程
# BSM动态过程：常数波动率 or 局部波动率
process = GeneralizedBSMProcess(spot=spot_price, interest=riskfree,
                                div=dividend, vol=volatility)
# Heston动态过程
process = HestonProcess(spot=spot_price, interest=riskfree, div=dividend, 
                        v0=0.025, var_theta=0.02, var_kappa=4.01,
                        var_vol=0.1, var_rho=-0.3)
```
接着，实例化一个定价引擎，定价引擎可以选择解析解、蒙特卡洛模拟、PDE有限差分法、数值积分法、树方法，  
其属性包含一个随机过程，以及定价方法参数，例如：
* Monte Carlo的路径数、随机数种类、方差缩减选项、低差异序列选项、随机数种子
* PDE有限差分的价格格点数、最大价格边界、有限差分算法(显式/隐式/Crank-Nicolson)
* 积分法的数值积分方法(梯形法则/辛普森法则)、价格格点数、最大价格边界
* 树方法的树分支数
```python
# 3. 定价引擎
an_engine = AnalyticVanillaEuEngine(process)
mc_engine = MCVanillaEngine(process, n_path=100000, seed=0,
                            rands_method=RandsMethod.LowDiscrepancy,
                            antithetic_variate=True, ld_method=LdMethod.Sobol) 
quad_engine = QuadVanillaEngine(process, quad_method=QuadMethod.Simpson,
                                n_points=801, n_max=4)
bitree_engine = BiTreeVanillaEngine(process, tree_branches=500)
pde_engine = FdmVanillaEngine(process, s_step=400, n_smax=4, fdm_theta=0.5)
```
最后，实例化一个期权产品对象，其属性包括一个定价引擎，以及产品参数，以香草期权为例，产品参数为看涨看跌、行权方式、行权价和到期时间。
```python
# 4. 定义产品：香草欧式期权
option = VanillaOption(maturity=1, strike=100, callput=Callput.Call,
                       exercise_type=ExerciseType.European)
# 使用Monte Carlo模拟定价
option.set_pricing_engine(mc_engine)
price_mc = option.price()
# 使用有限差分法定价
option.set_pricing_engine(pde_engine)
price_pde = option.price()
```
调用期权产品的price()方法，即可实现定价。各个衍生品结构的不同定价方法的使用示例，详见 https://gitee.com/lltech/pricelib/examples 中的demo。

定价引擎支持的波动率模型具体如下:

| **定价引擎**        | **波动率模型**               |
|-----------------|-------------------------|
| 1 解析解           | 常数波动率                   |
| 2 Monte Carlo模拟 | 常数波动率、局部波动率、Heston随机波动率 |
| 3 PDE有限差分法      | 常数波动率、局部波动率             |
| 4 积分法           | 常数波动率                   |
| 5 树方法           | 常数波动率                   |

### 3.3 时间参数的输入—— 只输入期限 / 输入起止日期

在上面的例子中，产品的到期时间直接输入了`maturity = 1`年:
```python
option = VanillaOption(maturity=1, strike=100, callput=Callput.Call,
                       exercise_type=ExerciseType.European)
```
但是，有时候我们需要输入起止日期，这时候可以使用以下方式：
```python
import datetime
start_date= datetime.date(2022, 1, 5)
end_date = datetime.date(2023, 1, 5)
option = VanillaOption(start_date=start_date, end_date=end_date, strike=100, 
                       callput=Callput.Call, exercise_type=ExerciseType.European)
```
需要注意的是，采用设置起止日期的方法时，一般需要指定估值日期，否则估值日期默认为今天的日期：
```python
set_evaluation_date(datetime.date(2022, 1, 5))
print(option.price())
```
在仅输入`maturity = 1`年时，实际上将起始日期设置为了估值日期，结束日期设置为了估值日期延后1年。  
如果您需要更精细地处理交易日历和年化系数，可以在产品对象中设置相关参数：
```python
option = VanillaOption(strike=100, callput=Callput.Call, 
                       exercise_type=ExerciseType.European,
                       start_date=start_date, end_date=end_date, 
                       trade_calendar=CN_CALENDAR, 
                       annual_days=annual_days, t_step_per_year=t_step_per_year)
set_evaluation_date(datetime.date(2022, 1, 5))
```
其中：
* trade_calendar：交易日历，默认使用中国内地的交易日历。
* annual_days：每年的自然日数量，默认为365天。
* t_step_per_year：每年的交易日数量，默认为243天。

对于一些结构复杂的衍生品，例如雪球、凤凰等自动赎回结构，涉及到敲出观察日。  
正如前面举过的例子，您可以不输入敲出观察日，仅输入存续期`maturity = 1`年和锁定期`lock_term = 3`个月，  
此时产品对象会自动生成一个根据交易日历调整后的敲出观察日期序列：
```python
option = StandardSnowball(maturity=1, lock_term=3, s0=100, barrier_out=103,
                          barrier_in=80, coupon_out=0.10, coupon_div=0.05)
```
这是一个"敲出票息10%，红利票息5%的1年期锁三103-80平敲雪球"，由于敲出观察日缺省，在定价过程中，程序会自动生成如下的敲出观察日序列（假设估值日期设为2022年1月5日）：
```
20220406_20220505_20220606_20220705_20220805_20220905_20221010_20221107_20221205_20230105
```
可以注意到自动生成的日期序列是每个月的工作日，遇到周六周日和法定节假日会向后顺延（如果顺延后月份变化，则会向前调整）。

另一种方式是直接输入敲出观察日序列，同时到期时间改为输入起止日期：
```python
from datetime import date
start_date= date(2022, 1, 5)
end_date = date(2023, 1, 5)
obs_dates = [date(2022, 4, 6), date(2022, 5, 5), date(2022, 6, 6), date(2022, 7, 5),
             date(2022, 8, 5), date(2022, 9, 5),date(2022, 10, 10), date(2022, 11, 7),
             date(2022, 12, 5), date(2023, 1, 5)]
option = StandardSnowball(start_date=start_date, end_date=end_date, 
                          obs_dates=obs_dates, lock_term=3, s0=100,
                          barrier_out=103, barrier_in=80,
                          coupon_out=0.10, coupon_div=0.05)
set_evaluation_date(date(2022, 1, 5))
print(option.price())
```
这里使用的是默认的CN_CALENDAR(中国内地交易日历)，年化系数为365和243，如果需要更改，可以在产品对象中设置相关参数。

### 3.4 注意事项

如果您在运行过定价库的代码之后，移动了定价库的位置，或者更改了定价库内文件夹的名称，这时候可能会出现报错。

这个问题产生的原因是定价库中使用了numba的JIT(Just in Time)即时编译技术进行加速，
并且开启了`cache=True`，这样会将编译好的中间代码缓存到本地，以提高下次运行的速度。如果您移动了定价库的位置，缓存的中间代码没有更新，于是就会引发报错。

遇到这种问题时，只需要在定价库目录搜索所有的`__pycache__`文件夹，删除其中的所有文件，然后重新运行代码即可。

## 4. 优势与应用
### 4.1 项目的优点

* 使用方便简单，只需两行代码即可完成定价。如果想要修改默认的定价引擎和随机过程，也有丰富的选项可供用户自行配置。PriceLib的代码注释、使用示例非常详细，易于学习上手。
* 完全采用Python语言，并且考虑了用户的代码经验各不相同的现实，源码简单易懂，适合只有Python基础的金融从业者使用，可以在业务上出现新产品时迅速二次开发，扩展新的产品和配套的定价引擎。
* 产品为中国市场上流行的结构化产品，如香草、二元、亚式、鲨鱼鳍、雪球、凤凰等，基本覆盖了常见的结构。对一些产品进行了重点的优化，例如雪球，对期进行了细化和分类，支持多种雪球变种的定价。
* 模型覆盖面广。除了常见的解析解和Monte Carlo模拟法以外，还提供了PDE有限差分法、数值积分法、二叉树等各种例子，支持局部波动率和Heston随机波动率，可供参考。
* 单个产品的定价模型文件可以单独进行版本管理。面对市场的新结构、新需求，业务部门可以迅速在老模型的基础上改写，然而风控部门在面对扩展新模型时，需要避免影响定价库中其他模型的定价结果，保持历史重估值的稳定。PriceLib采用模块化松耦合的设计，扩展新产品、新模型不影响公共组件，便于进行版本管理。
* 较高的计算速度。期权定价涉及大量数值计算，对计算效率有一定要求。相对于常规Python定价代码，本定价库具有较高的计算速度，考虑到很多用户的机器并没有GPU，我们使用了以下方法提高计算效率：
  * numpy向量化计算；
  * JIT(Just in Time)即时编译技术；
  * 最近最少使用缓存lru_cache(Last recently used cache)；
  * 对于Monte Carlo模拟法，支持方差缩减方法和低差异序列，加快收敛速度，减少需要的模拟路径数；支持复用已有的随机数矩阵和价格路径；
  * 对于PDE有限差分法，由于需要求解的线性方程组的系数矩阵都是三对角矩阵，本定价库使用Thomas算法(the Tridiagonal Matrix Algorithm)提高求解效率，本方法比矩阵求逆或LU分解更快。

### 4.2 目前的局限
当前库中暂未把资金交易作为重点，例如，在发行雪球时，在各个时点与对手方进行现金和利息的收付，后续我们会视情况增加。此外，我们目前聚焦于权益类衍生品，利率、汇率、商品等其他品种会在后续逐步扩展。

### 4.3 一些应用
由于Python语言是出名的胶水语言，和C/C++、Java等工业界常用的开发语言可以很好的融合，当然也可以使用Flask直接封装成SaaS服务。总之，基于PriceLib定价库，可以开发多种场外业务系统，包括但不限于：
* 估值系统: 加上参数管理模块，再调用定价模型，即可进行估值。
* 风险监控系统: 盈亏分解、参数敏感性分析、压力测试报表等。
* 自动对冲交易系统: 由于定价库中提供了PDE等方法，可以支持实时的持仓和Greeks监控，因此可以开发交易系统进行雪球等产品的自动化对冲。

PriceLib作为一个核心定价引擎库，没有界面，不太直观，因此在PriceLib的基础上，我们短时间快速开发了pyRisk这个小巧的工具作为展示定价引擎的窗口。pyRisk具有以下功能：

* 产品与模型管理：结合券商的模型风险管理工作的“非技术型”痛点，参考国际标准《模型风险管理指南》及其附件(简称SR11-7)，根据产品收录和管理对应的模型，同一产品下的模型遵照统一的产品参数，模型根据定价方法具有自己的方法参数，成交参数与估值参数字段统一，全模型通用。

![产品与模型管理](./docs/pyRisk_1.png)

* 簿记：通过可扩展的设计，避免了现在很多场外MIS系统经常发生的“记不进去，估不出来”、要不断升级系统从而持续为MIS系统付费的问题。即使业务中出现新结构、新参数，也可以方便地簿记好。
* 估值：支持日终估值与损益归因，可以按照估值日-标的资产-结构种类分组汇总统计，并绘制盈亏折线图与损益归因堆积图。

![成交簿记、持仓估值、损益归因](./docs/pyRisk_2.png)

* 其他常用工具：如Greeks对比、敏感性分析，S-σ压力测试等。

![敏感性分析](./docs/pyRisk_3.png)

pyRisk的定位是一个小巧的风险管理的辅助工具，主要目的只是为了展示应用PriceLib的经典场景，您可以通过以下网址免费下载使用pyRisk：  
https://api.galatech.com.cn/pyRisk/pyRisk_1.0.0_win10_x64.zip  

如果对于系统有较高性能和产品定制化特性要求，欢迎与我们联系，试用凌瓴科技采用C++开发的智能风险管理平台iRisk。

## 5. 项目结构

```
├─examples 使用示例                   
├─pricelib 金融衍生品定价库
│  ├─common  公共类和函数
│  │  ├─product_base 产品基类
│  │  ├─pricing_engine_base 定价引擎基类
│  │  ├─processes 随机过程
│  │  ├─term_structures 期限结构
│  │  ├─volmodels 波动率模型
│  │  ├─time 日期处理
│  │  └─utilities 工具函数
│  ├─products 产品
│  │  ├─vanilla 香草期权
│  │  ├─asian 亚式期权
│  │  ├─digital 二元期权
│  │  ├─barrier 障碍期权
│  │  ├─autocallable 自动赎回
│  │  └─accurals 累计期权
│  └─pricing_engines 定价引擎
│     ├─analytic_engines 解析解
│     ├─fdm_engines PDE有限差分法
│     ├─integral_engines 数值积分法
│     ├─mc_engines 蒙特卡洛模拟
│     └─tree_engines 树方法
├─tests 测试
├─setup.py 安装配置文件
├─.flake8 静态代码检查配置文件
├─.pylintrc 静态代码检查配置文件
├─LICENSE 许可证文件
├─NOTICE 项目许可
└─README.md 说明文档
```

## 6. 项目依赖

支持的Python解释器版本及第三方库详见setup.py文件

## 7. 静态代码检查与单元测试

本项目使用`flake8`和`pylint`进行静态代码检查，使用`pytest`进行单元测试
* 在项目根目录下执行以下命令进行静态代码检查：`flake8 pricelib`，`pylint pricelib`
* 在项目根目录下执行以下命令进行单元测试：`pytest`

## 8. 项目许可

pricelib is an open-source financial derivatives pricing library written in Python.

This file is part of pricelib.

Copyright (C) 2024 Galaxy Technologies

Licensed under the Apache License, Version 2.0 (the "License");
you may not use pricelib except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## 9. 特别鸣谢

在长期的默默地准备中，来自上海财经大学的多位实习生做出了重要的贡献。在这里我们要大声地感谢他们：夏鸿翔、张鹏任、张峻尉、刘振伟、裴子涵，谢谢你们！也让我们感谢上海财经大学培养了这么多优秀的学生，为金融强国贡献了力量！


