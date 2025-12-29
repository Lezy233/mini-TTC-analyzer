# Mini TTC Analyzer

这是一个基于实车轨迹 CSV 的自动 TTC（Time To Collision）分析工具。

## 功能特点

- **自动计算 TTC**：基于两车相对位置和速度计算碰撞时间。
- **最小安全距离**：自动识别全程最小车间距。
- **异常帧检测**：识别速度异常或数据跳变的非法帧。
- **高性能扫描**：支持千列数据秒级处理（实测 378 帧仅需 0.01s）。
- **数据可视化**：生成车辆轨迹图及 TTC 随时间变化曲线。
- **报表生成**：自动生成详细的 CSV 报表及汇总摘要。

## 输入数据格式 (data.csv)

CSV 文件包含 5 行（转置格式）：
1. `timestamp`: 时间戳 (ms)
2. `x1`: 第一辆车 x 轴坐标 (m)
3. `y1`: 第一辆车 y 轴坐标 (m)
4. `x2`: 第二辆车 x 轴坐标 (m)
5. `y2`: 第二辆车 y 轴坐标 (m)

## 快速开始

1. 确保已安装依赖：
   ```bash
   pip install pandas numpy matplotlib
   ```

2. 运行分析脚本：
   ```bash
   python src/ttc_analyzer.py
   ```

3. 查看结果：
   - `output/analysis_report.csv`: 详细逐帧数据。
   - `output/summary.txt`: 分析汇总（包含最小 TTC、异常帧数等）。
   - `output/visualization.png`: 可视化图表。

## 核心逻辑

- **速度计算**：$v = \frac{\Delta p}{\Delta t}$
- **闭合速度 (Closing Speed)**：$v_{closing} = -\frac{d(distance)}{dt}$
- **TTC 计算**：$TTC = \frac{distance}{v_{closing}}$ (仅当 $v_{closing} > 0$ 时)
- **安全阈值**：默认设为 1.5s。
