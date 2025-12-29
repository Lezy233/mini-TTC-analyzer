# Mini TTC & Trajectory Similarity Analyzer

本项目最初实现基于实车轨迹的 TTC（Time To Collision）分析，已工程化重构为可扩展的类结构，并新增基于重采样 + DTW 的轨迹相似度分析，以及 Plotly 可视化支持。

主要模块
- `src/base_analyzer.py`：基础抽象类 `BaseAnalyzer`，定义通用接口：`load()/preprocess()/compute()/report()/visualize()/run()`，便于封装与继承。
- `src/ttc_analyzer.py`：`TTCAnalyzer(BaseAnalyzer)`，实现 TTC、速度、异常帧检测与静态可视化（Matplotlib）。
- `src/similarity_analyzer.py`：`TrajectorySimilarityAnalyzer(BaseAnalyzer)`，实现轨迹重采样、2D DTW、逐点均距与 x/y 相关性计算，生成相似度评分，并使用 Plotly 输出交互式图表（`output/similarity.html`）。
- `src/run_analysis.py`：统一运行入口，支持 CLI 参数（`--input/--output/--resample`）。

输入格式
- 默认输入：转置格式 CSV（5 行，分别为 `timestamp, x1, y1, x2, y2`），示例见 `data/data.csv`。

依赖
- 项目依赖在 `requirements.txt` 中列出，主要包括：`pandas`, `numpy`, `plotly`, `kaleido`（用于将 Plotly 导出为静态图片）。

快速开始
1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行轨迹相似度分析（默认示例数据）：
```bash
python src/run_analysis.py --input data/data.csv --output output --resample 300
```

3. 运行 TTC 分析（如果需要调用特定类，可直接运行脚本或导入类）：
```bash
python -c "from ttc_analyzer import TTCAnalyzer; TTCAnalyzer().run()"
```

查看结果：
- `output/analysis_report.csv`: 逐帧计算结果（若运行 TTC 分析器）。
- `output/summary.txt`: 运行摘要（包含相似度/最小 TTC/异常统计等）。
- `output/resampled_distances.csv`: 轨迹重采样后每个样本的点对点距离序列。
- `output/similarity.html`: 交互式 Plotly 轨迹对比图。
- `output/similarity.png`（可选）：静态图像（需 `kaleido` 支持）。

可配置项
- 通过 `config` 字典或 `run_analysis.py` 的 CLI 参数配置：
  - `resample_points`：重采样点数（TrajectorySimilarityAnalyzer，默认 300）。
  - `ttc_threshold`：TTC 警报阈值（TTCAnalyzer，默认 1.5s）。
  - `max_speed_m_s`：异常判定最大速度阈值（TTCAnalyzer，默认 60 m/s）。

扩展建议
- 将相似度算法做成策略接口（支持 Fréchet、Hausdorff、归一化 DTW 等），以便插拔扩展。
- 添加单元测试覆盖 `dtw_distance`、重采样与可视化输出生成。
- 若需在线交互查看结果，考虑使用 Streamlit 或 Flask 封装一个简单的前端服务。

联系方式与贡献
- 欢迎在本仓库基础上扩展算法、改进异常检测策略或增强可视化展示。提交 PR 前请确保添加相应的测试和更新 `update.md`。

---
本 README 已更新以反映类化重构与轨迹相似度功能。

- **速度计算**：$v = \frac{\Delta p}{\Delta t}$
- **闭合速度 (Closing Speed)**：$v_{closing} = -\frac{d(distance)}{dt}$
- **TTC 计算**：$TTC = \frac{distance}{v_{closing}}$ (仅当 $v_{closing} > 0$ 时)
- **安全阈值**：默认设为 1.5s。
