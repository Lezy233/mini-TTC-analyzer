from abc import ABC, abstractmethod
import pandas as pd
import os
import time


class BaseAnalyzer(ABC):
    """基础分析器基类。定义通用接口和执行流程。

    子类需实现 `compute()` 方法。其他方法可被重写以定制行为。
    """

    def __init__(self, input_path: str = 'data/data.csv', output_dir: str = 'output', config: dict | None = None):
        self.input_path = input_path
        self.output_dir = output_dir
        self.config = config or {}
        self.start_time = None
        self.df: pd.DataFrame | None = None

    def ensure_output(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def load(self):
        """加载原始数据到 `self.df`。默认行为为读取转置格式的 CSV（5 行）。"""
        self.df = pd.read_csv(self.input_path, header=None, index_col=0).T
        # 默认列名，子类可重写
        self.df.columns = ['timestamp', 'x1', 'y1', 'x2', 'y2']
        self.df = self.df.apply(pd.to_numeric)

    def preprocess(self):
        """通用预处理（可被子类覆盖或扩展）。"""
        if self.df is None:
            raise RuntimeError('Data not loaded')

    @abstractmethod
    def compute(self):
        """核心计算逻辑，子类必须实现并保存结果到 `self.df`。"""

    def report(self):
        """默认生成 CSV 报表和 summary.txt。子类可调用 super().report() 后再扩展。"""
        if self.df is None:
            raise RuntimeError('Nothing to report')
        report_path = os.path.join(self.output_dir, 'analysis_report.csv')
        self.df.to_csv(report_path, index=False)

        summary_path = os.path.join(self.output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write('Analysis Summary\n')
            f.write('================\n')
            f.write(f'Total Frames: {len(self.df)}\n')
            f.write(f'Processing Time: {time.time() - self.start_time:.4f} s\n')

    def visualize(self):
        """可视化占位，子类可覆盖以绘图并保存文件到 output_dir。"""
        pass

    def run(self):
        self.ensure_output()
        self.start_time = time.time()
        self.load()
        self.preprocess()
        self.compute()
        self.report()
        self.visualize()
        return {
            'processing_time': time.time() - self.start_time,
            'output_dir': self.output_dir,
        }
