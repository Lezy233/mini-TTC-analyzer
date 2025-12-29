import os
import time
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from base_analyzer import BaseAnalyzer


class TTCAnalyzer(BaseAnalyzer):
    """TTC 分析器，实现具体的计算逻辑与可视化。"""

    def __init__(self, input_path: str = 'data/data.csv', output_dir: str = 'output', config: dict | None = None):
        super().__init__(input_path, output_dir, config)
        # 默认配置
        cfg = self.config
        cfg.setdefault('ttc_threshold', 1.5)
        cfg.setdefault('max_speed_m_s', 60.0)

    def preprocess(self):
        super().preprocess()
        # 计算距离占位列
        self.df['distance'] = np.sqrt((self.df['x1'] - self.df['x2']) ** 2 + (self.df['y1'] - self.df['y2']) ** 2)

    def compute(self):
        df = self.df
        if df is None:
            raise RuntimeError('Data not loaded')

        # dt in seconds (timestamp is in milliseconds)
        df['dt'] = df['timestamp'].diff() / 1000.0

        # Velocities
        df['vx1'] = df['x1'].diff() / df['dt']
        df['vy1'] = df['y1'].diff() / df['dt']
        df['vx2'] = df['x2'].diff() / df['dt']
        df['vy2'] = df['y2'].diff() / df['dt']

        # Closing speed
        df['v_closing'] = -df['distance'].diff() / df['dt']

        # TTC
        df['ttc'] = np.where(df['v_closing'] > 0, df['distance'] / df['v_closing'], np.inf)

        # Safety alert
        TTC_THRESHOLD = float(self.config.get('ttc_threshold', 1.5))
        df['safety_alert'] = df['ttc'] < TTC_THRESHOLD

        # Abnormal detection
        max_speed = float(self.config.get('max_speed_m_s', 60.0))
        df['is_abnormal'] = (
            df['vx1'].abs() > max_speed
        ) | (
            df['vy1'].abs() > max_speed
        ) | (
            df['vx2'].abs() > max_speed
        ) | (
            df['vy2'].abs() > max_speed
        ) | (
            df['distance'] < 0
        )

        # Summaries
        self.summary = {
            'min_distance': df['distance'].min(),
            'min_ttc': df[df['ttc'] > 0]['ttc'].min(),
            'abnormal_count': int(df['is_abnormal'].sum()),
            'safety_alerts': int(df['safety_alert'].sum()),
        }

    def report(self):
        super().report()
        # append summary info
        summary_path = os.path.join(self.output_dir, 'summary.txt')
        with open(summary_path, 'a') as f:
            f.write(f"Minimum Distance: {self.summary['min_distance']:.2f} m\n")
            f.write(f"Minimum TTC: {self.summary['min_ttc']:.2f} s\n")
            f.write(f"Abnormal Frames: {self.summary['abnormal_count']}\n")
            f.write(f"Safety Alerts (TTC < {self.config['ttc_threshold']}s): {self.summary['safety_alerts']}\n")

    def visualize(self, ttc_ylim: Optional[float] = 20.0):
        df = self.df
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(df['x1'], df['y1'], label='Car 1')
        plt.plot(df['x2'], df['y2'], label='Car 2')
        plt.title('Vehicle Trajectories')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(df['timestamp'], df['ttc'], color='red', label='TTC')
        plt.axhline(y=self.config['ttc_threshold'], color='gray', linestyle='--', label=f"Safety Threshold ({self.config['ttc_threshold']}s)")
        if ttc_ylim:
            plt.ylim(0, ttc_ylim)
        plt.title('Time To Collision (TTC) over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('TTC (s)')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualization.png'))


if __name__ == '__main__':
    # run with default parameters
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)
    analyzer = TTCAnalyzer()
    res = analyzer.run()
    print(f"Analysis complete. Results saved to {res['output_dir']}")
    print(f"Processing time: {res['processing_time']:.4f} seconds")
