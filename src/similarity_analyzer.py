import os
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from base_analyzer import BaseAnalyzer


def _resample_traj(times: np.ndarray, xs: np.ndarray, ys: np.ndarray, n_samples: int) -> np.ndarray:
    """Linearly resample trajectory to n_samples points. Returns shape (n_samples,2)."""
    if len(times) < 2:
        return np.column_stack((xs, ys))
    t_min, t_max = times.min(), times.max()
    t_new = np.linspace(t_min, t_max, n_samples)
    x_new = np.interp(t_new, times, xs)
    y_new = np.interp(t_new, times, ys)
    return np.column_stack((x_new, y_new))


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Simple 2D DTW distance (sum of Euclidean per-step cost)."""
    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = np.linalg.norm(a[i - 1] - b[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return cost[n, m]


class TrajectorySimilarityAnalyzer(BaseAnalyzer):
    """分析两条轨迹的相似度，支持 DTW、平均欧氏距离与相关性指标，使用 plotly 绘图。"""

    def __init__(self, input_path: str = 'data/data.csv', output_dir: str = 'output', config: dict | None = None):
        super().__init__(input_path, output_dir, config)
        cfg = self.config
        cfg.setdefault('resample_points', 300)

    def preprocess(self):
        super().preprocess()
        # nothing extra here; base class already parsed timestamp,x1,y1,x2,y2

    def compute(self):
        df = self.df
        if df is None:
            raise RuntimeError('Data not loaded')

        times = df['timestamp'].to_numpy()
        x1 = df['x1'].to_numpy()
        y1 = df['y1'].to_numpy()
        x2 = df['x2'].to_numpy()
        y2 = df['y2'].to_numpy()

        n_samples = int(self.config.get('resample_points', 300))
        traj1 = _resample_traj(times, x1, y1, n_samples)
        traj2 = _resample_traj(times, x2, y2, n_samples)

        # Mean Euclidean distance after resampling (pointwise)
        mean_dist = float(np.mean(np.linalg.norm(traj1 - traj2, axis=1)))

        # DTW distance (normalized by length)
        dtw_raw = float(dtw_distance(traj1, traj2))
        dtw_norm = dtw_raw / max(len(traj1), len(traj2))

        # Pearson correlation for x and y (on resampled)
        corr_x = np.corrcoef(traj1[:, 0], traj2[:, 0])[0, 1]
        corr_y = np.corrcoef(traj1[:, 1], traj2[:, 1])[0, 1]
        # handle NaN
        corr_x = 0.0 if np.isnan(corr_x) else float(corr_x)
        corr_y = 0.0 if np.isnan(corr_y) else float(corr_y)

        # similarity score: combine distance (inverse) and correlation
        sim_from_dist = 1.0 / (1.0 + mean_dist)
        sim_from_corr = max(0.0, (corr_x + corr_y) / 2.0)  # map to [0,1]
        similarity_score = 0.6 * sim_from_corr + 0.4 * sim_from_dist

        self.summary = {
            'mean_pointwise_distance': mean_dist,
            'dtw_normalized': dtw_norm,
            'corr_x': corr_x,
            'corr_y': corr_y,
            'similarity_score': float(similarity_score),
            'resampled_points': n_samples,
        }

        # store resampled distances per-sample for report
        per_sample_dist = np.linalg.norm(traj1 - traj2, axis=1)
        self.df_resampled = pd.DataFrame({
            'sample_index': np.arange(len(per_sample_dist)),
            'distance': per_sample_dist,
        })

    def report(self):
        # write base report (the original frame-level CSV) and then append similarity summary
        super().report()
        summary_path = os.path.join(self.output_dir, 'summary.txt')
        with open(summary_path, 'a') as f:
            f.write('\nTrajectory Similarity Summary\n')
            f.write('-----------------------------\n')
            f.write(f"Resampled Points: {self.summary['resampled_points']}\n")
            f.write(f"Mean Pointwise Distance: {self.summary['mean_pointwise_distance']:.4f} m\n")
            f.write(f"DTW (normalized): {self.summary['dtw_normalized']:.4f}\n")
            f.write(f"Correlation X: {self.summary['corr_x']:.4f}\n")
            f.write(f"Correlation Y: {self.summary['corr_y']:.4f}\n")
            f.write(f"Similarity Score (0-1): {self.summary['similarity_score']:.4f}\n")

        # save per-sample distances
        resampled_csv = os.path.join(self.output_dir, 'resampled_distances.csv')
        self.df_resampled.to_csv(resampled_csv, index=False)

    def visualize(self):
        df = self.df
        if df is None:
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['x1'], y=df['y1'], mode='lines+markers', name='Car 1'))
        fig.add_trace(go.Scatter(x=df['x2'], y=df['y2'], mode='lines+markers', name='Car 2'))

        title = (
            f"Trajectory Comparison — Similarity {self.summary['similarity_score']:.4f} | "
            f"MeanDist {self.summary['mean_pointwise_distance']:.2f} m | DTW {self.summary['dtw_normalized']:.2f}"
        )
        fig.update_layout(title=title, xaxis_title='X (m)', yaxis_title='Y (m)', legend=dict(x=0.01, y=0.99))

        out_html = os.path.join(self.output_dir, 'similarity.html')
        fig.write_html(out_html, include_plotlyjs='cdn')

        # attempt static export as PNG if kaleido is available
        try:
            out_png = os.path.join(self.output_dir, 'similarity.png')
            fig.write_image(out_png)
        except Exception:
            # silent fail: PNG requires kaleido
            pass


if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)
    analyzer = TrajectorySimilarityAnalyzer()
    res = analyzer.run()
    print('Similarity analysis done. Output:', res['output_dir'])
