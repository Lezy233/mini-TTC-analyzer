import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def analyze_ttc(input_path='data/data.csv', output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()

    # 1. Load and Transpose Data 加载数据
    # 5 rows: timestamp, x1, y1, x2, y2
    df_raw = pd.read_csv(input_path, header=None, index_col=0)
    df = df_raw.T
    df.columns = ['timestamp', 'x1', 'y1', 'x2', 'y2']
    df = df.apply(pd.to_numeric)
    
    # 2. Calculate Distance 计算距离
    df['distance'] = np.sqrt((df['x1'] - df['x2'])**2 + (df['y1'] - df['y2'])**2)
    
    # 3. Calculate Velocity and Relative Velocity
    # dt in seconds (timestamp is in milliseconds)
    df['dt'] = df['timestamp'].diff() / 1000.0
    
    # Velocity for car 1
    df['vx1'] = df['x1'].diff() / df['dt']
    df['vy1'] = df['y1'].diff() / df['dt']
    # Velocity for car 2
    df['vx2'] = df['x2'].diff() / df['dt']
    df['vy2'] = df['y2'].diff() / df['dt']
    
    # Relative velocity (closing speed)
    # v_closing = - d(distance) / dt
    df['v_closing'] = -df['distance'].diff() / df['dt']
    
    # 4. Calculate TTC 计算TTC
    # TTC = distance / v_closing (only if v_closing > 0)
    df['ttc'] = np.where(df['v_closing'] > 0, df['distance'] / df['v_closing'], np.inf)
    
    # 5. Safety Alert 检测安全警报
    TTC_THRESHOLD = 1.5 # seconds
    df['safety_alert'] = df['ttc'] < TTC_THRESHOLD

    # 6. Identify Abnormal Frames 识别异常帧
    # Abnormal if distance jumps too much or velocity is unrealistic
    # For this example, let's say v > 60 m/s (216 km/h) is abnormal
    # Or if distance is negative (shouldn't happen)
    df['is_abnormal'] = (df['vx1'].abs() > 60) | (df['vy1'].abs() > 60) | \
                        (df['vx2'].abs() > 60) | (df['vy2'].abs() > 60) | \
                        (df['distance'] < 0)
    
    # 7. Minimum Safety Distance 最小安全距离
    min_dist = df['distance'].min()
    min_ttc = df[df['ttc'] > 0]['ttc'].min()
    
    # 8. Generate Report 生成报告
    report_path = os.path.join(output_dir, 'analysis_report.csv')
    df.to_csv(report_path, index=False)
    
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Analysis Summary\n")
        f.write(f"================\n")
        f.write(f"Total Frames: {len(df)}\n")
        f.write(f"Minimum Distance: {min_dist:.2f} m\n")
        f.write(f"Minimum TTC: {min_ttc:.2f} s\n")
        f.write(f"Abnormal Frames: {df['is_abnormal'].sum()}\n")
        f.write(f"Safety Alerts (TTC < {TTC_THRESHOLD}s): {df['safety_alert'].sum()}\n")
        f.write(f"Processing Time: {time.time() - start_time:.4f} s\n")

    # 9. Visualization 数据可视化
    plt.figure(figsize=(12, 8))
    
    # Plot Trajectories 
    plt.subplot(2, 1, 1)
    plt.plot(df['x1'], df['y1'], label='Car 1')
    plt.plot(df['x2'], df['y2'], label='Car 2')
    plt.title('Vehicle Trajectories')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    
    # Plot TTC
    plt.subplot(2, 1, 2)
    plt.plot(df['timestamp'], df['ttc'], color='red', label='TTC')
    plt.axhline(y=1.5, color='gray', linestyle='--', label='Safety Threshold (1.5s)')
    plt.ylim(0, 20)
    plt.title('Time To Collision (TTC) over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('TTC (s)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualization.png'))
    
    print(f"Analysis complete. Results saved to {output_dir}")
    print(f"Processing time: {time.time() - start_time:.4f} seconds")

if __name__ == '__main__':
    # Ensure we are in the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)
    analyze_ttc()
