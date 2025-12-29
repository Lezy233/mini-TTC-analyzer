"""
一个小脚本，用于从NGSIM数据集中提取两辆车的轨迹数据，并保存为CSV文件。
"""

import os
import pandas as pd

def extract_data(input_path=None, output_path=None):
    base_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    if input_path is None:
        # input_path = os.path.join(base_data, 'NGSIM-US101-dataset', '0750-0805.txt')
        input_path = os.path.join(base_data, 'NGSIM-US101-dataset', '0820-0835.txt')
    if output_path is None:
        output_path = os.path.join(base_data, 'data.csv')

    df = pd.read_csv(input_path, sep=r'\s+', header=None, usecols=[0, 3, 4, 5], engine='python')
    df.columns = ['Vehicle_ID', 'Global_Time', 'Local_X', 'Local_Y']
    
    # 选出现次数最多的两个车辆ID（保证数据量够）
    top_two_ids = df['Vehicle_ID'].value_counts().index[:2]
    car1_id, car2_id = top_two_ids
    
    # 分别提取两车数据，按时间戳对齐
    car1 = df[df['Vehicle_ID'] == car1_id].set_index('Global_Time')[['Local_X', 'Local_Y']]
    car2 = df[df['Vehicle_ID'] == car2_id].set_index('Global_Time')[['Local_X', 'Local_Y']]
    
    # 合并两车数据（内连接，只保留时间戳重叠的部分）
    merged = pd.concat([car1.add_suffix('_car1'), car2.add_suffix('_car2')], axis=1).dropna()
    
    merged.columns = ['x1', 'y1', 'x2', 'y2']

    # Create a single-row DataFrame with timestamps as column labels
    # so it aligns with `merged.T` which has columns == merged.index
    timestamp_row = pd.DataFrame([merged.index.astype(str).tolist()],
                                  index=['timestamp'],
                                  columns=merged.index)

    # 行标题：timestamp, x1, y1, x2, y2
    result = pd.concat([timestamp_row, merged.T])
    result.to_csv(output_path, header=False)
    print(f"提取完成：车辆{car1_id}和{car2_id}，数据形状：{result.shape}")

if __name__ == '__main__':
    extract_data()