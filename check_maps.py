import os
import numpy as np
from skimage import io
from skimage.measure import block_reduce

def analyze_map(map_path):
    """分析地图文件的构成"""
    try:
        # 读取地图
        ground_truth = (io.imread(map_path, as_gray=True) * 255).astype(int)
        
        # 获取所有独特的像素值及其数量
        unique_values, counts = np.unique(ground_truth, return_counts=True)
        
        print(f"\n地图文件: {os.path.basename(map_path)}")
        print(f"地图尺寸: {ground_truth.shape}")
        print("\n像素值分布:")
        for value, count in zip(unique_values, counts):
            percentage = (count / ground_truth.size) * 100
            print(f"值 {value}: {count} 个像素 ({percentage:.2f}%)")
        
    except Exception as e:
        print(f"错误: 处理地图时发生错误: {str(e)}")

def main():
    map_dir = 'maps'
    map_path = os.path.join(map_dir, "162.png")
    
    if not os.path.exists(map_path):
        print(f"错误: 地图文件 '161.png' 不存在!")
        return
        
    analyze_map(map_path)

if __name__ == '__main__':
    main()