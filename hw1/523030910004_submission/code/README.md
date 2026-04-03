运行方式
1. 批量矫正并生成 60 张结果图：
   python restore_homography.py
2. 针对 raw_10_warp6 进行改进矫正：
   python restore_raw_10_warp6.py

输出说明
- 60 张结果图保存到 ../restored_images/，文件名与输入图像一致。
- 单图改进输出为 ../restored_images/raw_10_warp6_refined.png。
- 单应矩阵记录保存为 h_matrices.txt。

文件说明
- restore_homography.py：批量特征匹配与单应估计。
- restore_raw_10_warp6.py：单图改进处理脚本。
- h_matrices.txt：批量处理生成的单应矩阵记录。
