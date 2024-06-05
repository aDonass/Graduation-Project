#2024 NUAA 16院本人毕设
使用deeponet解决三类基础应用问题+工程技术难题（二维稳态对流反应扩散方程）

.mat/.npy类型的数据集下载：
https://yaleedu-my.sharepoint.com/personal/lu_lu_yale_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Flu%5Flu%5Fyale%5Fedu%2FDocuments%2Fdatasets%2F2022%5FCMAME%5FLu&ga=1


感谢@Lulu的论文给我巨大启发: L. Lu, X. Meng, S. Cai, Z. Mao, S. Goswami, Z. Zhang, & G. E. Karniadakis. A comprehensive and fair comparison of two neural operators (with practical extensions) based on FAIR data. Computer Methods in Applied Mechanics and Engineering, 393, 114778, 2022.
笔者尝试复现三类基础应用问题的代码时发现，作者使用Tensorflow1.7+Python2.7实现的DeepONet解方程已经落后，尤其是18年 Tensorflow2.0发布不兼容早期版本，tensorflow.addons库停止使用，导致原始代码无法运行。
毕设伊始试图仅替换tensorflow.addons，但做着做着发现尾大不掉，索性用pytorch2.2+python3.11完全重写（线性不稳定波问题例外）

于是本研究的deepxde.backend从奇怪的一堆bug的tensorflow.compat.v1换成了torch
运行代码前需要检查c盘.deepxde里的json文件的backend。

二维稳态对流反应扩散方程的数据用matlab做不来，选择了deepxde自带的GRF+有限差分法求解。运行时需先运行generate.py再运行netplot.py，后者同时集成了模型训练与结果评估（画热图）功能；画epoch-loss曲线则需要导出netplot.py的数据为一维数组再填充到huatu.py。

4年大学恍然就即将结束，风景不错，遗憾不少，
我曾跌跌撞撞迷失于浮光掠影，却在最后的最后找到了学习的正确路径。
感谢导师李先生，感谢母校，感谢永远支持我的家庭，感谢一路上遇到的所有人。

【世界如此美丽，这使你充满了决心。】
