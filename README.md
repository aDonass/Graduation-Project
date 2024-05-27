#2024 NUAA 16院本人毕设
使用deeponet解决三类基础应用问题+工程技术难题（二维稳态对流反应扩散方程）

.mat/.npy类型的数据集下载：
https://yaleedu-my.sharepoint.com/personal/lu_lu_yale_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Flu%5Flu%5Fyale%5Fedu%2FDocuments%2Fdatasets%2F2022%5FCMAME%5FLu&ga=1


感谢@Lulu的论文给我巨大启发。
笔者尝试复现三类基础应用问题的代码时希望淘汰落后的tensorflow.addons结构，
但做着做着发现尾大不掉，索性用pytorch完全重写（线性不稳定波问题例外）

于是这里的deepxde.backend从奇怪的一堆bug的tensorflow.compat.v1换成了pytorch
运行代码前需要检查c盘.deepxde里的json文件的backend是什么。

二维稳态对流反应扩散方程的数据用matlab做不来，选择了deepxde自带的GRF+有限差分法求解。运行时需先运行generate.py再运行plot.py，后者同时集成了模型训练与结果评估（画热图）功能；画epoch-loss曲线则需要导出plot.py的数据为一维数组再填充到huatu.py。

4年大学恍然就即将结束，风景不错，遗憾不少，
我曾跌跌撞撞迷失于浮光掠影，却在最后的最后找到了学习的正确路径……
感谢导师李先生，感谢母校，感谢永远支持我的家庭，感谢一路上遇到的所有人。

【世界如此美丽，这使你充满了决心。】
