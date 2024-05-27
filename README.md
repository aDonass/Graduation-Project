#2024 NUAA 16院本人毕设
使用deeponet解决三类基础应用问题+工程技术难题（二维稳态对流反应扩散方程）

.mat/.npy类型的数据集下载：
https://yaleedu-my.sharepoint.com/personal/lu_lu_yale_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Flu%5Flu%5Fyale%5Fedu%2FDocuments%2Fdatasets%2F2022%5FCMAME%5FLu&ga=1


感谢@Lulu的论文给我巨大启发。
笔者尝试复现三类基础应用问题的代码时希望淘汰落后的tensorflow.addons结构，
但做着做着发现尾大不掉，索性用pytorch完全重写（线性不稳定波问题例外）

于是这里的deepxde.backend从奇怪的一堆bug的tensorflow.compat.v1换成了pytorch
如需运行代码，记得在c盘.deepxde里修改。

二维稳态对流反应扩散方程的数据用matlab做不来，选择了deepxde自带的GRF+有限差分法求解。

4年大学浮光掠影，却在最后的最后找到了学习的正确路径……
感谢导师李先生，感谢母校，感谢一路上遇到的所有人。
【世界如此美丽，这使你充满了决心。】
