#2024 NUAA 16院本人毕设（的前一半代码）
使用deeponet解决三类偏微分方程问题，.mat/.npy类型的数据集下载：
https://yaleedu-my.sharepoint.com/personal/lu_lu_yale_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Flu%5Flu%5Fyale%5Fedu%2FDocuments%2Fdatasets%2F2022%5FCMAME%5FLu&ga=1
感谢@Lulu，感谢https://github.com/lu-group/deeponet-fno，
笔者尝试复现代码时希望淘汰落后的tensorflow.addons结构，
但做着做着发现尾大不掉，索性用pytorch完全重写（线性不稳定波问题例外）
于是这里的deepxde.backend从奇怪的一堆bug的tensorflow.compat.v1换成了pytorch
如需运行代码，记得在c盘.deepxde里修改。

4年大学浮光掠影，却在最后的最后找到了学习的正确路径……
感谢母校，感谢一路上遇到的所有人。
