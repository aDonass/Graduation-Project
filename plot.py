import deepxde as dde
import numpy as np
from deepxde.backend import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import io
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#34s=1000epoch

def get_data(filename1, filename2, ndata):
    x_branch = np.load(filename1)
    y = np.load(filename2)
    x_branch = x_branch[0:ndata].astype(np.float32)
    #print(type)
    y =y[0:ndata].astype(np.float32)
    s = x_branch.shape[1]



    grids = []
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    x_branch = x_branch.reshape(ndata, s * s)
    x = (x_branch, grid)
    y = y.reshape(ndata, s * s)
    return x, y


def dirichlet(inputs, output):
    x_trunk = inputs[1]
    x, y = x_trunk[:, 0], x_trunk[:, 1]
    return 20 * x * (1 - x) * y * (1 - y) * (output + 1)

class Branch(nn.Module):
    def __init__(self, m, activation):
        super(Branch, self).__init__()
        self.reshape = nn.Unflatten(1, (1, 29, 29))
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5, 5), stride=2)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(128 * 5 * 5, 128)  # Adjust the input features accordingly
        self.dense2 = nn.Linear(128, 128)

            # Store the activation function
        self.activation = activation

    def forward(self, x):
        x = self.reshape(x)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.flatten(x)
        x = self.activation(self.dense1(x))
        x = self.dense2(x)
        return x
def main():
    x_train, y_train = get_data("f.npy", "u.npy", 800)
    x_test, y_test = get_data("f.npy", "u.npy", 200)
    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

    # Example usage:
    # m is the size of the input feature vector
    m = 10000  # Example value, should be set according to your specific input
    activation = F.relu  # Example activation function, replace with your choice
    branch = Branch(m, activation)
    print(branch)

    net = dde.nn.pytorch.DeepONetCartesianProd(
        [m, 128], [2, 128, 128, 128, 128], activation, "Glorot normal"
    )

    scaler = StandardScaler().fit(y_train)
    std = np.sqrt(scaler.var_.astype(np.float32))

    def output_transform(inputs, outputs):
        # 确保输出和标准化参数都在相同的设备上
        mean = torch.tensor(scaler.mean_.astype(np.float32)).to(outputs.device)
        std = torch.tensor(std).to(outputs.device)

        # 分离输出，以便进行 NumPy 转换
        return outputs.detach().cpu() * std + mean

    net.apply_output_transform(output_transform)
    net.apply_output_transform(dirichlet)

    model = dde.Model(data, net)
    model.compile(
        torch.optim.AdamW(net.parameters(),1e-4),
        decay=("inverse time", 1, 1e-4),
        metrics=["mean l2 relative error"],
    )
    losshistory, train_state = model.train(epochs=30000, batch_size=None)
    y_pred = model.predict(x_test)

    # 反标准化，将预测结果转换回原始尺度
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)  # 确保真实值也是反标准化的

    y_pred =y_pred*1000
    y_test =y_test*1000
    s = int(np.sqrt(y_test.shape[1]))  # 假设网格点数是正方形

    # 绘制对比图
    for i in range(len(y_test)):
        plt.figure(figsize=(14, 6))

        # 绘制真实解的热图
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(1, 2, 1)
        plt.title('真实解')
        plt.imshow(y_test[i].reshape(s, s), cmap='viridis')
        plt.colorbar()

        # 绘制预测解的热图
        plt.subplot(1, 2, 2)
        plt.title('预测解')
        plt.imshow(y_pred[i].reshape(s, s), cmap='viridis')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(f"test_case_{i + 1}_heatmap.png")  # 保存图像
        plt.show()


if __name__ == "__main__":
        main()