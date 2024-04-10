import deepxde as dde
import numpy as np
from deepxde.backend import torch
import torch
from scipy import io
from sklearn.preprocessing import StandardScaler
#9s=1000 steps

def periodic(x):
    x = x * 2 * np.pi
    x = torch.tensor(x, dtype=torch.float32)
    return torch.cat(
        [torch.cos(x), torch.sin(x), torch.cos(2 * x), torch.sin(2 * x)], dim=1
    )


def get_data(ntrain, ntest):
    sub_x = 2 ** 6
    sub_y = 2 ** 6

    data = io.loadmat("burgers_data_R10.mat")
    x_data = data["a"][:, ::sub_x].astype(np.float32)
    y_data = data["u"][:, ::sub_y].astype(np.float32)
    x_branch_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :]
    x_branch_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :]

    s = 2 ** 13 // sub_y
    grid = np.linspace(0, 1, num=2 ** 13)[::sub_y, None]

    x_train = (x_branch_train, grid)
    x_test = (x_branch_test, grid)
    return x_train, y_train, x_test, y_test


def train(model, lr, epochs):
    decay = ("inverse time", epochs // 5, 0.5)
    model.compile("adam", lr=lr, metrics=["mean l2 relative error"], decay=decay)
    losshistory, train_state = model.train(epochs=epochs, batch_size=None)
    print("\nTraining done ...\n")


def main():
    x_train, y_train, x_test, y_test = get_data(1000, 200)

    m = 2 ** 7
    net = dde.maps.DeepONetCartesianProd(
        [m, 128, 128, 128, 128], [4, 128, 128, 128], "tanh", "Glorot normal"
    )
    net.apply_feature_transform(periodic)

    scaler = StandardScaler().fit(y_train)
    std = torch.sqrt(torch.tensor(scaler.var_).float())

    def output_transform(inputs, outputs):
        mean = torch.tensor(scaler.mean_, dtype=torch.float32)
        std = torch.sqrt(torch.tensor(scaler.var_, dtype=torch.float32))
        return outputs * std + mean

    net.apply_output_transform(output_transform)

    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)
    model = dde.Model(data, net)

    lr = 0.001
    epochs = 50000
    train(model, lr, epochs)


if __name__ == "__main__":
    main()
