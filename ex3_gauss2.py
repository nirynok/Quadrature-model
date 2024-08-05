import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from torch import autograd
import utils
import argparse
import os
import pickle


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--training_DT', default=1e-2, type=float, help='training period')
    parser.add_argument('--predicting_DT', default=10, type=float, help='predicting period')
    parser.add_argument('--n_particles', default=2, type=int, help='number of particles')
    parser.add_argument('--dimension', default=2, type=int, help='dimension of the problem')
    parser.add_argument('--nx', default=12, type=int, help=' the number of terms of the Taylor polynomial')
    parser.add_argument('--hidden_dim', default=32, type=int, help='dimension of the hidden layer')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--step_size', default=10, type=int, help='the period of learning rate decay')
    parser.add_argument('--gamma', default=0.8, type=float, help='multiplicative factor of learning rate decay')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--print_every', default=20, type=int, help='number of epochs between prints and plots')
    parser.add_argument('--predicting_points', default=200, type=int, help='number of predicting points on each plot')
    parser.add_argument('--training_points', default=1, type=int, help='number of training points on each plot')
    parser.add_argument('--training_samples', default=25, type=int, help='number of training samples')
    parser.add_argument('--testing_samples', default=100, type=int, help='number of testing samples')
    parser.add_argument('--noise', default=0, type=float, help='noise added on data')
    parser.add_argument('--eps', default=1e-3, type=float, help='eps used in integrator')
    parser.add_argument('--tol', default=1e-3, type=float, help='tolerance of neural network')
    parser.add_argument('--use_ReLU', dest='use_ReLU', action='store_true', help='use relu as activation function')
    parser.add_argument('--use_pde', dest='use_pde', action='store_true', help='use sin as activation function')
    parser.add_argument('--use_pade', dest='use_pade', action='store_true', help='use sin as activation function')
    parser.add_argument('--use_pade1', dest='use_pade1', action='store_true', help='use sin as activation function')
    parser.add_argument('--use_pade_type', dest='use_pade_type', action='store_true', help='use sin as activation function')
    parser.add_argument('--use_sin', dest='use_sin', action='store_true', help='use sin as activation function')
    parser.add_argument('--use_sin_act', dest='use_sin_act', action='store_true', help='use sin as activation function')
    parser.add_argument('--use_sin_para', dest='use_sin_para', action='store_true',
                        help='use sin as activation function')
    parser.add_argument('--use_prony', dest='use_prony', action='store_true', help='use sin as activation function')
    parser.add_argument('--L1', default=4, type=int, help='batch')
    parser.add_argument('--L2', default=4, type=int, help='batch')
    parser.add_argument('--l', default=4, type=int, help='batch')
    parser.add_argument("--save", type=str, default="experiments")
    parser.add_argument("--data", type=str, default="_num_25_1")
    parser.add_argument("--N", type=str, default="1")
    parser.set_defaults(feature=True)
    return parser.parse_args()


class Linear_BBB2(nn.Module):
    """
        Layer of our BNN.
    """
    def __init__(self, input_features, output_features, h):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features
        self.h = h

        # initialize mu and rho parameters for the weights of the layer

        self.f_w_1 = nn.Parameter((torch.rand([input_features, 16]) * math.sqrt(1. / 32)).requires_grad_(True))
        self.f_w_2 = nn.Parameter((torch.rand([16, output_features]) * math.sqrt(1. / 16)).requires_grad_(True))
        self.f_b_1 = nn.Parameter((torch.rand([16]) * math.sqrt(1. / 16)).requires_grad_(True))

        self.g_w_1 = nn.Parameter((torch.rand([input_features, 16]) * math.sqrt(1. / 32)).requires_grad_(True))
        self.g_w_2 = nn.Parameter((torch.rand([16, output_features]) * math.sqrt(1. / 16)).requires_grad_(True))
        self.g_b_1 = nn.Parameter((torch.rand([16]) * math.sqrt(1. / 16)).requires_grad_(True))

        self.K1 = nn.Parameter((torch.randn([input_features, 16]) * math.sqrt(1. / 32)).requires_grad_(True))
        self.a1 = nn.Parameter((torch.randn([16]) * math.sqrt(1. / 16)).requires_grad_(True))
        self.b1 = nn.Parameter((torch.randn([16]) * math.sqrt(1. / 16)).requires_grad_(True))

        self.K2 = nn.Parameter((torch.rand([input_features, 16]) * math.sqrt(1. / 32)).requires_grad_(True))
        self.a2 = nn.Parameter((torch.rand([16]) * math.sqrt(1. / 16)).requires_grad_(True))
        self.b2 = nn.Parameter((torch.rand([16]) * math.sqrt(1. / 16)).requires_grad_(True))
        self.c2 = nn.Parameter((torch.rand([input_features]) * math.sqrt(1. / input_features)).requires_grad_(True))

    def forward(self, input):
        H0 = torch.sigmoid(input @ self.f_w_1 + self.f_b_1) @ self.f_w_2
        H1 = torch.tanh(input @ self.g_w_1 + self.g_b_1) @ self.g_w_2
        f = (torch.sigmoid(input @ self.K1 + self.b1) * self.a1) @ self.K1.t()
        g = (torch.tanh(input @ self.K2 + self.b2) * self.a2) @ self.K2.t() + self.c2
        # H0 = H1
        # H0 = torch.tanh(input @ self.g_w_1 + self.g_b_1) @ self.g_w_2
        # H0 = (input[0]**2 + input[1]**2)/2
        # H1 = H0
        return f, g

    def diff1(self, x):
        J = torch.zeros(2, 2)
        J[0, 1] = 1.0
        J[1, 0] = -1.0
        x.requires_grad_()
        f, g = self(x)
        grad00 = f @ J
        grad11 = g @ J
        grad111 = autograd.grad(outputs=g, inputs=x, grad_outputs=grad11, retain_graph=True)[0] @ J
        return grad00, grad111, grad11

    def diff(self, x):
        J = torch.zeros(2, 2)
        J[0, 1] = 1.0
        J[1, 0] = -1.0
        x.requires_grad_()
        H0, H1 = self(x)
        grad0 = autograd.grad(outputs=H0, inputs=x, grad_outputs=torch.ones_like(H0), create_graph=True,
                              retain_graph=True)[0]
        grad00 = grad0 @ J
        grad1 = autograd.grad(outputs=H1, inputs=x, grad_outputs=torch.ones_like(H1), create_graph=True,
                              retain_graph=True)[0]
        grad11 = grad1 @ J
        grad111 = autograd.grad(outputs=grad1, inputs=x, grad_outputs=grad11, retain_graph=True)[0] @ J
        return grad00, grad111, grad11

    def cov1(self, W, W00, W111):
        W00_s = W00[:, :, 0, :] + W00[:, :, 1, :]
        W111_s = W111[:, :, 0, :] + W111[:, :, 1, :]
        Y = W[:, :, -1, :] - W[:, :, 0, :] - W00_s / 2 - W111_s / 4
        Y = Y.view(W.shape[0], W.shape[1], 1, W.shape[3])
        Y_T = torch.transpose(Y, dim0=2, dim1=3)
        Y_cov = (Y_T @ Y).sum(dim=0) / W.shape[0]
        return Y_cov

    def cov2(self, W):
        E1 = W[:, :, 0, :].view(W.shape[0], W.shape[1], 1, W.shape[3])
        E1_T = torch.transpose(E1, dim0=2, dim1=3)
        E2 = W[:, :, 1, :].view(W.shape[0], W.shape[1], 1, W.shape[3])
        E2_T = torch.transpose(E2, dim0=2, dim1=3)
        E_cov = (E1_T @ E1 + E2_T @ E2).sum(dim=0) / (W.shape[0]*2)
        return E_cov

    def sample_elbo2(self, E, W):
        W00, W111, W11 = self.diff1(W[:, :, 1:-1])
        # 期望
        E00 = (W00[:, :, 0, :] + W00[:, :, 1, :]).sum(axis=0) / (
                    W.shape[0] * 2)
        E111 = (W111[:, :, 0, :] + W111[:, :, 1, :]).sum(axis=0) / (
                    W.shape[0] * 2)
        LOSS = torch.nn.functional.mse_loss(E, W[0, :, 0, :] + E00 + E111 / 2,
                                            reduction='mean') + torch.nn.functional.mse_loss(
            self.cov1(W, W00, W111), self.cov2(W11), reduction='mean')
        return LOSS


def toy_function(x):
    return -x**4 + 3*x**2 + 1


def H(x):
    H0 = -4*torch.cos(x[:, :, :, 1])+x[:, :, :, 0]**2/2
    H1 = 0.2*torch.sin(x[:, :, :, 1])+1/2*x[:, :, :, 1]
    # H1 = x[:, :, :, 0]
    return H0.view(x.shape[0], x.shape[1], x.shape[2], 1), H1.view(x.shape[0], x.shape[1], x.shape[2], 1)


def H1(x):
    H0 = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 2)
    H1 = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 2)
    H0[:, :, :, 0] = x[:, :, :, 0]
    H0[:, :, :, 1] = 4*torch.sin(x[:, :, :, 1])
    H1[:, :, :, 0] = torch.zeros_like(x[:, :, :, 1])
    H1[:, :, :, 1] = 0.2*torch.cos(x[:, :, :, 1])+1/2
    return H0, H1


def diff1(x):
    J = torch.zeros(2, 2)
    J[0, 1] = 1.0
    J[1, 0] = -1.0
    x.requires_grad_()
    f, g = H1(x)
    grad00 = f @ J
    grad11 = g @ J
    grad111 = autograd.grad(outputs=g, inputs=x, grad_outputs=grad11, retain_graph=True)[0] @ J
    return grad00, grad111, grad11


def f(x):
    H0 = torch.zeros(x.shape[0], 2)
    H0[:, 0] = x[:, 0]
    H0[:, 1] = 4 * torch.sin(x[:, 1])
    return H0


def g(x):
    H1 = torch.zeros(x.shape[0], 2)
    H1[:, 0] = torch.zeros_like(x[:, 1])
    H1[:, 1] = 0.2 * torch.cos(x[:, 1]) + 1 / 2
    return H1


def diff(x):
    J = torch.zeros(2, 2)
    J[0, 1] = 1.0
    J[1, 0] = -1.0
    x.requires_grad_()
    H0, H1 = H(x)
    grad0 = autograd.grad(outputs=H0, inputs=x, grad_outputs=torch.ones_like(H0), create_graph=True,
                          retain_graph=True)[0]
    grad00 = grad0 @ J
    grad1 = autograd.grad(outputs=H1, inputs=x, grad_outputs=torch.ones_like(H1), create_graph=True,
                          retain_graph=True)[0]

    grad11 = grad1 @ J
    grad111 = autograd.grad(outputs=grad1, inputs=x, grad_outputs=grad11, retain_graph=True)[0] @ J
    return grad00, grad111, grad11


def cov1(W, W00, W111):
    W00_s = W00[:, :, 0, :] + W00[:, :, 1, :]
    W111_s = W111[:, :, 0, :] + W111[:, :, 1, :]
    Y = W[:, :, -1, :] - W[:, :, 0, :] - W00_s / 2 - W111_s / 4
    Y = Y.view(W.shape[0], W.shape[1], 1, W.shape[3])
    Y_T = torch.transpose(Y, dim0=2, dim1=3)
    Y_cov = (Y_T @ Y).sum(dim=0) / W.shape[0]
    return Y_cov


def cov2(W):
    E1 = W[:, :, 0, :].view(W.shape[0], W.shape[1], 1, W.shape[3])
    E1_T = torch.transpose(E1, dim0=2, dim1=3)
    E2 = W[:, :, 1, :].view(W.shape[0], W.shape[1], 1, W.shape[3])
    E2_T = torch.transpose(E2, dim0=2, dim1=3)
    E_cov = (E1_T @ E1 + E2_T @ E2).sum(dim=0) / (W.shape[0] * 2)
    return E_cov


def sample_elbo2(E, W):
    W00, W111, W11 = diff1(W[:, :, 1:-1])
    # 期望
    E00 = (W00[:, :, 0, :] + W00[:, :, 1, :]).sum(axis=0) / (
            W.shape[0] * 2)
    E111 = (W111[:, :, 0, :] + W111[:, :, 1, :]).sum(axis=0) / (
            W.shape[0] * 2)
    LOSS1 = torch.nn.functional.mse_loss(E, W[0, :, 0, :] + E00 + E111 / 2, reduction='mean')
    LOSS2 = torch.nn.functional.mse_loss(cov1(W, W00, W111), cov2(W11), reduction='mean')
    return LOSS1, LOSS2

def hamitonsolution_function(x, t0, t1):
    N = np.round((torch.tensor(t1-t0)/(1e-2)).max().item())
    h = (t1-t0)/N
    p0 = x[0]
    q0 = x[1]
    a = 1
    b = 1
    temp = np.random.normal(loc=0.0, scale=1.0, size=None)
    # temp = 0
    m = np.sqrt(h)
    X = [x]
    A = [[1, (a * h + b * m * temp) / 2],
         [-(a * h + b * m * temp) / 2, 1]]
    s = [p0 - q0 * (a * h + b * m * temp) / 2, q0 + p0 * (a * h + b * m * temp) / 2]
    # A = [[1, (a * h) / 2],
    #      [-(a * h) / 2, 1]]
    # s = [p0 - q0 * (a * h) / 2, q0 + p0 * (a * h) / 2 + m * temp]
    r = np.linalg.solve(A, s)
    p = r[0]
    q = r[1]
    for i in range(int(N-1)):
        p0 = p
        q0 = q
        X.append([p0, q0])
        temp = np.random.normal(loc=0.0, scale=1.0, size=None)
        A = [[1, (a * h + b * m * temp) / 2],
             [-(a * h + b * m * temp) / 2, 1]]
        s = [p0 - q0 * (a * h + b * m * temp) / 2, q0 + p0 * (a * h + b * m * temp) / 2]
        # A = [[1, (a * h) / 2],
        #      [-(a * h) / 2, 1]]
        # s = [p0 - q0 * (a * h) / 2, q0 + p0 * (a * h) / 2 + m * temp]
        r = np.linalg.solve(A, s)
        p = r[0]
        q = r[1]

    X.append([p, q])
    return np.array(X)
    # return np.hstack((p, q))

def hamitonsolution_function1(x, t0, t1):
    p0 = x[0]
    q0 = x[1]
    m = np.random.normal(loc=0.0, scale=np.sqrt(t1-t0), size=None)
    p = p0*np.cos(t1-t0 + m) - q0*np.sin(t1 - t0 + m)
    q = p0 * np.sin(t1 - t0 + m) + q0 * np.cos(t1 - t0 + m)
    return np.array([[p, q]])

def h_0(x):
    y = -4 * torch.cos(x[:, 1]) + x[:, 0] ** 2 / 2
    return y


def h_1(x):
    y = 0.2 * torch.sin(x[:, 1]) + 1 / 2 * x[:, 1]
    return y


def gen_data():
    W = torch.zeros(100, 1200, 4, 2)
    for i in range(1200):
        logger.info(i)
        X = []
        Y = []
        X.append([np.random.uniform(low=-4.0, high=4.0, size=None), np.random.uniform(low=-3.0, high=3.0, size=None)])
        for k in range(100):
            Z = []
            Z.append(X[-1])
            a = hamitonsolution_function(Z[-1], 0, 1 / 2 - 1 / (2 * 3 ** (1 / 2)))
            Z.append(a[-1])
            a = hamitonsolution_function(Z[-1], 1 / 2 - 1 / (2 * 3 ** (1 / 2)), 1 / 2 + 1 / (2 * 3 ** (1 / 2)))
            Z.append(a[-1])
            a = hamitonsolution_function(Z[-1], 1 / 2 + 1 / (2 * 3 ** (1 / 2)), 1)
            Z.append(a[-1])
            Y.append(Z)
        Y = np.array(Y)
        yt = torch.from_numpy(Y)
        yt = yt.to(torch.float32)
        W[:, i, :, :] = yt
    file = open('./experiments/train4_100_gauss2.pkl', 'wb')
    pickle.dump(W, file)
    file.close()


def gen_testdata():
    W = torch.zeros(100, 1000, 4, 2)
    for i in range(1000):
        logger.info(i)
        X = []
        Y = []
        X.append([np.random.uniform(low=-4.0, high=4.0, size=None), np.random.uniform(low=-3.0, high=3.0, size=None)])
        for k in range(100):
            Z = []
            Z.append(X[-1])
            a = hamitonsolution_function(Z[-1], 0, 1 / 2 - 1 / (2 * 3 ** (1 / 2)))
            Z.append(a[-1])
            a = hamitonsolution_function(Z[-1], 1 / 2 - 1 / (2 * 3 ** (1 / 2)), 1 / 2 + 1 / (2 * 3 ** (1 / 2)))
            Z.append(a[-1])
            a = hamitonsolution_function(Z[-1], 1 / 2 + 1 / (2 * 3 ** (1 / 2)), 1)
            Z.append(a[-1])
            Y.append(Z)
        Y = np.array(Y)
        yt = torch.from_numpy(Y)
        yt = yt.to(torch.float32)
        W[:, i, :, :] = yt
    file = open('./experiments/test4_100_gauss3.pkl', 'wb')
    pickle.dump(W, file)
    file.close()


if __name__ == "__main__":
    # data
    args = get_args()
    utils.makedirs(args.save)
    logger = utils.get_logger(logpath=os.path.join(args.save, "EG5logs_gauss2_{}.txt".format(args.N)), filepath=os.path.abspath(__file__))
    h = 0.01
    # gen_data()
    with open('./experiments/train5_2_gauss3.pkl', 'rb') as file:
        W1 = pickle.load(file)
    W1 = W1[:100]
    logger.info(W1.shape)
    x = W1[0, :, 0, :]
    y = W1[0, :, -1, :]
    E = W1[:, :, -1, :].sum(axis=0) / W1.shape[0]
    W = torch.zeros(W1.shape[0], W1.shape[1], 4, W1.shape[3])
    W[:, :, 0, :] = W1[:, :, 0, :]
    W[:, :, 1, :] = W1[:, :, 2, :]
    W[:, :, 2, :] = W1[:, :, 4, :]
    W[:, :, -1, :] = W1[:, :, -1, :]
    logger.info(sample_elbo2(E, W))
    #
    x1 = np.linspace(-1, 1, 100)
    y1 = np.linspace(-1, 1, 100)
    x_, y_ = np.meshgrid(x1, y1, indexing='ij')
    xy = torch.zeros(100, 100, 2)
    xy[:, :, 0] = torch.from_numpy(x_)
    xy[:, :, 1] = torch.from_numpy(y_)
    xy = xy.to(torch.float32)
    z0_true = -4 * np.cos(y_) + x_ ** 2 / 2
    z0_true = torch.from_numpy(z0_true)
    z0_true = z0_true.view(100, 100, 1)
    z1_true = 0.2 * np.sin(y_) + y_ / 2
    z1_true = torch.from_numpy(z1_true)
    z1_true = z1_true.view(100, 100, 1)
    #train
    net1 = Linear_BBB2(2, 1, h)
    optimizer = optim.Adam(net1.parameters(), lr=args.learning_rate)
    epochs = args.epochs
    Loss = nn.MSELoss()
    L = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        optimizer.zero_grad()
        loss = net1.sample_elbo2(E, W)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            logger.info('epoch: {}/{}'.format(epoch + 1, epochs))
            h0, h1 = net1(x)
            c0 = torch.nn.functional.mse_loss(f(x), h0, reduction='mean')
            c1 = torch.nn.functional.mse_loss(g(x), h1, reduction='mean')
            c2 = torch.nn.functional.mse_loss(g(x), -h1, reduction='mean')
            logger.info('Loss1:{}, test loss :{},{},{}'.format(loss.item(), c0.item(), c1.item(), c2.item()))
            L.append([loss.item(), c0.item(), c1.item(), c2.item()])
            # h0, h1 = net1(x)
            # a0 = h_0(x).view(x.shape[0], 1) - h0
            # b0 = torch.abs(a0 - a0.sum() / a0.shape[0]).sum() / a0.shape[0]
            # a1 = h_1(x).view(x.shape[0], 1) - h1
            # b1 = torch.abs(a1 - a1.sum() / a1.shape[0]).sum() / a1.shape[0]
            # a2 = -h_1(x).view(x.shape[0], 1) - h1
            # b2 = torch.abs(a2 - a2.sum() / a2.shape[0]).sum() / a2.shape[0]
            # z0_, z1_ = net1(xy)
            # l0 = torch.abs(z0_true - z0_ - (z0_true - z0_).sum() / 10000).sum() / 10000
            # l1 = torch.abs(z1_true - z1_ - (z1_true - z1_).sum() / 10000).sum() / 10000
            # l2 = torch.abs(-z1_true - z1_ - (-z1_true - z1_).sum() / 10000).sum() / 10000
            # logger.info('Loss1:{}, test loss :{},{},{},{},{},{}'.format(loss.item(), b0.item(), b1.item(), b2.item(), l0, l1, l2))
            # L.append([loss.item(), b0.item(), b1.item(), b2.item()])
    logger.info('Finished Training')
    file = open('./experiments/4_gauss2_{}.pkl'.format(args.N), 'wb')
    pickle.dump(L, file)
    file.close()
    torch.save(net1, './experiments/model_EG5_1_gauss2_{}.pkl'.format(args.N))

    J = torch.zeros(2, 2)
    J[0, 1] = 1.0
    J[1, 0] = -1.0

    z0_, z1_ = net1(xy)
    z0_origin, z1_origin = net1(torch.tensor([0., 0.]))
    l0 = torch.abs(z0_true - z0_ - (z0_true - z0_).sum() / 10000).sum() / 10000
    l1 = torch.abs(z1_true - z1_ - (z1_true - z1_).sum() / 10000).sum() / 10000
    logger.info(l0, l1)
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    sub = fig.add_subplot(111, projection='3d')
    surf0 = sub.plot_surface(x_, y_, z0_.view(100, 100).detach().numpy() - z0_origin.detach().numpy(), color='blue', label='$H_{net0}$')
    surf3 = sub.plot_surface(x_, y_, z0_true.view(100, 100).numpy(), color='yellow', label='$H_0$')
    sub.set_xlabel(r"x axis")
    sub.set_ylabel(r"y axis")
    sub.set_zlabel(r"z axis")
    plt.savefig("./experiments/h0_EG4_gauss2 Prediction.png")
    plt.show()

    fig = plt.figure(figsize=(10, 10), facecolor='white')
    sub = fig.add_subplot(111, projection='3d')
    surp1 = sub.plot_surface(x_, y_, z1_.view(100, 100).detach().numpy() - z1_origin.detach().numpy(), color='green', label='$H_{net1}$')
    surp2 = sub.plot_surface(x_, y_, z1_true.view(100, 100).numpy(), color='red', label='$H_1$')
    sub.set_xlabel(r"x axis")
    sub.set_ylabel(r"y axis")
    sub.set_zlabel(r"z axis")
    plt.savefig("./experiments/h1_EG4_gauss2 Prediction.png")
    plt.show()

    fig = plt.figure(figsize=(10, 10), facecolor='white')
    sub = fig.add_subplot(111, projection='3d')
    surp3 = sub.plot_surface(x_, y_, -z1_.view(100, 100).detach().numpy() + z1_origin.detach().numpy(), color='green',
                             label='$H_{net1}$')
    surp4 = sub.plot_surface(x_, y_, z1_true.view(100, 100).numpy(), color='red', label='$H_{1}$')
    sub.set_xlabel(r"x axis")
    sub.set_ylabel(r"y axis")
    sub.set_zlabel(r"z axis")
    plt.savefig("./experiments/h2_EG4_gauss2 Prediction_{}.png".format(args.h))
    plt.show()

    def f(X):
        X0_ = X0
        X0_ = torch.from_numpy(X0_)
        X0_ = X0_.to(torch.float32)
        X = torch.from_numpy(X)
        X = X.to(torch.float32)
        e = (X + X0_) / 2
        e.requires_grad_()
        h0, h1 = net1(e)
        grad0 = autograd.grad(outputs=h0, inputs=e, grad_outputs=torch.ones_like(h0))[0]
        grad1 = autograd.grad(outputs=h1, inputs=e, grad_outputs=torch.ones_like(h1))[0]
        y = X - X0_ - grad0 @ J * h - grad1 @ J * r * np.sqrt(h)
        return y.detach().numpy()

    def g(X):
        P = X[0]
        Q = X[1]
        p = X0[0]
        q = X0[1]
        return [P - p + (Q + q) / 2 * h + (Q + q) / 2 * np.sqrt(h) * r,
                Q - q - (P + p) / 2 * h - (P + p) / 2 * np.sqrt(h) * r]

    def g1(X):
        P = X[0]
        Q = X[1]
        p = X0[0]
        q = X0[1]
        return [P - p + (Q + q) / 2 * h,
                Q - q - (P + p) / 2 * h - np.sqrt(h) * r]
    YY = []
    YY_t = []
    for j in range(100):
        X0 = np.array([0, 0.5])
        r = np.random.normal(loc=0.0, scale=1.0, size=None)
        result = fsolve(f, X0)
        result_t = fsolve(g1, X0)
        # print(f(X0))
        Y = [result]
        Y_t = [result_t]
        for i in range(1000):
            X0_ = np.array(result)
            X0_t = np.array(result_t)
            r = np.random.normal(loc=0.0, scale=1.0, size=None)
            # plt.scatter(X0_[0], X0_[1], s=2, c='g')
            # plt.scatter(X0_t[0], X0_t[1], s=2, c='r')
            X0 = X0_
            result = fsolve(f, X0_)
            X0 = X0_t
            result_t = fsolve(g1, X0_t)
            Y.append(result)
            Y_t.append(result_t)
        YY.append(Y)
        YY_t.append(Y_t)
    YY = np.array(YY)
    YY_t = np.array(YY_t)
    DT = np.linspace(0, YY[0].shape[0] * 0.01, YY[0].shape[0])
    fig, axs = plt.subplots(1, 1, figsize=(9, 6), linewidth=3)
    axs.plot(DT, YY[0, :, 0], c='g', label='$p_t$ prediction')
    axs.plot(DT, YY_t[0, :, 0], c='r', label='ground truth')
    axs.set_xlabel('t', fontsize=12)
    axs.set_ylabel('$p_t$', fontsize=12)
    plt.savefig("./experiments/1_EG4_gauss2 Prediction.png")
    plt.show()
    fig2, axs = plt.subplots(1, 1, figsize=(9, 6), linewidth=3)
    axs.plot(DT, YY[0, :, 1], c='g', label='$q_t$ prediction')
    axs.plot(DT, YY_t[0, :, 1], c='r', label='ground truth')
    axs.set_xlabel('t', fontsize=12)
    axs.set_ylabel('$q_t$', fontsize=12)
    plt.savefig("./experiments/2_EG4_gauss2 Prediction.png")
    plt.show()
    fig3, axs = plt.subplots(1, 1, figsize=(9, 6), linewidth=3)
    axs.plot(DT, (YY[:, :, 0] ** 2 + YY[:, :, 1] ** 2).sum(axis=0) / YY.shape[0], c='blue')
    axs.plot(DT, (YY_t[:, :, 0] ** 2 + YY_t[:, :, 1] ** 2).sum(axis=0) / YY_t.shape[0], c='red')
    plt.savefig("./experiments/3_EG4_gauss2 Prediction.png")
    plt.show()


