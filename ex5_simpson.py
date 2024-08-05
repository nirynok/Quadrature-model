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

J = torch.zeros(2, 2)
J[0, 1] = 1.0
J[1, 0] = -1.0


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
    parser.add_argument("--save", type=str, default="experiments")
    parser.add_argument("--data", type=str, default="_num_25_1")
    parser.add_argument("--N", type=str, default="1")
    parser.add_argument("--T", type=float, default="1")
    parser.set_defaults(feature=True)
    return parser.parse_args()


class Linear_BBB2(nn.Module):
    """
        Layer of our BNN.
    """
    def __init__(self, input_features, output_features, T):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features
        self.T = T

        # initialize mu and rho parameters for the weights of the layer

        self.f_w_1 = nn.Parameter((torch.randn([input_features, 64]) * math.sqrt(1. / 128)).requires_grad_(True))
        self.f_w_2 = nn.Parameter((torch.randn([64, output_features]) * math.sqrt(1. / 64)).requires_grad_(True))
        self.f_b_1 = nn.Parameter((torch.randn([64]) * math.sqrt(1. / 64)).requires_grad_(True))
        self.f_b_2 = nn.Parameter((torch.randn([1]) * math.sqrt(1. / 1)).requires_grad_(True))

        self.g_w_1 = nn.Parameter((torch.rand([input_features, 32]) * math.sqrt(1. / 64)).requires_grad_(True))
        self.g_w_2 = nn.Parameter((torch.rand([32, output_features]) * math.sqrt(1. / 32)).requires_grad_(True))
        self.g_b_1 = nn.Parameter((torch.rand([32]) * math.sqrt(1. / 32)).requires_grad_(True))
        self.g_b_2 = nn.Parameter((torch.rand([1]) * math.sqrt(1. / 1)).requires_grad_(True))

    def forward(self, input):
        H0 = torch.tanh(input @ self.f_w_1 + self.f_b_1) @ self.f_w_2 + self.f_b_2
        H1 = torch.tanh(input @ self.g_w_1 + self.g_b_1) @ self.g_w_2 + self.g_b_2
        # H0 = H1
        # H0 = torch.tanh(input @ self.g_w_1 + self.g_b_1) @ self.g_w_2
        # H0 = (input[0]**2 + input[1]**2)/2
        # H1 = H0
        return H0, H1

    def H(self, x):
        H0 = (x[:, :, :, 0] ** 2 + x[:, :, :, 0] * x[:, :, :, 1] + x[:, :, :, 1]) / (
                1 * (1 + (x[:, :, :, 0] + x[:, :, :, 1]) ** 2))
        H1 = 0.1 * torch.log(1 + x[:, :, :, 0] ** 2 + x[:, :, :, 1] ** 2)
        return H0.view(x.shape[0], x.shape[1], x.shape[2], 1), H1.view(x.shape[0], x.shape[1], x.shape[2], 1)

    def diff(self, x):
        J = torch.zeros(2, 2)
        J[0, 1] = 1.0
        J[1, 0] = -1.0
        x.requires_grad_()
        H0, H1 = self(x)
        # H0, _ = H(x)
        grad0 = autograd.grad(outputs=H0, inputs=x, grad_outputs=torch.ones_like(H0), create_graph=True,
                              retain_graph=True)[0]
        grad00 = grad0 @ J
        grad1 = autograd.grad(outputs=H1, inputs=x, grad_outputs=torch.ones_like(H1), create_graph=True,
                              retain_graph=True)[0]
        grad11 = grad1 @ J
        grad111 = autograd.grad(outputs=grad1, inputs=x, grad_outputs=grad11, retain_graph=True)[0] @ J
        return grad00, grad111, grad11

    def cov1(self, W, W00, W111):
        W00_s = W00[:, :, 0, :] / 6 + W00[:, :, 1, :] / 3 * 2 + W00[:, :, 2, :] / 6
        W111_s = W111[:, :, 0, :] / 6 + W111[:, :, 1, :] / 3 * 2 + W111[:, :, 2, :] / 6
        Y = W[:, :, -1, :] - W[:, :, 0, :] - W00_s*self.T - W111_s / 2*self.T
        Y = Y.view(W.shape[0], W.shape[1], 1, W.shape[3])
        Y_T = torch.transpose(Y, dim0=2, dim1=3)
        Y_cov = (Y_T @ Y).sum(dim=0) / W.shape[0]
        return Y_cov

    def cov2(self, W):
        E1 = W[:, :, 0, :].view(W.shape[0], W.shape[1], 1, W.shape[3])
        E1_T = torch.transpose(E1, dim0=2, dim1=3)
        E2 = W[:, :, 1, :].view(W.shape[0], W.shape[1], 1, W.shape[3])
        E2_T = torch.transpose(E2, dim0=2, dim1=3)
        E3 = W[:, :, 2, :].view(W.shape[0], W.shape[1], 1, W.shape[3])
        E3_T = torch.transpose(E3, dim0=2, dim1=3)
        E_cov = (E1_T @ E1 / 6 + E2_T @ E2 / 3 * 2 + E3_T @ E3 / 6).sum(dim=0) / W.shape[0]
        return E_cov*self.T

    def sample_elbo2(self, E, W):
        W00, W111, W11 = self.diff(W)
        # 期望
        E00 = (W00[:, :, 0, :] / 6 + W00[:, :, 1, :] / 3 * 2 + W00[:, :, 2, :] / 6).sum(axis=0) / W.shape[0]*self.T
        E111 = (W111[:, :, 0, :] / 6 + W111[:, :, 1, :] / 3 * 2 + W111[:, :, 2, :] / 6).sum(axis=0) / W.shape[0]*self.T
        LOSS = torch.nn.functional.mse_loss(E, W[0, :, 0, :] + E00 + E111 / 2,
                                            reduction='mean') + torch.nn.functional.mse_loss(
            self.cov1(W, W00, W111), self.cov2(W11), reduction='mean')
        return LOSS


def toy_function(x):
    return -x**4 + 3*x**2 + 1


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
    # a = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 2, 2)
    # for i in range(2):
    #     a[:, :, :, i] = autograd.grad(outputs=grad1[:, :, :, i], inputs=x, grad_outputs=torch.ones_like(grad1[:, :, :, i]), create_graph=True,
    #                       retain_graph=True)[0]
    #     a[:, :, :, i] = \
    #     autograd.grad(outputs=grad1[:, :, :, i], inputs=x, grad_outputs=torch.ones_like(grad1[:, :, :, i]),
    #                   create_graph=True,
    #                   retain_graph=True)[0]

    grad11 = grad1 @ J
    grad111 = autograd.grad(outputs=grad1, inputs=x, grad_outputs=grad11, retain_graph=True)[0] @ J
    return grad00, grad111, grad11


def cov1(W, W00, W111):
    W00_s = W00[:, :, 0, :]/6 + W00[:, :, 1, :]/3*2 + W00[:, :, 2, :]/6
    W111_s = W111[:, :, 0, :]/6 + W111[:, :, 1, :]/3*2 + W111[:, :, 2, :]/6
    Y = W[:, :, -1, :] - W[:, :, 0, :] - W00_s*args.T - W111_s/2*args.T
    Y = Y.view(W.shape[0], W.shape[1], 1, W.shape[3])
    Y_T = torch.transpose(Y, dim0=2, dim1=3)
    Y_cov = (Y_T@Y).sum(dim=0)/W.shape[0]
    return Y_cov

def cov2(W):
    E1 = W[:, :, 0, :].view(W.shape[0], W.shape[1], 1, W.shape[3])
    E1_T = torch.transpose(E1, dim0=2, dim1=3)
    E2 = W[:, :, 1, :].view(W.shape[0], W.shape[1], 1, W.shape[3])
    E2_T = torch.transpose(E2, dim0=2, dim1=3)
    E3 = W[:, :, 2, :].view(W.shape[0], W.shape[1], 1, W.shape[3])
    E3_T = torch.transpose(E3, dim0=2, dim1=3)
    E_cov = (E1_T @ E1/6 + E2_T @ E2/3*2 + E3_T @ E3/6).sum(dim=0) / W.shape[0]
    return E_cov*args.T

def sample_elbo2(E, W):
    W00, W111, W11 = diff(W)
    #期望
    E00 = (W00[:, :, 0, :] / 6 + W00[:, :, 1, :] / 3 * 2 + W00[:, :, 2, :] / 6).sum(axis=0) / W.shape[0]*args.T
    E111 = (W111[:, :, 0, :] / 6 + W111[:, :, 1, :] / 3 * 2 + W111[:, :, 2, :] / 6).sum(axis=0) / W.shape[0]*args.T
    LOSS1 = torch.nn.functional.mse_loss(E, W[0, :, 0, :] + E00 + E111 / 2, reduction='mean')
    LOSS2 = torch.nn.functional.mse_loss(cov1(W, W00, W111), cov2(W11), reduction='mean')
    return LOSS1, LOSS2, torch.nn.functional.mse_loss(E, torch.zeros_like(E), reduction='mean')


def H(x):
    # Uq = 0.1 * (x[:, :, :, 1] * (x[:, :, :, 1] - 2)) ** 2 + 0.008 * x[:, :, :, 1] ** 3
    # dUq = 0.2 * (x[:, :, :, 1] * (x[:, :, :, 1] - 2)) * (2 * x[:, :, :, 1] - 2) + 0.024 * x[:, :, :, 1] ** 2
    # H0 = x[:, :, :, 0] ** 2 / (2 * (1 + dUq ** 2)) + Uq
    H0 = (x[:, :, :, 0] ** 2 + x[:, :, :, 0] * x[:, :, :, 1] + x[:, :, :, 1]) / (1 * (1 + (x[:, :, :, 0] + x[:, :, :, 1]) ** 2))
    H1 = 0.1*torch.log(1+x[:, :, :, 0]**2+x[:, :, :, 1]**2)
    return H0.view(x.shape[0], x.shape[1], x.shape[2], 1), H1.view(x.shape[0], x.shape[1], x.shape[2], 1)


def f(x):
    # Uq = 0.1*(x[1]*(x[1]-2))**2 + 0.008*x[1]**3
    # dUq = 0.2*(x[1]*(x[1]-2))*(2*x[1]-2) + 0.024*x[1]**2
    # H = x[0]**2/(2*(1+dUq**2)) + Uq
    y = (x[0] ** 2 + x[0] * x[1] + x[1]) / (1 * (1 + (x[0] + x[1]) ** 2))
    return y


def g(x):
    return 0.1*torch.log(1+x[0]**2+x[1]**2)


def df(x):
    x.requires_grad_()
    grad = autograd.grad(outputs=f(x), inputs=x, grad_outputs=torch.ones_like(f(x)), create_graph=True,
                          retain_graph=True)[0]@J
    # Uq = 0.1 * (x[1] * (x[1] - 2)) ** 2 + 0.008 * x[1] ** 3
    # dUq = 0.2 * x[1] * (x[1] - 2) * (2 * x[1] - 2) + 0.024 * x[1] ** 2
    # ddUq = 0.2*((x[1] - 2) * (2 * x[1] - 2)+2*x[1] * (x[1] - 2)+x[1] * (2*x[1] - 2))+0.048*x[1]
    # d = torch.tensor([-((-x[0]**2/(2*(1+dUq**2))**2)*4*dUq*ddUq+dUq), 2*x[0]/(2*(1+dUq**2))])
    return grad.detach()


def dg(x):
    # x.requires_grad_()
    # grad = autograd.grad(outputs=g(x), inputs=x, grad_outputs=torch.ones_like(g(x)), create_graph=True,
    #                       retain_graph=True)[0]@J
    d = torch.tensor(
        [-(0.1*2*x[1]/(1+x[0]**2+x[1]**2)), 0.1*2*x[0]/(1+x[0]**2+x[1]**2)])
    return d


def hamitonsolution_function(x, t0, t1):
    N = np.round((torch.tensor(t1-t0)/(1e-2)).max().item())
    h = (t1-t0)/N
    m = np.sqrt(h)
    # X = torch.zeros(int(N+1), 2)
    # X[0] = x
    for i in range(int(N)):
        temp = np.random.normal(loc=0.0, scale=1.0, size=None)
        x0 = x
        x_temp = x
        x = x0 + df((x0+x_temp)/2)*h+dg((x0+x_temp)/2)*m*temp
        while (x[0]-x_temp[0])**2 > 1e-12 and (x[1]-x_temp[1])**2 > 1e-12:
            x_temp = x
            x = x0 + df((x0 + x_temp) / 2) * h + dg((x0 + x_temp) / 2) * m * temp
            # print(j)
        # X[i+1] = x
    return x
    # return np.hstack((p, q))


def h_0(x):
    # Uq = 0.1 * (x[:, 1] * (x[:, 1] - 2)) ** 2 + 0.008 * x[:, 1] ** 3
    # dUq = 0.2 * (x[:, 1] * (x[:, 1] - 2)) * (2 * x[:, 1] - 2) + 0.024 * x[:, 1] ** 2
    # y = x[:, 0] ** 2 / (2 * (1 + dUq ** 2)) + Uq
    y = (x[:, 0] ** 2 + x[:, 0] * x[:, 1] + x[:, 1]) / (1 * (1 + (x[:, 0] + x[:, 1]) ** 2))
    return y


def h_1(x):
    y = 0.1 * torch.log(1 + x[:, 0] ** 2 + x[:, 1] ** 2)
    return y


def Z0_true(x0, x1):
    # Uq = 0.1 * (x1 * (x1 - 2)) ** 2 + 0.008 * x1 ** 3
    # dUq = 0.2 * (x1 * (x1 - 2)) * (2 * x1 - 2) + 0.024 * x1 ** 2
    # y = x0 ** 2 / (2 * (1 + dUq ** 2)) + Uq
    y = (x0 ** 2 + x0 * x1 + x1) / (1 * (1 + (x0 + x1) ** 2))
    return y


def Z1_true(x0, x1):
    y = 0.1 * np.log(1 + x0 ** 2 + x1 ** 2)
    return y


def gen_data():
    W = torch.zeros(500, 750, 7, 2)
    for i in range(750):
        logger.info(i)
        X = []
        X.append(torch.tensor([np.random.uniform(low=-4.0, high=4.0, size=None), np.random.uniform(low=-1.0, high=4.0, size=None)]))
        for k in range(500):
            W[k, i, 0] = X[-1]
            a = hamitonsolution_function(W[k, i, 0], 0, (1 / 2 - 15 ** (1 / 2) / 10)*args.T)
            W[k, i, 1] = a
            a = hamitonsolution_function(W[k, i, 1], (1 / 2 - 15 ** (1 / 2) / 10)*args.T, (1 / 2 - 1 / (2 * 3 ** (1 / 2)))*args.T)
            W[k, i, 2] = a
            a = hamitonsolution_function(W[k, i, 2], (1 / 2 - 1 / (2 * 3 ** (1 / 2)))*args.T, 1 / 2*args.T)
            W[k, i, 3] = a
            a = hamitonsolution_function(W[k, i, 3], 1 / 2*args.T, (1 / 2 + 1 / (2 * 3 ** (1 / 2)))*args.T)
            W[k, i, 4] = a
            a = hamitonsolution_function(W[k, i, 4], (1 / 2 + 1 / (2 * 3 ** (1 / 2)))*args.T, (1 / 2 + 15 ** (1 / 2) / 10)*args.T)
            W[k, i, 5] = a
            a = hamitonsolution_function(W[k, i, 5], (1 / 2 + 15 ** (1 / 2) / 10)*args.T, 1*args.T)
            W[k, i, 6] = a
    file = open('./experiments/train_non_gauss3_{}_{}.pkl'.format(args.N, args.T), 'wb')
    pickle.dump(W, file)
    file.close()


if __name__ == "__main__":
    # data
    args = get_args()
    utils.makedirs(args.save)
    logger = utils.get_logger(logpath=os.path.join(args.save, "EGnon_logs_simpson_{}_{}.txt".format(args.N, args.T)), filepath=os.path.abspath(__file__))
    # gen_data()
    with open('./experiments/train_non_gauss3_{}_{}.pkl'.format(args.N, args.T), 'rb') as file:
        W1 = pickle.load(file)
    logger.info(W1.shape)
    x = W1[0, :, 0, :]
    y = W1[0, :, -1, :]
    E = W1[:, :, -1, :].sum(axis=0) / W1.shape[0]
    W = torch.zeros(W1.shape[0], W1.shape[1], 3, W1.shape[3])
    W[:, :, 0, :] = W1[:, :, 0, :]
    W[:, :, 1, :] = W1[:, :, 3, :]
    W[:, :, -1, :] = W1[:, :, -1, :]
    logger.info(sample_elbo2(E, W))
    x1 = np.linspace(-1, 1, 100)
    y1 = np.linspace(-1, 1, 100)
    x_, y_ = np.meshgrid(x1, y1, indexing='ij')
    xy = torch.zeros(100, 100, 2)
    xy[:, :, 0] = torch.from_numpy(x_)
    xy[:, :, 1] = torch.from_numpy(y_)
    xy = xy.to(torch.float32)
    z0_true = Z0_true(x_, y_)
    z0_true = torch.from_numpy(z0_true)
    z0_true = z0_true.view(100, 100, 1)
    z1_true = Z1_true(x_, y_)
    z1_true = torch.from_numpy(z1_true)
    z1_true = z1_true.view(100, 100, 1)
    #train
    net1 = Linear_BBB2(2, 1, args.T)
    # net1 = torch.load('./experiments/model_EG4_2_gauss3_2.pkl', map_location=torch.device('cpu'))
    # net1.f_w_1.requires_grad_(False)
    # net1.f_w_2.requires_grad_(False)
    # net1.f_b_1.requires_grad_(False)
    # net1.g_w_1.requires_grad_(True)
    # net1.g_w_2.requires_grad_(True)
    # net1.g_b_1.requires_grad_(True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net1.parameters()), lr=args.learning_rate)
    epochs = args.epochs
    Loss = nn.MSELoss()
    L = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        optimizer.zero_grad()
        loss = net1.sample_elbo2(E, W)
        loss.backward()
        optimizer.step()
        if loss.item() < 0.1:
            optimizer.param_groups[0]['lr'] = 0.001
        if epoch % 10 == 0:
            logger.info('epoch: {}/{}'.format(epoch + 1, epochs))
            h0, h1 = net1(x)
            a0 = h_0(x).view(x.shape[0], 1) - h0
            b0 = torch.abs(a0 - a0.sum() / a0.shape[0]).sum() / a0.shape[0]
            a1 = h_1(x).view(x.shape[0], 1) - h1
            b1 = torch.abs(a1 - a1.sum() / a1.shape[0]).sum() / a1.shape[0]
            a2 = -h_1(x).view(x.shape[0], 1) - h1
            b2 = torch.abs(a2 - a2.sum() / a2.shape[0]).sum() / a2.shape[0]
            z0_, z1_ = net1(xy)
            l0 = torch.abs(z0_true - z0_ - (z0_true - z0_).sum() / 10000).sum() / 10000
            l1 = torch.abs(z1_true - z1_ - (z1_true - z1_).sum() / 10000).sum() / 10000
            l2 = torch.abs(-z1_true - z1_ - (-z1_true - z1_).sum() / 10000).sum() / 10000
            logger.info('Loss1:{}, test loss :{},{},{},{},{},{}'.format(loss.item(), b0.item(), b1.item(), b2.item(), l0, l1, l2))
            L.append([loss.item(), b0.item(), b1.item(), b2.item()])
    logger.info('Finished Training')
    file = open('./experiments/4_simpson_{}_{}.pkl'.format(args.N, args.T), 'wb')
    pickle.dump(L, file)
    file.close()
    torch.save(net1, './experiments/model_EG_non_simpson_{}_{}.pkl'.format(args.N, args.T))
    # net1 = torch.load('./experiments/model_EG4_gauss3_1.pkl', map_location=torch.device('cpu'))
    z0_, z1_ = net1(xy)
    l0 = torch.abs(z0_true - z0_ - (z0_true - z0_).sum() / 10000).sum() / 10000
    l1 = torch.abs(z1_true - z1_ - (z1_true - z1_).sum() / 10000).sum() / 10000
    l2 = torch.abs(-z1_true - z1_ - (-z1_true - z1_).sum() / 10000).sum() / 10000
    print(l0, l1, l2)
    # net1 = torch.load('./experiments/model_EG4_simpson_1.pkl', map_location=torch.device('cpu'))
    # z0_, z1_ = net1(xy)
    # l0 = torch.abs(z0_true - z0_ - (z0_true - z0_).sum() / 10000).sum() / 10000
    # l1 = torch.abs(z1_true - z1_ - (z1_true - z1_).sum() / 10000).sum() / 10000
    # l2 = torch.abs(-z1_true - z1_ - (-z1_true - z1_).sum() / 10000).sum() / 10000
    # print(l0, l1, l2)



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
    plt.savefig("./experiments/h0_EG4_gauss3 Prediction.png")
    plt.show()

    fig = plt.figure(figsize=(10, 10), facecolor='white')
    sub = fig.add_subplot(111, projection='3d')
    surp1 = sub.plot_surface(x_, y_, z1_.view(100, 100).detach().numpy() - z1_origin.detach().numpy(), color='green', label='$H_{net1}$')
    surp2 = sub.plot_surface(x_, y_, z1_true.view(100, 100).numpy(), color='red', label='$H_1$')
    sub.set_xlabel(r"x axis")
    sub.set_ylabel(r"y axis")
    sub.set_zlabel(r"z axis")
    plt.savefig("./experiments/h1_EG4_gauss3 Prediction.png")
    plt.show()

    fig = plt.figure(figsize=(10, 10), facecolor='white')
    sub = fig.add_subplot(111, projection='3d')
    surp3 = sub.plot_surface(x_, y_, -z1_.view(100, 100).detach().numpy() + z1_origin.detach().numpy(), color='green',
                             label='$H_{net1}$')
    surp4 = sub.plot_surface(x_, y_, z1_true.view(100, 100).numpy(), color='red', label='$H_{1}$')
    sub.set_xlabel(r"x axis")
    sub.set_ylabel(r"y axis")
    sub.set_zlabel(r"z axis")
    plt.savefig("./experiments/h2_EG4_gauss3 Prediction.png")
    plt.show()
    h = 0.01

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
    plt.savefig("./experiments/1_EG4_gauss3 Prediction.png")
    plt.show()
    fig2, axs = plt.subplots(1, 1, figsize=(9, 6), linewidth=3)
    axs.plot(DT, YY[0, :, 1], c='g', label='$q_t$ prediction')
    axs.plot(DT, YY_t[0, :, 1], c='r', label='ground truth')
    axs.set_xlabel('t', fontsize=12)
    axs.set_ylabel('$q_t$', fontsize=12)
    plt.savefig("./experiments/2_EG4_gauss3 Prediction.png")
    plt.show()
    fig3, axs = plt.subplots(1, 1, figsize=(9, 6), linewidth=3)
    axs.plot(DT, (YY[:, :, 0] ** 2 + YY[:, :, 1] ** 2).sum(axis=0) / YY.shape[0], c='blue')
    axs.plot(DT, (YY_t[:, :, 0] ** 2 + YY_t[:, :, 1] ** 2).sum(axis=0) / YY_t.shape[0], c='red')
    plt.savefig("./experiments/3_EG4_gauss3 Prediction.png")
    plt.show()


