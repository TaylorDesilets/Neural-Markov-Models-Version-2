import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time

from functions import (
    simulate_data,
    introduce_interval_censoring,
    create_time_matrix,
    right_censoring,
    loss_fn,
)

from functions_real_data import (
    predict,
    brier_score_real_world
)

###########################################################
####### Models ############################################
###########################################################

class TwoHeadModel(nn.Module):
    def __init__(self, input_dim, hidden=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head12 = nn.Linear(hidden, 1)
        self.head13 = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.net(x)
        return self.head12(h).squeeze(), self.head13(h).squeeze()


class LinearTwoHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.head12 = nn.Linear(input_dim, 1)
        self.head13 = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.head12(x).squeeze(), self.head13(x).squeeze()


class ConstantTwoHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.p12 = nn.Parameter(torch.zeros(1))
        self.p13 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n = x.shape[0]
        return self.p12.repeat(n), self.p13.repeat(n)


###########################################################
####### Training ##########################################
###########################################################
def run_simulation():
    def train_model(model, X, T, t12, t13, epochs, lr):
        opt = optim.Adam(model.parameters(), lr=lr)
        loss_hist = []

        start = time.time()

        for epoch in range(epochs):
            opt.zero_grad()
            loss = loss_fn(model, X, T, t12, t13)  # your translated loss_fn
            loss.backward()
            opt.step()

            loss_hist.append(loss.item())

        elapsed = time.time() - start
        return model, loss_hist, elapsed


    ###########################################################
    ####### Simulation ########################################
    ###########################################################

    Nid = 3000
    Nid_val = 1000
    Nruns = 100
    Sim_covs = 2

    # storage
    results = {
        "time_const": [], "time_lin": [], "time_NN": [],
        "loss_const": [], "loss_lin": [], "loss_NN": [],
        "loss_const_val": [], "loss_lin_val": [], "loss_NN_val": [],
        "BS_const": [], "BS_lin": [], "BS_NN": [],
        "MSE11_const_val": [], "MSE12_const_val": [], "MSE13_const_val": [],
        "MSE11_lin_val": [], "MSE12_lin_val": [], "MSE13_lin_val": [],
        "MSE11_NN_val": [], "MSE12_NN_val": [], "MSE13_NN_val": []
    }

    for run in range(Nruns):

        ######################## TRAIN DATA ########################
        TIME, t12, t13, state, X, lam12, lam13 = simulate_data(Nid, Sim_covs)
        lam11 = 1 - (lam12 + lam13)

        TIME = introduce_interval_censoring(TIME, t12, t13, 1)
        TIME = create_time_matrix(TIME)
        TIME, t12, t13 = right_censoring(TIME, t12, t13, 0.5)

        ######################## VALIDATION ########################
        TIME_v, t12_v, t13_v, state_v, X_v, lam12_v, lam13_v = simulate_data(Nid_val, Sim_covs)
        lam11_v = 1 - (lam12_v + lam13_v)

        TIME_v = introduce_interval_censoring(TIME_v, t12_v, t13_v, 1)
        TIME_v = create_time_matrix(TIME_v)
        TIME_v, t12_v, t13_v = right_censoring(TIME_v, t12_v, t13_v, 0.5)

        # torch tensors
        X = torch.tensor(X.T, dtype=torch.float32)
        X_v = torch.tensor(X_v.T, dtype=torch.float32)

        ######################## NN MODEL ########################
        nn_model = TwoHeadModel(2)

        nn_model, _, t1 = train_model(nn_model, X, TIME, t12, t13, 200, 1e-3)
        nn_model, nn_lossvec, t2 = train_model(nn_model, X, TIME, t12, t13, 300, 1e-4)

        nn_loss = nn_lossvec[-1]
        nn_loss_val = loss_fn(nn_model, X_v, TIME_v, t12_v, t13_v).item()

        pred11, pred12, pred13 = predict(nn_model, X_v)

        mse12 = np.mean(np.abs(lam12_v - pred12))
        mse13 = np.mean(np.abs(lam13_v - pred13))
        mse11 = np.mean(np.abs(lam11_v - pred11))

        BS_nn = brier_score_real_world(
            TIME_v, pred12, pred13, t12_v, t13_v, [5]
        )[0][0]

        ######################## LINEAR ########################
        lin_model = LinearTwoHead(2)

        lin_model, lin_lossvec, t_lin = train_model(
            lin_model, X, TIME, t12, t13, 500, 1e-2
        )

        lin_loss = lin_lossvec[-1]
        lin_loss_val = loss_fn(lin_model, X_v, TIME_v, t12_v, t13_v).item()

        pred11, pred12, pred13 = predict(lin_model, X_v)

        mse12_lin = np.mean(np.abs(lam12_v - pred12))
        mse13_lin = np.mean(np.abs(lam13_v - pred13))
        mse11_lin = np.mean(np.abs(lam11_v - pred11))

        BS_lin = brier_score_real_world(
            TIME_v, pred12, pred13, t12_v, t13_v, [5]
        )[0][0]

        ######################## CONSTANT ########################
        const_model = ConstantTwoHead()

        X_const = torch.ones((Nid, 1))
        X_const_v = torch.ones((Nid_val, 1))

        const_model, const_lossvec, t_const = train_model(
            const_model, X_const, TIME, t12, t13, 500, 1e-2
        )

        const_loss = const_lossvec[-1]
        const_loss_val = loss_fn(const_model, X_const_v, TIME_v, t12_v, t13_v).item()

        pred11, pred12, pred13 = predict(const_model, X_const_v)

        mse12_const = np.mean(np.abs(lam12_v - pred12))
        mse13_const = np.mean(np.abs(lam13_v - pred13))
        mse11_const = np.mean(np.abs(lam11_v - pred11))

        BS_const = brier_score_real_world(
            TIME_v, pred12, pred13, t12_v, t13_v, [5]
        )[0][0]

        ######################## STORE ########################
        results["time_NN"].append(t1 + t2)
        results["time_lin"].append(t_lin)
        results["time_const"].append(t_const)

        results["loss_NN"].append(nn_loss)
        results["loss_lin"].append(lin_loss)
        results["loss_const"].append(const_loss)

        results["loss_NN_val"].append(nn_loss_val)
        results["loss_lin_val"].append(lin_loss_val)
        results["loss_const_val"].append(const_loss_val)

        results["BS_NN"].append(BS_nn)
        results["BS_lin"].append(BS_lin)
        results["BS_const"].append(BS_const)

        results["MSE11_NN_val"].append(mse11)
        results["MSE12_NN_val"].append(mse12)
        results["MSE13_NN_val"].append(mse13)

        results["MSE11_lin_val"].append(mse11_lin)
        results["MSE12_lin_val"].append(mse12_lin)
        results["MSE13_lin_val"].append(mse13_lin)

        results["MSE11_const_val"].append(mse11_const)
        results["MSE12_const_val"].append(mse12_const)
        results["MSE13_const_val"].append(mse13_const)

    ###########################################################
    ####### Save Results ######################################
    ###########################################################

    df = pd.DataFrame(results)
    # df.to_csv("performance_metrics.csv", index=False)