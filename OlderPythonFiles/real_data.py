import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

###########################################################
####### Load Data #########################################
###########################################################

df_train = pd.read_csv("training_data.csv")
df_val = pd.read_csv("validation_data.csv")

TIME = df_train["time"].round().astype(int).values
state = df_train["state"].values

TIME_val = df_val["time"].round().astype(int).values
state_val = df_val["state"].values

# covariates
X = df_train[["x1", "x2", "x3", "x4"]].values
X_val = df_val[["x1", "x2", "x3", "x4"]].values

# torch tensors
X = torch.tensor(X, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
TIME = torch.tensor(TIME, dtype=torch.float32)
TIME_val = torch.tensor(TIME_val, dtype=torch.float32)
state = torch.tensor(state, dtype=torch.long)
state_val = torch.tensor(state_val, dtype=torch.long)

###########################################################
####### Models ############################################
###########################################################

class ConstantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(6))

    def forward(self, x):
        return self.param.repeat(x.shape[0], 1)


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 6)

    def forward(self, x):
        return self.linear(x)


class NNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 50),
            nn.SiLU(),
            nn.Linear(50, 50),
            nn.SiLU(),
            nn.Linear(50, 6)
        )

    def forward(self, x):
        return self.net(x)

def run_real_data():
    ###########################################################
    ####### Multi-sigmoid #####################################
    ###########################################################

    def multi_sigmoid(x):
        exp_x = torch.exp(x)
        denom = 1 + torch.sum(exp_x, dim=1, keepdim=True)
        return exp_x / denom

    ###########################################################
    ####### Loss ##############################################
    ###########################################################

    def loss_fn(model, X, T, state):
        preds = model(X)
        lam = multi_sigmoid(preds)

        survival = 1 - torch.sum(lam, dim=1) + 1e-5
        lam = torch.clamp(lam, min=1e-10)

        loss = 0.0

        for i in range(len(T)):
            if state[i] == 0:
                loss += T[i] * torch.log(survival[i])
            else:
                loss += torch.log(lam[i, state[i]-1]) + T[i] * torch.log(survival[i])

        return -loss

    ###########################################################
    ####### Training ##########################################
    ###########################################################

    def train_model(model, X, T, state, epochs, lr):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_hist = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            loss = loss_fn(model, X, T, state)
            loss.backward()
            optimizer.step()

            loss_hist.append(loss.item())

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

        return model, loss_hist

    ###########################################################
    ####### Train models ######################################
    ###########################################################

    const_model = ConstantModel()
    lin_model = LinearModel()
    nn_model = NNModel()

    # training
    const_model, _ = train_model(const_model, X, TIME, state, 200, 0.01)
    lin_model, _ = train_model(lin_model, X, TIME, state, 200, 0.01)
    nn_model, _ = train_model(nn_model, X, TIME, state, 400, 0.001)

    ###########################################################
    ####### Validation Loss ###################################
    ###########################################################

    const_loss_val = loss_fn(const_model, X_val, TIME_val, state_val)
    lin_loss_val = loss_fn(lin_model, X_val, TIME_val, state_val)
    nn_loss_val = loss_fn(nn_model, X_val, TIME_val, state_val)

    ###########################################################
    ####### Prediction ########################################
    ###########################################################

    def predict(model, X):
        preds = model(X)
        lam = multi_sigmoid(preds)
        lam0 = 1 - torch.sum(lam, dim=1, keepdim=True)
        return torch.cat([lam0, lam], dim=1).detach().numpy()

    const_pred = predict(const_model, X_val)
    lin_pred = predict(lin_model, X_val)
    nn_pred = predict(nn_model, X_val)

    ###########################################################
    ####### State Occupation ##################################
    ###########################################################

    def state_occupation_matrix(df, t_eval):
        df_filt = df[~((df["time"] < t_eval) & (df["state"] == 0))]

        TIME = df_filt["time"].round().astype(int).values
        state = df_filt["state"].values

        mat = np.zeros((len(TIME), 7))

        for i in range(len(TIME)):
            if TIME[i] > t_eval:
                mat[i, 0] = 1
            else:
                mat[i, state[i]] = 1

        return mat, df_filt["id"].values

    ###########################################################
    ####### Brier Score #######################################
    ###########################################################

    def brier_score_real_world(pred, obs, t_eval):
        N = pred.shape[0]
        BS = np.zeros(N)

        pi0 = np.zeros(7)
        pi0[0] = 1

        for i in range(N):
            P = np.zeros((7,7))
            P[0] = pred[i]
            np.fill_diagonal(P[1:], 1)

            P_t = np.linalg.matrix_power(P, t_eval)
            pred_state = pi0 @ P_t

            BS[i] = np.sum((pred_state - obs[i])**2)

        return BS

    ###########################################################
    ####### Evaluate ##########################################
    ###########################################################

    obs_mat, ids = state_occupation_matrix(df_val, 60)

    mask = df_val["id"].isin(ids)

    const_pred = const_pred[mask]
    lin_pred = lin_pred[mask]
    nn_pred = nn_pred[mask]

    BS_const = brier_score_real_world(const_pred, obs_mat, 60)
    BS_lin = brier_score_real_world(lin_pred, obs_mat, 60)
    BS_nn = brier_score_real_world(nn_pred, obs_mat, 60)

    print("Mean difference (lin - NN):", np.mean(BS_lin - BS_nn))