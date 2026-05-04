import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

###########################################################
####### Train/Test Split (Stratified by state) ############
###########################################################

def train_test_data_split(df, frac):
    test_indices = []

    for state_value in df["state"].unique():
        group_idx = df[df["state"] == state_value].index
        n_test = int(round(frac * len(group_idx)))

        sampled = np.random.choice(group_idx, size=n_test, replace=False)
        test_indices.extend(sampled)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return test_df, train_df


###########################################################
####### Multi-sigmoid (Softmax over 6 states) #############
###########################################################

def multi_sigmoid(x):
    exp_x = torch.exp(x)
    denom = 1 + torch.sum(exp_x)
    return exp_x / denom  # returns vector λ1,...,λ6


###########################################################
####### Neural Network ####################################
###########################################################

class NNModelTwoLayer(nn.Module):
    def __init__(self, n_covs, n_nodes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_covs, n_nodes),
            nn.SiLU(),  # swish
            nn.Linear(n_nodes, n_nodes),
            nn.SiLU(),
            nn.Linear(n_nodes, 6)  # output all 6 λ's at once
        )

    def forward(self, x):
        return self.net(x)


###########################################################
####### Loss Function #####################################
###########################################################

def loss_fn(model, X, T, state):
    preds = model(X)  # shape (N, 6)
    loss = 0.0

    for i in range(len(T)):
        lam = multi_sigmoid(preds[i])
        survival = 1 - torch.sum(lam) + 1e-5

        # clamp for stability
        lam = torch.clamp(lam, min=1e-10)

        t = T[i]

        if state[i] == 0:
            loss += t * torch.log(survival)

        else:
            loss += torch.log(lam[state[i]-1]) + t * torch.log(survival)

    return -loss


###########################################################
####### Training ##########################################
###########################################################

def train_model(model, X, T, state, epochs=100, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss = loss_fn(model, X, T, state)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model, loss_history


###########################################################
####### Prediction ########################################
###########################################################

def predict(model, X):
    preds = model(X)
    probs = []

    for i in range(len(preds)):
        lam = multi_sigmoid(preds[i])
        lam0 = 1 - torch.sum(lam)
        probs.append(torch.cat([lam0.unsqueeze(0), lam]))

    return torch.stack(probs)  # shape (N, 7)


###########################################################
####### State Occupation Matrix ############################
###########################################################

def state_occupation_matrix(df, time_eval):
    filtered_df = df[~((df["time"] < time_eval) & (df["state"] == 0))]

    TIME = filtered_df["time"].round().astype(int).values
    state = filtered_df["state"].values

    Nid = len(TIME)
    mat = np.zeros((Nid, 7))

    for i in range(Nid):
        if TIME[i] > time_eval:
            mat[i, 0] = 1
        else:
            mat[i, state[i]] = 1  # state already indexed correctly

    return mat, filtered_df["id"].values


###########################################################
####### Brier Score #######################################
###########################################################

def brier_score_real_world(pred, obs, T_eval):
    Nid = pred.shape[0]
    BS = np.zeros(Nid)

    pi0 = np.zeros(7)
    pi0[0] = 1

    for i in range(Nid):
        TP = pred[i]

        P = np.zeros((7,7))
        P[0] = TP
        np.fill_diagonal(P[1:], 1)

        P_t = np.linalg.matrix_power(P, T_eval)
        state_pred = pi0 @ P_t

        BS[i] = np.sum((state_pred - obs[i])**2)

    return BS