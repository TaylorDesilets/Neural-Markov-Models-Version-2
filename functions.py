import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import poisson, bernoulli, norm
from numpy.random import choice

###########################################################
####### Data manipulation functions #######################
###########################################################

def transform_transvec(transitionvector, statevector, endstate):
    transformed = transitionvector.copy()
    for i in range(len(statevector)):
        if statevector[i] != endstate:
            transformed[i] = 0
    return transformed


def introduce_interval_censoring(timevector, transition12, transition13, P):
    interval_vector = []
    for i in range(len(timevector)):
        eps1 = poisson.rvs(P)
        eps2 = poisson.rvs(P)

        if transition12[i] == 1 and timevector[i] >= 0:
            lower = max(0, timevector[i] - eps1)
            upper = timevector[i] + eps2
            interval_vector.append((lower, upper))

        elif transition13[i] == 1 and timevector[i] >= 0:
            lower = max(0, timevector[i] - eps1)
            upper = timevector[i] + eps2
            interval_vector.append((lower, upper))

        else:
            interval_vector.append((timevector[i], timevector[i]))

    return interval_vector


def create_time_matrix(timevector):
    return np.array(timevector, dtype=int)


###########################################################
####### Data simulation ###################################
###########################################################

def simulate_data(Nid, Ncovs):
    time_vector = []
    trans12 = np.zeros(Nid)
    trans13 = np.zeros(Nid)
    state_vec = []

    cov_mat = np.zeros((Nid, Ncovs))
    lambda_vec12 = np.zeros(Nid)
    lambda_vec13 = np.zeros(Nid)

    def h(X):
        return ((0.5*(X[0]**3)/(X[0]**3+0.3**3)) +
                (5*X[1]*(X[1]-0.9)*(X[1]-0.8)+0.05)) / 4

    def hz(X):
        return (np.exp(X[0]*2 + X[0]**3 - 4) +
                (5*X[1]*(X[1]-0.9)*(X[1]-0.8)+0.15)) / 3

    for i in range(Nid):
        day = 0
        phi = np.random.uniform(0.02, 0.98, Ncovs)
        cov_mat[i] = phi

        eps12 = norm.rvs(0, 0.05)
        eps13 = norm.rvs(0, 0.05)

        lam12 = h(phi) * (1 + eps12)
        lam13 = hz(phi) * (1 + eps13)

        lambda_vec12[i] = lam12
        lambda_vec13[i] = lam13

        lam11 = 1 - lam12 - lam13

        probs = [lam11, lam12, lam13]
        event = choice([1,2,3], p=probs)

        while event == 1:
            day += 1
            event = choice([1,2,3], p=probs)

        if event == 2:
            trans12[i] = 1
            state = 2
        else:
            trans13[i] = 1
            state = 3

        time_vector.append(day)
        state_vec.append(state)

    return (np.array(time_vector), trans12, trans13,
            np.array(state_vec), cov_mat.T,
            lambda_vec12, lambda_vec13)


###########################################################
####### Censoring #########################################
###########################################################

def right_censoring(TIME, trans12, trans13, prob):
    t12 = trans12.copy()
    t13 = trans13.copy()
    T = TIME.copy()

    for i in range(len(trans12)):
        if bernoulli.rvs(prob):
            t12[i] = 0
            t13[i] = 0

            remove_time = poisson.rvs(1)
            T[i] = T[i] - remove_time
            T[i] = max(T[i], 0)

    return T, t12, t13


###########################################################
####### Neural Network ####################################
###########################################################

class NNModelTwoLayer(nn.Module):
    def __init__(self, n_covs, n_nodes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_covs, n_nodes),
            nn.ReLU(),
            nn.Linear(n_nodes, n_nodes),
            nn.ReLU(),
            nn.Linear(n_nodes, 1)
        )

    def forward(self, x):
        return self.net(x)


def multi_sigmoid(x1, x2):
    exp1 = torch.exp(x1)
    exp2 = torch.exp(x2)
    denom = 1 + exp1 + exp2
    return exp1/denom, exp2/denom


###########################################################
####### Loss Function #####################################
###########################################################

def loss_fn(model12, model13, X, T, trans12, trans13):
    loss = 0.0

    pred12 = model12(X).squeeze()
    pred13 = model13(X).squeeze()

    for i in range(len(T)):
        lam12, lam13 = multi_sigmoid(pred12[i], pred13[i])
        survival = 1 - (lam12 + lam13) + 1e-9

        t = int(T[i])

        if trans12[i] == 1:
            loss += torch.log(lam12) + t * torch.log(survival)

        elif trans13[i] == 1:
            loss += torch.log(lam13) + t * torch.log(survival)

        else:
            loss += t * torch.log(survival)

    return -loss


###########################################################
####### Training ##########################################
###########################################################

def train(model12, model13, X, T, trans12, trans13, epochs=100):
    opt = optim.Adam(list(model12.parameters()) +
                     list(model13.parameters()), lr=0.01)

    for epoch in range(epochs):
        opt.zero_grad()

        loss = loss_fn(model12, model13, X, T, trans12, trans13)
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model12, model13


###########################################################
####### Brier Score #######################################
###########################################################

def state_occupation_probability(T, lam12, lam13):
    lam11 = 1 - (lam12 + lam13)

    P = np.array([
        [lam11, lam12, lam13],
        [0, 1, 0],
        [0, 0, 1]
    ])

    P_t = np.linalg.matrix_power(P, T)
    pi0 = np.array([1,0,0])

    return pi0 @ P_t


def brier_score(T, lam12_pred, lam13_pred, trans12, trans13, cutoff):
    BS = []

    for i in range(len(lam12_pred)):
        pi_t = state_occupation_probability(
            cutoff, lam12_pred[i], lam13_pred[i])

        if trans12[i] == 1:
            obs = np.array([0,1,0])
        elif trans13[i] == 1:
            obs = np.array([0,0,1])
        else:
            obs = np.array([1,0,0])

        BS.append(np.sum((pi_t - obs)**2))

    return np.array(BS)