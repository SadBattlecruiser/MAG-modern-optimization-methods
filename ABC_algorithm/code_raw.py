#!/usr/bin/env python
# coding: utf-8


import numpy as np
from tqdm import tqdm
from numba import njit

from multiprocessing import Pool



@njit
def spherical_func(X, X0=None):
    if X0 is None:
        X0 = np.zeros(X.shape[0])
    return np.sum(np.power(X - X0, 2))


@njit
def rastrigin_func(X):
    return np.sum(np.power(X, 2) - 10*np.cos(2*np.pi*X) + 10) - 330


#@njit
def rosenbrok_func(X):
    s = 0
    for i in range(X.shape[0] - 1):
        s += 100 * (X[i]**2 - X[i+1])**2 + (X[i] - 1)**2
    return s - 390

rastrigin_func(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))



# Матрица попарных расстояний
@njit
def calc_dist(X):
    dist = np.zeros((X.shape[0], X.shape[0]))
    for i in range(1, X.shape[0]):
        for j in range(i):
            dist[i, j] = np.linalg.norm(X[i]-X[j])
            dist[j, i] = dist[i, j]
    return dist

# Построить NNNS для Xi
def MNNS(X, FX, i):
    dist = calc_dist(X)
    sequence = [(i, X[i], FX[i])]
    while True:
        candidates = np.argwhere(FX < sequence[-1][2])
        if len(candidates) == 0:
            break
        min_dist = dist[sequence[-1][0], candidates].min()
        h = np.argwhere(dist[sequence[-1][0]] == min_dist)[0, 0]
        sequence.append((h, X[h], FX[h]))
    return sequence


# Стратегии поиска, входящие в SP
def ss1(X, FX, i):
    MSi = np.array(np.array([j[1] for j in MNNS(X, FX, i)]))
    MCXi = np.mean(MSi, axis=0)
    best = np.argwhere(FX == np.min(FX))[0, 0]
    k = np.random.randint(X.shape[0] - 1)
    if k >= i:
        k += 1
    Vi = X[i].copy()
    j = np.random.randint(X.shape[1])
    Vi[j] = MCXi[j] + np.random.uniform(-1., 1.)*(X[best, j] - X[k, j])
    return Vi

def ss2(X, FX, i):
    MSi = np.array(np.array([j[1] for j in MNNS(X, FX, i)]))
    best = np.argwhere(FX == np.min(FX))[0, 0]
    Vi = X[i].copy()
    j = np.random.randint(X.shape[1])
    if MSi.shape[0] == 1:
        k = np.random.randint(X.shape[0] - 1)
        if k >= i:
            k += 1
        Vi[j] = X[best, j] + np.random.uniform(-1., 1.)*(X[best, j] - X[k, j])
    else:
        h = np.random.randint(MSi.shape[0] - 1)
        Vi[j] = X[best, j] + np.random.uniform(-1., 1.)*(MSi[h+1, j] - MSi[h, j])
    return Vi



def initialize_X(SN, Xmin, Xmax, SP, func):
    if Xmin.shape != Xmax.shape:
        raise ValueError('initialize_X: Xmin.shape != Xmax.shape')
    X = np.zeros((SN, Xmin.shape[0]))
    FX = np.zeros(SN)
    for i in range(SN):
        X[i] = Xmin + np.random.uniform(0, 1, Xmin.shape[0])*(Xmax - Xmin)
        FX[i] = func(X[i])
    SP_idx = np.random.randint(len(SP), size=X.shape[0])
    return X, FX, SP_idx


def employed_phase(X, FX, SP, SP_idx, trial, func):
    V = np.zeros(X.shape)
    FV = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        V[i] = SP[SP_idx[i]](X, FX, i)
        FV[i] = func(V[i])
    for i in range(X.shape[0]):
        if FV[i] < FX[i]:
            X[i] = V[i]
            FX[i] = FV[i]
            trial[i] = 0
            # Обновляем стратегию поиска
            SP_idx_candidate = np.random.randint(len(SP) - 1)
            if SP_idx_candidate == SP_idx[i]:
                SP_idx[i] = SP_idx_candidate + 1
            else:
                SP_idx[i] = SP_idx_candidate
        else:
            trial[i] += 1
    return X, FX, SP_idx, trial


def onlooker_phase(X, FX, SP, SP_idx, trial, func):
    V = np.zeros(X.shape)
    FV = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        MSi = np.array(np.array([j[1] for j in MNNS(X, FX, i)]))
        r = np.random.randint(MSi.shape[0]-1) + 1 if MSi.shape[0] > 1 else 0
        h = np.argwhere(FX == func(MSi[r]))[0, 0]
        V[h] = SP[SP_idx[h]](X, FX, h)
        FV[h] = func(V[h])
    for i in range(X.shape[0]):
        if FV[i] == 0:
            continue
        if FV[i] < FX[i]:
            X[i] = V[i]
            FX[i] = FV[i]
            trial[i] = 0
            # Обновляем стратегию поиска
            SP_idx_candidate = np.random.randint(len(SP) - 1)
            if SP_idx_candidate == SP_idx[i]:
                SP_idx[i] = SP_idx_candidate + 1
            else:
                SP_idx[i] = SP_idx_candidate
        else:
            trial[i] += 1
    return X, FX, SP_idx, trial

def onlooker_old_phase(X, FX, SP, SP_idx, trial, func):
    def fit(f):
        return 1./(1+f) if f >= 0 else 1 + np.abs(f)
    
    fitX = np.vectorize(fit)(FX)
    idx_selected = np.random.choice(
        X.shape[0],
        p=fitX/np.sum(fitX)
    )
    X_selected = X[idx_selected].copy()
    
    V = np.zeros(X.shape)
    FV = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        k = np.random.randint(X.shape[0] - 1)
        if k >= i:
            k += 1
        V[i] = X[i] + np.random.uniform(-1., 1.)*(X[i] - X_selected)
        FV[i] = func(V[i])
    for i in range(X.shape[0]):
        if FV[i] < FX[i]:
            X[i] = V[i]
            FX[i] = FV[i]
            trial[i] = 0
        else:
            trial[i] += 1            
    return X, FX, SP_idx, trial


def scout_phase(X, FX, SP, SP_idx, trial, func, Xmin, Xmax, trial_lim):
    for i in range(X.shape[0]):
        if trial[i] >= trial_lim:
            XX, FXX, SP_idxXX = initialize_X(1, Xmin, Xmax, SP, func)
            X[i] = XX[0]
            FX[i] = FXX[0]
            SP_idx[i] = SP_idxXX[0]
            trial[i] = 0
    return X, FX, SP_idx, trial


def form_data(iteration, phase, X, FX, SP_idx, trial):
    return {
        'iteration': 0,
        'phase': 'initialize',
        'X': X.copy(),
        'FX': FX.copy(),
        'SP_idx': SP_idx.copy(),
        'trial': trial.copy()
    }

def NNSABC(SN, Xmin, Xmax, SP, func, max_iterations=1000, trial_lim=20, stagnation_lim=20, stagnation_eps=1e-6):
    path = []
    stagnation_cnt = 0
    stop_reason = 'max_iterations'
    X, FX, SP_idx = initialize_X(SN, Xmin, Xmax, SP, func)
    trial = np.zeros(X.shape[0])
    path.append(form_data(0, 'initialize', X, FX, SP_idx, trial))
    X_best_prev = X[np.argwhere(FX == np.min(FX))[0, 0]]
    #for iteration in tqdm(range(max_iterations)):
    for iteration in range(max_iterations):
        X_best_curr = X[np.argwhere(FX == np.min(FX))[0, 0]]
        if np.abs(func(X_best_curr) - func(X_best_prev)) < stagnation_eps:
            stagnation_cnt += 1
            if stagnation_cnt >= stagnation_lim:
                stop_reason = 'stagnation_lim'
                break
        else:
            stagnation_cnt = 0
        X_best_prev = X_best_curr
        X, FX, SP_idx, trial = employed_phase(X, FX, SP, SP_idx, trial, func)
        path.append(form_data(iteration, 'employed', X, FX, SP_idx, trial))
        X, FX, SP_idx, trial = onlooker_phase(X, FX, SP, SP_idx, trial, func)
        #X, FX, SP_idx, trial = onlooker_old_phase(X, FX, SP, SP_idx, trial, func)
        path.append(form_data(iteration, 'onlooker', X, FX, SP_idx, trial))
        X, FX, SP_idx, trial = scout_phase(X, FX, SP, SP_idx, trial, func, Xmin, Xmax, trial_lim=10)
    return stop_reason, X_best_curr, X, FX, path


def NNSABC_wrapper(start_num, SN, Xmin, Xmax, SP, func, max_iterations=1000, trial_lim=20, stagnation_lim=20, stagnation_eps=1e-6):
    print(f'size {Xmin.shape[0]}, start {start_num}')
    stop_reason, X_best, X, FX, path = NNSABC(SN, Xmin, Xmax, SP, func, max_iterations, trial_lim, stagnation_lim, stagnation_eps)
    FX_best = func(X_best)
    return {
        'start_num': start_num,
        'stop_reason': stop_reason,
        'FX_best': FX_best,
        'X_best': X_best,
        'path': path,
        'Xmin': Xmin[0],
        'Xmax': Xmax[0]
    }


# Тестируем
from multiprocessing import Pool
from functools import partial

if __name__ == '__main__':
    dimensions_list = [2, 4, 8, 16, 32]
    #dimensions_list = [16]

    func = rastrigin_func
    alpha = 0.2
    SN = 70
    SP = [ss1, ss2]
    max_iterations = 1000

    res = []

    for dimensions in dimensions_list:
        Xmin = np.array([- alpha] * dimensions)
        Xmax = np.array([alpha] * dimensions)
        NNSABC_func = partial(
            NNSABC_wrapper,
            SN=SN,
            Xmin=Xmin,
            Xmax=Xmax,
            SP=SP,
            func=func,
            max_iterations=max_iterations
        )
        with Pool(12) as p:
            res += p.map(NNSABC_func, range(100))
            #for start in range(100):
            #    print('dimensions:', dimensions)
            #    print('start num: ', start)
            #    print('Xmin:      ')
            #    print(Xmin)
            #    print('Xmax:      ')
            #    print(Xmax)
            #    stop_reason, X_best,  X,FX, path = NNSABC(SN, Xmin, Xmax, SP, func, max_iterations)
            #    FX_best = func(X_best)
            #    print('stop_reason', stop_reason)
            #    print('FX_best', FX_best)
            #    res.append({
            #        'start_num': start,
            #        'stop_reason': stop_reason,
            #        'FX_best': FX_best,
            #        'X_best': X_best,
            #        'path': path
            #    })




    import pandas as pd

    rastrigin_df = pd.DataFrame([{
        'start_num': i['start_num'],
        'stop_reason': i['stop_reason'],
        'FX_best': i['FX_best'],
        'X_best': i['X_best'],
        'Xmin': i['Xmin'],
        'Xmax': i['Xmax']
    } for i in res])
    rastrigin_df['dimensions'] = rastrigin_df['X_best'].apply(lambda x: np.shape(x)[0])
    #rastrigin_df['error_norm'] = (rastrigin_df['FX_best'] + 330.).apply(np.abs)
    rastrigin_df['error_norm'] = (rastrigin_df['FX_best'] + 390.).apply(np.abs)
    print(rastrigin_df)
    rastrigin_df.to_csv(f'rastrigin_{Xmax[0]}_df.csv')


    # Считаем вероятность попадания в глобальный минимум
    global_prob = []
    for dimensions in rastrigin_df['dimensions'].unique():
        global_prob.append({
            'dimensions': dimensions,
            'global_prob': float(np.sum((rastrigin_df['dimensions'] == dimensions) & (rastrigin_df['error_norm'] <= 1e-2))) / \
                        float(np.sum(rastrigin_df['dimensions'] == dimensions))
        })
    global_prob_df = pd.DataFrame(global_prob)
    print(global_prob_df)
    global_prob_df.to_csv(f'rastrigin_global_prob_{Xmax[0]}_df.csv')

