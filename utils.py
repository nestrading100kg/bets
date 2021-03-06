import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import glob

import os
import pandas as pd
import numpy as np
import pickle
import re
import datetime
from statsmodels.formula.api import ols
import statsmodels
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from sklearn.metrics import make_scorer, r2_score, mean_squared_error

from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from xgboost import XGBRegressor

from sklearn.ensemble import GradientBoostingRegressor
import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import poisson,skellam
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from numba import njit


def hist(z):
    x = np.array(z)
    
    for q in np.arange(0, 1.1, 0.1):
        print(np.round(q, 2), "\t", np.round(np.quantile(x, q), 3))
    
    plt.figure(figsize=(10, 5))
    
    plt.hist(x, bins=100)
    
    plt.show()


@njit
def prob_x(probs1, probs2):
    prob_x = 0.0
    for i in range (0, 20):
        for j in range (0, 20):
            if (i == j):
                prob_x += probs1[i] * probs2[j]
    return prob_x, 1 - prob_x


def prob_ind_total(a, probs1):
    normed = 1.0
    under = 0.0
    for i in range (0, 20):
        if (i < a):
            under += probs1[i]
        if (i == a):
            normed -= probs1[i]
    under = under / normed
    return under, 1 - under


from scipy.stats import poisson


@njit
def factorial(x):
    if x == 0:
        return 1

    return x * factorial(x - 1)

@njit
def pois(a, mu=1):
    return np.exp(mu*(-1)) * (mu ** a) / factorial(a)
    # return poisson.pmf(a, mu)
    

@njit
def prob_hand(a, probs1, probs2):
    normed = 1.0
    win1 = 0.0
    for i in range (0, 20):
        for j in range (0, 20):
            if (i + a > j):
                win1 += probs1[i] * probs2[j]
            if (i + a == j):
                normed -= probs1[i] * probs2[j]
    win1 = win1 / normed
    return win1, 1 - win1


@njit
def prob_hand_1(a, probs1, probs2): #?????? +0.25
    normed = 1.0
    win1 = 0.0
    half_win = 0.0
    for i in range (0, 20):
        for j in range (0, 20):
            if (i + a > j + 0.5):
                win1 += probs1[i] * probs2[j]
            if (np.round(i + a - 0.25) == j):
                half_win += probs1[i] * probs2[j]
    return win1, half_win, 1 - win1 - half_win


@njit
def prob_hand_3(a, probs1, probs2): #?????? -0.25
    normed = 1.0
    win1 = 0.0
    half_lose = 0.0
    for i in range (0, 20):
        for j in range (0, 20):
            if (i + a > j + 0.5):
                win1 += probs1[i] * probs2[j]
            if (np.round(i + a + 0.25) == j):
                half_lose += probs1[i] * probs2[j]
    return win1, half_lose, 1 - win1 - half_lose

@njit
def prob_total(a, probs1, probs2):
    normed = 1.0
    under = 0.0
    for i in range (0, 20):
        for j in range (0, 20):
            if (i + j < a):
                under += probs1[i] * probs2[j]
            if (i + j == a):
                normed -= probs1[i] * probs2[j]
    under = under / normed
    return under, 1 - under

@njit
def prob_total_1(a, probs1, probs2): #?????? 2.25
    normed = 1.0
    win1 = 0.0
    half_win = 0.0
    for i in range (0, 20):
        for j in range (0, 20):
            if (i + j < a - 0.5):
                win1 += probs1[i] * probs2[j]
            if (np.round(i + j + 0.25, 2) == a):
                half_win += probs1[i] * probs2[j]
    return win1, half_win, 1 - win1 - half_win

@njit
def prob_total_3(a, probs1, probs2): #?????? 2.75
    normed = 1.0
    win1 = 0.0
    half_lose = 0.0
    for i in range (0, 20):
        for j in range (0, 20):
            if (i + j < a - 0.5):
                win1 += probs1[i] * probs2[j]
            if (np.round(i + j - 0.25, 2) == a):
                half_lose += probs1[i] * probs2[j]
    return win1, half_lose, 1 - win1 - half_lose


def prob_total_sum(a, probs):
    normed = 1.0
    under = 0.0
    for i in range (0, 20):
        if (i < a):
            under += probs[i]
        if (i == a):
            normed -= probs[i]
    under = under / normed
    return under, 1 - under


@njit
def value_x_new(odd, exp1, exp2):
    probs1 = []
    probs2 = []
    for i in range(0, 20):
        probs1.append(pois(i, exp1))
        probs2.append(pois(i, exp2))
        
    value = prob_x(probs1, probs2)[0] * odd
    return value

def x_income(row):
    if row['res'] == 0:
        return 1
    return 0

def value_1(row):
    odd = row["f1_open"]
    hand = row["hand_open"]
    exp1 = row["pred_home"]
    exp2 = row["pred_away"]

    probs1 = []
    probs2 = []
    for i in range(0, 20):
        probs1.append(pois(i, exp1))
        probs2.append(pois(i, exp2))
        
    odd1 = odd
    h = np.round(hand * 4)

    if (h % 2 == 0):
        value = prob_hand(hand, probs1, probs2)[0] * odd1
        return value
    elif (h % 4 == 1):
        value = \
            prob_hand_1(hand, probs1, probs2)[0] * odd1 + \
            prob_hand_1(hand, probs1, probs2)[1] * ((odd1 - 1) / 2 + 1)
        return value
    else:
        value = \
            prob_hand_3(hand, probs1, probs2)[0] * odd1 + \
            prob_hand_3(hand, probs1, probs2)[1] * 0.5
        return value


@njit
def value_1_new(odd, hand, exp1, exp2):
    probs1 = []
    probs2 = []
    for i in range(0, 20):
        probs1.append(pois(i, exp1))
        probs2.append(pois(i, exp2))
        
    odd1 = odd
    h = np.round(hand * 4)

    if (h % 2 == 0):
        value = prob_hand(hand, probs1, probs2)[0] * odd1
        return value
    elif (h % 4 == 1):
        value = \
            prob_hand_1(hand, probs1, probs2)[0] * odd1 + \
            prob_hand_1(hand, probs1, probs2)[1] * ((odd1 - 1) / 2 + 1)
        return value
    else:
        value = \
            prob_hand_3(hand, probs1, probs2)[0] * odd1 + \
            prob_hand_3(hand, probs1, probs2)[1] * 0.5
        return value


def value_2(row):
    odd = row["f2_open"]
    hand = row["hand_open"]
    exp1 = row["pred_home"]
    exp2 = row["pred_away"]

    probs1 = []
    probs2 = []
    for i in range(0, 20):
        probs1.append(pois(i, exp1))
        probs2.append(pois(i, exp2))

    odd1 = odd
    h = np.round(hand * 4)
    
    if (h % 2 == 0):
        value = prob_hand(hand, probs1, probs2)[1] * odd1
        return value
    elif (h % 4 == 3):
        value = prob_hand_3(hand, probs1, probs2)[2] * odd1 + prob_hand_3(hand, probs1, probs2)[1] * ((odd1 - 1) / 2 + 1)
        return value
    else:
        value = prob_hand_1(hand, probs1, probs2)[2] * odd1 + prob_hand_1(hand, probs1, probs2)[1] * 0.5
        return value


@njit
def value_2_new(odd, hand, exp1, exp2):
    probs1 = []
    probs2 = []
    for i in range(0, 20):
        probs1.append(pois(i, exp1))
        probs2.append(pois(i, exp2))

    odd1 = odd
    h = np.round(hand * 4)
    
    if (h % 2 == 0):
        value = prob_hand(hand, probs1, probs2)[1] * odd1
        return value
    elif (h % 4 == 3):
        value = prob_hand_3(hand, probs1, probs2)[2] * odd1 + prob_hand_3(hand, probs1, probs2)[1] * ((odd1 - 1) / 2 + 1)
        return value
    else:
        value = prob_hand_1(hand, probs1, probs2)[2] * odd1 + prob_hand_1(hand, probs1, probs2)[1] * 0.5
        return value


def value_under(row):
    odd = row["under_open"]
    total = row["total_open"]
    exp1 = row["pred_home"]
    exp2 = row["pred_away"]
    
    probs1 = []
    probs2 = []
    for i in range(0, 20):
        probs1.append(pois(i, exp1))
        probs2.append(pois(i, exp2))

    odd1 = odd
    h = np.round(total * 4)
    
    if (h % 2 == 0):
        value = prob_total(total, probs1, probs2)[0] * odd1
        return value
    elif (h % 4 == 1):
        value = prob_total_1(total, probs1, probs2)[0] * odd1 + prob_total_1(total, probs1, probs2)[1] * ((odd1 - 1) / 2 + 1)
        return value
    else:
        value = prob_total_3(total, probs1, probs2)[0] * odd1 + prob_total_3(total, probs1, probs2)[1] * 0.5
        return value


@njit
def value_under_new(odd, total, exp1, exp2):
    probs1 = []
    probs2 = []
    for i in range(0, 20):
        probs1.append(pois(i, exp1))
        probs2.append(pois(i, exp2))

    odd1 = odd
    h = np.round(total * 4)
    
    if (h % 2 == 0):
        value = prob_total(total, probs1, probs2)[0] * odd1
        return value
    elif (h % 4 == 1):
        value = prob_total_1(total, probs1, probs2)[0] * odd1 + prob_total_1(total, probs1, probs2)[1] * ((odd1 - 1) / 2 + 1)
        return value
    else:
        value = prob_total_3(total, probs1, probs2)[0] * odd1 + prob_total_3(total, probs1, probs2)[1] * 0.5
        return value


def value_over(row):
    odd = row["over_open"]
    total = row["total_open"]
    exp1 = row["pred_home"]
    exp2 = row["pred_away"]
    
    odd1 = odd
    h = np.round(total * 4)
    probs1 = []
    probs2 = []
    for i in range(0, 20):
        probs1.append(pois(i, exp1))
        probs2.append(pois(i, exp2))

    if (h % 2 == 0):
        value = prob_total(total, probs1, probs2)[1] * odd1
        return value
    elif (h % 4 == 3):
        value = prob_total_3(total, probs1, probs2)[2] * odd1 + prob_total_3(total, probs1, probs2)[1] * ((odd1 - 1) / 2 + 1)
        return value
    else:
        value = prob_total_1(total, probs1, probs2)[2] * odd1 + prob_total_1(total, probs1, probs2)[1] * 0.5
        return value

@njit
def value_over_new(odd, total, exp1, exp2):
    odd1 = odd
    h = np.round(total * 4)
    probs1 = []
    probs2 = []
    for i in range(0, 20):
        probs1.append(pois(i, exp1))
        probs2.append(pois(i, exp2))

    if (h % 2 == 0):
        value = prob_total(total, probs1, probs2)[1] * odd1
        return value
    elif (h % 4 == 3):
        value = prob_total_3(total, probs1, probs2)[2] * odd1 + prob_total_3(total, probs1, probs2)[1] * ((odd1 - 1) / 2 + 1)
        return value
    else:
        value = prob_total_1(total, probs1, probs2)[2] * odd1 + prob_total_1(total, probs1, probs2)[1] * 0.5
        return value


def hand_income1(row):
    if row['res'] + row['hand_open'] == 0:
        return 0
    if row['res'] + row['hand_open'] == 0.25:    
        return (row['f1_open'] - 1) / 2
    if row['res'] + row['hand_open'] == -0.25:
        return -0.5
    if row['res'] + row['hand_open'] < -0.4:
        return -1
    if row['res'] + row['hand_open'] > 0.4:
        return row['f1_open'] - 1

def hand_income2(row):
    if row["res"] + row['hand_open'] == 0:
        return 0
    if row["res"] + row['hand_open'] == -0.25:
        return (row['f2_open'] - 1) / 2
    if row["res"] + row['hand_open'] == 0.25:
        return -0.5
    if (row["res"] + row['hand_open'] < -0.4):
        return row['f2_open'] - 1
    if (row["res"] + row['hand_open'] > 0.4):
        return -1

def over_income(row):
    if row['sum_res'] - row['total_open'] == 0:
        return 0
    if row['sum_res'] - row['total_open'] == 0.25:
        return (row['over_open'] - 1) / 2
    if row['sum_res'] - row['total_open'] == -0.25:
        return -0.5
    if (row['sum_res'] - row['total_open'] < -0.4):
        return -1
    if (row['sum_res'] - row['total_open'] > 0.4):
        return row['over_open'] - 1


def under_income(row):
    if row['sum_res'] - row['total_open'] == 0:
        return 0
    if row['sum_res'] - row['total_open'] == -0.25:
        return (row['under_open'] - 1) / 2
    if row['sum_res'] - row['total_open'] == 0.25:
        return -0.5
    if (row['sum_res'] - row['total_open'] < -0.4):
        return row['under_open'] - 1
    if (row['sum_res'] - row['total_open'] > 0.4):
        return -1

def convert_score_diff(x):
    try:
        s = x.split(":")
        
        return float(s[0]) - float(s[1])
    except:
        print(x)
        return -10000

def convert_score_sum(x):
    try:
        s = x.split(":")
        
        return float(s[0]) - float(s[1])
    except:
        print(x)
        return -10000