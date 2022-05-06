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


def hist(z):
    x = np.array(z)
    
    for q in np.arange(0, 1.1, 0.1):
        print(np.round(q, 2), "\t", np.round(np.quantile(x, q), 3))
    
    plt.figure(figsize=(10, 5))
    
    plt.hist(x, bins=100)
    
    plt.show()


def convert(line):
    line = line[1:-1]
    return np.array([int(item) for item in line.split(' ') if item != ''])

def convert1(line):
    line = line[1:-1]
    return np.array([float(item) for item in line.split(' ') if item != ''])

def drop_odd(arr):
    return arr[np.where(arr < 91)[0]]

def make_label(arr):
    ar = arr[np.where(arr > 0)[0]]
    return len(ar[np.where(ar < 46)[0]])

def number_in_period(arr, start=-1, end=91):
    return len(np.where(np.logical_and(arr > start, arr < end))[0])
#from lightgbm import LGBMRegressor


probs1 = []
probs2 = []

def pois(a, mu=1):
    return np.exp(mu*(-1)) * (mu ** a) / math.factorial(a)
def prob_hand(a, probs1=probs1, probs2=probs2):
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

def prob_x():
    prob_x = 0.0
    for i in range (0, 20):
        for j in range (0, 20):
            if (i == j):
                prob_x += probs1[i] * probs2[j]
    return prob_x, 1 - prob_x

def prob_total(a, probs1=probs1, probs2=probs2):
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

def prob_hand_1(a, probs1=probs1, probs2=probs2): #для +0.25
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

def prob_hand_3(a, probs1=probs1, probs2=probs2): #для -0.25
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

def prob_total_1(a, probs1=probs1, probs2=probs2): #для +2.25
    normed = 1.0
    win1 = 0.0
    half_win = 0.0
    for i in range (0, 20):
        for j in range (0, 20):
            if (i + j < a - 0.5):
                win1 += probs1[i] * probs2[j]
            if (np.round(i + j + 0.25) == a):
                half_win += probs1[i] * probs2[j]
    return win1, half_win, 1 - win1 - half_win


def prob_total_3(a, probs1=probs1, probs2=probs2): #для 2.75
    normed = 1.0
    win1 = 0.0
    half_lose = 0.0
    for i in range (0, 20):
        for j in range (0, 20):
            if (i + j < a - 0.5):
                win1 += probs1[i] * probs2[j]
            if (np.round(i + j - 0.25) == a):
                half_lose += probs1[i] * probs2[j]
    return win1, half_lose, 1 - win1 - half_lose