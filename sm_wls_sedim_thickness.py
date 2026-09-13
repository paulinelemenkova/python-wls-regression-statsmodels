#!/usr/bin/env python
# coding: utf-8
"""Weighted Least Squares (WLS) Regression with Python and statsmodels

Author:  Polina Lemenkova
ORCID:   https://orcid.org/0000-0002-5759-1089
Archive: https://doi.org/10.13140/RG.2.2.10302.95042
License: MIT

See README.md for details.
"""
# In[35]:


from __future__ import print_function

import os

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from patsy import dmatrices
from scipy import stats
from statsmodels.iolib.table import SimpleTable, default_txt_fmt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

sns.set_style('whitegrid')

# Step-2. Import data
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv("Tab-Morph.csv")
df = df.dropna()
nsample = 25
# x = np.linspace(0, 25, nsample)
x = df.sedim_thick
X = np.column_stack((x, (x - 5)**2))
X = sm.add_constant(X)
beta = [5., 0.5, -0.01]
sig = 0.5
w = np.ones(nsample)
w[nsample * 6//10:] = 3
y_true = np.dot(X, beta)
e = np.random.normal(size=nsample)
y = y_true + sig * w * e
X = X[:, [0, 1]]

# Step-3.
mod_wls = sm.WLS(y, X, weights=1./(w ** 2))
res_wls = mod_wls.fit()
print(res_wls.summary())

# Step-4.
res_ols = sm.OLS(y, X).fit()
print(res_ols.params)
print(res_wls.params)

# Step-5.
se = np.vstack([[res_wls.bse], [res_ols.bse], [res_ols.HC0_se],
                [res_ols.HC1_se], [res_ols.HC2_se], [res_ols.HC3_se]])
se = np.round(se, 4)
colnames = ['x1', 'const']
rownames = ['WLS', 'OLS', 'OLS_HC0', 'OLS_HC1', 'OLS_HC3', 'OLS_HC3']
tabl = SimpleTable(se, colnames, rownames, txt_fmt=default_txt_fmt)
print(tabl)

# Step-6.
covb = res_ols.cov_params()
prediction_var = res_ols.mse_resid + (X * np.dot(covb, X.T).T).sum(1)
prediction_std = np.sqrt(prediction_var)
tppf = stats.t.ppf(0.975, res_ols.df_resid)

# Step-7.
prstd_ols, iv_l_ols, iv_u_ols = wls_prediction_std(res_ols)

# Step-8.
prstd, iv_l, iv_u = wls_prediction_std(res_wls)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, 'o', label="Bathymetric \nObservations", linewidth=.7, c='#0095d9')
ax.plot(x, y_true, '-', c='#1e50a2', label="True", linewidth=.9)
# OLS
ax.plot(x, res_ols.fittedvalues, 'r--', linewidth=.7)
ax.plot(x, iv_u_ols, 'r--', label="Ordinary Least Squares", linewidth=.7)
ax.plot(x, iv_l_ols, 'r--', linewidth=.7)
# WLS
ax.plot(x, res_wls.fittedvalues, '--.', c='#65318e', linewidth=.7, )
ax.plot(x, iv_u, '--', c='#65318e', label="Weighted Least Squares", linewidth=.7)
ax.plot(x, iv_l, '--', c='#65318e', linewidth=.7)
ax.legend(loc="best")
ax.set_xlabel('Sediment thickness, m', fontsize=10)

plt.title("Weighted Least Squares \nof sediment thickness at Mariana Trench by 25 bathymetric profiles", fontsize=14)
plt.annotate('D', xy=(-0.01, 1.06), xycoords="axes fraction", fontsize=18,
             bbox=dict(boxstyle='round, pad=0.3', fc='w', edgecolor='grey', linewidth=1, alpha=0.9))
plt.show()
