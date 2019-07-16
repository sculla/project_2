#!/anaconda3/envs/metis/bin/python

#if __name__ == '__main__':
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib


sns.set_style('darkgrid')

import pickle


def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

train_val_test = ['X_tr', 'X_val', 'X_test', 'y_tr', 'y_val', 'y_test']
k = 81

for na in train_val_test:
    with open(f'data/{na}.{k}.pickle', 'rb') as f:
        globals()[na] = pickle.load(f)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_tr.values)
X_val_scaled = scaler.transform(X_val.values)
X_test_scaled = scaler.transform(X_test.values)

#Lasso test
alphalist = 10 ** (np.linspace(-2, 2, 200))
err_vec_val = np.zeros(len(alphalist))  # pre allocation is faster than appending
err_vec_train = np.zeros(len(alphalist))

for i, curr_alpha in enumerate(alphalist):
    # note the use of a new sklearn utility: Pipeline to pack
    # multiple modeling steps into one fitting process
    steps = [('standardize', StandardScaler()),
             ('lasso', Lasso(alpha=curr_alpha))]

    pipe = Pipeline(steps)
    pipe.fit(X_tr.values, y_tr)

    val_set_pred = pipe.predict(X_val.values)
    err_vec_val[i] = mse(y_val, val_set_pred)
plt.plot(np.log10(alphalist), err_vec_val)
plt.savefig('images/Lasso_Test.svg')

print('The best value for test \u03BB in a Lasso is {}'.format(alphalist[np.argmin(err_vec_val)]))


#Lasso CV
alphavec = 10 ** np.linspace(-2, 2, 200)

lasso_model = LassoCV(alphas=alphavec, cv=5, tol=.00001, max_iter=500)
lasso_model.fit(X_scaled, y_tr)
print('The best value for CV \u03BB in a Lasso is {}, and CV == test: {}'.format(lasso_model.alpha_, lasso_model.alpha_ == alphalist[np.argmin(err_vec_val)]))

# Scaled Coefficients
# [('Assessed_Value', 642530.2972543074),
#  ('BG^2', -14357.300046032118),
#  ('BC^2', -5186.510696270045),
#  ('Stories', -2429.167857371234),
#  ('Living_units', 15981.988512924796),
#  ('Above_grade_living_area', -22421.49345814874),
#  ('Below_grade_living_area', -7575.529280712608),
#  ('Total_basement', 6394.6453378824635),
#  ('Finished_basement', -5218.793270781661),
#  ('Sq_ft_lot', 16080.982407121579),
#  ('Topography', -901.792153504179),
#  ('Environmental', -421.4572205665248),
#  ('Nuisances', -4003.3215791324296),
#  ('Building_Age', 8297.759457703842)]

test_set_pred = lasso_model.predict(X_val_scaled)
resid = y_val - test_set_pred

#CV Lasso Model Graph
f, ax = plt.subplots()

ax.scatter(test_set_pred, resid, alpha=.2, color='#33C2FF')
from matplotlib.ticker import NullFormatter, LogLocator, FixedLocator
ax.set_title('Residuals of CV Lasso Model \u03BB:100, R^2 Score: {}'.format(round(r2_score(y_val, test_set_pred),3)), fontsize=14, color='w')
ax.set_ylim(resid.std()*-3.1, resid.std()*3.1)
ax.set_ylabel('y - \u0177: Residual', color='w', fontsize=14)
ax.set_xlabel('Test Set Prediction', color='w', fontsize=14)
ax.set_yscale('linear')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.set_xscale('log')
locmin = LogLocator(base=10.0, subs='auto')
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(NullFormatter())
locmaj = LogLocator(base=10.0, subs='auto')
ax.xaxis.set_major_locator(locmaj)
resid_std = [resid.std() * x for x in [-3,-2,-1,0,1,2,3]]
ax.yaxis.set_major_locator(FixedLocator(resid_std))
ylab = '-3\u03C3 -2\u03C3 -1\u03C3 x\u0305 1\u03C3 2\u03C3 3\u03C3'.split()
ax.set_yticklabels(ylab)
ax.set_xticks(list(plt.xticks()[0])+[1e6]+[8e5])
ax.set_xlim(np.min(test_set_pred), np.max(test_set_pred))
plt.tight_layout()
plt.savefig('images/Lasso_Residual.svg', transparent=True)
