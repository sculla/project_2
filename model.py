#!/anaconda3/envs/metis/bin/python

#if __name__ == '__main__':
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter, LogLocator, FixedLocator
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

def build_graph(resid, val_pred_internal, y_val, mod, name='test', model_name='N/A'):
    #CV Lasso Model Graph
    f, ax = plt.subplots()

    ax.scatter(val_pred_internal, resid, alpha=.2, color='#33C2FF')

    ax.set_title('Residuals of CV {} Model \u03BB:{:2f}, R^2 Score: {}'.format(model_name, mod.alpha_,round(r2_score(y_val, val_pred_internal), 3)), fontsize=14, color='w')
    # ax.set_ylim(resid.std()*-3, resid.std()*3)
    ax.set_ylabel(f'y - \u0177 Residual: \u0305x :{resid.mean():.2f}, \u03C3:{resid.std():.2f}', color='w', fontsize=12)
    ax.set_xlabel('Test Set Prediction', color='w', fontsize=14)
    ax.set_yscale('linear')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_xscale('linear')
    # locmin = LogLocator(base=10.0, subs='auto')
    # ax.xaxis.set_minor_locator(locmin)
    # ax.xaxis.set_minor_formatter(NullFormatter())
    # locmaj = LogLocator(base=10.0, subs='auto')
    # ax.xaxis.set_major_locator(locmaj)
    # resid_std = [resid.std() * x for x in [-4,-3,-2,-1,0,1,2,3,4]]
    # ax.yaxis.set_major_locator(FixedLocator(resid_std))
    # ylab = '-4\u03C3 -3\u03C3 -2\u03C3 -1\u03C3 0 1\u03C3 2\u03C3 3\u03C3 4\u03C3'.split()
    # ax.set_yticklabels(ylab)
    # ax.set_xticks(list(plt.xticks()[0])+[1e6]+[8e5])
    ax.set_xlim(np.min(val_pred_internal), np.max(val_pred_internal))
    plt.tight_layout()
    plt.savefig(f'images/{model_name}_Residual.{name}.svg', transparent=True)

train_val_test = ['X_tr', 'X_val', 'X_test', 'y_tr', 'y_val', 'y_test']
k = 22


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

lasso_model = LassoCV(alphas=alphavec, cv=50, tol=.000000000001, max_iter=5000)
lasso_model.fit(X_scaled, y_tr)
print('The best value for CV \u03BB in a Lasso is {}, and CV == test: {}'.format(lasso_model.alpha_, lasso_model.alpha_ == alphalist[np.argmin(err_vec_val)]))

val_predicted = lasso_model.predict(X_val_scaled)
resid = y_val - val_predicted

build_graph(resid=resid, val_pred_internal=val_predicted, y_val=y_val, model_name='Lasso',mod=lasso_model)

#Ridge Test
alphalist = 10 ** (np.linspace(-2, 2, 200))
err_vec_val = np.zeros(len(alphalist))  # pre allocation is faster than appending
err_vec_train = np.zeros(len(alphalist))
for i, curr_alpha in enumerate(alphalist):
    # note the use of a new sklearn utility: Pipeline to pack
    # multiple modeling steps into one fitting process
    steps = [('standardize', StandardScaler()),
             ('ridge', Ridge(alpha=curr_alpha))]
    pipe = Pipeline(steps)
    pipe.fit(X_tr.values, y_tr)
    val_set_pred = pipe.predict(X_val.values)
    err_vec_val[i] = mse(y_val, val_set_pred)
plt.plot(np.log10(alphalist), err_vec_val)
plt.savefig('images/Ridge_Test.svg')
print('The best value for test \u03BB in a Ridge is {}'.format(alphalist[np.argmin(err_vec_val)]))

#Ridge CV
alphavec = 10 ** np.linspace(-2, 2, 200)

ridge_model = RidgeCV(alphas=alphavec, cv=10)

ridge_model.fit(X_scaled, y_tr)
print('The best value for CV \u03BB in a Ridge is {}, and CV == test: {}'.format(ridge_model.alpha_, ridge_model.alpha_ == alphalist[np.argmin(err_vec_val)]))

val_predicted = ridge_model.predict(X_val_scaled)
resid = y_val - val_predicted

build_graph(resid=resid, val_pred_internal=val_predicted, y_val=y_val, model_name='Ridge',mod=ridge_model)

# Scaled Coefficients
# [('Assessed_Value', 642090.3065471586),
#  ('BG^2', -14207.056660080996),
#  ('BC^2', -5208.552177667601),
#  ('Stories', -2235.653642116267),
#  ('Living_units', 15946.370330110187),
#  ('Above_grade_living_area', -22222.477622288512),
#  ('Below_grade_living_area', -7434.095015982243),
#  ('Total_basement', 6336.554021618317),
#  ('Finished_basement', -5169.566954538874),
#  ('Sq_ft_lot', 16062.115665920513),
#  ('Topography', -893.3418014063818),
#  ('Environmental', -419.54935799268833),
#  ('Nuisances', -4001.854694924439),
#  ('Building_Age', 8516.435194134543)]




def lasso_path(name='test'):
    from sklearn.linear_model import lars_path
    print("Computing regularization path using the LARS ...")
    alphas, _, coefs = lars_path(X_scaled, y_tr.values, method='lasso')
    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]
    plt.figure(figsize=(10, 10))
    plt.plot(xx, coefs.T)
    ymin, ymax = plt.ylim()
    plt.vlines(xx, ymin, ymax, linestyle='dashed', colors='grey')
    plt.xlabel('|coef| / max|coef|', color='w')
    plt.ylabel('Coefficients', color='w')
    plt.title('LASSO Path')
    plt.axis('tight')
    plt.legend(X_tr.columns)
    plt.yticks(c='w')
    plt.xticks(c='w')
    plt.tight_layout()
    plt.savefig(f'images/Lasso_Path{name}.svg', transparent=True)

