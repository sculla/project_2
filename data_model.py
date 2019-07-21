#!/anaconda3/envs/metis/bin/python

# log values so far... sales_price, assessed_value, sq_ft_lot
# age home 1/x

"""
This module is for the modeling, graphing, and testing of the data.
It runs both LassoCV and RidgeCV.
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sns.set_style('darkgrid')


def rat_test(col, val, model):
    """Ratio test for incrementing values"""
    X = homes.drop(['Sale_price', 'Address'], axis=1)
    x2 = X.copy()
    x2[col] = 0
    x3 = X.copy()
    x3[col] = val
    thing2 = np.median(model.predict(x2.values))
    thing3 = np.median(model.predict(x3.values))
    return (np.exp(thing3 - thing2) - 1) * 100


def build_graph(resid, val_pred_internal, y_val, model_return, name='test', model_name='N/A'):
    """Graph builder"""
    # CV Model Graph of resids
    f, ax = plt.subplots()

    ax.scatter(val_pred_internal, resid, alpha=.2, color='#33C2FF')

    ax.set_title('Residuals of CV {} Model \u03BB:{:2f}, R^2 Score: {}'.format(model_name, model_return.alpha_,
                                                                               round(r2_score(y_val, val_pred_internal),
                                                                                     3)), fontsize=14, color='w')
    # ax.set_ylim(resid.std()*-3, resid.std()*3)
    ax.set_ylabel(f'ln(y) - ln(\u0177) Residual: \u0305x :{resid.mean():.2f}, \u03C3:{resid.std():.2f}', color='w',
                  fontsize=12)
    ax.set_xlabel('ln(y): Test Set Sales Price', color='w', fontsize=14)
    ax.set_yscale('linear')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_xscale('linear')

    ##log axis before transformation of y to log(y)
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


def lasso_path(X_scaled, y_tr, columns, name='test', ):
    """Lasso Path"""
    from sklearn.linear_model import lars_path
    print("Computing regularization path using the LARS ...")
    _, _, coefs = lars_path(X_scaled, y_tr.values, method='lasso')
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
    plt.legend(columns)
    plt.yticks(c='w')
    plt.xticks(c='w')
    plt.tight_layout()
    plt.savefig(f'images/Lasso_Path.{name}.svg', transparent=True)


# Loading Data

homes = pd.read_pickle(f'data/.2018_house_data_frame.pickle')

scaler = StandardScaler()

results = dict()


def run_tests(row=''):
    """Main Cross validation for the data"""

    X, y = homes.drop(['Sale_price'], axis=1), homes['Sale_price']

    # CV set

    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=.2)

    # pulling addresses

    X_tr, X_test, addr = X_tr.drop(['Address'], axis=1), X_test.drop(['Address'], axis=1), X_test['Address']

    # single run set
    # X_tr, X_val_test, y_tr, y_val_test = train_test_split(X, y, test_size=.25)
    # X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=.5)
    X_scaled = scaler.fit_transform(X_tr.values)
    # X_val_scaled = scaler.transform(X_val.values)
    X_test_scaled = scaler.transform(X_test.values)

    alphavec = 10 ** np.linspace(-5, 1, 200)

    lasso_model = LassoCV(alphas=alphavec, cv=50, tol=.000000000001, max_iter=5000)
    lasso_model.fit(X_scaled, y_tr)
    print('The best value for CV \u03BB in a Lasso is {}, and CV == test: ??'.format(lasso_model.alpha_))

    test_predicted = lasso_model.predict(X_test)
    resid = y_test - test_predicted

    build_graph(name=row, resid=resid, val_pred_internal=test_predicted, y_val=y_test, model_name='Lasso',
                model_return=lasso_model)

    # Ridge CV

    alphavec = 10 ** np.linspace(0, 2, 200)

    ridge_model = RidgeCV(alphas=alphavec, cv=50)

    ridge_model.fit(X_scaled, y_tr)
    print('The best value for CV \u03BB in a Ridge is {}, and CV == test: ??'.format(ridge_model.alpha_))

    test_predicted = ridge_model.predict(X_test_scaled)
    resid = y_test - test_predicted

    build_graph(name=row, resid=resid, val_pred_internal=test_predicted, y_val=y_test, model_name='Ridge',
                model_return=ridge_model)

    results[row] = [ridge_model.score(X_test_scaled, y_test), ridge_model.alpha_,
                    lasso_model.score(X_test_scaled, y_test), lasso_model.alpha_]
    print(results)
    # lasso_path(X_scaled, y_tr, X_test.columns, 'slide')
    return lasso_model, ridge_model, X_test_scaled, y_test, addr


if __name__ == '__main__':
    lasso_m, ridge_m, X_test_scaled, y_test, Addresses = run_tests()

    #err = np.exp(ridge_m.predict(X_test_scaled))
    err = ridge_m.predict(X_test_scaled) - y_test
    err_add = list(zip(err, Addresses))

    err_df = pd.DataFrame(err_add)
    err_df1 = err_df[err_df[0] > -.01]
    err_df2 = err_df1[(err_df1[0] < .01)]
    print(f'{err_df2.shape[0]}/{err_df.index.max()} or '
          f'{err_df2.shape[0] / err_df.index.max() * 100:.1f}%'
          f' of test within 1%')

    err_df = pd.DataFrame(err_add)
    err_df1 = err_df[err_df[0] > -.05]
    err_df2 = err_df1[(err_df1[0] < .05)]
    print(f'{err_df2.shape[0]}/{err_df.index.max()} or '
          f'{err_df2.shape[0] / err_df.index.max() * 100:.1f}%'
          f' of test within 5%')

    err_df1 = err_df[err_df[0] > -.1]
    err_df2 = err_df1[(err_df1[0] < .1)]
    print(f'{err_df2.shape[0]}/{err_df.index.max()} or '
          f'{err_df2.shape[0] / err_df.index.max() * 100:.1f}%'
          f' of test within 10%')
