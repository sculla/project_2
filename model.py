#!/anaconda3/envs/metis/bin/python

# log values so far... sales_price, assessed_value, sq_ft_lot
#(0.7723763538844225, 6) ALL OTHER
#[0.4115104767372036, 3] UNIQUE

#if __name__ == '__main__':

import pandas as pd
import pickle
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


# train_val_test = ['X_tr', 'X_val', 'X_test', 'y_tr', 'y_val', 'y_test']
#
# k=16
#
# for na in train_val_test:
#     with open(f'data/{na}.{k}.pickle', 'rb') as f:
#         globals()[na] = pickle.load(f)
#
# #x feature fixing
# for nam in ['X_tr', 'X_val', 'X_test']:
#     globals()[nam]['Sq_ft_lot'] = np.log(globals()[nam]['Sq_ft_lot'])



#Loading Data

homes = pd.read_pickle(f'data/.2018_house_data_frame.pickle')

X, y = homes.drop('Sale_price',axis=1), homes['Sale_price']
X_tr, X_val_test, y_tr, y_val_test = train_test_split(X, y, test_size=.25)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=.5)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_tr.values)
X_val_scaled = scaler.transform(X_val.values)
X_test_scaled = scaler.transform(X_test.values)

# #Lasso test
# alphalist = 10 ** (np.linspace(0, 5, 200))
# err_vec_val = np.zeros(len(alphalist))  # pre allocation is faster than appending
# err_vec_train = np.zeros(len(alphalist))
#
# for i, curr_alpha in enumerate(alphalist):
#     # note the use of a new sklearn utility: Pipeline to pack
#     # multiple modeling steps into one fitting process
#     steps = [('standardize', StandardScaler()),
#              ('lasso', Lasso(alpha=curr_alpha))]
#
#     pipe = Pipeline(steps)
#     pipe.fit(X_tr.values, y_tr)
#
#     val_set_pred = pipe.predict(X_val.values)
#     err_vec_val[i] = mse(y_val, val_set_pred)
# plt.plot(np.log10(alphalist), err_vec_val)
# plt.savefig('images/Lasso_Test.svg')
#
# print('The best value for test \u03BB in a Lasso is {}'.format(alphalist[np.argmin(err_vec_val)]))
#

#Lasso CV

alphavec = 10 ** np.linspace(-5, 1, 200)

lasso_model = LassoCV(alphas=alphavec, cv=50, tol=.000000000001, max_iter=5000)
lasso_model.fit(X_scaled, y_tr)
print('The best value for CV \u03BB in a Lasso is {}, and CV == test: ??'.format(lasso_model.alpha_))

val_predicted = lasso_model.predict(X_val_scaled)
resid = y_val - val_predicted

build_graph(resid=resid, val_pred_internal=val_predicted, y_val=y_val, model_name='Lasso',mod=lasso_model)

#Ridge Test
# alphalist = 10 ** (np.linspace(0, 5, 200))
# err_vec_val = np.zeros(len(alphalist))  # pre allocation is faster than appending
# err_vec_train = np.zeros(len(alphalist))
# for i, curr_alpha in enumerate(alphalist):
#     # note the use of a new sklearn utility: Pipeline to pack
#     # multiple modeling steps into one fitting process
#     steps = [('standardize', StandardScaler()),
#              ('ridge', Ridge(alpha=curr_alpha))]
#     pipe = Pipeline(steps)
#     pipe.fit(X_tr.values, y_tr)
#     val_set_pred = pipe.predict(X_val.values)
#     err_vec_val[i] = mse(y_val, val_set_pred)
# plt.plot(np.log10(alphalist), err_vec_val)
# plt.savefig('images/Ridge_Test.svg')
# print('The best value for test \u03BB in a Ridge is {}'.format(alphalist[np.argmin(err_vec_val)]))

#Ridge CV

alphavec = 10 ** np.linspace(0, 2, 200)

ridge_model = RidgeCV(alphas=alphavec, cv=50)

ridge_model.fit(X_scaled, y_tr)
print('The best value for CV \u03BB in a Ridge is {}, and CV == test: ??'.format(ridge_model.alpha_))

val_predicted = ridge_model.predict(X_val_scaled)
resid = y_val - val_predicted

build_graph(resid=resid, val_pred_internal=val_predicted, y_val=y_val, model_name='Ridge',mod=ridge_model)


ridge_model.score(X_test_scaled,y_test)
lasso_model.score(X_test_scaled,y_test)


# Scaled Coefficients
# [('Assessed_Value', 0.5241055124342983),
#  ('BG^2', 0.003825368032531888),
#  ('BC^2', 0.003915232924695901),
#  ('Stories', -0.00020517454579227362),
#  ('Living_units', 0.00045125888528731997),
#  ('Above_grade_living_area', -0.0060953623144073404),
#  ('Sq_ft_lot', -1.4846575922853973e-05),
#  ('Topography', 0.0009098444043948711),
#  ('Environmental', 0.004875474030402848),
#  ('Nuisances', -0.003437971930275706),
#  ('Building_Age', 0.013027452736148753),
#  ('Lake_Washington_AVERAGE', 0.002137089944589778),
#  ('Lake_Washington_EXCELLENT', 0.005139604827937412),
#  ('Lake_Washington_FAIR', 0.003545991084165392),
#  ('Lake_Washington_GOOD', 0.0006937466551024443),
#  ('Puget_Sound_AVERAGE', -0.005179857217334168),
#  ('Puget_Sound_EXCELLENT', -0.002402961671907304),
#  ('Puget_Sound_FAIR', -0.00213469525360545),
#  ('Puget_Sound_GOOD', -9.607150769138991e-05),
#  ('Lake_Sammamish_AVERAGE', 0.000762859026248759),
#  ('Lake_Sammamish_EXCELLENT', -0.005164491722961763),
#  ('Lake_Sammamish_FAIR', -0.00039276853477181834),
#  ('Lake_Sammamish_GOOD', 0.0013860217905181992),
#  ('Small_Lake/River_AVERAGE', -0.002749309170468741),
#  ('Small_Lake/River_EXCELLENT', -0.0028947327938783777),
#  ('Small_Lake/River_GOOD', -0.004339248038981542),
#  ('Seattle_Skyline_AVERAGE', -0.001538703537774605),
#  ('Seattle_Skyline_EXCELLENT', 0.0018081056781539524),
#  ('Seattle_Skyline_GOOD', 0.0012554156965545649),
#  ('Mt._Rainier_AVERAGE', -0.0021589242316619795),
#  ('Mt._Rainier_EXCELLENT', -0.0007988814536543745),
#  ('Mt._Rainier_GOOD', -0.001362080207977857),
#  ('Olympics_Mt._AVERAGE', 0.004362819126885326),
#  ('Olympics_Mt._EXCELLENT', 0.00036429182395798825),
#  ('Olympics_Mt._GOOD', 0.0008856555499238446),
#  ('Cascades_Mt._AVERAGE', 0.0004192962974472708),
#  ('Cascades_Mt._EXCELLENT', 0.0018845636243585502),
#  ('Cascades_Mt._GOOD', 0.0012647358932003818),
#  ('Other_view_AVERAGE', -0.0035394989111668077),
#  ('Other_view_EXCELLENT', -0.0010675075179140895),
#  ('Other_view_GOOD', 0.0003953782369497768)]
