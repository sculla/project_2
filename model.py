#!/anaconda3/envs/metis/bin/python

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.pipeline import Pipeline

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

    lasso_model = LassoCV(alphas=alphavec, cv=5, tol=.01, max_iter=50)
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
    resid = test_set_pred - y_val
    sns.scatterplot(np.arange(len(resid))[20:], resid[20:], alpha=.1, color='#33C2FF')
    plt.title('Residuals of CV Lasso Model \u03BB:100', color='w')
    plt.ylabel('Y_p - Y :Residual', color='w')
    plt.xlabel('Observations R^2 Score: {}'.format(round(r2_score(y_val, test_set_pred),3)),color='w')
    plt.xticks(color='w')
    plt.yticks(color='w')
    plt.tight_layout()
    plt.savefig('images/Lasso_Residual.svg',transparent=True)

