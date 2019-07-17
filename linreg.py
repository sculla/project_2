
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge #ordinary linear regression + w/ ridge regularization
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def lin_reg():
    import pickle
    import numpy as np
    homes = pd.read_pickle(f'data/.2018_house_data_frame.pickle')

    X, y = homes.drop('Sale_price',axis=1), homes['Sale_price']

    # train_val_test = ['X', 'X_val', 'X_test', 'y', 'y_val', 'y_test']
    # for name in train_val_test:
    #     with open(f'data/{name}.pickle', 'rb') as f:
    #         print(f'{name} loaded correctly')
    #         globals()[name] = pickle.load(f)
    old = (0,0)
    # old_2 = (0,0)
    scaler = StandardScaler()
    for alphas in [10**x for x in np.linspace(-2,2,200)]:
    for k in range(50):
        #hold out 20% of the data for validation & final testing then split
        X_tr, X_val_test, y_tr, y_val_test = train_test_split(X, y, test_size=.25)

        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=.5)

        #set up the 3 models we're choosing from:

        lm = LinearRegression()

        #Feature scaling for X, val, and test so that we can run our ridge model on each


        X_scaled = scaler.fit_transform(X_tr.values)
        X_val_scaled = scaler.transform(X_val.values)
        # X_test_scaled = scaler.transform(X_test.values)

        lm_reg = Ridge(alpha=)

        #Feature transforms for X, val, and test so that we can run our poly model on each
        # poly = PolynomialFeatures(degree=2)


        # X_poly = poly.fit_transform(X_tr.values)
        # X_val_poly = poly.transform(X_val.values)
        # # X_test_poly = poly.transform(X_test.values)

        lm_poly = LinearRegression()

        # validate
        print(f'k: {k}')
        lm.fit(X_tr, y_tr)
        print(f'Linear Regression val R^2: {lm.score(X_val, y_val):.3f}')

        lm_reg.fit(X_scaled, y_tr)
        print(f'Ridge Regression val R^2: {lm_reg.score(X_val_scaled, y_val):.3f}')
        new = [lm_reg.score(X_val_scaled, y_val),k]
        if new[0] > old[0]:
            old = new
        # lm_poly.fit(X_poly, y_tr)
        # print(f'Degree 2 polynomial regression val R^2: {lm_poly.score(X_val_poly, y_val):.3f}')
        train_na = ['X_tr', 'X_val', 'X_test', 'y_tr', 'y_val', 'y_test']
        train_val_test = [X_tr, X_val, X_test, y_tr, y_val, y_test]
        for idx, names in enumerate(train_val_test):
            with open(f'data/test/{train_na[idx]}.{pick}.{k}.pickle', 'wb') as f:
                pickle.dump(names, f)
    print(old)

#def fitting():

if __name__ == '__main__':
    lin_reg()
    # import pickle
    # k=1
    # train_val_test = ['X_tr', 'X_val', 'X_test', 'y_tr', 'y_val', 'y_test']
    # homes = pd.read_pickle('data/.2018_house_data_frame.pickle')
    # X, y = homes.drop('Sale_price',axis=1), homes['Sale_price']
    # X_tr, X_val_test, y_tr, y_val_test = train_test_split(X, y, test_size=.125, random_state=k)
    # X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=.5)
    # for na in train_val_test:
    #     with open(f'data/test/{na}.{k}.pickle', 'wb') as f:
    #         pickle.dump(globals()[na], f)



