# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 08:51:22 2020

@author: niket
"""
'''
    Version:3.1
    The input data must have no empty csee and should be one hot encoaded/lable encoded.
    The best possible output of variance score is 1.0, and the lower are worse(they can also come negative.)
'''
def TestSize(n):
    if n<50:
        return 0.15
    return 0.2

class MultiRegressor:
    def __init__(self,X,y,X_predict):
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TestSize(len(X)))
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        self.regressor=regressor
        self.X=X
        self.y=y
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        self.X_predict = X_predict
        self.y_predicted=regressor.predict(X_predict)
        self.y_new=regressor.predict(X_test)
    def R2score(self):
        return self.regressor.score(self.X,self.y)
    def variance(self):
        from sklearn.metrics import explained_variance_score
        return explained_variance_score(self.y_test, self.y_new, multioutput='uniform_average')
    def predict(self):
        return self.y_predicted 
    def visualise(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.X, self.y, color = 'red')
        plt.scatter(self.X_predict,self.y_predicted, color='blue')
        plt.plot(self.X_predict, self.y_predicted , color = 'black')
        plt.title('Red-> Given Points\nblue-> Predicted points\nblack->Predicted Curve')
        plt.show()
        
class SVR:
    def __init__(self,X,y,X_predict):
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        regressor = SVR(kernel = 'rbf')
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        sc_X.fit(X)
        scX=sc_X.transform(X)
        scX_predict=sc_X.transform(X_predict)
        sc_y.fit(y)
        scy=sc_y.transform(y)
        X_train, X_test, y_train, y_test = train_test_split(scX, scy, test_size = TestSize(X.shape[0]))
        regressor.fit(X_train, y_train)
        y_new=regressor.predict(X_train)
        y_predicted=regressor.predict(scX_predict)
        self.regressor=regressor
        self.sc_X=sc_X
        self.sc_y=sc_y
        self.X=X
        self.y=y
        self.y_new=y_new
        self.scy=scy
        self.X_predict=X_predict
        self.y_predicted=y_predicted
    def variance(self):
        from sklearn.metrics import explained_variance_score
        return explained_variance_score(self.scy, self.y_new,multioutput='uniform_average')
    def prediict(self):
        return self.sc_y.inverse_transform(self.y_predicted)
    def visualise(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.X, self.y, color = 'red')
        plt.scatter(self.sc_X.inverse_transform(self.X_predict),self.sc_y.inverse_transform(self.y_predicted), color='blue')
        plt.plot(self.sc_X.inverse_transform(self.X_predict),self.sc_y.inverse_transform(self.y_predicted) , color = 'black')
        plt.title('Red-> Given Points\nblue-> Predicted points\nblack->Predicted Curve')
        plt.show()
        
class Tree:
    def __init__(self,X,y,X_predict):
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TestSize(X.shape[0]))
        regressor = DecisionTreeRegressor()
        regressor.fit(X_train, y_train)
        y_new=regressor.predict(X_predict)
        y_predicted=regressor.predict(X_test)
        self.regressor=regressor
        self.X=X
        self.y=y
        self.y_test=y_test
        self.y_new=y_new
        self.y_predicted=y_predicted
    def variance(self):
        from sklearn.metrics import explained_variance_score
        return explained_variance_score(self.y_test, self.y_new,multioutput='uniform_average')
    def predict(self):
        return self.y_predicted
    def visualise(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.X, self.y, color = 'red')
        plt.scatter(self.X.inverse_transform(self.X_predict),self.y.inverse_transform(self.y_predicted), color='blue')
        plt.plot(self.X.inverse_transform(self.X_predict),self.y.inverse_transform(self.y_predicted) , color = 'black')
        plt.title('Red-> Given Points\nblue-> Predicted points\nblack->Predicted Curve')
        plt.show()

class Forest:
    def __init__(self,X,y,X_predict,n_estimated):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TestSize(X.shape[0]))
        regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
        regressor.fit(X_train, y_train)
        y_new=regressor.predict(X_test)
        y_predicted=regressor.predict(X_predict)
        self.regressor=regressor
        self.y_new=y_new
        self.y_predicted=y_predicted
        self.y_test=y_test
    def errerscore(self):
        from sklearn.metrics import explained_variance_score
        return explained_variance_score(self.y_test, self.y_new,multioutput='uniform_average')
    def predict(self):
        return self.y_predicted
    def visualise(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.X, self.y, color = 'red')
        plt.scatter(self.X.inverse_transform(self.X_predict),self.y.inverse_transform(self.y_predicted), color='blue')
        plt.plot(self.X.inverse_transform(self.X_predict),self.y.inverse_transform(self.y_predicted) , color = 'black')
        plt.title('Red-> Given Points\nblue-> Predicted points\nblack->Predicted Curve')
        plt.show()
        
class PolyRegressor:
    def __init__(self,X,y,X_predict,Degree):
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import PolynomialFeatures
        poly_reg = PolynomialFeatures(degree = Degree)
        X_poly = poly_reg.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size = TestSize(X.shape[0]))
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        self.regressor=regressor
        self.X=X
        self.y=y
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        self.X_predict = X_predict
        self.y_predicted=regressor.predict(self.X_predict)
        self.y_new=regressor.predict(X_test)
    def R2score(self):
        return self.regressor.score(self.X,self.y)
    def variance(self):
        from sklearn.metrics import explained_variance_score
        return explained_variance_score(self.y_test, self.y_new,multioutput='uniform_average')
    def predict(self):
        return self.y_predicted 
    def visualise(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.X, self.y, color = 'red')
        plt.scatter(self.X_predict,self.y_predicted, color='blue')
        plt.plot(self.X_predict, self.y_predicted , color = 'black')
        plt.title('Red-> Given Points\nblue-> Predicted points\nblack->Predicted Curve')
        plt.show()
        
        
       
    
        
    
    
    
    