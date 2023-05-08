from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor, StackingRegressor
from sklearn.model_selection import GridSearchCV


class ML_Models():
    def __init__(self):
        pass

    def LR(self): # Basert på plot og prediction så ser denne grei ut.
        model = LinearRegression()
        return model

    def DTR(self): # Må kanskje tweakes litt på.
        # Might tweak on the parameters to see different results. Left on default for now.
        model = DecisionTreeRegressor(criterion="friedman_mse")
        return model

    def MLP(self): # Basert på plot og prediction så ser denne grei ut.
        # Might tweak on the parameters to see different results.
        model = MLPRegressor()
        return model

    def XGBoost(self): # Basert på plot og prediction så må denne tweakes. Funker kun for TEL.OL
        # Might tweak on the parameters to see different results. Left on default for now. 
        model = XGBRegressor()
        return model

    def XGBoost_LR(self): # Basert på plot og prediction så ser denne grei ut.
        # Used "gblinear" as booster param to have Linear Regression as the base learner.
        model = XGBRegressor(booster="gblinear")
        return model
    
    def ADA(self): # Basert på plot og prediction så må denne tweakes. Funker kun for TEL.OL
        # Might tweak on the parameters to see different results.
        model = AdaBoostRegressor()
        return model
    
    def GBR(self): # Basert på plot og prediction så må denne tweakes. Funker kun for TEL.OL
        # Might tweak on the parameters to see different results. Left on default for now.
        model = GradientBoostingRegressor()
        return model
    
    def Bagging(self): # Basert på plot og prediction så må denne tweakes. Funker kun for TEL.OL
        model = BaggingRegressor()
        return model
    
    # Stacking
    # Named SER to be recognized by def cross_evaluate from sklearn.
    def SER(self): # Basert på plot og prediction så ser denne grei ut.
        estimators = []
        estimators.append(("LR", self.LR()))
        estimators.append(("DTR", self.DTR()))
        estimators.append(("MLP", self.MLP()))
        estimators.append(("XGBoost", self.XGBoost()))
        estimators.append(("ADA", self.ADA()))
        estimators.append(("GRB", self.GBR()))
        estimators.append(("Bagging", self.Bagging()))

        model = StackingRegressor(estimators=estimators)
        return model  

    # Helper function that choose model from models class
    def pick_model(self, model):
        if model == "LR":
            return self.LR()
        elif model == "DTR":
            return self.DTR()
        elif model == "MLP":
            return self.MLP()
        elif model == "XGBoost":
            return self.XGBoost()
        elif model == "XGBoost_LR":
            return self.XGBoost_LR()
        elif model == "ADA":
            return self.ADA()
        elif model == "GBR":
            return self.GBR()
        elif model == "Bagging":
            return self.Bagging()
        elif model == "StackedRegressor":
            return self.SER()