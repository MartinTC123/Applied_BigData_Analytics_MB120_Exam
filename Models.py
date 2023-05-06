from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor, StackingRegressor


class ML_Models():
    def __init__(self):
        pass

    def LR(self):
        model = LinearRegression()
        return model

    def DTR(self):
        # Might tweak on the parameters to see different results. Left on default for now.
        model = DecisionTreeRegressor()
        return model

    def MLP(self):
        # Might tweak on the parameters to see different results.
        model = MLPRegressor(random_state=42, max_iter=500)
        return model

    def XGBoost(self):
        # Might tweak on the parameters to see different results. Left on default for now.
        model = XGBRegressor()  
        return model

    def XGBoost_LR(self):
        # Used "gblinear" as booster param to have Linear Regression as the base learner.
        # Might tweak on the parameters to see different results.
        model = XGBRegressor(booster="gblinear")
        return model
    
    def ADA(self):
        # Might tweak on the parameters to see different results.
        model = AdaBoostRegressor(random_state=42)
        return model
    
    def GBR(self):
        # Might tweak on the parameters to see different results. Left on default for now.
        model = GradientBoostingRegressor()
        return model
    
    def Bagging(self):
        # Might tweak on the parameters to see different results. Left on default for now.
        model = BaggingRegressor()
        return model
    
    # Stacking
    def SER(self):
        estimators = []
        estimators.append(("LR", self.LR()))
        estimators.append(("DTR", self.DTR()))
        estimators.append(("MLP", self.MLP()))
        estimators.append(("XGBoost", self.XGBoost()))
        #estimators.append(("XGBoost_LR", self.XGBoost_LR)) blir ikke gjenkjent som en regressor av cross-validate.
        estimators.append(("ADA", self.ADA()))
        estimators.append(("GRB", self.GBR()))
        estimators.append(("Bagging", self.Bagging()))

        model = StackingRegressor(estimators=estimators)
        return model  

    # NOTE: Trenger vi ADA_MLP?