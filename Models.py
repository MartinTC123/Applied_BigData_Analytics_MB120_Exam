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
        model = DecisionTreeRegressor()
        return model

    def MLP(self): 
        # Might tweak on the parameters to see different results.
        # Added max_iter as a parameter to reduce the runtime.
        model = MLPRegressor(max_iter=100)
        return model

    def XGBoost(self): 
        model = XGBRegressor()
        return model

    def XGBoost_LR(self): 
        # Used "gblinear" as booster param to have Linear Regression as the estimator.
        model = XGBRegressor(booster="gblinear")
        return model
    
    def ADA(self): 
        # Left on default. Estimator is DTR.
        model = AdaBoostRegressor()
        return model

    def ADA_LR(self): 
        # Takes Linear Regression as estimator to compare results against default version.
        # If "estimator=" gives error when running def train_models in notebook, change "estimator=" to "base_estimator="" 
        model = AdaBoostRegressor(estimator=self.LR())
        return model
    
    def GBR(self): 
        # Left on default. Estimator is DTR.
        model = GradientBoostingRegressor()
        return model
    
    def Bagging(self): 
        # Left on default. Estimator is DTR.
        model = BaggingRegressor()
        return model

    def Bagging_LR(self):
        # Takes Linear Regression as estimator to compare results against default version.
        # If "estimator=" gives error when running def train_models in notebook, change "estimator=" to "base_estimator="" 
        model = BaggingRegressor(estimator=self.LR())
        return model

    def Bagging_MLP(self):
        # Takes MLP as estimator to compare results against default version.
        # If "estimator=" gives error when running def train_models in notebook, change "estimator=" to "base_estimator="" 
        model = BaggingRegressor(estimator=self.MLP())
        return model

    
    # Stacking
    # final_estimator not specified due to good performance without it. (RidgeCV is default. Source: Scikit-learn)
    def Stacking(self): 
        estimators = []
        estimators.append(("LR", self.LR()))
        estimators.append(("DTR", self.DTR()))
        estimators.append(("MLP", self.MLP()))
        estimators.append(("XGBoost", self.XGBoost()))
        estimators.append(("ADA", self.ADA()))
        estimators.append(("ADA_LR", self.ADA_LR()))
        estimators.append(("GBR", self.GBR()))
        estimators.append(("Bagging", self.Bagging()))
        estimators.append(("Bagging_LR", self.Bagging_LR()))
        estimators.append(("Bagging_MLP", self.Bagging_MLP()))

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
        elif model == "ADA_LR":
            return self.ADA_LR()
        elif model == "GBR":
            return self.GBR()
        elif model == "Bagging":
            return self.Bagging()
        elif model == "Bagging_LR":
            return self.Bagging_LR()
        elif model == "Bagging_MLP":
            return self.Bagging_MLP()
        elif model == "StackedRegressor":
            return self.Stacking()