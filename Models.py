from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

class ML_Models():
    def __init__(self):
        pass

    # MODEL
    def model_Linear(self):
        model = LinearRegression()

        return model

    # MODEL
    def model_DTR(self):
        #friedmans_mse score for potential splits provided better results than default 
        #samples required for a leaf node was found best at 5
        model = DecisionTreeRegressor(criterion="friedman_mse", min_samples_leaf = 5)

        return model

    def model_MLP(self):
        model = MLPRegressor(random_state=1, max_iter=500)

        return model
