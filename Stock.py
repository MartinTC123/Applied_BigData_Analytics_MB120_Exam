import yfinance as yf
import pandas as pd
import inspect
import datetime as dt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class Stock_analysis(): # This class with its function will serve as a framework for this assignment.

    # Creating a constructor that will treat inputs when creating an instance of the Stock_analysis() class.
    def __init__(self, input_tickers, start_date=None, end_date=dt.date.today(), interval="1d", main_feature="Adj Close", indicators=None, selected_models=None):
        # check if input is a single stock (string).
        if isinstance(input_tickers, str):
            self.input_tickers = input_tickers.split(" ")
            self.input_tickers_string = input_tickers
        # check if input is a list of stocks.
        elif pd.api.types.is_list_like(input_tickers):
            self.input_tickers = input_tickers
            self.input_tickers_string = " ".join(input_tickers)

        # sets the selected interval (1d is default if there are no input).
        self.interval = interval
        # sets the main feature (column) to try and predict ("Adj Close" is default if there are no input).
        self.main_feature = main_feature
 
        # if the input of indicators=None (empty), use default list below. 
        # List below contains acceptable indicators for this assignment.
        if indicators is None:
            self.indicators = ["MA5", "MA20", "MA50", "MA200", "MIN", "MAX", "LOG_RET", "MOM", "VOLA", "DIFF"]
        elif  pd.api.types.is_list_like(indicators) | isinstance(indicators, str):
            self.indicators = indicators

        # default error metrics used in this assignment.
        # SE OVER OM DET ER DETTE VI SKAL BRUKE
        self.metric_names = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error", "neg_mean_absolute_percentage_error"]
        
        # Prettier version of scoring names for printing, error is appended by print method.
        # SE OVER OM DET ER DETTE VI SKAL BRUKE
        self.metric_names_printing = {"r2":"R-Squared", "neg_mean_absolute_error":"Mean Absolute Error", "neg_root_mean_squared_error":"Root Mean Squared Error", "neg_mean_absolute_percentage_error":"Mean Absolute Percentage Error"}
        
        # Getting the selected models
        if selected_models is None:
            self.selected_models = {}
            all_models = dir(self)
            model_methods = [model_name for model_name in all_models if model_name.startswith("model_")]
            for method_name in model_methods:
                self.selected_models[method_name] = getattr(self, method_name)
        elif isinstance(selected_models, dict):
            self.selected_models = selected_models

        # SE OVER OM DET ER VERDT Å BRUKE DE 4 DICT NEDENFOR
        # prepare dict for train/test split indexes per ticker.
        #self.all_splits = {}
        # prepare dict for cross validation results per ticker.
        self.cv_results = {}
        # prepare dict for trained models per ticker.
        self.trained_models = {}
        # prepare dict for error metrics per ticker.
        self.metrics_df = {}

        # download given stock data and add them to a dict.
        self.stock_data = {}
        for ticker in self.input_tickers:
            print(f"Downloading {ticker} data")
            raw_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            self.stock_data[ticker] = raw_data

        # prepare dict for stock prediction data.
        self.stock_predictions = {}
        
        print("Downloading stock data is completed")
    
    # Preprocessing - Adding indicator columns.
    def add_indicator_columns(self, data):
        # Creating label and shifting the selected main_feature value by 1.
        label_name = "Label"
        self.label_name = label_name
        data[label_name] = data[self.main_feature].shift(periods=1)

        # Checking which of the different indicators that should be added as a column (based on input to self.indicators list).
        if "MA5" in self.indicators:
            data["MA5"] = data[label_name].rolling(5).mean()
        if "MA20" in self.indicators:
            data["MA20"] = data[label_name].rolling(20).mean()
        if "MA50" in self.indicators:
            data["MA50"] = data[label_name].rolling(50).mean()
        if "MA200" in self.indicators:
            data["MA200"] = data[label_name].rolling(200).mean()
        if "MIN" in self.indicators:
            data["MIN"] = data[label_name].rolling(20).min()
        if "MAX" in self.indicators:
            data["MAX"] = data[label_name].rolling(20).max()
        log_ret = np.log(data[label_name] / data[label_name].shift(1))
        if "LOG_RET" in self.indicators:
            data["LOG_RET"] = log_ret
        if "MOM" in self.indicators:
            data["MOM"] = log_ret.rolling(20).mean()
        if "VOLA" in self.indicators:
            data["VOLA"] = log_ret.rolling(20).std()
        if "DIFF" in self.indicators:
            data["DIFF"] = data[label_name] - data[label_name].shift(1)

        # remove empty vals.
        data.dropna(axis=0, inplace=True)
        
    # Preprocessing - Creating X and y arrays (used to for instance create train and test sets).
    def create_X_y_arrays(self, data):
        # array that contains the indicators data.
        X = data.loc[:, self.indicators].to_numpy()
        # array with the target data (based on main_feature).
        y = data[self.label_name].to_numpy()
        return X, y

    # Preprocessing - Uses the X and y arrays to create train and test splits.
    def create_X_y_train_test_split(self, X, y):
        data = self.stock_data[self.current_stock]

        n_splits = 5
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

        for train_index, test_index in self.tscv.split(data):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return X_train, X_test, y_train, y_test
    
    # Preprocessing, training & evaluating the chosen models for this assignment.
    def train_models(self, input_models):
        for ticker, data in self.stock_data.items():
            # Setting the ticker to current stock.
            self.current_stock = ticker

            # Adding indicators.
            self.add_indicator_columns(data=data)            

            # Creating X and y arrays for train and test sets.
            X, y = self.create_X_y_arrays(data=data)

            # Creating training and test splits.
            X_train, X_test, y_train, y_test = self.create_X_y_train_test_split(X=X, y=y)

            # Evaluating and training selected models.
            for model_i in input_models:
                model = self.selected_models["model_"+model_i]()
                metric_dict = {}
                for metric_name in self.metric_names:
                    metric_dict[metric_name] = metric_name
                    # using method from sci-kit lib to cross-validate
                    cross_val_results = cross_validate(
                        model,
                        X,
                        y,
                        cv=self.tscv,
                        scoring=metric_dict,
                        return_train_score=True,
                        n_jobs=-1,
                        verbose=0  
                    )
                    model.fit(X_train, y_train)
                self.cv_results[ticker+"_model_"+model_i] = cross_val_results 
                self.trained_models["trained_model_"+model_i+"_"+ticker] = model

        return self.cv_results
    
    def predict_trained_models(self, trained_models):
        for ticker, data in self.stock_data.items():
            # Setting the ticker to current stock.
            self.current_stock = ticker

            # Creating X and y arrays for train and test sets.
            X, y = self.create_X_y_arrays(data=data)

            last_train_index, last_test_index = None, None

            for train_index, test_index in self.tscv.split(data):
                last_train_index, last_test_index = train_index, test_index

            X_train, X_test, y_train, y_test = self.create_X_y_train_test_split(X=X, y=y)

            prediction = data.loc[data.index[last_test_index], [self.main_feature]].copy(deep=True)
            self.stock_predictions[ticker] = prediction

            for model_i in trained_models:
                self.current_model = model_i
                model = self.trained_models["trained_model_"+model_i+"_"+ticker]
                y_pred = model.predict(X_test)
                prediction.loc[:, model_i+" Prediction"] = y_pred
                mse = mean_squared_error(y_test, y_pred)
                print(f"--------{ticker} {model_i}--------")
                print(f"Mean squared error for: ",mse)
                r2 = r2_score(y_test, y_pred)
                print(f"R-squared score for: ", r2)
                mae = mean_absolute_error(y_test, y_pred)
                print(f"Mean absolute error: ", mae)
        
        return self.stock_predictions


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