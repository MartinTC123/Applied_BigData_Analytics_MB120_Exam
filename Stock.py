import yfinance as yf
import pandas as pd
import inspect
import datetime as dt
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

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
            self.indicators = ["MA5", "MA20", "MA50", "MA200", "MIN", "MAX", "LOGR", "MOM", "VOLA", "DIFF"]
        elif  pd.api.types.is_list_like(indicators) | isinstance(indicators, str):
            self.indicators = indicators

        # default error metrics used in this assignment.
        # SE OVER OM DET ER DETTE VI SKAL BRUKE
        self.metric_names = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error", "neg_mean_absolute_percentage_error"]
        
        # Prettier version of scoring names for printing, error is appended by print method.
        # SE OVER OM DET ER DETTE VI SKAL BRUKE
        self.metric_names_printing = {"r2":"R-Squared", "neg_mean_absolute_error":"Mean Absolute Error", "neg_root_mean_squared_error":"Root Mean Squared Error", "neg_mean_absolute_percentage_error":"Mean Absolute Percentage Error"}
        
        
        # default models used for analysis in this assigment.
        if selected_models is None:
            self.selected_models = {}
            # get all methods in the class
            for method_name, method_object in inspect.getmembers(self, predicate=inspect.ismethod):
                # check name for correct pattern to identify models
                if method_name.startswith("model_") == True:
                    self.selected_models[method_name] = method_object
        elif isinstance(selected_models, dict):
            # save user version
            self.selected_models = selected_models
        
        # LSTM NN model that cant be evaluated using sci-kit. 
        # dict for LSTM NN model that will be evaluated using keras.
        self.keras_models = ["LSTM"]

        # SE OVER OM DET ER VERDT Å BRUKE DE 4 DICT NEDENFOR
        # prepare dict for train/test split indexes per ticker.
        self.all_splits = {}
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
        self.stock_data_predictions = {}
        
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
        log_r = np.log(data[label_name] / data[label_name].shift(1))
        if "LOGR" in self.indicators:
            data["LOGR"] = log_r
        if "MOM" in self.indicators:
            data["MOM"] = log_r.rolling(20).mean()
        if "VOLA" in self.indicators:
            data["VOLA"] = log_r.rolling(20).std()
        if "DIFF" in self.indicators:
            data["DIFF"] = data[label_name] - data[label_name].shift(1)

        # remove empty vals.
        data.dropna(axis=0, inplace=True)
        
    # Preprocessing - Creating X and y arrays (used to for instance create train and test sets).
    def create_X_y_arrays(self, data):
        # array that contains the indicators data.
        X = data.loc[:, self.indicators].to_numpy()
        print("Shape of X:", X.shape)
        # array with the target data (based on main_feature).
        y = data[self.label_name].to_numpy()
        print("Shape of y:", y.shape)
        return X, y

    # Preprocessing - Uses the X and y arrays to create train and test splits.
    def create_X_y_train_test_split(self, X, y):
        data = self.stock_data[self.current_stock]

        # Splitting the data by 0.8.
        self.split = int(len(data) * 0.8)

        # allocating 80% to training and 20% to testing.
        X_train, X_test = X[:self.split], X[self.split:]
        y_train, y_test = y[:self.split], y[self.split:]

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

            # Training and evaluating selected models.
            for model_i in input_models:
                model = self.selected_models["model_"+model_i]()
                if model_i not in self.keras_models:
                    metric_dict = {}
                    for metric_name in self.metric_names:
                        metric_dict[metric_name] = metric_name
                    cross_val_results = cross_validate(
                        model,
                        X,
                        y,
                        cv=self.split,
                        scoring=metric_dict,
                        return_train_score=True,
                        n_jobs=-1,
                        verbose=0  
                    )
                    model.fit(X_train, y_train)
                # MÅ LEGGE TIL ELSE CLAUSE DERSOM MODEL ER I keras_models
                self.cv_results["cv_results_"+model_i+"_"+ticker] = cross_val_results
                self.trained_models["trained_model_"+model_i+"_"+ticker] = model

        return self.cv_results

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