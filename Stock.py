import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np

class Stock_analysis():
    def __init__(self, input_tickers, start_date=None, end_date=dt.date.today(), interval="1d", main_feature="Adj Close", indicators=None):
        
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
 
        # if the input of features=None (empty), use default list below. 
        # List below contains acceptable features for this assignment.
        if indicators is None:
            self.indicators = ["MA5", "MA20", "MA50", "MA200", "MIN", "MAX", "LOGR", "MOM", "VOLA", "DIFF"]
        elif  pd.api.types.is_list_like(indicators) | isinstance(indicators, str):
            self.indicators = indicators

        # default error metrics used in this assignment.
        # SE OVER OM DET ER DETTE VI SKAL BRUKE
        self.metric_names = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error", "neg_mean_absolute_percentage_error"]
        
        # Prettier version of scoring names for printing, error is appended by print method.
        # SE OVER OM DET ER DETTE VI SKAL BRUKE
        self.metric_names_printing = {"r2":"R^2", "neg_mean_absolute_error":"Mean Absolute Error", "neg_root_mean_squared_error":"Root Mean Squared Error", "neg_mean_absolute_percentage_error":"Mean Absolute Percentage Error"}
        
        
        # default models used for analysis in this assigment.
        self.selected_models = ["FYLLE INN MODELLENE VI SKAL TESTE"]
        
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
    
    
    def add_indicator_columns(self, data):
        # Creating label and shifting the selected main_feature value by 1.
        label_name = "Label"
        self.label_name = label_name
        data[label_name] = data[self.main_feature].shift(periods=1)

        # Checking which of the different indicators that should be added as a column.
        if "MA5" in self.indicators:
            data["MA5"] = data[label_name].rolling(5).mean()
        if "MA20" in self.features:
            data["MA20"] = data[label_name].rolling(20).mean()
        if "MA50" in self.features:
            data["MA50"] = data[label_name].rolling(50).mean()
        if "MA200" in self.features:
            data["MA200"] = data[label_name].rolling(200).mean()
        if "MIN" in self.features:
            data["MIN"] = data[label_name].rolling(20).min()
        if "MAX" in self.features:
            data["MAX"] = data[label_name].rolling(20).max()
        log_r = np.log(data[label_name] / data[label_name].shift(periods=1))
        if "LOGR" in self.features:
            data["LOGR"] = log_r
        if "MOM" in self.features:
            data["MOM"] = log_r.rolling(20).mean()
        if "VOLA" in self.features:
            data["VOLA"] = log_r.rolling(20).std()
        if "DIFF" in self.features:
            data["DIFF"] = data[label_name] - data[label_name].shift(periods=1)

        # remove empty vals.
        data.dropna(axis=0, inplace=True)
        

    def create_X_y_arrays(self, data):
        # array that contains the indicators data
        X = data.loc[:, self.indicators].to_numpy()
        print("Shape of X:", X.shape())
        # array with the target data (based on main_feature)
        y = data[self.label_name].to_numpy()
        print("Shape of y:", y.shape())
        return X, y

    