import logging
from sklearn import linear_model
from sklearn import model_selection
import statsmodels.api as sm
import pandas
import numpy

logger = logging.getLogger('DATA POSTPROCESSING')

def log_basic_info(data_frame, title):
    log_message = '\n' + '-' * 60
    log_message += '\n' + title + ' basic info:'
    log_message += '\n' + '-' * 60
    log_message += '\n' + str(data_frame.describe())
    log_message += '\n' + '-' * 60
    logger.info(log_message)

def elastic_net_regression(predictor, target, test_size = 0.2, random_state = None):
    
    # split the data into training and test sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(predictor, target, test_size = test_size, random_state = random_state)
    
    elastic_net = linear_model.LinearRegression()
    result = elastic_net.fit(X_train, y_train)
    
    model = sm.OLS(y_train, X_train)
    results = model.fit()
    
    # Make predictions on the full set
    y_pred = results.predict(sm.add_constant(predictor))
    
    y_train = 10**target.squeeze()
    y_pred = 10**y_pred.squeeze()

    # Calculate the errors
    errors = y_train - y_pred
    
    print("Summary")
    print(results.summary())
    