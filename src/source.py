import logging
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.regression 
import pandas
import numpy
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics

logger = logging.getLogger('DATA POSTPROCESSING')

def log_basic_info(data_frame, title):
    log_message = '\n' + '-' * 60
    log_message += '\n' + title + ' basic info:'
    log_message += '\n' + '-' * 60
    log_message += '\n' + str(data_frame.describe())
    log_message += '\n' + '-' * 60
    logger.info(log_message)
    
def calculate_vifs(predictor):
    # For each column, calculate VIF and add it to the DataFrame
    vifs = pandas.DataFrame()
    vifs["VIF"] = [variance_inflation_factor(predictor.values, i) for i in range(predictor.shape[1])]
    return vifs

def get_R2pred(predictor, target, n_folds = 30):
    model = linear_model.LinearRegression()
    scores = model_selection.cross_val_score(model, predictor, target, cv = n_folds)
    return scores.mean()

def get_OLS_result(predictor, target, test_size = 0.2):
    predictor_train, predictor_test, target_train, target_test = model_selection.train_test_split(predictor, target, test_size=test_size, random_state=None)
    model = statsmodels.regression.linear_model.OLS(target_train, predictor_train)
    results = model.fit()
    return results.summary()
    