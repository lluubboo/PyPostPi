import src.io_utilities as io_utilities
import src.tools as tools
import src.visualize as visualize
import src.source as source
import pandas
import statsmodels.api as statistics
import scipy

# ************************************************** Set up logging ******************************************************
logger = io_utilities.init_logger()
logger.info("Data postprocessing pipeline started...")

# *************************************************** Select files *******************************************************
result_file = io_utilities.select_file()
logger.info("Result file selected: " + result_file)

predictors_file = io_utilities.select_file()
logger.info("Predictors file selected: " + predictors_file)

target_file = io_utilities.select_file()
logger.info("Target file selected: " + target_file)

# ************************************************* Create dataframes ***************************************************
regression_result = pandas.read_csv(result_file, sep = io_utilities.guess_delimiter(result_file))
# TODO: remove last row
regression_result = regression_result.drop(regression_result.index[-1])
logger.info("Regression result data frame created.")

predictors = pandas.read_csv(predictors_file, sep = io_utilities.guess_delimiter(predictors_file))
logger.info("Predictors data frame created.")

target = pandas.read_csv(target_file, sep = io_utilities.guess_delimiter(target_file))
logger.info("Predictors data frame created.")

# elastic net regression
logger.info("Elastic net regression started...")
elastic_net = tools.elastic_net_regression(predictors, target)

# ****************************************************** Basic info ********************************************************
source.log_basic_info(regression_result, "Regression result")
# ***************************************************** Correlation ********************************************************
logger.info('\n' + str(predictors.corr()))
# ************************************************  Residuals analyses  ****************************************************
visualize.residuals_plot(regression_result.iloc[:, 0], regression_result.iloc[:, 2])
# residuals independence test
logger.info("Residuals independence test:")
durbin_watson_value = statistics.stats.stattools.durbin_watson(regression_result.iloc[:, 2])
logger.info("Durbin-Watson test: " + str(durbin_watson_value))
# ************************************************ Residuals normality *****************************************************
# normality test
logger.info("Normality tests:")
jb_value, p_value, skewness, kurtosis = statistics.stats.stattools.jarque_bera(regression_result.iloc[:, 2])
k2, p_value = scipy.stats.shapiro(regression_result.iloc[:, 2])
result = scipy.stats.anderson(regression_result.iloc[:, 2])
logger.info("Jarque-Bera test: " + str(jb_value) + " p-value: " + str(p_value) + " skewness: " + str(skewness) + " kurtosis: " + str(kurtosis))
logger.info("Shapiro-Wilk test: " + str(k2) + " p-value: " + str(p_value))
logger.info("Anderson-Darling test: " + str(result))
# ************************************************* Heteroscedasticity *****************************************************
# qq plot
visualize.qq_residuals_plot(regression_result.iloc[:, 2])
# Heteroscedasticity tests
logger.info("Heteroscedasticity tests:")
test = statistics.stats.het_breuschpagan(regression_result.iloc[:, 2], predictors)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
logger.info("Breusch-Pagan test: " + str(dict(zip(labels, test))))
