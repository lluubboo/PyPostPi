import src.io_utilities as io_utilities
import src.tools as tools
import src.visualize as visualize
import src.source as source
import pandas
import statsmodels.api as statistics
import scipy

# ************************************************** Set up pandas df ******************************************************

# Set pandas to display all rows and columns
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', None)
pandas.set_option('display.max_colwidth', None)

# ************************************************** Set up logging ********************************************************

logger = io_utilities.init_logger()
logger.info(tools.get_header_separator() + "INITIALIZATION:")
logger.info("Data postprocessing pipeline started...")

# *************************************************** Select files *********************************************************

result_file = io_utilities.select_file()
logger.info("Result file selected: " + result_file)

dataset_file = io_utilities.select_file()
logger.info("Dataset file selected: " + dataset_file)

# ************************************************* Create residuals dataframe *********************************************

result = pandas.read_csv(result_file, sep = io_utilities.guess_delimiter(result_file))
dataset = pandas.read_csv(dataset_file, sep = io_utilities.guess_delimiter(dataset_file))

# TODO: remove last row
result = result.drop(result.index[-1])
logger.info("Regression result data frame created with shape: " + str(result.shape))

predictor = dataset.iloc[:, :-1]
target = dataset.iloc[:, -1]

logger.info("Predictor data frame created with shape: " + str(predictor.shape))
logger.info("Target data frame created with shape: " + str(target.shape))

# ****************************************************** Basic info ********************************************************

logger.info(tools.get_header_separator() + 'C++ EVOREGR REGRESSION RESULT BASIC INFO:' + '\n' + str(result.describe()))
logger.info("OLS result: \n" + str(source.get_OLS_result(predictor, target, 0.2)))

# ************************************************** R2 prediction *********************************************************

logger.info(tools.get_header_separator() + "R2 PREDICTION:")
logger.info("R2 prediction: " + str(source.get_R2pred(predictor, target)))

# **************************************  Pearson correlation coefficient matrix *******************************************

logger.info(tools.get_header_separator() + "COVARIANCE MATRIX:")
logger.info("Covariance matrix: \n" + str(dataset.corr()))

# ************************************************  Variance inflation factor **********************************************

logger.info(tools.get_header_separator() + "VARIANCE INFLATION FACTORS:")
logger.info("VIF-s: \n" + str(source.calculate_vifs(predictor)))

# ************************************************  Residuals analyses  ****************************************************

logger.info(tools.get_header_separator() + "RESIDUALS HOMOSCEDASTICITY:")

# export residuals - y plot
visualize.residuals_plot(result.iloc[:, 0], result.iloc[:, 2])
logger.info("Residuals simple plot exported.")

# export residuals QQ plot
visualize.qq_residuals_plot(result.iloc[:, 2])
logger.info("Residuals QQ plot exported.")

# residuals autocorrelation test
logger.info(tools.get_header_separator() + "RESIDUALS AUTOCORRELATION:")
durbin_watson_value = statistics.stats.stattools.durbin_watson(result.iloc[:, 2])

interpretation = ""
if durbin_watson_value < 1.5:
    interpretation = "Positive autocorrelation"
elif durbin_watson_value > 2.5:
    interpretation = "Negative autocorrelation"
else:
    interpretation = "No autocorrelation"

logger.info(f'Durbin-Watson test value: {durbin_watson_value}, Interpretation: {interpretation}')

# ************************************************ Residuals normality *****************************************************

logger.info(tools.get_header_separator() + "RESIDUALS NORMALITY TESTS:")

significance_level = 0.05

# Jarque-Bera test
jb_value, p_value, skewness, kurtosis = statistics.stats.stattools.jarque_bera(result.iloc[:, 2])
jb_interpretation = "Jarque-Bera test: Fail to reject the null hypothesis, data has a normal distribution." if p_value > significance_level else "Jarque-Bera test: Reject the null hypothesis, data does not have a normal distribution."
logger.info(f"Jarque-Bera test: {jb_value}, p-value: {p_value}, skewness: {skewness}, kurtosis: {kurtosis}")
logger.info(f"Significance level: {significance_level}, interpretation: {jb_interpretation}")

# Shapiro-Wilk test
k2, p_value = scipy.stats.shapiro(result.iloc[:, 2])
sw_interpretation = "Shapiro-Wilk test: Fail to reject the null hypothesis, data has a normal distribution." if p_value > significance_level else "Shapiro-Wilk test: Reject the null hypothesis, data does not have a normal distribution."
logger.info(f"Shapiro-Wilk test: {k2}, p-value: {p_value}")
logger.info(f"Significance level: {significance_level}, interpretation: {sw_interpretation}")

# Anderson-Darling test
result = scipy.stats.anderson(result.iloc[:, 2])
ad_interpretation = "Anderson-Darling test: Fail to reject the null hypothesis, data has a normal distribution." if result.statistic < result.critical_values[2] else "Anderson-Darling test: Reject the null hypothesis, data does not have a normal distribution."
logger.info(f"Anderson-Darling test statistic: {result.statistic}, critical value at 5% significance level: {result.critical_values[2]}")
logger.info(f"Significance level: {significance_level}, interpretation: {ad_interpretation}")



