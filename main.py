import pandas
import io_utilities
import post_processing
import statsmodels.api as statistics
import scipy

# Set up logging
logger = io_utilities.init_logger()

logger.info("Data postprocessing pipeline started...")

# select file
file = io_utilities.select_file()
logger.info("File selected: " + file)

# create dataframe
data_frame = pandas.read_csv(file, sep = io_utilities.guess_delimiter(file))
logger.info("Data frame created.")

# basic info
logger.info('\n' + str(data_frame.describe()))

# residuals plot
post_processing.residuals_plot(data_frame.iloc[:, 0], data_frame.iloc[:, 2])

# normality test
jb_value, p_value, skewness, kurtosis = statistics.stats.stattools.jarque_bera(data_frame.iloc[:, 2])
k2, p_value = scipy.stats.shapiro(data_frame.iloc[:, 2])
result = scipy.stats.anderson(data_frame.iloc[:, 2])
logger.info("Jarque-Bera test: " + str(jb_value) + " p-value: " + str(p_value) + " skewness: " + str(skewness) + " kurtosis: " + str(kurtosis))
logger.info("Shapiro-Wilk test: " + str(k2) + " p-value: " + str(p_value))
logger.info("Anderson-Darling test: " + str(result))