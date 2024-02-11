import pandas
import io_utilities
import post_processing

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