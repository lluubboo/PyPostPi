import logging

logger = logging.getLogger('DATA POSTPROCESSING')

def log_basic_info(data_frame, title):
    log_message = '\n' + '-' * 60
    log_message += '\n' + title + ' basic info:'
    log_message += '\n' + '-' * 60
    log_message += '\n' + str(data_frame.describe())
    log_message += '\n' + '-' * 60
    logger.info(log_message)
    