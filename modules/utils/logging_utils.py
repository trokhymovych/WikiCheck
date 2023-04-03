from typing import *
import logging
from pathlib import Path
from datetime import datetime
import csv

from modules.utils.measurer import TimeMeasurer

DEFAULT_MEASURER = TimeMeasurer()

DEFAULT_FORMAT_PARAMS = dict(
    fmt='[{asctime}][{name}][{levelname}] {message}',
    datefmt='%Y-%m-%d %H:%M:%S',
    style='{'
)

ROOT_LOGGER_NAME = 'NLI_API'

DEFAULT_LOGGER = logging.getLogger("DEFAULT")


def get_logger(name: str,
               log_level: Union[str, int] = None,
               console: bool = False,
               log_file: Union[str, Path] = None,
               additional_handlers: bool = False,
               propagate: bool = True):
    """
    Get logger based on settings
    If root logger name is available in project scope
    and name in format f"{root_logger_name}.{some_name}"
    than logger will have all root logger handlers
    :param name: name of logger
    :param log_level: level of logging str ("INFO") or int (0, 10, 20 ..)
    :param console: log to console or not
    :param log_file: if set -> will log to specific file
    :param additional_handlers: add handlers even they exist
    :param propagate: send records to parent
    :return:
    """
    log_level = log_level or "NOTSET"
    logger = logging.getLogger(name)
    logger.propagate = propagate
    if not logger.handlers or additional_handlers:
        formatter = logging.Formatter(**DEFAULT_FORMAT_PARAMS)
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level)
            logger.addHandler(console_handler)
        logger.setLevel(log_level)
    return logger


def check_if_none(x):
    if type(x) == str:
        return x
    else:
        return ""

class CSVLogger:
    def __init__(self, config):
        date_string = datetime.today().strftime('%Y-%m-%d')
        self.file_path = f"{config['log_file_path']}logs_{date_string}.csv"
        self.fields = ["datetime", "model_name", "request", "response", "time_spend", "ip"]
        with open(self.file_path, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(self.fields)

    def add_log(self, log_attributes: dict):
        log_string = [log_attributes.get(f) for f in self.fields]
        with open(self.file_path, 'a') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(log_string)
