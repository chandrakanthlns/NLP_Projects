import functools
import time
import logging
import sys
from os import path, makedirs
import json


def readWriteJson(file, json_file):
    """
    This function writes and reads json file from coreNLP annotator method

    """
    json_folder= '../data'
    json_file= json_file + ".json"
    if not path.exists(json_folder):
        makedirs(json_folder, exist_ok=True)

    file_path = path.join(json_folder, json_file)
    
    with open(file_path, 'w', encoding='utf-8-sig') as filehandle:
        json.dump(file, filehandle, ensure_ascii=False)

    with open(file_path, encoding='utf-8-sig') as fh:
        data = json.load(fh)

    return data

def create_logger():
    """
    Creates a logging object and returns it
    """
    logger = logging.getLogger("example_logger")
    logger.setLevel(logging.INFO)
    # create the logging file handler
    fh = logging.FileHandler("../logs/test.log")
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    # add handler to logger object
    logger.addHandler(fh)
    return logger


def exception(function):
    """
    A decorator that wraps the passed in function and logs 
    exceptions should one occur
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        logger = create_logger()
        try:
            return function(*args, **kwargs)
        except Exception as e:
            # log the exception
            err = "There was an exception in  "
            err += function.__name__
            logger.exception(err)
            # re-raise the exception
            return e
    return wrapper


def timer(func):
    """Print the runtime of the decorated function"""
    
    @functools.wraps(func)
    def wrapper_time(*args,**kwargs):
        start_time = time.perf_counter()
        value = func(*args,**kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        
        return value
    return wrapper_time