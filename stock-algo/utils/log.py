#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 11:41:42 2025

@author: cjymain
"""

import logging
import traceback
from functools import wraps


def log_setup(level, file_location):
    log_format = logging.Formatter(
        '%(asctime)s FROM %(filename)s VIA %(funcName)s ON %(lineno)d [%(levelname)s]: %(message)s',
        '%Y-%m-%d %I:%M:%S %p'
    )

    # Configure boto3 and botocore to not excessively log
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)

    logger = logging.getLogger("technically")
    if level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif level == "INFO":
        logger.setLevel(logging.INFO)
    elif level == "WARNING":
        logger.setLevel(logging.WARNING)
    elif level == "ERROR":
        logger.setLevel(logging.ERROR)
    elif level == "CRITICAL":
        logger.setLevel(logging.CRITICAL)

    file_handler = logging.FileHandler(file_location, mode='w')
    file_handler.setFormatter(log_format)

    logger.addHandler(file_handler)

    return logger

def get_logger():
    logger = logging.getLogger("technically")
    return logger

def log_class(cls=None, exclude=[], critical=False):
    "Class decorator to log method calls."
    exclude.extend(["execute"])

    def decorator(cls):
        for name, method in cls.__dict__.items():
            if (callable(method) and
                    not name.startswith("__") and
                    not hasattr(method, '_is_logged') and
                    name not in exclude):
                setattr(cls, name, log_method(method, class_name=cls.__name__,
                                              critical=critical)
                )
        return cls

    if cls is None:
        return decorator
    return decorator(cls)

def log_method(func=None, class_name=None, critical=False):
    "Method decoratro that can be used directly or from log_class."
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logging.getLogger('technically')

            # Checks whether method is in a class
            if args and hasattr(args[0], '__class__'):
                real_class_name = class_name or args[0].__class__.__name__
                method_name = func.__name__
                log.debug(f'Entering {real_class_name}.{method_name}()')

                try:
                    result = func(*args, **kwargs)
                    log.debug(f'Exiting {real_class_name}.{method_name}()')
                    return result
                except Exception as e:
                    # Log exception with traceback
                    log.error(f'Exception in {real_class_name}.{method_name}(): {str(e)}')
                    log.error(f'Function parameters: {args}')
                    log.error(f'Traceback: {traceback.format_exc()}')

                    if critical:
                        raise
                    return
            else:
                log.debug(f'Entering {func.__name__}()')
                try:
                    result = func(*args, **kwargs)
                    log.debug(f'Exiting {func.__name__}()')
                    return result
                except Exception as e:
                    log.error(f'Exception in {func.__name__}(): {str(e)}')
                    log.error(f'Function parameters: {args}')
                    log.error(f'Traceback: {traceback.format_exc()}')

                    if critical:
                        raise
                    return

        wrapper._is_logged = True
        return wrapper

    if func is None:
        return decorator
    return decorator(func)
