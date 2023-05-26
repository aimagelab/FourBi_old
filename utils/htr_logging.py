import logging
import sys

DEBUG = False

get_trace = getattr(sys, 'gettrace', None)
if get_trace():
    print('Program runs in Debug mode')
    DEBUG = True


def get_logger(name: str):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

    formatter = logging.Formatter(fmt='[%(asctime)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG if DEBUG else logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    return logger
