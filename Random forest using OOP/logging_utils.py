from enum import auto, Enum
import logging


class StreamType(Enum):
    CONSOLE = auto()
    FILE = auto()


def get_handler_stream(logger, level, stream=StreamType.CONSOLE):
    info_handler = tuple(filter(lambda handler: handler.level <= level, logger.handlers))
    info_handler = info_handler[0] if len(info_handler) > 0 else None
    file = None
    if (isinstance(info_handler, logging.StreamHandler) and stream == StreamType.CONSOLE
            or isinstance(info_handler, logging.FileHandler) and stream == StreamType.FILE):
        file = info_handler.stream
    return file
