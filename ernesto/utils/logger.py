import logging


def setup_logger(verbose=False):
    logger = logging.getLogger("ErNESTO-DT")

    # Avoid adding multiple handlers if setup_logger is called more than once
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set the logging level
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # Custom format
    ch.setFormatter(CustomFormatter())

    # Add handler
    logger.addHandler(ch)
    
    # Optional: prevent propagation to root logger
    logger.propagate = True

    return logger


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + fmt + reset,
        logging.INFO: grey + fmt + reset,
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: bold_red + fmt + reset
    }

    def __init__(self):
        super().__init__()
        # Pre-create formatters for each level
        self._formatters = {
            level: logging.Formatter(fmt_str)
            for level, fmt_str in self.FORMATS.items()
        }

    def format(self, record):
        formatter = self._formatters.get(record.levelno, self._formatters[logging.INFO])
        return formatter.format(record)
