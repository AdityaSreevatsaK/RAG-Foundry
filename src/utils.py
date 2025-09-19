import logging
import time
from functools import wraps


def setup_logger(name: str = "AppLogger"):
    """
    Configures and returns a logger with a specified name.

    Args:
        name (str): The name of the logger. Defaults to "AppLogger".

    Returns:
        logging.Logger: A configured logger instance with a console handler and a specific format.
    """
    # Get a logger instance with the specified name
    logger = logging.getLogger(name)
    # Set the minimum level of messages to handle (DEBUG is the lowest)
    logger.setLevel(logging.DEBUG)

    # This check prevents adding duplicate handlers if the function is called multiple times
    if not logger.handlers:
        # Create a handler to send log messages to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Create a formatter to define the structure of the log messages
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        # Apply the formatter to the handler
        console_handler.setFormatter(formatter)

        # Add the configured handler to the logger
        logger.addHandler(console_handler)

    # Return the configured logger instance
    return logger


def track_time(wrapped_function):
    """
    Decorator that logs the execution time of the decorated function.

    Args:
        wrapped_function (Callable): The function whose execution time is to be measured.

    Returns:
        Callable: A wrapper function that logs the time taken for execution and returns the result of the original function.
    """
    log = setup_logger("Track Time Decorator")

    @wraps(wrapped_function)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = wrapped_function(*args, **kwargs)
        end_time = time.time()
        log.info(f"Time taken for {wrapped_function.__name__} function execution: {end_time - start_time:.4f} seconds.")
        return result

    return wrapper
