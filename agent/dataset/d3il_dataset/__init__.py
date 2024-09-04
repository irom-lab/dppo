import os

FRAMEWORK_DIR = os.path.dirname(__file__)


def sim_framework_path(*args) -> str:
    """
    Abstraction from os.path.join()
    Builds absolute paths from relative path strings with SIM_FRAMEWORK/ as root.
    If args already contains an absolute path, it is used as root for the subsequent joins
    Args:
        *args:

    Returns:
        absolute path

    """
    return os.path.abspath(os.path.join(FRAMEWORK_DIR, *args))
