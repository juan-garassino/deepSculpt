from matplotlib import pyplot as plt
from colorama import Fore, Style
import os
import numpy as np
import errno


class Manager:  # make manager work with and with out epochs
    def __init__(self, model_name, data_name):
        self.model_name = model_name

        self.data_name = data_name

        self.comment = "{}_{}".format(model_name, data_name)

        self.data_subdir = "{}/{}".format(model_name, data_name)

    @staticmethod
    def make_directory(directory):
        try:
            os.makedirs(directory)

            print(
                "\n⏹ "
                + Fore.GREEN
                + f"This directory has been created {directory}"
                + Style.RESET_ALL
            )

        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


if __name__ == "__main__":
    pass
