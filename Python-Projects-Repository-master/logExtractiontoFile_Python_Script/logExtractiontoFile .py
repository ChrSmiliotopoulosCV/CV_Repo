import os
import sys
import numpy as np


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.txt", "a")
        self.log = open("logfile.csv", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


sys.stdout = Logger()

conf_matrix_result = np.array([[235, 33, 2],
                               [56, 0, 44],
                               [0, 7, 12]])

print(0.005 / 0.00035)

print(conf_matrix_result / 10)

print("Hello World")

# sys.stdout = open('stdout.txt', 'w')

