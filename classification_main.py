from model import Model
import random
from typing import List
from skmultiflow.data import DataStream
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from debugger import Debugger
from parameters import Parameters
# Ignore specific RuntimeWarnings (division by zero in this case)
warnings.filterwarnings("ignore", message="overflow encountered in double_scalars")

def main():
    Parameters.ACTIONS = [0,1,2]
    Parameters.NUM_OBSERVATIONS = 4
    model = Model()
    debugger = Debugger()
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    for i in range(100):
        best_score=model.generation(X,y,i)

if __name__ == "__main__":
    main()
