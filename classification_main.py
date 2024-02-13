from model import Model
import random
from typing import List
from skmultiflow.data import DataStream
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
import warnings
from debugger import Debugger
from parameters import Parameters
# Ignore specific RuntimeWarnings (division by zero in this case)
warnings.filterwarnings("ignore", message="overflow encountered in double_scalars")

def main():
    Parameters.ACTIONS = [0,1,2]
    Parameters.NUM_OBSERVATIONS = 4
    Parameters.POPULATION_SIZE = 30
    Parameters.INITIAL_PROGRAM_POPULATION =100
    Parameters.LUCKY_BREAK_NUM = 1
    model = Model()
    debugger = Debugger()
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    for i in range(100):
        champion_team=model.generation(X,y,i)
        if i%10 == 0:
            teampath = "champion_team_"+str(i)
            modelpath = "champion_model_"+str(i)
            model.saveChampionTeam(champion_team,teampath)       
            model.saveChampionModel(modelpath)   

def champion_run():
    iris = datasets.load_iris()
    Parameters.ACTIONS = [0,1,2]
    Parameters.NUM_OBSERVATIONS = 4
    X = iris.data
    y = iris.target
    model = Model()
    champion_model = model.loadChampionModel('/home/amous/Research/pytpg-vanilla/champion_model_90')
    champion_team = champion_model.loadChampionTeam('/home/amous/Research/pytpg-vanilla/champion_team_90')

    y_pred = champion_model.predict(X,champion_team)
    score = f1_score(y, y_pred, average='macro')
    print("accuracy is:",score)
    
if __name__ == "__main__":
    #main()
   champion_run()
