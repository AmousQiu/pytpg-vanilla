
from environment import Environment
from program import Program
from team import Team
from mutator import Mutator
from parameters import Parameters

import random
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
import pickle

class Model:

    # A model consists of a population of teams and a population of programs.
    """ A model consists of a population of teams and a population of programs."""
    def __init__(self):

        # Initialize the program population
        self.programPopulation: List[Program] = [ Program() for _ in range(Parameters.INITIAL_PROGRAM_POPULATION)]

        # Initialize the team population
        self.teamPopulation: List[Team] = [ Team(self.programPopulation) for _ in range(Parameters.POPULATION_SIZE)]

        for team in self.teamPopulation:
            team.referenceCount = 0

    def getRootTeams(self) -> List[Team]:
        rootTeams: List[Team] = []
        for team in self.teamPopulation:
            if team.referenceCount == 0:
                rootTeams.append(team)

        return rootTeams

    
    def predict(self,X,team):
        y_pred = []
        for i in range(len(X)):
            try:
                action = team.getAction(self.teamPopulation,X[i])
                y_pred.append(action)
            except Exception as e:
                print(f"Error: {e}")
        return y_pred 
    
    def generation(self, X,y,gen) -> None:
        print("Teams here:",len(self.getRootTeams()))
        for teamNum, team in enumerate(self.getRootTeams()):
            score = 0
            y_pred = self.predict(X,team)
            #score = f1_score(y, y_pred, average='macro')
            score = accuracy_score(y,y_pred)
            team.scores.append(score)

        
        print("\nGeneration", gen, "complete.\n")
        
        sortedTeams: List[Team] = list(sorted(self.getRootTeams(), key=lambda team: team.getFitness()))
        for team in sortedTeams[-Parameters.LUCKY_BREAK_NUM:]:
            team.luckyBreaks += 1

        print("Best performing teams:")
        championTeam = sortedTeams[-1]
        print(f"Team {championTeam.id} score: {championTeam.getFitness()}, lucky breaks: {championTeam.luckyBreaks}")
        self.select(sortedTeams)
        self.evolve()
        return championTeam
    
    def saveChampionModel(self,filepath):
        with open(filepath,'wb') as f:
            pickle.dump(self,f)
        print(f"Champion model saved to {filepath}")
        
    def loadChampionModel(self,filepath):
        with open(filepath, 'rb') as f:
            championModel = pickle.load(f)
        return championModel
        
    def saveChampionTeam(self,championTeam,filepath):
        with open(filepath,'wb') as f:
            pickle.dump(championTeam,f)
        print(f"Champion team saved to {filepath}")
    
    def loadChampionTeam(self,filepath):
        with open(filepath, 'rb') as f:
            championTeam = pickle.load(f)
        return championTeam
    
    def cleanProgramPopulation(self) -> None:
        inUseProgramIds: List[str] = []
        for team in self.teamPopulation:
            for program in team.programs:
                inUseProgramIds.append(program.id)

        for program in self.programPopulation:
            if program.id not in inUseProgramIds:
                self.programPopulation.remove(program)

    # Remove uncompetitive teams from the population
    def select(self, sortedTeams) -> None:

        #sortedTeams: List[Team] = list(sorted(self.getRootTeams(), key=lambda team: team.getFitness()))

        # Remove a POPGAP fraction of teams
        remainingTeamsCount: int = int(Parameters.POPGAP * len(self.getRootTeams()))

        for team in sortedTeams[:remainingTeamsCount]:
            
            if team.luckyBreaks > 0:
                team.luckyBreaks -= 1
                print(f"Tried to remove team {team.id} but they had a lucky break! {team.getFitness()} (remaining breaks: {team.luckyBreaks})")
            else:
                self.teamPopulation.remove(team)

        # Clean up, if there are programs that are not referenced by any teams.
        # Remove them
        self.cleanProgramPopulation() 

    # Create new teams cloned from the remaining root teams
    def evolve(self) -> None:
        while len(self.getRootTeams()) < Parameters.POPULATION_SIZE:
            parent_team = random.choice(self.getRootTeams())
            child_team =parent_team.copy()
            Mutator.mutateTeam(self.programPopulation, self.teamPopulation, child_team)
            self.teamPopulation.append(child_team)
