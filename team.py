from typing import List
import random
from program import Program
import numpy as np
from uuid import uuid4
from typing import List
from copy import deepcopy
from parameters import Parameters

import numpy as np

class Team:
    def __init__(self, programPopulation: List[Program],programs=None):
        self.id: UUID = uuid4()

        # A team is a champion if it is the best performing team in a generation
        self.scores: List[float] = []
        self.luckyBreaks: int = 0
        
        # If programs are not provided, select them from the program population
        if programs is None:
            self.programs = self.select_unique_programs(programPopulation)
        else:
            self.programs = programs        

    def select_unique_programs(self, programPopulation: List[Program]) -> List[Program]:
        """
        Selects a unique set of programs ensuring at least two distinct actions.
        """
        size = random.randint(2, Parameters.MAX_INITIAL_TEAM_SIZE)
        selected_programs = random.sample(programPopulation, k=size)
        actions = {program.action for program in selected_programs}

        # Attempt to ensure at least two distinct actions if possible
        attempts = 0
        while len(actions) < 2 and attempts < 100:
            selected_programs = random.sample(programPopulation, k=size)
            actions = {program.action for program in selected_programs}
            attempts += 1

        return selected_programs
        
    # Choose the program with the highest confidence
    def getAction(self, teamPopulation: List['Team'], state: np.array, visited: List[str] = []) -> str:
        visited.append(self)

        sortedPrograms = sorted(self.programs, key=lambda program: program.bid(state)['confidence'])
            
        for program in sortedPrograms:
            if program.action in Parameters.ACTIONS:
                if program.action == None:
                    raise RuntimeError("A NONE ACTION WAS ENCOUNTERED HERE")
                return program.action
            else:
                for team in teamPopulation:
                    if str(team.id) == program.action:
                        return team.getAction(teamPopulation, state, visited)
                return random.choice(Parameters.ACTIONS)
                raise RuntimeError(f"Team {self.id} points to team {program.action}, and that team does not exist within the population.")

    def getFitness(self):
            return self.scores[-1]

    
    # Given a parent team, a new offspring team is cloned and mutated
    def copy(self):
        clone: 'Team' = deepcopy(self)
        clone.referenceCount = 0
        clone.luckyBreaks = 0
        clone.id = uuid4()
        return clone
