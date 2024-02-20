from instruction import Instruction

import random
from uuid import uuid4
from typing import List, Dict
import numpy as np
from parameters import Parameters
from copy import deepcopy

class Program:
    def __init__(self,instructions =None,action=None):
        self.id: UUID = uuid4()
        self.registers: np.array = np.zeros(Parameters.NUM_REGISTERS)
        
        if action == None:
            self.action: str = random.choice(Parameters.ACTIONS)
        else:
            self.action = action
            
        if instructions == None:
            self.instructions: List[Instruction] = []
         # Generate a list of instructions ranging from 1 to the maximum number of instructions
            for _ in range(random.randint(4, Parameters.MAX_INSTRUCTION_COUNT)):
                self.instructions.append(Instruction())
        else:
            self.instructions = instructions

    def __str__(self) -> str:
        header: str = f"Program {self.id}:\n"
        instructions: str = '\n'.join(map(str, self.instructions))
        return f"{header}{instructions}"

    def __hash__(self) -> int:
        return hash(str(self))
    
    def reset(self):
        self.registers = np.zeros(Parameters.NUM_REGISTERS)
        
    def execute(self, state: np.array) -> None:
        self.reset()
        for instruction in self.instructions:
            instruction.execute(state, self.registers)

    def bid(self, state: np.array) -> Dict[float, str]:
        self.execute(state)
        
        return {
            "confidence": self.registers[0],
            "action": self.action
        }

    def clone(self):
        clone: 'Program' = deepcopy(self)
        #clone = Program(self.instructions,self.action)
        return clone
    
    def copy(self):
        return deepcopy(self)