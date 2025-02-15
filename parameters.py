from typing import List

class Parameters:
    POPULATION_SIZE: int = 100
    LUCKY_BREAK_NUM = 1
    INITIAL_PROGRAM_POPULATION: int = 1000
    POPGAP: float = 0.2
    ACTIONS: List[str] = [ "DO_NOTHING", "STEER_LEFT", "STEER_RIGHT", "GAS", "BRAKE" ]
    ENVIRONMENT: str = "CarRacing-v2"
    NUM_OBSERVATIONS: int = 27648
    NUM_REGISTERS: int = 8
    DELETE_INSTRUCTION_PROBABILITY: float = 1
    ADD_INSTRUCTION_PROBABILITY: float = 1
    SWAP_INSTRUCTION_PROBABILITY: float = 1
    MUTATE_INSTRUCTION_PROBABILITY: float = 0.8#0.8
    ADD_PROGRAM_PROBABILITY: float = 0.8
    DELETE_PROGRAM_PROBABILITY: float = 0.8
    NEW_PROGRAM_PROBABILITY: float = 0.8
    CROSSOVER_PROBABILITY:float = 0.2
    MUTATE_PROGRAM_PROBABILITY: float = 0.2
    MUTATE_PROGRAM_ACTION_PROBABILITY: float = 0.8
    TEAM_POINTER_PROBABILITY: float = 0.5#1
    MAX_INSTRUCTION_COUNT: float = 64
    MAX_INITIAL_TEAM_SIZE: float = 5
    
