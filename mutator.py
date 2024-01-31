from program import Program, Instruction
from team import Team
import random
from typing import List
from parameters import Parameters

class Mutator:

    @staticmethod
    def mutateInstruction(instruction: Instruction) -> None:
        parts: List[str] = [ "ADDRESSING_MODE", "OPERATION", "SOURCE REGISTER", "DESTINATION REGISTER"]
        mutatedPart: str = random.choice(parts)

        originalHash: int = hash(instruction)
        
        if mutatedPart == "ADDRESSING_MODE":
            instruction.mode = random.choice(["INPUT", "REGISTERS"])

            # ensure that the registers remain within bounds
            if instruction.mode == "INPUT" and instruction.source > Parameters.NUM_OBSERVATIONS - 1:
                instruction.source = instruction.source % Parameters.NUM_OBSERVATIONS
            elif instruction.mode == "REGISTERS" and instruction.source > Parameters.NUM_REGISTERS - 1:
                instruction.source = instruction.source % Parameters.NUM_REGISTERS
                
        elif mutatedPart == "OPERATION":
            instruction.operation = random.choice(['+', '-', '*', '/'])

        elif mutatedPart == "SOURCE REGISTER":
            if instruction.mode == "INPUT":
                instruction.source = random.randint(0, Parameters.NUM_OBSERVATIONS - 1)
            elif instruction.mode == "REGISTERS":
                instruction.source = random.randint(0, Parameters.NUM_REGISTERS - 1)

        elif mutatedPart == "DESTINATION REGISTER":
            instruction.destination = random.randint(0, Parameters.NUM_REGISTERS - 1) 
            
        newHash: int = hash(instruction)
        
        # Mutation is the same as the original, try again.
        if newHash == originalHash:
            Mutator.mutateInstruction(instruction)

    @staticmethod
    def mutateProgram(program: Program) -> None:

        originalHash: int = hash(program)

        # delete an instruction
        if len(program.instructions) > 1:
            if random.random() < Parameters.DELETE_INSTRUCTION_PROBABILITY:
                program.instructions.remove(random.choice(program.instructions))

        # add an instruction
        if random.random() < Parameters.ADD_INSTRUCTION_PROBABILITY:
            program.instructions.append(Instruction())

        # swap an instruction
        if random.random() < Parameters.SWAP_INSTRUCTION_PROBABILITY:
            if len(program.instructions) > 2:
                i, j = random.sample(range(len(program.instructions)), 2)
                program.instructions[i], program.instructions[j] = program.instructions[j], program.instructions[i]

        # mutate an instruction
        if random.random() < Parameters.MUTATE_INSTRUCTION_PROBABILITY:
            Mutator.mutateInstruction(random.choice(program.instructions))

        newHash: int = hash(program)

        # Mutation is the same as the original, try again.
        if newHash == originalHash:
            Mutator.mutateProgram(program)

    # TODO: Add a hash for teams so we know each team is unique after mutation
    @staticmethod
    def mutateTeam(programPopulation: List[Program], teamPopulation: List[Team], team: Team):
        # add a program
        if random.random() < Parameters.ADD_PROGRAM_PROBABILITY:
            
            newProgram: Program = random.choice(programPopulation) 

            ids = [ program.id for program in team.programs ]
            # Ensure we're not adding a duplicate program to the team
            while newProgram.id in ids:
                newProgram = random.choice(programPopulation)
            
            team.programs.append(newProgram)

        # delete a program
        if random.random() < Parameters.DELETE_PROGRAM_PROBABILITY:
            if len(team.programs) > 1:
                team.programs.remove(random.choice(team.programs))
        
        # create a new program
        if random.random() < Parameters.NEW_PROGRAM_PROBABILITY:
            program: Program = Program()
            programPopulation.append(program)
            team.programs.append(program)

        # mutate a program's action
        if random.random() < Parameters.MUTATE_PROGRAM_PROBABILITY:
            numAtomicActions: int = 0
            for program in team.programs:
                if program.action in Parameters.ACTIONS:
                    numAtomicActions += 1
            
            program: Program = random.choice(team.programs)
            
            # A team must have at least one atomic action!
            if numAtomicActions > 1 and random.random() < Parameters.TEAM_POINTER_PROBABILITY:
                newTeam: Team = random.choice(teamPopulation)
                while newTeam.id == team.id:
                    newTeam = random.choice(teamPopulation)

                newTeam.referenceCount += 1    
                program.action = str(newTeam.id) 
            else:
                if program.action not in Parameters.ACTIONS:
                    for t in teamPopulation:
                        if str(t.id) == program.action:
                            t.referenceCount -= 1
                
                program.action = random.choice(Parameters.ACTIONS)
