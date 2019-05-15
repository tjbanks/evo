import random
from deap import creator, base, tools, algorithms
import cell
import numpy as np

def evo_example(user_entry):

    #creator is a meta factory class to register functions
    #Creates a new class named FitnessMax inheriting from base in the creator module.
    #The fitness is a measure of quality of a solution.
    #If values are provided as a tuple, the fitness is initalized using those values, otherwise it is empty (or invalid).
    #anything after the base class will be passed into the class when called, optional (weights)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    #Create the core class to operate on, Individual inherits from list class, fitness is an argument to this class
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    #Register a function in the toolbox under the name alias
    #toolbox.register(alias, method[, argument[, ...]])
    toolbox.register("attr_bool", random.randint, 0, 150)

    toolbox.register("attr_na", random.randint, 30, 140)
    toolbox.register("attr_k", random.randint, 10, 40)
    toolbox.register("attr_leak", random.randint, 15, 105)
    #Initialize the individual with 3 randinit (attr_bool is a randint between 0 and 1)
    #toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=3)
    toolbox.register("individual", tools.initCycle, creator.Individual, [lambda:random.randint(30,140), lambda:random.randint(10,40), lambda:random.randint(15,105)],n=1)
    #register a population to be a list of individuals, no specified number until creation
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #tools. -> https://deap.readthedocs.io/en/master/api/tools.html

    def fitness_function(individual):
        try:
            val = cell.get_cell_properties(*individual, as_list=True)
            fitness = np.abs(np.divide(val,user_entry)).sum()
            print("fitness: " + str(fitness))
            return fitness,
        except:
            print("not viable")
            return 0,

    toolbox.register("evaluate", fitness_function)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma= 15, indpb=0.25)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=10)
    
    NGEN=5
    for gen in range(NGEN):
        print("Generation " + str(gen))
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.25)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
    top2 = tools.selBest(population, k=2)

    print(top2)

if __name__ == '__main__':
    
    user_entry = [0.004, 0.058, 0.06, 0.062, 0.064, 3.0, 15.166572661351978, 3.592743625725153, -73.04588861070172]
    #Hope to get back 90, 30, 45
    #First run yields 91, 23.92646581169069, 34
    #Result: [0.004, 0.052, 0.054, 0.056, 0.058, 1.2000000000000002, 21.286041343955873, 5.042358023991584, -72.95101008264628]
    evo_example(user_entry)