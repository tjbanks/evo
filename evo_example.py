import random
from deap import creator, base, tools, algorithms

def evo_example():

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
    toolbox.register("attr_bool", random.randint, 0, 1)
    #Initialize the individual with 100 randinit (attr_bool is a randint between 0 and 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100)
    #register a population to be a list of individuals, no specified number until creation
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #tools. -> https://deap.readthedocs.io/en/master/api/tools.html

    def evalOneMax(individual):
        return sum(individual),

    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=300)
    print(population)
    NGEN=40
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
    top10 = tools.selBest(population, k=10)

    print(top10)

if __name__ == '__main__':
    evo_example()