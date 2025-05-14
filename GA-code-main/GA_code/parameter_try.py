import numpy
import ga_try
import random
import os
import subprocess
import shutil
import time
import matplotlib.pyplot as plt

"""
The y=target is to maximize this equation ASAP:
    y = w1x1+w2x2+w3x3+w4x4+w5x5+6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7)
    What are the best values for the 6 weights w1 to w6?
    We are going to use the genetic algorithm for the best possible values after a number of generations.
"""

start = time.time()

# Number of the weights we are looking to optimize.
num_weights = 20
m=0
n=0

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 48
num_parents_mating = 16

# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
#Creating the initial population.
new_population = numpy.random.randint(low = 1.0, high = 3.0,size= pop_size)

#creating a population with forty 1's and sixty 2's
for i in range(0,sol_per_pop):
    for j in range(0,num_weights):
        if j<8:
            new_population[i,j]=int(1)
        else:
            new_population[i,j]=int(2)

#generating randomness
for i in range(0,sol_per_pop):
    random.shuffle(new_population[i])

print(new_population)

extra = 0
gen_count = 0
gen_arr = []
num_generations =10
fit_max = []
for generation in range(num_generations):
    gen_count = gen_count + 1
    print("Generation : ", generation)
    # Measing the fitness of each chromosome in the population.
    fitness,fit_total = ga_try.cal_pop_fitness(new_population,sol_per_pop,num_weights,m,n)

    # Selecting the best parents in the population for mating.
    parents = ga_try.select_mating_pool(new_population, fitness, 
                                      num_parents_mating)

    # Generating next generation using crossover.
    offspring_crossover = ga_try.crossover(parents,
                                       offspring_size=(pop_size[0]-parents.shape[0], num_weights))

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = ga_try.mutation(offspring_crossover)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

    m=m+sol_per_pop
    n=n+sol_per_pop
    # The best result in the current iteration.

    if generation==num_generations-1:
        time.sleep(5)
        for i in range(0,sol_per_pop):
            if os.path.exists("AvgRg_"+str(i)+".data"):
                os.remove("AvgRg_"+str(i)+".data")
        time.sleep(30)   
        fitness,fit_total = ga_try.cal_pop_fitness(new_population,sol_per_pop,num_weights,m,n)

    arr_length = len(fit_total)-extra
    extra = len(fit_total)

    for i in range(0,arr_length):
        gen_arr.append(gen_count)


    test = fitness[0:num_parents_mating-1]
    ele = test[0]
    chk = "True"

    for item in test:
        if ele != item:
            chk = "False"
            break

    if (chk == "True"):
        for change in range(int(num_parents_mating/2),int(num_parents_mating)):
            random.shuffle(new_population[change])


    fit_max.append(str(min(fitness)))
    fit_max.append(" ")
    compilation = open("compilation.txt","w")
    compilation.writelines(fit_max)
    compilation.close()
    
    time.sleep(20)
    for i in range(0,sol_per_pop):
        if os.path.exists("AvgRg_"+str(i)+".data"):
            os.remove("AvgRg_"+str(i)+".data")

    time.sleep(30)
 
# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
#fitness = ga_try.cal_pop_fitness(new_population,sol_per_pop,num_weights,m,n)


# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.min(fitness))
best_match_idx = best_match_idx[0][0]
print(fit_total)
print("Best solution : ", new_population[best_match_idx, :])
new_population = numpy.asarray(new_population)
#print("Best solution fitness : ", fitness[best_match_idx])
f = open("sequence_min.txt","w")
for i in range(0,num_weights):
    f.write(str(new_population[best_match_idx,i]))
    f.write("\n")
f.close()
print("Best solution fitness: ", fitness[best_match_idx]) 
stop = time.time()
print(gen_arr)
print(f"Runtime of the program is {stop-start}")
'''
#Plot for Rg value vs Gen Num
fig, ax = plt.subplots()
ax.plot(gen_arr, fit_total, 'o')
plt.xticks([0, 10, 20, 30, 40],fontsize=9, weight='bold')
plt.yticks(numpy.arange(numpy.min(fit_total),numpy.max(fit_total)+0.1,step=(numpy.max(fit_total)-numpy.min(fit_total))/4),fontsize=9,weight='bold')
ax.spines["bottom"].set_linewidth(2)
ax.spines["top"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
ax.spines["right"].set_linewidth(2)
plt.xlabel("Generation Number", fontweight='bold')
plt.ylabel("Rg Values", fontweight='bold')
plt.savefig("Rg_vs_N.png")
'''
