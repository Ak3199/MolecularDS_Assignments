import numpy
import time
import random
import os
import subprocess
import shutil

cwd = os.getcwd()
# This project is extended and a library called PyGAD is released to build the genetic algorithm.
# PyGAD documentation: https://pygad.readthedocs.io
# Install PyGAD: pip install pygad
# PyGAD source code at GitHub: https://github.com/ahmedfgad/GeneticAlgorithmPython

fit_total = []

def cal_pop_fitness(new_population,sol_per_pop,num_weights,m,n):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    #write separate .txt files each with different solution sequences in a population 
    for i in range(0,sol_per_pop):    
        file2 = open("sequence"+str(i)+".txt", "w")
        file1 = open(cwd + "/sequences_min/sequence"+str(m)+".txt","w")
        m=m+1
        for j in range(0,num_weights):
            file2.write(str(new_population[i,j]))
            file1.write(str(new_population[i,j]))
            file1.write("\n")
            file2.write("\n")
        file2.close()
        file1.close()
    
    #generate sol_per_pop number of polymer 1 to 2 data conversion .cpp files
    for i in range(0,sol_per_pop):
        shutil.copy('datatoData1.cpp', 'datatoData1'+str(i)+'.cpp')

    #changing the content of .cpp files according to sequence files to be read
    for i in range(0,sol_per_pop):
        cpp_file = open("datatoData1"+str(i)+".cpp", "r")
        list_of_lines = cpp_file.readlines()
        list_of_lines[21] = '        infile1.open("sequence'+str(i)+'.txt");\n'
        list_of_lines[37] = '        outfile.open("polymer2_'+str(i)+'.data");\n'
    
        cpp_file = open("datatoData1"+str(i)+".cpp", "w")
        cpp_file.writelines(list_of_lines)
        cpp_file.close()

    #polymer1.data to new_polymer data conversion
    for i in range(0,sol_per_pop):
        os.system('g++ datatoData1'+str(i)+'.cpp')
        subprocess.call("./a.out")

    #duplicate lammps files
    for i in range(0,sol_per_pop):
        shutil.copy('in.lammps', 'in_'+str(i)+'.lammps')

    #change lammps file content
    for i in range(0,sol_per_pop):
        lammps_file = open("in_"+str(i)+".lammps", "r")
        list_of_content = lammps_file.readlines()
        list_of_content[13] = ' read_data polymer2_'+str(i)+'.data\n'
        list_of_content[58] = 'fix        	RgAve  all ave/time 1000 1000 1000000 c_1 file AvgRg_'+str(i)+'.data\n'
    
        lammps_file = open("in_"+str(i)+".lammps", "w")
        lammps_file.writelines(list_of_content)
        lammps_file.close()
    
    #evaluate generate script
    subprocess.call("./try.sh")

    while True:
        count_file = 0
        for i in range(0,sol_per_pop):
            if os.path.exists("AvgRg_"+str(i)+".data"):
                count_file = count_file + 1
        if count_file == sol_per_pop:
            break
        else:
            time.sleep(350)

    time.sleep(150)
   
    #get fitness value
    fitness = []
    for i in range(0,sol_per_pop):
        rad_file = open("AvgRg_"+str(i)+".data", "r")
        list_of_content = rad_file.readlines()
        fitness.append(float(list_of_content[2].split(" ")[1]))
        
        count=0
        for k in new_population[i]:
            if k==1:
                count+count+1
        if count==40:
            fit_total.append(float(list_of_content[2].split(" ")[1]))
        rad_file.close()
        
        file2=open(cwd + "/sequences_min/sequence"+str(n)+".txt","r")
        rad_add = file2.readlines()
        rad_add.append(" ")
        rad_add[20]=str(list_of_content[2].split(" ")[1])
        file2=open(cwd + "/sequences_min/sequence"+str(n)+".txt","w")
        file2.writelines(rad_add)
        file2.close()
        n=n+1

    fitness = numpy.array(fitness)
    print(numpy.min(fitness))
    print(fitness)
    return fitness, fit_total

def select_mating_pool(pop, fit, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        min_fitness_idx = numpy.where(fit == numpy.min(fit))
        min_fitness_idx = min_fitness_idx[0][0]
        parents[parent_num, :] = pop[min_fitness_idx, :]
        fit[min_fitness_idx] = 99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    #crossover_point = numpy.uint8(offspring_size[1]/2)
    crossover_point = random.randint(1,offspring_size[1])

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        #parent1_idx = k%parents.shape[0]
        parent1_idx = random.randint(0, len(parents)-1)
        # Index of the second parent to mate.
        #parent2_idx = (k+1)%parents.shape[0]
        parent2_idx = random.randint(0, len(parents)-1)
        while True:
            if parent1_idx==parent2_idx:
                parent2_idx = random.randint(0, len(parents)-1)
            else:
                break
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    
    return offspring
'''    
    k=0
    for i in offspring:
        count=0
        for j in i:
            if j==1:
                count = count+1

        if count==10:
            k=k+1
            continue
        else:
            if count>10:
                excess = count - 10
                excess_counter = 0

                for z in range(0,i.size):
                    if i[z]==1:
                        offspring[k,z] = 2
                        excess_counter = excess_counter + 1
                    if excess_counter==excess:
                        break

            if count<10:
                excess = 10-count
                excess_counter = 0

                for z in range(0,i.size):
                    if i[z]==2:
                        offspring[k,z]=1
                        excess_counter=excess_counter+1
                    if excess_counter==excess:
                        break
        k=k+1
'''


def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        random_value = numpy.random.uniform(1.0, 1.0, 1)
        loc = random.randint(0,19)
        if offspring_crossover[idx,loc]==1:
            offspring_crossover[idx,loc] = offspring_crossover[idx,loc]+random_value
        else:
            offspring_crossover[idx, loc] = offspring_crossover[idx, loc] - random_value
    
    
    

    k=0
    
    for i in offspring_crossover:
        count=0
        for j in i:
            if j==1:
                count = count+1

        if count==40:
            k=k+1
            continue
        else:
            if count>8:
                excess = count - 8
                excess_counter = 0

                for z in range(0,i.size):
                    if i[z]==1:
                        offspring_crossover[k,z] = 2
                        excess_counter = excess_counter + 1
                    if excess_counter==excess:
                        break

            if count<8:
                excess = 8-count
                excess_counter = 0

                for z in range(0,i.size):
                    if i[z]==2:
                        offspring_crossover[k,z]=1
                        excess_counter=excess_counter+1
                    if excess_counter==excess:
                        break
        k=k+1

    return offspring_crossover
