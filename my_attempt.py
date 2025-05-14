import numpy as np
import random
import matplotlib.pyplot as plt
import time

def parse_lammps_file(file_path):
    """Parse a LAMMPS data file to extract atoms and bonds."""
    atoms = {}
    bonds = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    reading_atoms = False
    reading_bonds = False
    
    for line in lines:
        line = line.strip()
        
        if line == "Atoms":
            reading_atoms = True
            continue
        elif line == "Bonds":
            reading_atoms = False
            reading_bonds = True
            continue
        
        if reading_atoms and line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 5:
                atom_id = int(parts[0])
                molecule_id = int(parts[1])
                atom_type = int(parts[2])
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                atoms[atom_id] = {'type': atom_type, 'x': x, 'y': y, 'z': z}
        
        if reading_bonds and line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 4:
                bond_id = int(parts[0])
                bond_type = int(parts[1])
                atom1 = int(parts[2])
                atom2 = int(parts[3])
                bonds.append((atom1, atom2))
    
    return atoms, bonds

def create_adjacency_list(bonds, num_atoms):
    """Create an adjacency list from bond information."""
    adjacency_list = [[] for _ in range(num_atoms + 1)]  # +1 because atom IDs start from 1
    
    for atom1, atom2 in bonds:
        adjacency_list[atom1].append(atom2)
        adjacency_list[atom2].append(atom1)  # Undirected graph
    
    return adjacency_list

def calculate_energy(spins, adjacency_list):
    """Calculate the energy of an Ising model configuration."""
    energy = 0
    
    for i in range(1, len(spins) + 1):
        for neighbor in adjacency_list[i]:
            if neighbor > i:  # Count each interaction only once
                energy += spins[i-1] * spins[neighbor-1]
    
    return energy

def calculate_fitness(spins, adjacency_list):
    """Calculate fitness (negative energy) for the GA."""
    return -calculate_energy(spins, adjacency_list)

def initialize_population(pop_size, num_spins):
    """Initialize a random population of spin configurations."""
    return [np.random.choice([-1, 1], size=num_spins) for _ in range(pop_size)]

def random_selection(population, fitnesses):
    """Select an individual randomly from the population."""
    return random.choice(population)

def roulette_wheel_selection(population, fitnesses):
    """Select an individual using roulette wheel selection."""
    # Shift fitnesses to ensure all are positive
    min_fitness = min(fitnesses)
    if min_fitness < 0:
        shifted_fitnesses = [f - min_fitness + 1 for f in fitnesses]
    else:
        shifted_fitnesses = fitnesses
    
    total_fitness = sum(shifted_fitnesses)
    if total_fitness == 0:
        return random.choice(population)
    
    selection_point = random.uniform(0, total_fitness)
    current = 0
    
    for i, fitness in enumerate(shifted_fitnesses):
        current += fitness
        if current > selection_point:
            return population[i]
    
    return population[-1]

def tournament_selection(population, fitnesses, tournament_size=3):
    """Select an individual using tournament selection."""
    indices = random.sample(range(len(population)), tournament_size)
    tournament = [(population[i], fitnesses[i]) for i in indices]
    winner = max(tournament, key=lambda x: x[1])
    return winner[0]

def elitist_selection(population, fitnesses, elite_size):
    """Select the best individuals from the population."""
    indices = np.argsort(fitnesses)[-elite_size:]
    return [population[i] for i in indices]

def single_point_crossover(parent1, parent2):
    """Perform single-point crossover between two parents."""
    point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def two_point_crossover(parent1, parent2):
    """Perform two-point crossover between two parents."""
    point1 = random.randint(1, len(parent1) - 2)
    point2 = random.randint(point1 + 1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
    child2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))
    return child1, child2

def uniform_crossover(parent1, parent2, probability=0.5):
    """Perform uniform crossover between two parents."""
    child1, child2 = parent1.copy(), parent2.copy()
    for i in range(len(parent1)):
        if random.random() < probability:
            child1[i], child2[i] = child2[i], child1[i]
    return child1, child2

def mutate(chromosome, mutation_rate):
    """Mutate a chromosome by flipping spins with a given probability."""
    mutated = chromosome.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] *= -1  # Flip the spin
    return mutated

def run_genetic_algorithm(adjacency_list, num_spins, selection_method="tournament", 
                         crossover_method="uniform", use_elitism=True, 
                         population_size=100, max_generations=500, 
                         crossover_rate=0.8, mutation_rate=0.01, elite_size=2):
    """
    Run the genetic algorithm to find the ground state of the Ising model.
    
    Parameters:
    -----------
    adjacency_list : list
        List of lists representing the adjacency structure of the polymer
    num_spins : int
        Number of spins in the system
    selection_method : str
        Method for selecting parents: "random", "roulette", or "tournament"
    crossover_method : str
        Method for crossover: "single", "two", or "uniform"
    use_elitism : bool
        Whether to use elitism (preserving the best individuals)
    population_size : int
        Size of the population
    max_generations : int
        Maximum number of generations to run
    crossover_rate : float
        Probability of crossover
    mutation_rate : float
        Probability of mutation for each spin
    elite_size : int
        Number of elite individuals to preserve if using elitism
        
    Returns:
    --------
    best_solution : numpy.ndarray
        The best spin configuration found
    best_fitnesses : list
        List of the best fitness values for each generation
    """
    # Initialize population
    population = initialize_population(population_size, num_spins)
    
    # Track best fitness over generations
    best_fitnesses = []
    
    start_time = time.time()
    
    for generation in range(max_generations):
        # Calculate fitness for each individual
        fitnesses = [calculate_fitness(ind, adjacency_list) for ind in population]
        
        # Track best fitness
        best_fitness = max(fitnesses)
        best_fitnesses.append(best_fitness)
        
        # Print progress every 50 generations
        if generation % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Generation {generation}: Best fitness = {best_fitness}, Time elapsed: {elapsed:.2f}s")
        
        # Create new population
        new_population = []
        
        # Elitism
        if use_elitism:
            elite = elitist_selection(population, fitnesses, elite_size)
            new_population.extend(elite)
        
        # Fill the rest of the population
        while len(new_population) < population_size:
            # Selection
            if selection_method == "random":
                parent1 = random_selection(population, fitnesses)
                parent2 = random_selection(population, fitnesses)
            elif selection_method == "roulette":
                parent1 = roulette_wheel_selection(population, fitnesses)
                parent2 = roulette_wheel_selection(population, fitnesses)
            elif selection_method == "tournament":
                parent1 = tournament_selection(population, fitnesses)
                parent2 = tournament_selection(population, fitnesses)
            else:
                raise ValueError(f"Unknown selection method: {selection_method}")
            
            # Crossover
            if random.random() < crossover_rate:
                if crossover_method == "single":
                    offspring1, offspring2 = single_point_crossover(parent1, parent2)
                elif crossover_method == "two":
                    offspring1, offspring2 = two_point_crossover(parent1, parent2)
                elif crossover_method == "uniform":
                    offspring1, offspring2 = uniform_crossover(parent1, parent2)
                else:
                    raise ValueError(f"Unknown crossover method: {crossover_method}")
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()
            
            # Mutation
            offspring1 = mutate(offspring1, mutation_rate)
            offspring2 = mutate(offspring2, mutation_rate)
            
            # Add to new population
            new_population.append(offspring1)
            if len(new_population) < population_size:
                new_population.append(offspring2)
        
        # Replace old population
        population = new_population
    
    # Return best individual and fitness history
    fitnesses = [calculate_fitness(ind, adjacency_list) for ind in population]
    best_index = fitnesses.index(max(fitnesses))
    
    total_time = time.time() - start_time
    print(f"GA completed in {total_time:.2f} seconds")
    
    return population[best_index], best_fitnesses

def compare_strategies(adjacency_list, num_spins):
    """Compare different GA strategies and plot results."""
    selection_methods = ["random", "roulette", "tournament"]
    crossover_methods = ["single", "two", "uniform"]
    elitism_options = [False, True]
    
    results = {}
    
    for selection in selection_methods:
        for crossover in crossover_methods:
            for elitism in elitism_options:
                key = f"{selection}_{crossover}_elitism={elitism}"
                print(f"\nRunning GA with strategy: {key}")
                
                best_solution, fitness_history = run_genetic_algorithm(
                    adjacency_list, num_spins,
                    selection_method=selection,
                    crossover_method=crossover,
                    use_elitism=elitism,
                    max_generations=100  # Reduced for demonstration
                )
                
                results[key] = fitness_history
                final_energy = -fitness_history[-1]
                print(f"Strategy: {key}, Final Energy: {final_energy}")
    
    return results

def visualize_polymer_spins(atoms, bonds, spins):
    """Visualize the polymer with spins colored according to their values."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot atoms
    for atom_id, atom_data in atoms.items():
        spin_value = spins[atom_id-1]
        color = 'red' if spin_value == 1 else 'blue'
        ax.scatter(atom_data['x'], atom_data['y'], atom_data['z'], c=color, s=100)
    
    # Plot bonds
    for atom1, atom2 in bonds:
        x1, y1, z1 = atoms[atom1]['x'], atoms[atom1]['y'], atoms[atom1]['z']
        x2, y2, z2 = atoms[atom2]['x'], atoms[atom2]['y'], atoms[atom2]['z']
        ax.plot([x1, x2], [y1, y2], [z1, z2], 'k-', lw=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Polymer with Spin Configuration (Red: +1, Blue: -1)')
    plt.savefig('polymer_spin_configuration.png')
    plt.show()

def save_lammps_data_to_file(lammps_data, file_path):
    """Save LAMMPS data to a file."""
    with open(file_path, 'w') as f:
        f.write(lammps_data)

def main():
    # Example LAMMPS data (replace with your file path)
    lammps_data = """
    20 atoms
    19 bonds
    2 atom types
    1 bond types
    
    -13.5721 13.5721 xlo xhi
    -13.5721 13.5721 ylo yhi
    -13.5721 13.5721 zlo zhi
    
    Masses
    
    1  1
    2  1
    
    Atoms
    
    1  1  2  -2.03212  -6.40088  -7.14687  0  0  0
    2  2  2  -2.38817  -6.50992  -8.04255  0  0  0
    3  3  2  -2.36552  -6.27412  -8.98318  0  0  0
    4  4  2  -2.47895  -6.95579  -9.66389  0  0  0
    5  5  1  -1.77558  -6.94416  -10.3317  0  0  0
    6  6  1  -1.30506  -7.6656  -9.88563  0  0  0
    7  7  1  -1.16092  -8.20008  -9.0891  0  0  0
    8  8  1  -0.716761  -9.06234  -9.10083  0  0  0
    9  9  2  -1.50481  -9.35022  -9.58766  0  0  0
    10  10  1  -2.27599  -8.76187  -9.58234  0  0  0
    11  11  2  -3.07885  -8.89269  -10.1107  0  0  0
    12  12  1  -4.03118  -9.04589  -10.2132  0  0  0
    13  13  1  -4.51863  -8.20726  -10.2161  0  0  0
    14  14  1  -4.4168  -7.25609  -10.0554  0  0  0
    15  15  1  -3.89086  -6.50434  -9.74054  0  0  0
    16  16  1  -3.58668  -5.71975  -9.25806  0  0  0
    17  17  2  -2.61857  -5.76938  -9.22322  0  0  0
    18  18  2  -2.85907  -6.02768  -8.3197  0  0  0
    19  19  2  -2.79907  -6.98136  -8.48638  0  0  0
    20  20  2  -3.17694  -7.21514  -7.62414  0  0  0
    
    Bonds
    
    1 1 1 2
    2 1 2 3
    3 1 3 4
    4 1 4 5
    5 1 5 6
    6 1 6 7
    7 1 7 8
    8 1 8 9
    9 1 9 10
    10 1 10 11
    11 1 11 12
    12 1 12 13
    13 1 13 14
    14 1 14 15
    15 1 15 16
    16 1 16 17
    17 1 17 18
    18 1 18 19
    19 1 19 20
    """
    
    # Save LAMMPS data to a file
    lammps_file = "polymer_data.txt"
    save_lammps_data_to_file(lammps_data, lammps_file)
    
    # Parse LAMMPS data
    atoms, bonds = parse_lammps_file(lammps_file)
    num_atoms = len(atoms)
    
    # Create adjacency list
    adjacency_list = create_adjacency_list(bonds, num_atoms)
    
    print(f"Loaded polymer with {num_atoms} atoms and {len(bonds)} bonds")
    
    # Option 1: Run a single GA with specific parameters
    selection_method = "tournament"
    crossover_method = "uniform"
    use_elitism = True
    
    print(f"\nRunning GA with {selection_method} selection, {crossover_method} crossover, elitism={use_elitism}")
    best_solution, fitness_history = run_genetic_algorithm(
        adjacency_list, num_atoms,
        selection_method=selection_method,
        crossover_method=crossover_method,
        use_elitism=use_elitism
    )
    
    # Plot the energy evolution
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fitness_history) + 1), [-f for f in fitness_history])
    plt.xlabel('Generation')
    plt.ylabel('Energy')
    plt.title(f'Energy Evolution with {selection_method} selection, {crossover_method} crossover, elitism={use_elitism}')
    plt.grid(True)
    plt.savefig('energy_evolution.png')
    plt.show()
    
    # Visualize the best solution
    print("\nVisualizing the best spin configuration found")
    visualize_polymer_spins(atoms, bonds, best_solution)
    
    # Option 2: Compare different strategies
    print("\nComparing different GA strategies")
    results = compare_strategies(adjacency_list, num_atoms)
    
    # Plot comparison results
    plt.figure(figsize=(12, 8))
    for key, history in results.items():
        plt.plot(range(1, len(history) + 1), [-f for f in history], label=key)
    plt.xlabel('Generation')
    plt.ylabel('Energy')
    plt.title('Energy Evolution for Different GA Strategies')
    plt.legend()
    plt.grid(True)
    plt.savefig('strategy_comparison.png')
    plt.show()

if __name__ == "__main__":
    main()
