import numpy as np
import matplotlib.pyplot as plt
import random
import time
from matplotlib.ticker import MaxNLocator
from copy import deepcopy
import matplotlib

matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # Use a font that supports Unicode

# Set matplotlib to use 'Agg' backend which is more robust for various environments
matplotlib.use('Agg')


# Generate random city coordinates
def generate_cities(num_cities, seed=42):
    """Generate city coordinates"""
    np.random.seed(seed)
    cities = np.random.rand(num_cities, 2) * 100
    return cities


# Calculate distance matrix
def calculate_distance_matrix(cities):
    """Calculate distance matrix between all cities"""
    n = len(cities)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate Euclidean distance
                distance_matrix[i, j] = np.sqrt(np.sum((cities[i] - cities[j]) ** 2))
    return distance_matrix


# Calculate path length for a single salesman
def calculate_path_length(path, distance_matrix):
    """Calculate the length of a path"""
    if not path:  # If path is empty
        return 0

    length = 0
    for i in range(len(path) - 1):
        length += distance_matrix[path[i], path[i + 1]]
    # Add distance from last city back to starting city
    length += distance_matrix[path[-1], path[0]]
    return length


# Calculate total path length for multiple salesmen
def calculate_total_path_length(paths, distance_matrix):
    """Calculate the total length of paths for all salesmen"""
    total_length = 0
    for path in paths:
        total_length += calculate_path_length(path, distance_matrix)
    return total_length


# Create a random solution
def create_random_solution(num_cities, num_salesmen=2):
    """Create a random solution with multiple salesmen"""
    # Create a list of all cities except depot city (0)
    cities = list(range(1, num_cities))
    random.shuffle(cities)

    # Determine split points for salesmen
    split_points = sorted(random.sample(range(1, len(cities)), num_salesmen - 1))

    # Split cities among salesmen
    paths = []
    start_idx = 0

    for point in split_points:
        # Add depot city (0) at the beginning of each path
        path = [0] + cities[start_idx:point]
        paths.append(path)
        start_idx = point

    # Add the last group
    paths.append([0] + cities[start_idx:])

    return paths


# Create a greedy solution
def create_greedy_solution(distance_matrix, num_salesmen=2, depot_city=0):
    """Create an initial solution using greedy algorithm"""
    num_cities = distance_matrix.shape[0]

    # Create a list of all cities except depot
    cities = list(range(1, num_cities))

    # Sort cities by distance from depot
    cities.sort(key=lambda city: distance_matrix[depot_city, city])

    # Determine split points for salesmen (try to balance workload)
    paths = [[] for _ in range(num_salesmen)]

    # Add depot city to all paths
    for i in range(num_salesmen):
        paths[i].append(depot_city)

    # Assign cities in a round-robin fashion (starting with the closest ones)
    current_salesman = 0
    for city in cities:
        paths[current_salesman].append(city)
        current_salesman = (current_salesman + 1) % num_salesmen

    return paths


# Apply 2-opt local search
def apply_2opt(paths, distance_matrix, max_iterations=100):
    """Apply 2-opt local search to improve solution"""
    best_paths = deepcopy(paths)
    best_distance = calculate_total_path_length(paths, distance_matrix)

    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        # Try 2-opt moves for each salesman's path
        for s in range(len(best_paths)):
            path = best_paths[s]

            # Skip paths that are too short
            if len(path) <= 3:  # depot + 2 cities minimum for 2-opt
                continue

            for i in range(1, len(path) - 1):  # Skip depot
                for j in range(i + 1, len(path)):
                    if j - i == 1:
                        continue  # Skip adjacent cities

                    # 2-opt swap
                    new_path = path.copy()
                    new_path[i:j] = new_path[i:j][::-1]

                    # Create new solution
                    new_paths = deepcopy(best_paths)
                    new_paths[s] = new_path

                    new_distance = calculate_total_path_length(new_paths, distance_matrix)

                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_paths = new_paths
                        improved = True
                        break

                if improved:
                    break

            if improved:
                break

    return best_paths, best_distance


# Balance workloads between salesmen
def balance_workload(paths, distance_matrix, max_iterations=50):
    """Balance workload between salesmen by moving cities"""
    best_paths = deepcopy(paths)
    best_distance = calculate_total_path_length(paths, distance_matrix)

    # Calculate path lengths
    path_lengths = [calculate_path_length(path, distance_matrix) for path in best_paths]

    for _ in range(max_iterations):
        # Find salesman with longest and shortest routes
        longest_idx = np.argmax(path_lengths)
        shortest_idx = np.argmin(path_lengths)

        if longest_idx == shortest_idx or len(best_paths[longest_idx]) <= 2:  # Just depot and one city
            break

        # Try to move a city from longest to shortest path
        for i in range(1, len(best_paths[longest_idx])):  # Skip depot
            city = best_paths[longest_idx][i]

            # Create new solution by moving city
            new_paths = deepcopy(best_paths)
            new_paths[longest_idx].pop(i)
            insert_pos = random.randint(1, len(new_paths[shortest_idx]))
            new_paths[shortest_idx].insert(insert_pos, city)

            new_distance = calculate_total_path_length(new_paths, distance_matrix)

            if new_distance < best_distance:
                best_distance = new_distance
                best_paths = new_paths
                path_lengths = [calculate_path_length(path, distance_matrix) for path in best_paths]
                break

    return best_paths, best_distance


# Base GWO algorithm for multiple TSP
class mTSP_GWO:
    def __init__(self, cities, population_size=30, max_iterations=100, num_salesmen=2,
                 use_subgroups=True, use_uniform_init=True, use_2opt=True, use_balancing=True):
        self.cities = cities
        self.num_cities = len(cities)
        self.distance_matrix = calculate_distance_matrix(cities)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.num_salesmen = num_salesmen
        self.convergence_curve = np.zeros(max_iterations)
        self.fitness_curve = np.zeros(max_iterations)  # Added fitness curve
        self.execution_time = 0

        # Optimization methods to use
        self.use_subgroups = use_subgroups
        self.use_uniform_init = use_uniform_init
        self.use_2opt = use_2opt
        self.use_balancing = use_balancing

        # Subgroup settings
        self.num_subgroups = 3 if use_subgroups else 1
        self.subgroup_size = population_size // self.num_subgroups if use_subgroups else population_size

        # Initialize population
        self.initialize_population()

        # Global best solution
        self.global_best_pos = None
        self.global_best_score = float('inf')

        # For fitness calculation (maximization)
        self.initial_distance = None  # Will store initial solution distance

    def initialize_population(self):
        """Initialize population"""
        # Initialize subgroups
        self.subgroups = []

        for sg_idx in range(self.num_subgroups):
            # Determine subgroup size
            if sg_idx == self.num_subgroups - 1:
                sg_size = self.population_size - sg_idx * self.subgroup_size
            else:
                sg_size = self.subgroup_size

            # Create wolves (solutions)
            wolves = []

            if self.use_uniform_init:
                # Uniform initialization: mix of random and greedy solutions
                num_greedy = sg_size // 3  # 1/3 greedy solutions
                num_random = sg_size - num_greedy

                # Generate random solutions
                for _ in range(num_random):
                    wolves.append(create_random_solution(self.num_cities, self.num_salesmen))

                # Generate greedy solutions
                for _ in range(num_greedy):
                    wolves.append(create_greedy_solution(self.distance_matrix, self.num_salesmen))

                # Apply 2-opt to some solutions if enabled
                if self.use_2opt:
                    num_2opt = max(1, sg_size // 10)  # 10% of solutions
                    for i in range(num_2opt):
                        improved_paths, _ = apply_2opt(wolves[i], self.distance_matrix, max_iterations=10)
                        wolves[i] = improved_paths

                # Apply balancing to some solutions if enabled
                if self.use_balancing:
                    num_balance = max(1, sg_size // 5)  # 20% of solutions
                    for i in range(num_balance):
                        balanced_paths, _ = balance_workload(wolves[i], self.distance_matrix)
                        wolves[i] = balanced_paths
            else:
                # Simple random initialization
                for _ in range(sg_size):
                    wolves.append(create_random_solution(self.num_cities, self.num_salesmen))

            # Create subgroup
            subgroup = {
                'wolves': wolves,
                'fitness': np.zeros(sg_size),
                'alpha_pos': None,
                'alpha_score': float('inf'),
                'beta_pos': None,
                'beta_score': float('inf'),
                'delta_pos': None,
                'delta_score': float('inf')
            }

            self.subgroups.append(subgroup)

    def evaluate_fitness(self, subgroup_idx):
        """Evaluate fitness for wolves in a subgroup"""
        subgroup = self.subgroups[subgroup_idx]

        for i in range(len(subgroup['wolves'])):
            # Calculate distance (lower is better)
            distance = calculate_total_path_length(subgroup['wolves'][i], self.distance_matrix)
            subgroup['fitness'][i] = distance

            # Store initial distance for fitness scaling if not already set
            if self.initial_distance is None:
                self.initial_distance = distance

        # Update alpha, beta, and delta wolves
        for i in range(len(subgroup['wolves'])):
            if subgroup['fitness'][i] < subgroup['alpha_score']:
                subgroup['delta_score'] = subgroup['beta_score']
                subgroup['delta_pos'] = deepcopy(subgroup['beta_pos']) if subgroup['beta_pos'] is not None else None

                subgroup['beta_score'] = subgroup['alpha_score']
                subgroup['beta_pos'] = deepcopy(subgroup['alpha_pos']) if subgroup['alpha_pos'] is not None else None

                subgroup['alpha_score'] = subgroup['fitness'][i]
                subgroup['alpha_pos'] = deepcopy(subgroup['wolves'][i])

                # Update global best solution
                if subgroup['alpha_score'] < self.global_best_score:
                    self.global_best_score = subgroup['alpha_score']
                    self.global_best_pos = deepcopy(subgroup['alpha_pos'])

            elif subgroup['fitness'][i] < subgroup['beta_score']:
                subgroup['delta_score'] = subgroup['beta_score']
                subgroup['delta_pos'] = deepcopy(subgroup['beta_pos']) if subgroup['beta_pos'] is not None else None

                subgroup['beta_score'] = subgroup['fitness'][i]
                subgroup['beta_pos'] = deepcopy(subgroup['wolves'][i])

            elif subgroup['fitness'][i] < subgroup['delta_score']:
                subgroup['delta_score'] = subgroup['fitness'][i]
                subgroup['delta_pos'] = deepcopy(subgroup['wolves'][i])

    def crossover_mtsp(self, parent1, parent2, a):
        """Crossover operation for mTSP"""
        # Deep copy to avoid modifying originals
        child = deepcopy(parent1)

        # Skip crossover if a parameter is too low
        if random.random() > a:
            return child

        # Randomly select a salesman to apply crossover
        salesman_idx = random.randint(0, self.num_salesmen - 1)

        # Get the paths of the selected salesman
        path1 = parent1[salesman_idx]
        path2 = parent2[salesman_idx]

        # Skip crossover if paths are too short
        if len(path1) <= 2 or len(path2) <= 2:
            return child

        # Create a new path using order crossover (OX)
        # Keep the depot city at position 0
        # Apply crossover only to the cities (positions 1 onwards)
        new_path = [path1[0]]  # Start with depot

        # Select a segment from path1 (excluding depot)
        start = random.randint(1, len(path1) - 2)
        end = random.randint(start, len(path1) - 1)

        # Add the selected segment from path1
        segment = path1[start:end + 1]
        new_path.extend(segment)

        # Add remaining cities from path2 in their original order
        for city in path2[1:]:  # Skip depot in path2
            if city not in segment:
                new_path.append(city)

        # Update the child's path for the selected salesman
        child[salesman_idx] = new_path

        # Ensure no city is assigned to more than one salesman
        # Get all cities in the new solution
        all_cities = []
        for i, path in enumerate(child):
            if i != salesman_idx:  # Skip the path we just modified
                all_cities.extend(path[1:])  # Skip depot

        # Find and remove duplicates
        duplicates = []
        unique_cities = set()

        for city in all_cities:
            if city in new_path[1:]:  # If city is in new_path (except depot)
                duplicates.append(city)
            else:
                unique_cities.add(city)

        # Remove duplicates from other paths
        for i in range(len(child)):
            if i != salesman_idx:
                child[i] = [city for city in child[i] if city == 0 or city not in new_path]

        # Check if any cities are missing
        all_cities_set = set(range(1, self.num_cities))
        current_cities = set()

        for path in child:
            current_cities.update(path[1:])  # Skip depot

        missing_cities = all_cities_set - current_cities

        # Randomly assign missing cities
        for city in missing_cities:
            salesman_idx = random.randint(0, self.num_salesmen - 1)
            insert_pos = random.randint(1, len(child[salesman_idx]))
            child[salesman_idx].insert(insert_pos, city)

        return child


