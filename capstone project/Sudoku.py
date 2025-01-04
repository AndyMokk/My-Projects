from Algorithm import solve_with_backtracking, solve_with_simulated_annealing, solve_with_bitmasks, \
    solve_with_crosshatching, solve_with_constraint_programming, solve_with_custom_csp, solve_with_genetic, \
    solve_with_crosshatching_new, solve_with_bitmasks_new, solve_merge, solver_gen
import random as rand
import numpy as np
import copy
import constraint
import time
import statistics
from tqdm import tqdm


class Board:

    def __init__(self, size):
        self.size = size
        self.grid = [['.' for _ in range(size)] for _ in range(size)]
        self.c = copy.deepcopy(self.grid)
        self.ans = []

    def print_board(self):  # Renamed to avoid conflict with built-in print
        bd = int(np.sqrt(self.size))
        end = self.size - 1
        if self.size < 10:
            for i in range(self.size):
                if i % bd == 0 and i != 0:
                    print("- " * (self.size + bd - 1))

                for j in range(len(self.grid[0])):
                    if j % bd == 0 and j != 0:
                        print("| ", end="")

                    if j == end:
                        print(self.grid[i][j], end="\n")
                    else:
                        print(str(self.grid[i][j]) + " ", end="")
        else:
            for i in range(len(self.grid)):
                if i % bd == 0 and i != 0:
                    print("- " * (self.size + 3 * bd - 1))

                for j in range(len(self.grid[0])):
                    if j % bd == 0 and j != 0:
                        print("| ", end="")

                    if j == end:
                        print(self.grid[i][j], end="\n")
                    else:
                        if self.grid[i][j] == "." or self.grid[i][j] < int(10):
                            print(str(self.grid[i][j]) + "  ", end="")
                        else:
                            print(str(self.grid[i][j]) + " ", end="")
        print("\n")

    def nextEmpty(self):  # Returns the next empty pair of coordinates on the grid
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.grid[i][j] == '.':
                    return (i, j)  # _y, _x
        return None

    def isValid(self, num, pos):  # Checks if the placement is valid
        for i in range(len(self.grid[0])):
            if self.grid[pos[0]][i] == num and pos[1] != i:
                return False

        for i in range(len(self.grid)):
            if self.grid[i][pos[1]] == num and pos[0] != i:
                return False

        v1 = int(np.sqrt(self.size))
        _x = pos[1] // v1
        _y = pos[0] // v1

        for i in range(_y * v1, _y * v1 + v1):
            for j in range(_x * v1, _x * v1 + v1):
                if self.grid[i][j] == num and (i, j) != pos:
                    return False

        return True

    def generate(self, difficulty):
        self.grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        sqrt_size = int(np.sqrt(self.size))
        rand_row = [_ for _ in range(self.size)]
        rand.shuffle(rand_row)
        rand_col = [_ for _ in range(self.size)]
        rand.shuffle(rand_col)
        rand_num = [_ for _ in range(1, self.size + 1)]
        rand.shuffle(rand_num)

        for i in range(sqrt_size):
            self.grid[rand_col[i]][rand_row[i]] = rand_num[i]
        self.solver_gen()
        self.ans = copy.deepcopy(self.grid)
        cells = [(i, j) for i in range(self.size) for j in range(self.size)]
        rand.shuffle(cells)
        cells_removed = 0

        
        # Existing logic for other difficulties
        if difficulty == 0:  # Easy
            clues = int(self.size ** 2 * 0.6)
        elif difficulty == 1:  # Medium
            clues = int(self.size ** 2 * 0.5)
        else:
            clues = int(self.size ** 2 * 0.4)  #Default to medium
            if self.size ==16:
                clues=115

        cells_removed = 0
        print('start')
        for (row, col) in cells:
            if cells_removed >= (self.size ** 2 - clues):
                break

            # Temporarily remove the cell's value
            removed_value = self.grid[row][col]
            self.grid[row][col] = '.'
            # Check if the puzzle still has a unique solution
            num_solutions = self.count_solutions(self.grid)
            print(cells_removed,num_solutions)
            if num_solutions:
                # If not unique, put the value back
                self.grid[row][col] = removed_value
            else:
                cells_removed += 1
        self.c = copy.deepcopy(self.grid)

    def count_solutions(self, grid):
        problem = constraint.Problem(solver=constraint.RecursiveBacktrackingSolver())
        size = self.size

        # Define variables with domains (1 to size or specific values if fixed)
        for i in range(size):
            for j in range(size):
                var = f'V{i}_{j}'
                if grid[i][j] == '.':
                    problem.addVariable(var, range(1, size + 1))
                else:
                    problem.addVariable(var, [grid[i][j]])

        # Add row constraints
        for i in range(size):
            row_vars = [f'V{i}_{j}' for j in range(size)]
            problem.addConstraint(constraint.AllDifferentConstraint(), row_vars)

        # Add column constraints
        for j in range(size):
            col_vars = [f'V{i}_{j}' for i in range(size)]
            problem.addConstraint(constraint.AllDifferentConstraint(), col_vars)

        # Add subgrid constraints
        n = int(np.sqrt(size))
        for i in range(0, size, n):
            for j in range(0, size, n):
                subgrid_vars = [f'V{x}_{y}' for x in range(i, i + n) for y in range(j, j + n)]
                problem.addConstraint(constraint.AllDifferentConstraint(), subgrid_vars)

        first_solution = problem.getSolution()
        if not first_solution:
            # No solution exists
            return False  # Or handle as per your requirement

        # Step 2: Add a constraint to exclude the first solution
        def exclude_first_solution(*args):
            # Compare each variable's value with the first solution
            return not all(args[i] == first_solution[var] for i, var in enumerate(problem._variables))

        problem.addConstraint(exclude_first_solution, problem._variables)

        # Step 3: Attempt to find a second solution
        second_solution = problem.getSolution()

        return second_solution

    def binput(self, _y, _x, num):  # Inputs a number into the grid
        if 1 <= num <= self.size:
            if self.c[_y - 1][_x - 1] == '.':
                if self.isValid(num, (_y - 1, _x - 1)):
                    self.grid[_y - 1][_x - 1] = num
                    return True
                else:
                    return False
            else:
                # Check if the cell was originally empty
                if self.c[_y - 1][_x - 1] == '.':
                    if self.isValid(num, (_y - 1, _x - 1)):
                        self.grid[_y - 1][_x - 1] = num
                        self.print_board()
                    else:
                        return False
                else:
                    return False
        return False

    def solve_with_backtracking(self):
        start_time = time.time()
        success = solve_with_backtracking(self.grid)
        end_time = time.time()
        computation_time = end_time - start_time
        return success, computation_time

    def solve_with_simulated_annealing(self):
        start_time = time.time()
        success = solve_with_simulated_annealing(self.grid, self.c)
        end_time = time.time()
        computation_time = end_time - start_time
        return success, computation_time

    def solve_with_bitmasks(self):
        start_time = time.time()
        success = solve_with_bitmasks(self.grid)
        end_time = time.time()
        computation_time = end_time - start_time
        return success, computation_time

    def solve_with_bitmasks_new(self):
        start_time = time.time()
        success = solve_with_bitmasks_new(self)
        end_time = time.time()
        computation_time = end_time - start_time
        return success, computation_time

    def solve_with_crosshatching(self):
        start_time = time.time()
        success = solve_with_crosshatching(self.grid)
        end_time = time.time()
        computation_time = end_time - start_time
        return success, computation_time

    def solve_with_constraint_programming(self):
        start_time = time.time()
        success = solve_with_constraint_programming(self.grid)
        end_time = time.time()
        computation_time = end_time - start_time
        return success, computation_time

    def solve_with_genetic(self):
        start_time = time.time()
        success = solve_with_genetic(self.grid)
        end_time = time.time()
        computation_time = end_time - start_time
        return success, computation_time

    def solve_with_custom_csp(self):
        start_time = time.time()
        success = solve_with_custom_csp(self.grid)
        end_time = time.time()
        computation_time = end_time - start_time
        return success, computation_time

    def solve_with_crosshatching_new(self):
        start_time = time.time()
        success = solve_with_crosshatching_new(self)
        end_time = time.time()
        computation_time = end_time - start_time
        return success, computation_time

    def solve_merge(self):
        start_time = time.time()
        success = solve_merge(self)
        end_time = time.time()
        computation_time = end_time - start_time
        return success, computation_time

    def solver_gen(self):
        solver_gen(self.grid)


import multiprocessing as mp


def run_algorithm(board, algo_name):
    if algo_name == "Backtracking":
        return board.solve_with_backtracking()
    elif algo_name == "Simulated Annealing":
        return board.solve_with_simulated_annealing()
    elif algo_name == "Constraint Programming":
        return board.solve_with_constraint_programming()
    elif algo_name == "Custom CSP":
        return board.solve_with_custom_csp()
    elif algo_name == "Bitmasking":
        return board.solve_with_bitmasks()
    elif algo_name == "Bitmasking New":
        return board.solve_with_bitmasks_new()
    elif algo_name == "Cross-Hatching":
        return board.solve_with_crosshatching()
    elif algo_name == "Cross-Hatching New":
        return board.solve_with_crosshatching_new()
    #elif algo_name == "Genetic":
        #return board.solve_with_genetic()
    elif algo_name == "Merge":
        return board.solve_merge()


def run_single_trial(params):
    trial_num, size, difficulty, algo_names = params

    # Initialize results dictionary for this trial
    trial_results = {name: {difficulty: None} for name in algo_names}

    # Generate a new puzzle for this trial
    print('grid',trial_num,'generating')
    board = Board(size)
    board.generate(difficulty)
    print('grid',trial_num,'generated')

    # Test each algorithm
    for name in algo_names:
        # Create a new board copy for each algorithm
        board_copy = Board(size)
        board_copy.grid = [row[:] for row in board.grid]
        board_copy.c = [row[:] for row in board.c]

        try:
            success, solve_time = run_algorithm(board_copy, name)
            if board_copy.grid != board.ans:
                print('Not same solution:', name)
                board_copy.print_board()
                print(board.ans)
            if success:
                trial_results[name][difficulty] = solve_time
            else:
                print(f"{name}: Failed to solve")
        except Exception as e:
            print(f"{name}: Error occurred - {str(e)}")

    return trial_results

def compare_algorithms(num_trials=10, size=9, level=0):


    algo_names = [
        "Backtracking",
        "Simulated Annealing",
        "Constraint Programming",
        "Custom CSP",
        "Bitmasking",
        "Bitmasking New",
        "Cross-Hatching",
        "Cross-Hatching New",
        #"Genetic",
        "Merge"
    ]
    print('start comparing')
    difficulty_levels = [level]
    results = {name: {diff: [] for diff in difficulty_levels}
               for name in algo_names}

    # Create a pool of workers
    num_processes = min(mp.cpu_count(), num_trials)
    pool = mp.Pool(processes=num_processes)

    try:
        # Prepare parameters for each trial
        params = [(i, size, difficulty_levels[0], algo_names) for i in range(num_trials)]

        # Run trials in parallel
        trial_results = tqdm(pool.imap(run_single_trial, params),total=num_trials)

        # Aggregate results
        for trial_result in trial_results:
            for algo_name in algo_names:
                for diff in difficulty_levels:
                    if trial_result[algo_name][diff] is not None:
                        results[algo_name][diff].append(trial_result[algo_name][diff])
                        for name in algo_names:
                            for diff in difficulty_levels:
                                times = results[name][diff]
                                if times:
                                    mean_time = statistics.mean(times)
                        print("\nSummary Statistics:")
                        print("-" * 100)
                        print(
                            f"{'Algorithm':<25} {'Difficulty':<10} {'Mean':<12} {'Min':<12} {'Max':<12} {'Median':<12} {'Success Rate':<12}")
                        print("-" * 100)

                        for name in algo_names:
                            for diff in difficulty_levels:
                                times = results[name][diff]
                                if times:
                                    mean_time = statistics.mean(times)
                                    min_time = min(times)
                                    max_time = max(times)
                                    median_time = statistics.median(times)
                                    success_rate = len(times) / num_trials * 100
                                    print(
                                        f"{name:<25} {diff:<10} {mean_time:.6f}  {min_time:.6f}  {max_time:.6f}  {median_time:.6f}  {success_rate:.1f}%")
                                else:
                                    print(f"{name:<25} {diff:<10} No successful solutions")

    finally:
        pool.close()
        pool.join()

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 100)
    print(
        f"{'Algorithm':<25} {'Difficulty':<10} {'Mean':<12} {'Min':<12} {'Max':<12} {'Median':<12} {'Success Rate':<12}")
    print("-" * 100)

    for name in algo_names:
        for diff in difficulty_levels:
            times = results[name][diff]
            if times:
                mean_time = statistics.mean(times)
                min_time = min(times)
                max_time = max(times)
                median_time = statistics.median(times)
                success_rate = len(times) / num_trials * 100
                print(
                    f"{name:<25} {diff:<10} {mean_time:.6f}  {min_time:.6f}  {max_time:.6f}  {median_time:.6f}  {success_rate:.1f}%")
            else:
                print(f"{name:<25} {diff:<10} No successful solutions")

    print("\nDetailed Results:")
    print("-" * 80)
    print(f"{'Algorithm':<25} {'Difficulty':<10} {'All Times (seconds)'}")
    print("-" * 80)

    for name in algo_names:
        for diff in difficulty_levels:
            times = results[name][diff]
            times_str = ", ".join([f"{t:.6f}" for t in times])
            print(f"{name:<25} {diff:<10} [{times_str}]")
