from Algorithm import solve_with_backtracking, solve_with_simulated_annealing, solve_with_bitmasks, solve_with_crosshatching, solve_with_constraint_programming, solve_with_custom_csp, solve_with_genetic
import random as rand
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import pandas as pd

class Board:

    def __init__(self, size):
        self.size = size
        self.grid = [['.' for _ in range(size)] for _ in range(size)]
        self.copy = copy.deepcopy(self.grid)
        self.difficulty = 2


    def print_board(self):  # Renamed to avoid conflict with built-in print
        bd = int(np.sqrt(self.size))
        end = self.size - 1
        if self.size <10:
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
                    print("- " * (self.size + 3*bd -1))

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

    def generate(self, difficulty):  # Generates a board
        self.grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        n = int(np.sqrt(self.size))

        for _ in range(n):
            _l = list(range(1, self.size + 1))
            for _y in range(n):
                for _x in range(n):
                    _num = rand.choice(_l)
                    self.grid[_y][_x] = _num
                    _l.remove(_num)

        self.solve_with_backtracking()  # Solves the board to ensure a valid solution
        self.removeNumbers(difficulty)  # Removes numbers based on difficulty
        self.copy = copy.deepcopy(self.grid)

    def removeNumbers(self, difficulty):  # Removes numbers to set difficulty
        n = int(np.sqrt(self.size))
        fn = int(n * n * 4.5 - 9.5 * n + 5)
        if difficulty == 0:
            _squares_to_remove = fn
        elif difficulty == 1:
            _squares_to_remove = int(self.size * self.size / 2)
        elif difficulty == 2:
            _squares_to_remove = int(self.size * self.size - fn)
        else:
            return

        for _ in range(n):
            _counter = 0
            while _counter < n + 1:
                _y = rand.randint(_ * n, _ * n + n - 1)
                _x = rand.randint(_ * n, _ * n + n - 1)
                if self.grid[_y][_x] != '.':
                    self.grid[_y][_x] = '.'
                    _counter += 1

        _squares_to_remove -= int(n * (n + 1))
        _counter = 0
        while _counter < _squares_to_remove:
            _y = rand.randint(0, self.size - 1)
            _x = rand.randint(0, self.size - 1)

            if self.grid[_y][_x] != '.':
                self.grid[_y][_x] = '.'
                _counter += 1

        return self.grid

    def findvnum(self,_y, _x):
        vnum = [x for x in range(1, self.size + 1)]

        for i in range(self.size):
            if self.grid[_y][i] != '.' and self.grid[_y][i] in vnum:
                vnum.remove(self.grid[_y][i])

        for i in range(self.size):
            if self.grid[i][_x] != '.' and self.grid[i][_x] in vnum:
                vnum.remove(self.grid[i][_x])

        v1 = int(np.sqrt(self.size))
        gx = _x // v1
        gy = _y // v1

        for i in range(gy * v1, gy * v1 + v1):
            for j in range(gx * v1, gx * v1 + v1):
                if self.grid[i][j] != '.' and self.grid[i][j] in vnum:
                    vnum.remove(self.grid[i][j])

        return vnum

    def binput(self, _y, _x, num):  # Inputs a number into the grid
        if 1 <= num <= self.size:
            if self.copy[_y - 1][_x - 1] == '.':
                if self.isValid(num, (_y - 1, _x - 1)):
                    self.grid[_y - 1][_x - 1] = num
                    return True
                else:
                    return False
            else:
                # Check if the cell was originally empty
                if self.copy[_y - 1][_x - 1] == '.':
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
        success = solve_with_simulated_annealing(self.grid, self.copy)
        end_time = time.time()
        computation_time = end_time - start_time
        return success, computation_time

    def solve_with_bitmasks(self):
        start_time = time.time()
        success = solve_with_bitmasks(self.grid)
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

    def compare_algorithms(self, num_puzzles):
        algorithms = {
            'Backtracking': self.solve_with_backtracking,
            'Bitmasks':self.solve_with_bitmasks,
            'Crosshatching': self.solve_with_crosshatching,
            #'Simulated Annealing': self.solve_with_simulated_annealing,
            #'Genetic': self.solve_with_genetic,
            'Constraint Programming': self.solve_with_constraint_programming,
            'Custom CSP': self.solve_with_custom_csp
        }

        # Initialize dictionaries to accumulate total times and success counts
        total_times = {name: 0.0 for name in algorithms}
        success_counts = {name: 0 for name in algorithms}

        for puzzle_num in range(1, num_puzzles + 1):
            print(f"\n=== Generating and Solving Puzzle {puzzle_num} ===")
            # Generate a new Sudoku puzzle
            self.generate(self.difficulty)
            self.copy = copy.deepcopy(self.grid)  # Preserve the original puzzle
            print(f"Puzzle {puzzle_num}:")
            self.print_board()

            for name, solver in algorithms.items():
                print(f"Running {name}...")
                # Reset the grid to the original puzzle before solving
                self.grid = copy.deepcopy(self.copy)

                # Run the solver and measure the time
                success, comp_time = solver()
                total_times[name] += comp_time
                if success:
                    success_counts[name] += 1
                    print(f"{name}: Success in {comp_time:.6f} seconds.")
                else:
                    print(f"{name}: Failure in {comp_time:.6f} seconds.")

            # Optionally, print a separator or any additional information
            print(f"=== Completed Puzzle {puzzle_num} ===\n")

        # Calculate average times
        average_times = {name: total_times[name] / num_puzzles for name in algorithms}
        average_success_rates = {name: (success_counts[name] / num_puzzles) * 100 for name in algorithms}

        # Display the results in a table
        results_df = pd.DataFrame({
            'Algorithm': list(average_times.keys()),
            'Average Time (s)': list(average_times.values()),
            'Success Rate (%)': list(average_success_rates.values())
        })

        print("\n=== Comparison Results ===")
        print(results_df)

        # Plot the results
        plt.figure(figsize=(14, 7))

        # Bar plot for Average Time
        plt.subplot(1, 2, 1)
        solver_names = list(average_times.keys())
        solver_avg_times = list(average_times.values())
        plt.bar(solver_names, solver_avg_times, color='skyblue')
        plt.xlabel('Algorithms')
        plt.ylabel('Average Computation Time (seconds)')
        plt.title('Average Computation Time by Algorithm')
        plt.xticks(rotation=45, ha='right')
        for i, v in enumerate(solver_avg_times):
            plt.text(i, v + max(solver_avg_times) * 0.01, f"{v:.6f}", ha='center', va='bottom')

        # Bar plot for Success Rate
        plt.subplot(1, 2, 2)
        solver_success = list(average_success_rates.values())
        plt.bar(solver_names, solver_success, color='lightgreen')
        plt.xlabel('Algorithms')
        plt.ylabel('Success Rate (%)')
        plt.title('Success Rate by Algorithm')
        plt.xticks(rotation=45, ha='right')
        for i, v in enumerate(solver_success):
            plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom')

        plt.tight_layout()
        plt.show()
