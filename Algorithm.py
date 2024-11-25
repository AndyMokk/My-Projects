import random as rand
import numpy as np
import copy
import constraint
import itertools



def solve_with_backtracking(grid):
    def isValid(num, pos):
        for i in range(len(grid[0])):
            if grid[pos[0]][i] == num and pos[1] != i:
                return False

        for i in range(len(grid)):
            if grid[i][pos[1]] == num and pos[0] != i:
                return False

        v1 = int(np.sqrt(len(grid)))
        _x = pos[1] // v1
        _y = pos[0] // v1

        for i in range(_y * v1, _y * v1 + v1):
            for j in range(_x * v1, _x * v1 + v1):
                if grid[i][j] == num and (i, j) != pos:
                    return False

        return True

    def _brute_force_helper(row, col):
        if row == len(grid):
            return True

        if col == len(grid):
            return _brute_force_helper(row + 1, 0)

        if grid[row][col] != '.':
            return _brute_force_helper(row, col + 1)

        for num in range(1, len(grid) + 1):
            if isValid(num, (row, col)):
                grid[row][col] = num
                if _brute_force_helper(row, col + 1):
                    return True
                grid[row][col] = '.'

        return False

    return _brute_force_helper(0, 0)

def solve_with_simulated_annealing(grid, c_grid, max_iter=500000, initial_temp=1.0, cooling_rate=0.999995):
    def cost(grid):
        conflicts = 0
        size = len(grid)
        n = int(np.sqrt(size))
        for row in range(size):
            row_vals = grid[row]
            conflicts += size - len(set(row_vals))
        for col in range(size):
            col_vals = [grid[row][col] for row in range(size)]
            conflicts += size - len(set(col_vals))
        # Block conflicts
        for block_row in range(n):
            for block_col in range(n):
                block_vals = []
                for i in range(block_row * n, (block_row + 1) * n):
                    for j in range(block_col * n, (block_col + 1) * n):
                        block_vals.append(grid[i][j])
                conflicts += size - len(set(block_vals))
        return conflicts

    def initialize_grid(grid, c_grid):
        size = len(grid)
        n = int(np.sqrt(size))
        for block_row in range(n):
            for block_col in range(n):
                block_nums = set()
                for i in range(block_row * n, (block_row + 1) * n):
                    for j in range(block_col * n, (block_col + 1) * n):
                        if c_grid[i][j] != '.':
                            block_nums.add(c_grid[i][j])
                nums_to_fill = list(set(range(1, size + 1)) - block_nums)
                rand.shuffle(nums_to_fill)
                idx = 0
                for i in range(block_row * n, (block_row + 1) * n):
                    for j in range(block_col * n, (block_col + 1) * n):
                        if c_grid[i][j] == '.':
                            grid[i][j] = nums_to_fill[idx]
                            idx += 1
                        else:
                            grid[i][j] = c_grid[i][j]

    def random_swap(grid, c_grid):
        size = len(grid)
        n = int(np.sqrt(size))
        while True:
            block_row = rand.randint(0, n - 1)
            block_col = rand.randint(0, n - 1)
            cells = []
            for i in range(block_row * n, (block_row + 1) * n):
                for j in range(block_col * n, (block_col + 1) * n):
                    if c_grid[i][j] == '.':
                        cells.append((i, j))
            if len(cells) < 2:
                continue
            (row1, col1), (row2, col2) = rand.sample(cells, 2)
            grid[row1][col1], grid[row2][col2] = grid[row2][col2], grid[row1][col1]
            break

    # Initialize grid
    initialize_grid(grid, c_grid)
    current_temp = initial_temp
    current_cost = cost(grid)

    for iteration in range(max_iter):
        if current_cost == 0:
            return True

        new_grid = [row[:] for row in grid]
        random_swap(new_grid, c_grid)
        new_cost = cost(new_grid)
        delta_cost = new_cost - current_cost

        if delta_cost <= 0 or rand.random() < np.exp(-delta_cost / current_temp):
            grid[:] = new_grid
            current_cost = new_cost

        current_temp *= cooling_rate

    return False


def solve_with_bitmasks(grid):
    def nextEmpty():
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '.':
                    return (i, j)
        return None

    def findvnum( _y, _x):
        size = len(grid)
        vnum = [x for x in range(1, size + 1)]

        for i in range(size):
            if grid[_y][i] != '.' and grid[_y][i] in vnum:
                vnum.remove(grid[_y][i])

        for i in range(size):
            if grid[i][_x] != '.' and grid[i][_x] in vnum:
                vnum.remove(grid[i][_x])

        v1 = int(np.sqrt(size))
        gx = _x // v1
        gy = _y // v1

        for i in range(gy * v1, gy * v1 + v1):
            for j in range(gx * v1, gx * v1 + v1):
                if grid[i][j] != '.' and grid[i][j] in vnum:
                    vnum.remove(grid[i][j])

        return vnum

    _find = nextEmpty()
    if not _find:
        return True
    else:
        _y, _x = _find

    vnum = findvnum(_y, _x)

    for i in vnum:
        grid[_y][_x] = i
        if solve_with_bitmasks(grid):
            return True

        grid[_y][_x] = '.'

    return False

def solve_with_crosshatching(grid):
    def nextEmpty():
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '.':
                    return (i, j)
        return None

    def findvnum(_y, _x):
        size = len(grid)
        vnum = [x for x in range(1, size + 1)]

        for i in range(size):
            if grid[_y][i] != '.' and grid[_y][i] in vnum:
                vnum.remove(grid[_y][i])

        for i in range(size):
            if grid[i][_x] != '.' and grid[i][_x] in vnum:
                vnum.remove(grid[i][_x])

        v1 = int(np.sqrt(size))
        gx = _x // v1
        gy = _y // v1

        for i in range(gy * v1, gy * v1 + v1):
            for j in range(gx * v1, gx * v1 + v1):
                if grid[i][j] != '.' and grid[i][j] in vnum:
                    vnum.remove(grid[i][j])

        return vnum

    def get_setting():
        vnum_grid = [[[] for _ in range(len(grid))] for _ in range(len(grid))]
        num_vnum = np.array([[int(99) for _ in range(len(grid[0]))] for _ in range(len(grid[0]))])
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '.':
                    vnum_grid[i][j] = findvnum(i,j)
                    num_vnum[i][j] = len(vnum_grid[i][j])
        return vnum_grid, num_vnum

    def update_vgrid(grid, vnum_grid, num_vnum,_y,_x,num): #i in for loop
        deleted_position=[]
        for i in range(len(grid[0])):
            if grid[i][_x] == '.' and (num in vnum_grid[i][_x]):
                vnum_grid[i][_x].remove(num)
                num_vnum[i][_x]-=1
                deleted_position.append([i,_x])

        for j in range(len(grid)):
            if grid[_y][j] == '.' and (num in vnum_grid[_y][j]):
                vnum_grid[_y][j].remove(num)
                num_vnum[_y][j]-=1
                deleted_position.append([_y,j])

        v1 = int(np.sqrt(len(grid)))
        _x1 = _x // v1
        _y1 = _y // v1

        for i in range(_y1 * v1, _y1 * v1 + v1):
            for j in range(_x1 * v1, _x1 * v1 + v1):
                if grid[i][j] == '.' and (num in vnum_grid[i][j]):
                    vnum_grid[i][j].remove(num)
                    num_vnum[i][j] -= 1
                    deleted_position.append([i, j])

        return deleted_position

    def reset_vgrid(grid, vnum_grid, num_vnum,i,dp):
        for _y,_x in dp:
            vnum_grid[_y][_x].append(i)
            num_vnum[_y][_x]+=1

    def ch_backtracking(grid, vnum_grid, num_vnum):
        if not nextEmpty():
            return True
        least_num = num_vnum.min()
        _y,_x = np.argwhere(num_vnum == least_num)[0]
        vnum = vnum_grid[_y][_x]
        num_vnum[_y][_x] = 99
        for i in vnum:
            grid[_y][_x] = i
            dp = update_vgrid(grid, vnum_grid, num_vnum,_y,_x,i)
            if ch_backtracking(grid, vnum_grid, num_vnum):
                return True
            reset_vgrid(grid, vnum_grid, num_vnum,i,dp)
            grid[_y][_x] = '.'
        num_vnum[_y][_x] = least_num
        return False

    vnum_grid, num_vnum = get_setting()
    return ch_backtracking(grid,vnum_grid, num_vnum)

def solve_with_genetic(grid):

    def create_individual():
        size = len(grid)
        individual = []
        for i in range(size):
            valid_num = [_ for _ in range(1, size + 1)]
            for j in range(size):
                if grid[i][j] != '.':
                    valid_num.remove(grid[i][j])
            rand.shuffle(valid_num)
            individual.append(valid_num)
        return individual

    def isValid(g, num, pos):
        for i in range(len(g[0])):
            if g[pos[0]][i] == num and pos[1] != i:
                return False

        for i in range(len(g)):
            if g[i][pos[1]] == num and pos[0] != i:
                return False

        v1 = int(np.sqrt(len(g)))
        _x = pos[1] // v1
        _y = pos[0] // v1

        for i in range(_y * v1, _y * v1 + v1):
            for j in range(_x * v1, _x * v1 + v1):
                if g[i][j] == num and (i, j) != pos:
                    return False

        return True

    def fitness(individual):
        n = len(grid)
        score = 0
        t_grid = copy.deepcopy(grid)
        no_valid_num = []

        for i in range(n):
            t_value = copy.deepcopy(individual[i])
            vnum_i = 0
            for j in range(n):
                if t_grid[i][j] == '.':
                    if isValid(t_grid, int(t_value[0]), (i, j)):
                        t_grid[i][j] = int(t_value.pop(0))
                    else:
                        no_valid_num.append(t_value.pop(0))
                        vnum_i += 1
            if vnum_i == 0:
                score += 10
            else:
                score -= 5

        score -= len(no_valid_num)
        return score

    def selection(population):
        temp = []
        for _ in range(len(population) // 2):
            comper = rand.choices(population, k=2)
            comper.sort(key=fitness, reverse=True)
            temp.append(comper[0])
        temp.sort(key=fitness, reverse=True)
        return temp

    def crossover(parent1, parent2):
        child = copy.deepcopy(parent1)
        size = len(parent2)
        for i in range(size):
            if rand.random() < 0.5:
                child[i] = parent2[i]
        return child

    def mutate(individual):
        if rand.random() < 0.1:  # Reduced mutation rate
            idx = rand.randint(0, len(individual) - 1)
            if individual[idx]:
                individual[idx][rand.randint(0, len(individual[idx]) - 1)] = rand.randint(1, len(grid))
        return individual

    def genetic_algorithm():
        size = len(grid)
        population_size = max(100, size**4)  # Ensure a minimum population size
        num_it = 1000
        population = [create_individual() for _ in range(population_size)]
        best_score = -float('inf')
        scores = []

        for i in range(num_it):
            new_population = selection(population)
            score = fitness(new_population[0])
            scores.append(score)

            # Check for convergence
            if len(scores) >= 3 and scores[-1] == scores[-2] == scores[-3]:
                new_population=new_population[:size]
                for _ in range(len(new_population) // 2):
                    new_population.append(create_individual())


            print(f"Iteration {i}: Best Score: {score}")

            if score > best_score:
                best_score = score

            # Check for a solution
            if best_score >= (size * 10):
                return True, new_population[0]

            next_gen = new_population
            while len(next_gen) < population_size:
                parent1, parent2 = rand.choices(new_population, k=2)
                child = crossover(parent1, parent2)
                child = mutate(child)
                next_gen.append(child)

            population = next_gen

        return False, None  # No solution found

    solved, solution = genetic_algorithm()
    for i in range(len(grid)):
        t_value = solution[i]
        for j in range(len(grid)):
            if grid[i][j] == '.':
                grid[i][j] = int(t_value.pop(0))

    return solved

def solve_with_constraint_programming(grid):
    problem = constraint.Problem()
    size = len(grid)

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

    # Get the solution
    solution = problem.getSolution()

    if solution:
        for i in range(size):
            for j in range(size):
                var = f'V{i}_{j}'
                grid[i][j] = solution[var]
        return True
    return False

def solve_with_custom_csp(grid):
    def findvnum(grid, _y, _x):
        size = len(grid)
        vnum = [x for x in range(1, size + 1)]

        for i in range(size):
            if grid[_y][i] != '.' and grid[_y][i] in vnum:
                vnum.remove(grid[_y][i])

        for i in range(size):
            if grid[i][_x] != '.' and grid[i][_x] in vnum:
                vnum.remove(grid[i][_x])

        v1 = int(np.sqrt(size))
        gx = _x // v1
        gy = _y // v1

        for i in range(gy * v1, gy * v1 + v1):
            for j in range(gx * v1, gx * v1 + v1):
                if grid[i][j] != '.' and grid[i][j] in vnum:
                    vnum.remove(grid[i][j])

        return vnum

    def apply_single_candidate(grid, candidates):
        """Fill cells that have only one candidate."""
        progress = False
        size = len(grid)
        for i in range(size):
            for j in range(size):
                if grid[i][j] == '.' and len(candidates[i][j]) == 1:
                    num = candidates[i][j].pop()
                    grid[i][j] = num
                    update_candidates(grid, candidates, i, j, num)
                    progress = True
        return progress

    def apply_naked_pairs(grid, candidates):
        """Apply the Naked Pairs strategy."""
        progress = False
        size = len(grid)
        for unit in get_all_units(size):
            pairs = {}
            for cell in unit:
                i, j = cell
                if len(candidates[i][j]) == 2:
                    key = tuple(sorted(candidates[i][j]))
                    pairs.setdefault(key, []).append(cell)
            for key, cells in pairs.items():
                if len(cells) == 2:
                    for cell in unit:
                        if cell not in cells:
                            before = len(candidates[cell[0]][cell[1]])
                            candidates[cell[0]][cell[1]].difference_update(key)
                            after = len(candidates[cell[0]][cell[1]])
                            if before != after:
                                progress = True
        return progress

    def apply_hidden_pairs(grid, candidates):
        """Apply the Hidden Pairs strategy."""
        progress = False
        size = len(grid)
        for unit in get_all_units(size):
            counts = {}
            for cell in unit:
                for num in candidates[cell[0]][cell[1]]:
                    counts[num] = counts.get(num, 0) + 1
            hidden_pairs = [pair for pair in itertools.combinations(counts, 2) if
                            counts[pair[0]] == 2 and counts[pair[1]] == 2]
            for pair in hidden_pairs:
                cells_with_pair = [cell for cell in unit if
                                   pair[0] in candidates[cell[0]][cell[1]] and pair[1] in candidates[cell[0]][cell[1]]]
                if len(cells_with_pair) == 2:
                    for cell in cells_with_pair:
                        before = len(candidates[cell[0]][cell[1]])
                        candidates[cell[0]][cell[1]].intersection_update(pair)
                        after = len(candidates[cell[0]][cell[1]])
                        if before != after:
                            progress = True
        return progress

    def apply_x_wing(grid, candidates):
        """Apply the X-Wing strategy."""
        progress = False
        size = len(grid)
        for num in range(1, size + 1):
            rows_with_num = []
            for i in range(size):
                cols = [j for j in range(size) if num in candidates[i][j]]
                if len(cols) == 2:
                    rows_with_num.append((i, cols))
            for (row1, cols1), (row2, cols2) in itertools.combinations(rows_with_num, 2):
                if cols1 == cols2:
                    c1, c2 = cols1
                    for i in range(size):
                        if i != row1 and i != row2:
                            if num in candidates[i][c1]:
                                candidates[i][c1].remove(num)
                                progress = True
                            if num in candidates[i][c2]:
                                candidates[i][c2].remove(num)
                                progress = True
        return progress

    def get_all_units(size):
        """Retrieve all units (rows, columns, subgrids)."""
        units = []
        for i in range(size):
            units.append([(i, j) for j in range(size)])
        for j in range(size):
            units.append([(i, j) for i in range(size)])
        n = int(np.sqrt(size))
        for i in range(0, size, n):
            for j in range(0, size, n):
                units.append([(x, y) for x in range(i, i + n) for y in range(j, j + n)])
        return units

    def update_candidates(grid, candidates, row, col, num):
        """Update candidates after placing a number."""
        size = len(grid)
        for k in range(size):
            candidates[row][k].discard(num)
            candidates[k][col].discard(num)
        n = int(np.sqrt(size))
        start_row, start_col = row - row % n, col - col % n
        for i in range(start_row, start_row + n):
            for j in range(start_col, start_col + n):
                candidates[i][j].discard(num)
        return None

    def initialize_candidates(grid):
        """Initialize candidate lists for each empty cell."""
        size = len(grid)
        candidates = [[set() for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if grid[i][j] == '.':
                    candidates[i][j] = set(findvnum(grid, i, j))
                else:
                    candidates[i][j] = set()
        return candidates

    candidates = initialize_candidates(grid)
    progress = True
    while progress:
        progress = apply_single_candidate(grid, candidates) or \
                   apply_naked_pairs(grid, candidates) or \
                   apply_hidden_pairs(grid, candidates) or \
                   apply_x_wing(grid, candidates)

    # If not solved, use fallback (backtracking or other approach)
    if all(grid[i][j] != '.' for i in range(len(grid)) for j in range(len(grid))):
        return True
    return solve_with_backtracking(grid)  # Use backtracking as a fallback


