import Sudoku as sdku
import copy

print('Size of grid (choose from 4, 9, 16):')
while True:
    try:
        x = int(input())
        if x in [4, 9, 16]:
            break
        else:
            print('Please input a valid number (4, 9, 16)')
    except ValueError:
        print('Please enter an integer.')


Board = sdku.Board(x)


while True:

    print('\nWhich function would you like to use? (1-Solve manually, 2-Solve by Algorithm, 3-Compare algorithms, 0-Exit)')
    try:
        z = int(input())
    except ValueError:
        print('Please enter an integer.')
        continue


    if z in [1, 2, 3, 0,11]:
        if z == 1:
            # Manual Solve
            print('Select difficulty (0-2: Higher the number, more difficult the game):')
            while True:
                try:
                    Board.difficulty = int(input())
                    if 0 <= Board.difficulty <= 2:
                        break
                    else:
                        print('Please input a valid number (0-2)')
                except ValueError:
                    print('Please enter an integer.')

            Board.generate(Board.difficulty)
            Board.copy = copy.deepcopy(Board.grid)

            print('The Sudoku puzzle is:')

            Board.print_board()
            while True:
                print(
                    'Enter the row and column number of the cell you want to change and the value (e.g., "1 2 3"), or enter "0 0 0" to stop:')
                try:
                    row, col, value = map(int, input().split())
                except ValueError:
                    print('Please enter three integers separated by spaces.')
                    continue

                if value == 0:
                    break

                if 1 <= row <= x and 1 <= col <= x and 1 <= value <= x:
                    if not Board.nextEmpty():
                        print('The board is already solved')
                        break
                    elif Board.binput(row, col, value):
                        Board.print_board()
                    else:
                        print('The number is not valid')
                else:
                    print('Please input valid numbers within the grid range.')

                if not Board.nextEmpty():
                    print('The board is already solved')
                    break
        elif z == 2:

            print('Select difficulty (0-2: Higher the number, more difficult the game):')
            while True:
                try:
                    Board.difficulty = int(input())
                    if 0 <= Board.difficulty <= 2:
                        break
                    else:
                        print('Please input a valid number (0-2)')
                except ValueError:
                    print('Please enter an integer.')


            Board.generate(Board.difficulty)
            Board.copy = copy.deepcopy(Board.grid)

            while True:

                Board.grid = copy.deepcopy(Board.copy)
                print('The Sudoku puzzle is:')
                Board.print_board()

                print('\nChoose the solving algorithm:')
                print('1 - Backtracking')
                print('2 - Simulated Annealing')
                print('3 - Constraint Programming')
                print('4 - Custom CSP (Human-like)')
                print('5 - Bitmasks Algorithm')
                print('6 - Crosshatching Algorithm')
                print('7 - Genetic Programming')
                print('0 - Exit')
                while True:
                    try:
                        algo = int(input())
                        if algo in [0, 1, 2, 3, 4, 5, 6, 7]:
                            break
                        else:
                            print('Please choose a number between 1 and 0.')
                    except ValueError:
                        print('Please enter an integer.')

                if algo in [1, 2, 3, 4, 5, 6, 7]:
                    algo_dict = {
                        1: ('Backtracking', Board.solve_with_backtracking),
                        2: ('Simulated Annealing', Board.solve_with_simulated_annealing),
                        3: ('Constraint Programming', Board.solve_with_constraint_programming),
                        4: ('Custom CSP (Human-like)', Board.solve_with_custom_csp),
                        5: ('Bitmasks Algorithm', Board.solve_with_bitmasks),
                        6: ('Crosshatching Algorithm', Board.solve_with_crosshatching),
                        7: ('Genetic Programming',Board.solve_with_genetic)
                    }
                    name, solver = algo_dict[algo]
                    success, comp_time = solver()
                    Board.print_board()
                    if success:
                        print(f'Solved by {name} in {comp_time:.6f} seconds.')
                    else:
                        print(f'No solution found using {name}.')
                    Board.grid = copy.deepcopy(Board.copy)

                if algo ==0:
                    break

        elif z == 3:

            print('Select difficulty (0-2: Higher the number, more difficult the game):')
            while True:
                try:
                    Board.difficulty = int(input())
                    if 0 <= Board.difficulty <= 2:
                        break
                    else:
                        print('Please input a valid number (0-2)')
                except ValueError:
                    print('Please enter an integer.')


            print('\nEnter the number of Sudoku puzzles to compare:')
            while True:
                try:
                    num_puzzles = int(input())
                    if num_puzzles > 0:
                        break
                    else:
                        print('Please enter a positive integer.')
                except ValueError:
                    print('Please enter an integer.')

            # Optional: Ask if the user wants to set a specific difficulty or use the previously set one
            print(f'\nComparing algorithms using {num_puzzles} Sudoku puzzles with difficulty level {Board.difficulty}.')
            Board.compare_algorithms(num_puzzles)
        elif z == 0:
            print('Exiting the game. Goodbye!')
            break
    else:
        print('Please input a valid number (1, 2, or 9)')