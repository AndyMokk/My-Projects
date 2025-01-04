# main.py
import Sudoku as sdku

def main():
    while True:
        print("Size of grid (choose from 4, 9, 16):")
        size_input = input().strip()
        if size_input not in ['4', '9', '16']:
            print("Invalid grid size. Please choose from 4, 9, or 16.")
            continue
        size = int(size_input)

        print("\nWhich function would you like to use? (1-Solve manually, 2-Solve by Algorithm, 3-Compare algorithms, 0-Exit)")
        func_input = input().strip()
        if func_input == '0':
            print("Exiting...")
            break
        elif func_input not in ['1', '2', '3']:
            print("Invalid option. Please choose 1, 2, 3, or 0.")
            continue
        func = int(func_input)

        if func == 1:
            # Solve manually (Placeholder: Implement manual solving if needed)
            print("\nManual solving feature is not implemented yet.")
            continue

        elif func == 2:
            print("\nSelect difficulty (0-2: Higher the number, more difficult the game):")
            difficulty_input = input().strip()
            if difficulty_input not in ['0', '1', '2']:
                print("Invalid difficulty. Please choose 0, 1, or 2.")
                continue
            difficulty = int(difficulty_input)

            board = sdku.Board(size)
            board.generate(difficulty)
            print("The Sudoku puzzle is:")
            board.print_board()

            # Choose solving algorithm
            print("\nChoose the solving algorithm:")
            print("1 - Backtracking")
            print("2 - Simulated Annealing")
            print("3 - Constraint Programming")
            print("4 - Custom CSP (Human-like)")
            print("5 - Bitmasks Algorithm")
            print("6 - Crosshatching Algorithm")
            #print("7 - Genetic Programming")
            print("0 - Exit")
            algo_input = input().strip()

            algo_map = {
                '1': 'Backtracking',
                '2': 'Simulated Annealing',
                '3': 'Constraint Programming',
                '4': 'Custom CSP',
                '5': 'Bitmasking',
                '6': 'Cross-Hatching',
                #'7': 'Genetic',
                '0': 'Exit'
            }

            if algo_input not in algo_map:
                print("Invalid algorithm choice.")
                continue

            if algo_input == '0':
                print("Returning to main menu.")
                continue

            algo_name = algo_map[algo_input]
            print(f"Solving with {algo_name}...")

            success, solve_time = sdku.run_algorithm(board, algo_name)
            if success:
                print(f"Solved by {algo_name} in {solve_time:.6f} seconds.")
            else:
                print(f"Failed to solve using {algo_name}.")

            print("The Sudoku puzzle is:")
            board.print_board()

        elif func == 3:
            print("\nSelect difficulty for comparison (0=Easy, 1=Medium, 2=Hard):")
            difficulty_input = input().strip()
            if difficulty_input not in ['0', '1', '2']:
                print("Invalid difficulty. Please choose 0, 1, or 2.")
                continue
            difficulty = int(difficulty_input)

            print("\nStarting comparison of algorithms...")
            sdku.compare_algorithms(num_trials=10, size=size, level=difficulty)
            print("\nComparison completed.")

if __name__ == '__main__':
    main()
