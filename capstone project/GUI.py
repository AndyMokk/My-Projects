import tkinter as tk
from tkinter import messagebox
import Sudoku as sdku
import threading
import copy
import math

def convert_grid_to_int(grid):
    """Convert displayed cell strings to integers (or keep '.' if empty)."""
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            val = grid[i][j]
            if val == '' or val == '.':
                grid[i][j] = '.'
            else:
                try:
                    grid[i][j] = int(val)
                except ValueError:
                    grid[i][j] = '.'

def convert_grid_to_str(grid):
    """Convert numeric cells to string for display; '.' is treated as empty."""
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '.':
                grid[i][j] = ''
            else:
                grid[i][j] = str(grid[i][j])

class SudokuApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sudoku Solver")
        self.root.geometry("1000x600")

        self.board = None
        self.grid_size = tk.IntVar(value=9)
        self.difficulty = tk.IntVar(value=0)

        # Frames
        self.control_frame = tk.Frame(root, padx=10, pady=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.board_frame = tk.Frame(root, padx=10, pady=10)
        self.board_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Start with a main menu
        self.create_main_menu()

        self.initial_grid = None  # To store a copy of the puzzle before solving
        self.entries = []         # Will hold references to cell widgets

    def create_main_menu(self):
        self.clear_frame(self.control_frame)
        self.clear_frame(self.board_frame)

        tk.Label(
            self.control_frame,
            text="Sudoku Solver",
            font=("Arial", 24, "bold")
        ).pack(pady=20)

        # Grid Size
        tk.Label(
            self.control_frame,
            text="Select Grid Size:",
            font=("Arial", 16)
        ).pack(pady=10)

        grid_size_menu = tk.OptionMenu(
            self.control_frame,
            self.grid_size,
            4, 9, 16
        )
        grid_size_menu.config(font=("Arial", 14), width=15)
        grid_size_menu.pack(pady=5)

        # Difficulty
        tk.Label(
            self.control_frame,
            text="Select Difficulty:",
            font=("Arial", 16)
        ).pack(pady=10)

        difficulty_menu = tk.OptionMenu(
            self.control_frame,
            self.difficulty,
            0, 1, 2
        )
        difficulty_menu.config(font=("Arial", 14), width=15)
        difficulty_menu.pack(pady=5)

        # Start Button
        tk.Button(
            self.control_frame,
            text="Start",
            command=self.start_game,
            font=("Arial", 16),
            width=20,
            bg="#4CAF50",
            fg="white"
        ).pack(pady=30)

    def start_game(self):
        size = self.grid_size.get()
        self.board = sdku.Board(size)
        self.open_main_menu()

    def open_main_menu(self):
        self.clear_frame(self.control_frame)
        self.clear_frame(self.board_frame)

        tk.Label(
            self.control_frame,
            text=f"Sudoku Solver ({self.board.size}x{self.board.size})",
            font=("Arial", 20, "bold")
        ).pack(pady=20)

        tk.Button(
            self.control_frame,
            text="Solve Manually",
            command=self.solve_manually,
            font=("Arial", 16),
            width=20,
            bg="#2196F3",
            fg="white"
        ).pack(pady=10)

        tk.Button(
            self.control_frame,
            text="Solve by Algorithm",
            command=self.solve_by_algorithm,
            font=("Arial", 16),
            width=20,
            bg="#FF9800",
            fg="white"
        ).pack(pady=10)

        tk.Button(
            self.control_frame,
            text="Compare Algorithms",
            command=self.compare_algorithms,
            font=("Arial", 16),
            width=20,
            bg="#9C27B0",
            fg="white"
        ).pack(pady=10)

        tk.Button(
            self.control_frame,
            text="Generate New Puzzle",
            command=self.generate_new_puzzle,
            font=("Arial", 16),
            width=20,
            bg="#f44336",
            fg="white"
        ).pack(pady=10)

    def solve_manually(self):
        # Generate puzzle
        self.board.generate(self.difficulty.get())
        # Display puzzle in manual mode
        self.display_board(manual=True)

    def solve_by_algorithm(self):
        # Generate puzzle
        self.board.generate(self.difficulty.get())

        # Store initial grid
        self.initial_grid = copy.deepcopy(self.board.grid)

        self.clear_frame(self.control_frame)
        tk.Label(
            self.control_frame,
            text="Solve by Algorithm",
            font=("Arial", 20, "bold")
        ).pack(pady=20)

        # Algorithm selection
        algorithms = {
            "Backtracking": "backtracking",
            "Crosshatching": "crosshatching",
            "Simulated Annealing": "simulated_annealing",
            "Constraint Programming": "constraint_programming",
            "Custom CSP (Human-like)": "custom_csp",
            "Bitmasks Algorithm": "bitmasks",
            "Genetic Programming": "genetic",
            "Merge Method": "merge",
        }
        self.selected_algo = tk.StringVar(value="Backtracking")

        tk.Label(
            self.control_frame,
            text="Select Solving Algorithm:",
            font=("Arial", 16)
        ).pack(pady=10)

        algo_menu = tk.OptionMenu(
            self.control_frame,
            self.selected_algo,
            *algorithms.keys()
        )
        algo_menu.config(font=("Arial", 14), width=20)
        algo_menu.pack(pady=5)

        # Solve Button
        solve_button = tk.Button(
            self.control_frame,
            text="Solve",
            command=lambda: self.run_solver_thread(algorithms),
            font=("Arial", 16),
            width=20,
            bg="#4CAF50",
            fg="white"
        )
        solve_button.pack(pady=20)

        # Reset Grid
        reset_button = tk.Button(
            self.control_frame,
            text="Reset Grid",
            command=self.reset_grid,
            font=("Arial", 16),
            width=20,
            bg="#f44336",
            fg="white"
        )
        reset_button.pack(pady=10)

        # Back to Menu
        back_button = tk.Button(
            self.control_frame,
            text="Back to Menu",
            command=self.open_main_menu,
            font=("Arial", 16),
            width=20,
            bg="#757575",
            fg="white"
        )
        back_button.pack(pady=10)

        # Display the board
        self.display_board(manual=False)

    def run_solver_thread(self, algorithms):
        # Disable controls
        for widget in self.control_frame.winfo_children():
            widget.config(state='disabled')

        algo_name = self.selected_algo.get()
        solver_key = algorithms[algo_name]

        solving_thread = threading.Thread(target=self.run_solver, args=(solver_key,))
        solving_thread.daemon = True
        solving_thread.start()

    def run_solver(self, solver_key):
        success = False
        comp_time = 0.0
        convert_grid_to_int(self.board.grid)
        try:
            if solver_key == "backtracking":
                success, comp_time = self.board.solve_with_backtracking()
            elif solver_key == "crosshatching":
                success, comp_time = self.board.solve_with_crosshatching()
            elif solver_key == "simulated_annealing":
                success, comp_time = self.board.solve_with_simulated_annealing()
            elif solver_key == "constraint_programming":
                success, comp_time = self.board.solve_with_constraint_programming()
            elif solver_key == "custom_csp":
                success, comp_time = self.board.solve_with_custom_csp()
            elif solver_key == "bitmasks":
                success, comp_time = self.board.solve_with_bitmasks()
            elif solver_key == "genetic":
                success, comp_time = self.board.solve_with_genetic()
            elif solver_key == "merge":
                success, comp_time = self.board.solve_merge()
        except Exception as e:
            print(f"Error in run_solver: {e}")
            success = False

        self.root.after(0, self.on_solver_finished, success, comp_time)

    def on_solver_finished(self, success, comp_time):
        # Re-enable controls
        for widget in self.control_frame.winfo_children():
            widget.config(state='normal')

        if success:
            convert_grid_to_str(self.board.grid)
            self.update_gui_with_solution()
            msg = f"Solved in {comp_time:.4f} seconds."
            messagebox.showinfo("Success", msg)
        else:
            messagebox.showwarning("Failure", "Could not solve the Sudoku.")

    def update_gui_with_solution(self):
        """Push the solved board into the GUI fields."""
        for r in range(self.board.size):
            for c in range(self.board.size):
                entry = self.entries[r][c]
                entry.config(state='normal')
                entry.delete(0, tk.END)
                val = self.board.grid[r][c]
                if val != '':
                    entry.insert(0, val)
                    entry.config(disabledforeground='blue')
                    entry.config(state='disabled')
                else:
                    entry.insert(0, "")
                    entry.config(disabledforeground='black')
                    entry.config(state='disabled')

    def compare_algorithms(self):
        """
        Create a new window to collect parameters for comparing algorithms, then
        call the compare_algorithms function from sudoku.py in a background thread,
        and show the output in real-time (or after it finishes).
        """
        compare_window = tk.Toplevel(self.root)
        compare_window.title("Compare Algorithms")
        compare_window.geometry("700x500")

        # Number of trials
        tk.Label(compare_window, text="Number of Trials:", font=("Arial", 14)).grid(row=0, column=0, sticky="e", padx=5, pady=5)
        trials_var = tk.IntVar(value=5)  # default
        tk.Entry(compare_window, textvariable=trials_var, width=10, font=("Arial", 14)).grid(row=0, column=1, padx=5, pady=5)

        # Difficulty
        tk.Label(compare_window, text="Difficulty (0=Easy, 1=Med, 2=Hard):", font=("Arial", 14)).grid(row=1, column=0, sticky="e", padx=5, pady=5)
        diff_var = tk.IntVar(value=self.difficulty.get()) 
        tk.Entry(compare_window, textvariable=diff_var, width=10, font=("Arial", 14)).grid(row=1, column=1, padx=5, pady=5)

        # Grid size
        tk.Label(compare_window, text="Grid Size (4, 9, 16):", font=("Arial", 14)).grid(row=2, column=0, sticky="e", padx=5, pady=5)
        size_var = tk.IntVar(value=self.board.size if self.board else 9)
        tk.Entry(compare_window, textvariable=size_var, width=10, font=("Arial", 14)).grid(row=2, column=1, padx=5, pady=5)

        # -- Output Section --
        output_frame = tk.Frame(compare_window)
        output_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        compare_window.grid_rowconfigure(4, weight=1)
        compare_window.grid_columnconfigure(1, weight=1)

        output_text = tk.Text(output_frame, wrap="word", font=("Consolas", 11))
        output_text.pack(fill=tk.BOTH, expand=True)

        # -- Button to run comparison --
        def run_comparison():
            """
            Run the compare_algorithms function in a background thread and
            redirect stdout to capture the printed results.
            """
            def worker():
                import sys
                import io
                old_stdout = sys.stdout
                mystdout = io.StringIO()

                sys.stdout = mystdout
                try:
                    # Call the compare_algorithms function from Sudoku.py
                    sdku.compare_algorithms(
                        num_trials=trials_var.get(),
                        size=size_var.get(),
                        level=diff_var.get()
                    )
                except Exception as e:
                    # Print any exceptions into our captured output
                    print(f"Error during comparison: {e}")

                # Restore stdout
                sys.stdout = old_stdout
                output = mystdout.getvalue()

                # Insert the results into the Text widget
                self.root.after(0, lambda: output_text.insert(tk.END, output))

            # Start the worker thread
            t = threading.Thread(target=worker, daemon=True)
            t.start()

        run_button = tk.Button(
            compare_window,
            text="Run Comparison",
            command=run_comparison,
            font=("Arial", 14),
            bg="#9C27B0",
            fg="white"
        )
        run_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

    def generate_new_puzzle(self):
        if self.board:
            self.board.generate(self.difficulty.get())
            convert_grid_to_str(self.board.grid)  # For display
            self.display_board(manual=False)
        else:
            messagebox.showwarning(
                "Warning",
                "Please start a new game first."
            )

    def display_board(self, manual):
        self.clear_frame(self.board_frame)
        self.entries = [[None for _ in range(self.board.size)] for _ in range(self.board.size)]

        size = self.board.size
        subgrid_size = int(math.sqrt(size))
        if subgrid_size ** 2 != size:
            messagebox.showerror("Error", "Grid size must be a perfect square (e.g., 4, 9, 16).")
            return

        # Convert '.' to '' for display
        convert_grid_to_str(self.board.grid)

        for sg_row in range(subgrid_size):
            for sg_col in range(subgrid_size):
                subframe = tk.Frame(self.board_frame, bd=2, relief="solid")
                subframe.grid(row=sg_row, column=sg_col, padx=1, pady=1)

                for r in range(subgrid_size):
                    for c in range(subgrid_size):
                        grid_row = sg_row * subgrid_size + r
                        grid_col = sg_col * subgrid_size + c

                        val = self.board.grid[grid_row][grid_col]
                        e = tk.Entry(
                            subframe,
                            width=4,
                            font=("Arial", 18),
                            justify="center",
                            bd=1,
                            relief="ridge"
                        )
                        e.grid(row=r, column=c, padx=1, pady=1)

                        if val != '':
                            e.insert(0, str(val))
                            if manual:
                                # Allow user input if manual
                                e.config(state="normal", disabledforeground="blue")
                            else:
                                # Otherwise, disable
                                e.config(state="disabled", disabledforeground="blue")
                        else:
                            if manual:
                                e.config(state="normal")
                            else:
                                e.config(state="disabled", disabledforeground="black")

                        self.entries[grid_row][grid_col] = e

        if manual:
            # Submit button to check user solution
            submit_btn = tk.Button(
                self.control_frame,
                text="Submit",
                command=self.submit_manual_solution,
                font=("Arial", 16),
                width=20,
                bg="#2196F3",
                fg="white"
            )
            submit_btn.pack(pady=20)

            back_btn = tk.Button(
                self.control_frame,
                text="Back to Menu",
                command=self.open_main_menu,
                font=("Arial", 16),
                width=20,
                bg="#757575",
                fg="white"
            )
            back_btn.pack(pady=10)

    def submit_manual_solution(self):
        # Pull user entries into board
        for r in range(self.board.size):
            for c in range(self.board.size):
                val = self.entries[r][c].get()
                if val.isdigit():
                    self.board.grid[r][c] = int(val)
                else:
                    self.board.grid[r][c] = '.'
        if self.is_solved_simple():
            messagebox.showinfo("Success", "Congratulations! You have solved the Sudoku.")
        else:
            messagebox.showwarning("Incomplete", "The Sudoku is not yet solved.")

    def reset_grid(self):
        if self.initial_grid is None:
            messagebox.showwarning("Warning", "No initial grid to reset to.")
            return

        self.board.grid = copy.deepcopy(self.initial_grid)
        convert_grid_to_str(self.board.grid)
        for i in range(self.board.size):
            for j in range(self.board.size):
                entry = self.entries[i][j]
                entry.config(state='normal')
                entry.delete(0, tk.END)
                val = self.board.grid[i][j]
                if val != '':
                    entry.insert(0, val)
                    entry.config(disabledforeground='blue')
                    entry.config(state='disabled')
                else:
                    entry.insert(0, "")
                    entry.config(disabledforeground='black')
                    entry.config(state='disabled')

        messagebox.showinfo("Reset", "The grid has been reset to its initial state.")

    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    # A quick "is_solved" check for user-submitted solution
    def is_solved_simple(self):
        """A simple check that each row, col, and subgrid contains unique numbers."""
        size = self.board.size
        # Check for any '.' left
        for i in range(size):
            for j in range(size):
                if self.board.grid[i][j] == '.':
                    return False

        # Check rows
        for r in range(size):
            row_nums = set()
            for c in range(size):
                val = self.board.grid[r][c]
                if val in row_nums:
                    return False
                row_nums.add(val)
        # Check cols
        for c in range(size):
            col_nums = set()
            for r in range(size):
                val = self.board.grid[r][c]
                if val in col_nums:
                    return False
                col_nums.add(val)
        # Check subgrids
        subgrid = int(math.sqrt(size))
        for sg_row in range(subgrid):
            for sg_col in range(subgrid):
                seen = set()
                for r in range(subgrid):
                    for c in range(subgrid):
                        rr = sg_row*subgrid + r
                        cc = sg_col*subgrid + c
                        val = self.board.grid[rr][cc]
                        if val in seen:
                            return False
                        seen.add(val)
        return True


if __name__ == "__main__":
    root = tk.Tk()
    app = SudokuApp(root)
    root.mainloop()