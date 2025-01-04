Below is a sample **README.md** that you can include in your GitHub repository. It explains the project’s purpose, how it’s structured, and how to run and use the files. Feel free to customize any part of this document to fit your needs.

---

# Sudoku Solver and Generator with Multiple Algorithms

This repository contains a Sudoku solver and generator project written in Python. The codebase explores multiple algorithms—both traditional and AI-inspired—for generating and solving Sudoku puzzles of varying sizes (4×4, 9×9, 16×16). It also provides a GUI for interactive usage.

## Overview

1. **`main.py`**  
   - Command-line interface where you can select a grid size (4, 9, or 16) and choose different modes:
     1. **Manual Solve** – Generate a puzzle, then fill in numbers yourself in the terminal.  
     2. **Solve by Algorithm** – Generate a puzzle, then pick an algorithm to solve it automatically.  
     3. **Compare Algorithms** – (Work in progress) Compare the performance of multiple algorithms over several puzzles.

2. **`sudoku.py`**  
   - Defines the `Board` class, which holds the state of the Sudoku grid.
   - Implements puzzle generation (with different difficulties), checking for unique solutions, and solver method dispatch.
   - Also defines a `compare_algorithms` function that can run multiple trials, generate puzzles, solve them with various algorithms, and summarize performance.

3. **`algorithm.py`**  
   - Contains implementations of several solver algorithms:
     - **Backtracking**  
     - **Simulated Annealing**  
     - **Bitmask** (classic and “new” variant)  
     - **Crosshatching** (classic and “new” variant)  
     - **Constraint Programming** (using the `python-constraint` library)  
     - **Genetic Algorithm**  
     - **Custom CSP** (human-like strategies)  
     - **Merge** (hybrid approach for puzzle generation/solving)

4. **`GUI.py`**  
   - A Tkinter-based graphical user interface for generating, solving, and comparing Sudoku puzzles.
   - Provides buttons and dropdown menus to select grid size, difficulty, and solution methods.
   - Displays the Sudoku board and allows manual edits or algorithmic solutions.

## Features

- **Puzzle Generation**  
  Generates Sudoku puzzles of different sizes and difficulties. Ensures each puzzle has a (potentially) unique solution by repeatedly checking solutions.
  
- **Multiple Solver Algorithms**  
  Compare classic algorithms like Backtracking, as well as AI-based approaches like Simulated Annealing and Genetic Algorithms.

- **Graphical User Interface (GUI)**  
  A Tkinter-based GUI for easy interaction:
  - **Manual Solve** – Enter numbers directly on the board.
  - **Algorithmic Solve** – Let one of the supported methods solve it automatically.
  - **Compare Algorithms** – Runs time comparisons (displayed in the console output inside the GUI).


## Getting Started



1. **Option A: Run the GUI**:
   ```bash
   python GUI.py
   ```
   - Select grid size, difficulty, then press **Start**.
   - Choose “Solve Manually” to fill in your own answers, or “Solve by Algorithm” to see the solver in action.
   - “Compare Algorithms” opens a subwindow to run performance comparisons (results shown in a text box).

2. **Option B: Run from Command Line (`main.py`)**:
   ```bash
   python main.py
   ```
   - **Select Grid Size** (4, 9, or 16).
   - **Choose Function**:
     - **1 – Solve manually**: generate a puzzle and type your moves in the terminal.  
     - **2 – Solve by Algorithm**: generate a puzzle and pick the solver method.  
     - **3 – Compare algorithms**: specify how many puzzles to test and gather performance data.

## Usage Notes

- **Puzzle generation** for larger sizes (9×9, 16×16) can be slow, especially if you’re ensuring unique solutions at higher difficulties. This is normal for Sudoku generation, which can require solving many times behind the scenes.
- **Comparison** uses the `multiprocessing` library to parallelize puzzle generation and solving. If you experience issues on Windows, ensure you’re calling it in a `if __name__ == "__main__":` guard.


