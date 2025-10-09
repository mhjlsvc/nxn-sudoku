from sudoku_model import SudokuSolver
import numpy as np

if __name__ == "__main__":

    # 9x9 Sudoku za demonstraciju
    puzzle_9x9 = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    
    solver = SudokuSolver(puzzle_9x9)
    
    # Prikaz početnog stanja
    solver.display_grid("Početno stanje Sudokua (Ulaz)")
    
    # Prikaz maske fiksnih polja
    print("\nFiksirana maska (True = fiksno, False = promenljivo):")
    print(solver.fixed_mask)

    
