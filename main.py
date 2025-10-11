from sudoku_model import SudokuSolver
import numpy as np

if __name__ == "__main__":

    
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
    
    
    solver.display_grid("Početno stanje Sudokua (Ulaz)")
    
    
    print("\nFiksirana maska (True = fiksno, False = promenljivo):")
    print(solver.fixed_mask)

    """
    np.random.seed(42) 
    possible_values = np.arange(1, solver.N + 1)

    for i in range(solver.N):
        for j in range(solver.N):
            if not solver.fixed_mask[i, j]:
                solver.grid[i, j] = np.random.choice(possible_values)

    solver.display_grid("Nasumicna inicijalizacija")
    """

    solver.greedy_init(passes = 2, seed = 42 )
    solver.display_grid("Posle inicijalizacije")

    total_conflicts = solver.objective_f()

    print(f"\nUkupno konflikata (objective_f): {total_conflicts}")
    
    print(f"Konflikti u redovima: {solver._row_conflicts()}")
    print(f"Konflikti u kolonama: {solver._col_conflicts()}")
    print(f"Konflikti u blokovima: {solver._block_conflicts()}")

    pos_conflicts = solver.get_conflicts()
    print(f"Pozicija konflikata:{pos_conflicts}")

    
