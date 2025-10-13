from sudoku_model import ILS_CP
from sudoku_model import SudokuCP
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
    
    solver = ILS_CP(puzzle_9x9, seed=42)
    
    
    solver.display_grid("Početno stanje Sudokua (Ulaz)")
    
    """
    print("\nFiksirana maska (True = fiksno, False = promenljivo):")
    print(solver.fixed_mask)

    
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
    """
    total_conflicts = solver.objective_f()

    print(f"\nUkupno konflikata (objective_f): {total_conflicts}")
    
    print(f"Konflikti u redovima: {solver._row_conflicts()}")
    print(f"Konflikti u kolonama: {solver._col_conflicts()}")
    print(f"Konflikti u blokovima: {solver._block_conflicts()}")

    pos_conflicts = solver.get_conflicts()
    print(f"Pozicija konflikata:{pos_conflicts}")
    """

    print(f"\nPočetna ILS cena: {solver.current_cost}")

    solver._min_conflicts_with_tabu(
        iteration_limit=5000, 
        acceptance_prob=0.01, 
        tabu_size=10
    )

    solver.display_grid("Nakon Lokalnog Pretraživanja:")

    print(f"Konačna ILS cena: {solver.current_cost}")
    print(f"Najbolja postignuta cena: {solver.best_cost}")
    print(f"Da li je trenutno rešenje validno? {solver.is_valid()}")

    cp_solver = SudokuCP(puzzle_9x9, seed = 42)
    cp_solver._build_cp_model()

    print("\nBroj promenljivih", cp_solver.N**2)

    status = cp_solver.cp_refinement(time_limit=10.0, fix_noncon=True, hints=True)
    cp_solver.display_grid()

    cp_final_cost = cp_solver.objective_f()
    print(f"Konačna CP cena konflikata: {cp_final_cost}")
    print(f"Da li je rešenje validno? {cp_solver.is_valid()}")
