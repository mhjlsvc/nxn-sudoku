from sudoku_model import ILS_CP
from sudoku_model import SudokuCP
import numpy as np
from ortools.sat.python.cp_model import OPTIMAL, FEASIBLE

if __name__ == "__main__":
    
    puzzle_9x9 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 6, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 7, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 9]
    ]
    
    solver = ILS_CP(puzzle_9x9, seed=42)
    solver.display_grid("Početno stanje Sudokua (Ulaz): ")
    
    total_conflicts_pre_greedy = solver.objective_f()
    print(f"\nUkupno konflikata PRE Greedy inicijalizacije: {total_conflicts_pre_greedy}")

    solver.greedy_init(passes = 2, seed = 42 )

    solver._initialize_auxiliary_structures() 
    total_conflicts_post_greedy = solver.objective_f() 
    
    solver.current_cost = total_conflicts_post_greedy 
    solver.best_cost = solver.current_cost
    solver.best_grid = solver.grid.copy() 

    solver.display_grid("Stanje NAKON Greedy Inicijalizacije:")
    print(f"\nUkupno konflikata NAKON Greedy inicijalizacije: {total_conflicts_post_greedy}")

    print(f"\nPočetna ILS cena: {solver.current_cost}")

    solver.solve_ils_cp(
        total_iterations=200,      
        ls_iterations=5000,       
        acceptance_prob=0.01,
        tabu_size=10,
        cp_limit=1.0,             
        empty_factor_init=0.25, 
        alpha=0.97                 
    )

    solver.grid = solver.best_grid.copy()
    solver.display_grid("Konačno rešenje nakon ILS-CP:")

    print(f"\nNajbolja postignuta cena: {solver.best_cost}")
    print(f"\nDa li je konačno rešenje validno? {solver.is_valid()}")
    print("\n")