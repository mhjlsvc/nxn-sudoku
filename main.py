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
    solver.display_grid("Posle inicijalizacije: ")

    solver._initialize_auxiliary_structures() 

    total_conflicts_post_greedy = solver.objective_f() 
    solver.current_cost = total_conflicts_post_greedy 
    solver.best_cost = solver.current_cost
    solver.best_grid = solver.grid.copy() 

    print(f"\nUkupno konflikata NAKON Greedy inicijalizacije: {total_conflicts_post_greedy}")

    print(f"\nPočetna ILS cena: {solver.current_cost}")

    #ZA TEST _min_conflicts_with_tabu
    ls_iterations_test = 5000 
    acceptance_prob_test = 0.01
    tabu_size_test = 10

    solver._min_conflicts_with_tabu(
        iteration_limit=ls_iterations_test, 
        acceptance_prob=acceptance_prob_test, 
        tabu_size=tabu_size_test
    )

    solver.display_grid("Nakon LS: ")

    final_test_cost_objective_f = solver.objective_f() 

    print(f"\nLS Test, interna cena: {solver.current_cost}")
    print(f"LS Test, konačna cena : {final_test_cost_objective_f}")
    print(f"Najbolja postignuta cena: {solver.best_cost}")
    print(f"Da li je rešenje validno? {solver.current_cost == 0}")

    cp_solver = SudokuCP(puzzle_9x9, seed = 42)

    cp_solver.grid = solver.best_grid.copy()
    cp_solver.fixed_mask = (cp_solver.grid != 0)
    cp_solver._build_cp_model()

    print("\nCP: ")
    print(f"Broj promenljivih: {cp_solver.N**2}")

    status = cp_solver.cp_refinement(time_limit=10.0, fix_noncon=True, hints=True)
    cp_solver.display_grid("Rešenje nakon CP:")

    cp_final_cost = cp_solver.objective_f()

    print(f"Konačna CP cena konflikata: {cp_final_cost}")
    print(f"Da li je rešenje validno? {cp_solver.is_valid()}")