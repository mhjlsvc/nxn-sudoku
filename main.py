from sudoku_model import ILS_CP
import time
import csv
import os
from ortools.sat.python.cp_model import OPTIMAL, FEASIBLE

def load_puzzles_from_file(filepath: str) -> list[list[list[int]]]:
    
    puzzles = []
    current_puzzle_str = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('#') or line == '---':
                    
                    if len(current_puzzle_str) > 0:
                        N = len(current_puzzle_str)
                        if N > 0 and all(len(row) == N for row in current_puzzle_str):
                            puzzle_matrix = []
                            for row_list in current_puzzle_str:
                                row = [int(num) for num in row_list]
                                puzzle_matrix.append(row)
                            
                            puzzles.append(puzzle_matrix)
                            
                        current_puzzle_str = [] # Reset
                    continue
                
                if line:

                    numbers = [n for n in line.split(' ') if n]
                    if numbers:
                        current_puzzle_str.append(numbers)

        if len(current_puzzle_str) > 0:
            N = len(current_puzzle_str)
            if N > 0 and all(len(row) == N for row in current_puzzle_str):
                puzzle_matrix = []
                for row_list in current_puzzle_str:
                    row = [int(num) for num in row_list]
                    puzzle_matrix.append(row)
                puzzles.append(puzzle_matrix)

    except FileNotFoundError:
        print(f"GREŠKA: Fajl '{filepath}' nije pronađen.")
        return []

    return puzzles

def save_results_to_csv(results: dict, filename: str, run_id: str):
    
    results['Run_ID'] = run_id 
    fieldnames = list(results.keys())
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';') 
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(results)

if __name__ == "__main__":

    PUZZLE_FILE_PATH = 'puzzles.txt'

    all_puzzles = load_puzzles_from_file(PUZZLE_FILE_PATH)

    if not all_puzzles:
        print(f"Nije pronađen nijedan Sudoku problem u fajlu: {PUZZLE_FILE_PATH}. Završavanje programa.")
        exit()
        
    res_dat = "Rezultati"

    for idx, puzzle in enumerate(all_puzzles):
        N = len(puzzle)
        print(f"\n--- REŠAVANJE PROBLEMA #{idx + 1} (VELIČINA: {N}x{N}) ---")

        solver = ILS_CP(puzzle, seed=42)
        solver.display_grid("Početno stanje Sudokua (Ulaz): ")
                   
        solver.greedy_init(passes=2, seed=42)
        solver._initialize_auxiliary_structures()

        total_conflicts_post_greedy = solver.objective_f()
        solver.current_cost = total_conflicts_post_greedy
        solver.best_cost = solver.current_cost
        solver.best_grid = solver.grid.copy()

        solver.display_grid("Stanje NAKON Greedy Inicijalizacije:")
        print(f"\nPočetna ILS cena: {solver.current_cost}")

        start_time = time.time()

        final_metrics = solver.solve_ils_cp(
            total_iterations=50,
            ls_iterations=50,
            acceptance_prob=0.15,
            tabu_size=10,
            cp_limit=15,
            empty_factor_init=0.2,
            alpha=0.995
        )

        save_results_to_csv(
        results=final_metrics, 
        filename="experiment_results3.csv", 
        run_id=res_dat
        )
    
        print(f"\nRezultati uspešno zapisani u CSV datoteku.")

        end_time = time.time()
        execution_time = end_time - start_time

        solver.grid = solver.best_grid.copy()
        solver.display_grid(f"Konačno rešenje za problem #{idx + 1}")

        print(f"\nNajbolja postignuta cena: {solver.best_cost}")
        print(f"Rešenje validno: {solver.is_valid()}")
        print("\nStatistika: \n")
        print(f"Vreme izvršenja (ILS-CP): {execution_time:.4f} sekundi")
        print(f"Ukupan broj ILS iteracija: {solver.total_iterations_run}")
        print(f"LS poboljšanja: {solver.ls_success_count}")
        print(f"CP poziva/uspeha: {solver.cp_call_count} / {solver.cp_success_count}")
        print("\n")