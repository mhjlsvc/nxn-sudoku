import numpy as np
from collections import deque
from typing import Set, Tuple, Optional
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import OPTIMAL, FEASIBLE
import random
import time, math


class SudokuSolver:

    def __init__(self, puzzle):

        self.grid = np.array(puzzle, dtype=int)
        self.N = self.grid.shape[0]
        self.K = int(np.sqrt(self.N))

        self.fixed_mask = (self.grid != 0)

    def display_grid(self, title="Trenutna Sudoku ploča"):

        print(f"\n--- {title} (Veličina: {self.N}x{self.N}) ---")
    
        width = len(str(self.N))
    
        fmt_str = f"{{:<{width}}}"

        def print_separator():
            cell_width = width + 1 
            cell_line = ("-" * cell_width) 
        
            line = ""
            for _ in range(self.K):
                block_line = ("+" + cell_line) * self.K + "+"
                line += block_line
        
                separator_unit = ('-' * (width + 1)) + '+' 
        
            separator_part = ("-" * (width + 1)) 
        
            block_sep = (separator_part + "+") * self.K

            final_separator = "+" + block_sep * self.K
        
            print(final_separator)

        for i in range(self.N):
            if i % self.K == 0:
                print_separator()

            row_str = "|"
            for j in range(self.N):
                val = self.grid[i, j]
            
                if val != 0:
                    cell_content = fmt_str.format(val)
                else:
                    cell_content = fmt_str.format(".") 

                row_str += f" {cell_content}" 
            
                if (j + 1) % self.K == 0:
                    row_str += " |" 
        
            print(row_str)

        print_separator()

    def _line_conflicts(self, line):

        vals = line[(line > 0)]

        if len(vals) == 0:
            return 0

        unique, counts = np.unique(vals, return_counts = True)

        return int(np.sum(counts[counts > 1] - 1) )

    def _row_conflicts(self):

        total = 0

        for i in range(self.N):
            row = self.grid[i,:]
            conflicts = self._line_conflicts(row)
            total += conflicts

        return total

    def _col_conflicts(self):

        total = 0

        for j in range(self.N):
            col = self.grid[:,j]
            conflicts = self._line_conflicts(col)
            total += conflicts

        return total

    def _block_conflicts(self):

        total = 0

        for i in range(self.K):
            for j in range(self.K):
                i_start = i * self.K
                i_end = (i + 1) * self.K
                j_start = j * self.K
                j_end = (j + 1) * self.K
                block = self.grid[i_start:i_end, j_start:j_end].ravel()
                conflicts = self._line_conflicts(block)
                total += conflicts

        return total

    def objective_f(self):
        con_row = self._row_conflicts()
        con_col = self._col_conflicts()
        con_block = self._block_conflicts()

        total_con = con_row + con_col + con_block

        return total_con

    def is_valid(self) -> bool:
        if np.any((self.grid < 1) | (self.grid > self.N)):
            return False

        return self.objective_f() == 0

    def get_conflicting_cells(self) -> Set[Tuple[int, int]]:

        N = self.N
        conflicting_cells = set()

        for i in range(N):
            for j in range(N):
                val = self.grid[i, j]

                if val == 0: continue

                is_conflicting = False

                if np.count_nonzero(self.grid[i, :] == val) > 1:
                    is_conflicting = True

                if np.count_nonzero(self.grid[:, j] == val) > 1:
                    is_conflicting = True

                ik_start = (i // self.K) * self.K
                ik_end = ik_start + self.K
                jk_start = (j // self.K) * self.K
                jk_end = jk_start + self.K
                block = self.grid[ik_start:ik_end, jk_start:jk_end]

                if np.count_nonzero(block == val) > 1:
                    is_conflicting = True

                if is_conflicting and not self.fixed_mask[i, j]:
                    conflicting_cells.add((i, j))

        return conflicting_cells

    def _placment_cost(self, i:int, j:int, val:int ) -> int:
        curr = self.grid[i, j]
        self.grid[i, j] = val

        row_duplicates = max(0, np.count_nonzero(self.grid[i, :] == val ) - 1)
        col_duplicates = max(0, np.count_nonzero(self.grid[:, j] == val ) - 1)

        cost = row_duplicates + col_duplicates

        self.grid[i, j] = curr

        return cost

    def _fill_block_greedy(self, ik:int, jk:int, rand: np.random.Generator ) -> None:
        if rand is None:
            rand = np.random._rng()

        k = self.K
        ik_start = ik * k
        ik_end = (ik + 1) * k
        jk_start = jk * k
        jk_end = (jk + 1) * k

        block = self.grid[ik_start:ik_end,jk_start:jk_end]

        curr = block[block > 0]
        all = np.arange(1, self.N + 1)

        missing = []

        for val in all:
            count_val = np.count_nonzero(curr == val)

            if count_val == 0:
                missing.append(val)

        empty = []

        for i in range(ik_start, ik_end):
            for j in range(jk_start, jk_end):
                if self.grid[i, j] == 0:
                    empty.append((i, j))

        if not missing or not empty:
            return

        rand.shuffle(missing)
        rand.shuffle(empty)

        for val in missing:
            best_pos = None
            best_cost = None

            for(i, j) in empty:
                if self.grid[i, j] != 0:
                    continue

                cost = self._placment_cost(i, j, val)

                if ( best_pos is None ) or ( cost < best_cost ) or ( cost == best_cost and rand.random() < 0.5 ):
                    best_pos = (i, j)
                    best_cost = cost

            if best_pos is not None:
                i, j = best_pos
                self.grid[i, j]  = val
                empty.remove((i, j))
                if not empty:
                     break

    def greedy_init(self, passes: int = 2, seed: int | None = None ) -> None:

        rand = np.random.default_rng(seed)

        for _ in range(passes):
            for ik in range(self.K):
                for jk in range(self.K):
                    self._fill_block_greedy(ik = ik, jk= jk, rand = rand)

    def _calculate_delta_cost(self, i: int, j: int, new_value: int) -> int:

        N = self.N
        K = self.K
        current_value = self.grid[i, j]

        if current_value == new_value:
            return 0

        old_cost = 0

        if np.count_nonzero(self.grid[i, :] == current_value) > 1:
            old_cost += (np.count_nonzero(self.grid[i, :] == current_value) - 1)

        if np.count_nonzero(self.grid[:, j] == current_value) > 1:
            old_cost += (np.count_nonzero(self.grid[:, j] == current_value) - 1)

        ik_start = (i // K) * K
        ik_end = ik_start + K
        jk_start = (j // K) * K
        jk_end = jk_start + K
        block = self.grid[ik_start:ik_end, jk_start:jk_end]
        if np.count_nonzero(block == current_value) > 1:
            old_cost += (np.count_nonzero(block == current_value) - 1)

        self.grid[i, j] = new_value
        new_cost = 0

        if np.count_nonzero(self.grid[i, :] == new_value) > 1:
            new_cost += (np.count_nonzero(self.grid[i, :] == new_value) - 1)

        if np.count_nonzero(self.grid[:, j] == new_value) > 1:
            new_cost += (np.count_nonzero(self.grid[:, j] == new_value) - 1)

        new_block = self.grid[ik_start:ik_end, jk_start:jk_end]
        if np.count_nonzero(new_block == new_value) > 1:
            new_cost += (np.count_nonzero(new_block == new_value) - 1)

        self.grid[i, j] = current_value

        delta = new_cost - old_cost

        return delta


class ILS_CP(SudokuSolver):

    def __init__(self, puzzle, seed: int = 42):

        super().__init__(puzzle)

        import numpy as np
        self.random = np.random.default_rng(seed)

        self.tabu_list = {}
        self.current_cost = self.objective_f()
        self.best_cost = self.current_cost
        self.best_grid = self.grid.copy()

        self.row_counts = np.zeros((self.N, self.N + 1), dtype=int)
        self.col_counts = np.zeros((self.N, self.N + 1), dtype=int)
        self.row_missing = np.zeros(self.N, dtype=int)
        self.col_missing = np.zeros(self.N, dtype=int)

        self.model = None
        self.cp_vars = None
        self.cp_solver = None

        self.current_cost = self.objective_f()
        self.best_cost = self.current_cost
        self.best_grid = self.grid.copy()

        self.total_iterations_run = 0
        self.ls_success_count = 0  
        self.cp_call_count = 0     
        self.cp_success_count = 0

    def _initialize_auxiliary_structures(self):

        self.row_counts.fill(0)
        self.col_counts.fill(0)

        for i in range(self.N):
            for j in range(self.N):
                v = self.grid[i, j]
                if v != 0:
                    self.row_counts[i, v] += 1
                    self.col_counts[j, v] += 1

        for i in range(self.N):
            self.row_missing[i] = sum(1 for v in range(1, self.N + 1) if self.row_counts[i, v] == 0)
            self.col_missing[i] = sum(1 for v in range(1, self.N + 1) if self.col_counts[i, v] == 0)

    def _subgrid_cells(self, bi: int, bj: int) -> list[Tuple[int, int]]:

        cells = []
        i_start, i_end = bi * self.K, (bi + 1) * self.K
        j_start, j_end = bj * self.K, (bj + 1) * self.K

        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                if not self.fixed_mask[i, j]:
                    cells.append((i, j))
        return cells

    def _delta_cost_swap(self, i: int, j: int, ii: int, jj: int) -> int:

        v1, v2 = self.grid[i, j], self.grid[ii, jj]
        delta = 0

        counts = self.row_counts[i]
        if counts[v1] == 1: delta += 1
        if counts[v2] == 0: delta -= 1

        counts = self.row_counts[ii]
        if counts[v2] == 1: delta += 1
        if counts[v1] == 0: delta -= 1

        counts = self.col_counts[j]
        if counts[v1] == 1: delta += 1
        if counts[v2] == 0: delta -= 1

        counts = self.col_counts[jj]
        if counts[v2] == 1: delta += 1
        if counts[v1] == 0: delta -= 1

        return delta

    def _best_swap_in_subgrid(self, i: int, j: int, tabu: Set[Tuple[Tuple[int,int], Tuple[int,int]]],  aspiration_cost: int) -> Tuple[Optional[Tuple[int,int,int,int]], int, bool]:

        bi, bj = i // self.K, j // self.K
        cells = self._subgrid_cells(bi, bj)
        best = None
        best_delta = None
        best_tabu_aspire = False

        for ii, jj in cells:
            if (ii == i and jj == j): continue
            if self.fixed_mask[ii, jj] or self.fixed_mask[i, j]: continue

            swap_key = tuple(sorted(((i, j), (ii, jj))))
            is_tabu = swap_key in tabu

            d = self._delta_cost_swap(i, j, ii, jj)
            allow = is_tabu and (self.current_cost + d < aspiration_cost)

            if (best is None) or (d < best_delta) or (d == best_delta and self.random.random() < 0.5):
                if not is_tabu or allow:
                    best = (i, j, ii, jj)
                    best_delta = d
                    best_tabu_aspire = allow

        if best is None:
            return None, 0, False
        return best, best_delta, best_tabu_aspire

    def _cell_in_conflict(self, i: int, j: int) -> bool:

        v = self.grid[i, j]
        if v == 0: return False
        return self.row_counts[i, v] > 1 or self.col_counts[j, v] > 1

    def _apply_swap(self, i: int, j: int, ii: int, jj: int):

        v1, v2 = self.grid[i, j], self.grid[ii, jj]

        self.row_counts[i, v1] -= 1
        self.row_counts[i, v2] += 1
        self.row_counts[ii, v2] -= 1
        self.row_counts[ii, v1] += 1
        self.col_counts[j, v1] -= 1
        self.col_counts[j, v2] += 1
        self.col_counts[jj, v2] -= 1
        self.col_counts[jj, v1] += 1

        self.row_missing[i] = sum(1 for v in range(1, self.N+1) if self.row_counts[i, v] == 0)
        self.row_missing[ii] = sum(1 for v in range(1, self.N+1) if self.row_counts[ii, v] == 0)
        self.col_missing[j] = sum(1 for v in range(1, self.N+1) if self.col_counts[j, v] == 0)
        self.col_missing[jj] = sum(1 for v in range(1, self.N+1) if self.col_counts[jj, v] == 0)

        self.grid[i, j], self.grid[ii, jj] = v2, v1

        self.current_cost = np.sum(self.row_missing) + np.sum(self.col_missing)

        if self.current_cost < self.best_cost:
            self.best_cost = self.current_cost
            self.best_grid = self.grid.copy()

    def _min_conflicts_with_tabu(self, iteration_limit: int, tabu_size: int, no_improvement_limit: int = 50) -> bool:        
    
        initial_cost = self.objective_f()
    
        if initial_cost == 0:
            return False

        best_grid_ls = self.grid.copy()
        best_cost_in_ls = initial_cost
        self.current_cost = initial_cost
    
        N = self.N
        no_improvement_counter = 0 
        self.tabu_list.clear()

        moves_used = iteration_limit

        for k in range(1, iteration_limit + 1):
        
            if best_cost_in_ls == 0:
                break
            
            conflicting_cells = self.get_conflicting_cells()

            if not conflicting_cells:
                best_cost_in_ls = 0
                self.current_cost = 0
                moves_used = k
                break

            i, j = self.random.choice(list(conflicting_cells))
            current_value = self.grid[i, j]

            min_delta = float('inf')
            best_moves = []

            for new_value in range(1, N + 1):
                if new_value == current_value:
                    continue

                delta = self._calculate_delta_cost(i, j, new_value)

                if delta < min_delta:
                    min_delta = delta
                    best_moves = [(i, j, new_value)]
                elif delta == min_delta:
                    best_moves.append((i, j, new_value))

                if best_moves:
                    r, c, new_value = self.random.choice(best_moves)
                    old_value = current_value 
            
                is_tabu = self.tabu_list.get((r, c)) == old_value
                is_aspirated = (self.current_cost + min_delta < self.best_cost) 
            
                if is_tabu and not is_aspirated and min_delta >= 0:
                    continue 
                            
                self.grid[r, c] = new_value

                self.tabu_list[(r, c)] = old_value 
                if len(self.tabu_list) > tabu_size:
                    key_to_delete = next(iter(self.tabu_list)) 
                    del self.tabu_list[key_to_delete]

                new_full_cost = self.objective_f() 
                self.current_cost = new_full_cost 

                if new_full_cost < best_cost_in_ls:
                    best_cost_in_ls = new_full_cost
                    best_grid_ls = self.grid.copy()
                    no_improvement_counter = 0

                if new_full_cost == 0:
                    moves_used = k
                    self.current_cost = 0
                    best_cost_in_ls = 0
                    best_grid_ls = self.grid.copy()
                    break 
                else:
                    no_improvement_counter += 1
        
                if no_improvement_counter >= no_improvement_limit:
                    self.grid = best_grid_ls.copy()
                    self.current_cost = best_cost_in_ls
    
        self.grid = best_grid_ls.copy()
        self.current_cost = best_cost_in_ls 

        if self.current_cost < initial_cost:
            self.ls_success_count += 1 

        return initial_cost != self.current_cost, moves_used

    def _accept(self, old_cost: int, new_cost: int, T: float, mode: str = "metropolis", accept_prob: float = 0.0) -> bool:
        
        mode = mode.lower()

        if mode == "hill":
            if new_cost < old_cost:
                return True
        
            return self.random.random() < float(accept_prob)
        else:
            
            if new_cost <= old_cost:
                return True
            if T <= 1e-12:
                return False
            
            delta = new_cost - old_cost
            p = math.exp(-float(delta) / float(T))

            return self.random.random() < p

    def _temperature(self, it: int, T0: float, alpha: float) -> float:
        
        return float(T0) * (float(alpha) ** int(it))
    
    def ils_run(self,
            time_limit_s: float = 10.0,
            mode: str = "metropolis",        
            T0: float = 1.25, alpha: float = 0.995,
            accept_prob: float = 0.01,       
            ls_callback=None,                
            perturb_callback=None,           
            ls_kwargs: dict | None = None,
            perturb_kwargs: dict | None = None,
            verbose: bool = False):
        
        start = time.time()
        ls_kwargs = ls_kwargs or {}
        perturb_kwargs = perturb_kwargs or {}

        cur_cost = int(self.objective_f())
        best_cost = cur_cost
        best_grid = self.grid.copy()

        it = 0
        while time.time() - start < time_limit_s and best_cost > 0:
            if ls_callback is not None:
                ls_callback(self, **ls_kwargs)
            cur_cost = int(self.objective_f())

            if cur_cost < best_cost:
                best_cost = cur_cost
                best_grid = self.grid.copy()
                if verbose:
                    print(f"LS best={best_cost}")

            if perturb_callback is not None:
                perturb_callback(self, **perturb_kwargs)

            new_cost = int(self.objective_f())
            T = self._temperature(it, T0, alpha) if mode != "hill" else 0.0
            if self._accept(cur_cost, new_cost, T, mode=mode, accept_prob=accept_prob):
                cur_cost = new_cost
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_grid = self.grid.copy()
                    if verbose:
                        print(f"Prihvaćeno bolje: ={best_cost}")
            else:
           
                self.grid[:, :] = best_grid
                cur_cost = best_cost
                if verbose:
                    print(f"Vrati se na bolje ={best_cost}")

            it += 1

        self.grid[:, :] = best_grid
        return best_cost

    def _free_cells_in_block(self, ik: int, jk: int):
        K = self.K
        ik_start = ik * K
        ik_end = (ik + 1)*K
        jk_start = jk * K
        jk_end = (jk + 1)*K
        
        free_cells = []

        for i in range(ik_start, ik_end):
            for j in range(jk_start, jk_end):
                if not self.fixed_mask[i, j]:
                    free_cells.append((i, j))

        return free_cells
    
    def _perturb_one_swap_in_block(self):
        K = self.K
    
        order = []  
        for ik in range(K):
            for jk in range(K):
                order.append((ik, jk))

        self.random.shuffle(order)

        for (ik, jk) in order:
            cells = self._free_cells_in_block(ik, jk)
            if len(cells) >= 2:
                (i1, j1), (i2, j2) = self.random.sample(cells, 2)
                self.grid[i1, j1], self.grid[i2, j2] = self.grid[i2, j2], self.grid[i1, j1]
                return  
            
    def _perturb_k_swaps(self, k: int = 3):
        for _ in range(k):
            self._perturb_one_swap_in_block()
            
    def _perturb_shuffle_block(self):
        K = self.K

        blocks = []
        for ik in range(K):
            for jk in range(K):
                blocks.append((ik, jk))

        self.random.shuffle(blocks)

        for (bi, bj) in blocks:
            cells = self._free_cells_in_block(bi, bj)   
            if len(cells) >= 2:
                vals = []
                for(i, j) in cells:
                    vals.append(self.grid[i, j])

                self.random.shuffle(vals)

                for (i, j), v in zip(cells, vals):
                    self.grid[i, j] = v
                return
            
    def perturb(self, p_rate: float, cp_time_limit: float) -> None:
 
        N = self.N
        
        total_cells_to_empty = int(N * N * p_rate)
        
        mutable_cells = []
        for i in range(N):
            for j in range(N):
                if self.grid[i, j] != 0 and not self.fixed_mask[i, j]:
                    mutable_cells.append((i, j))

        if len(mutable_cells) < total_cells_to_empty:
            total_cells_to_empty = len(mutable_cells)

        if not mutable_cells:
            return
             
        indices_to_empty = self.random.choice(
            mutable_cells, 
            size=total_cells_to_empty, 
            replace=False  
        )
        
        for i, j in indices_to_empty:
            self.grid[i, j] = 0
            
        self.current_cost = self.objective_f()
        
        cp_refiner = SudokuCP(self.grid.copy(), seed=None) 
        
        cp_refiner.fixed_mask = self.fixed_mask.copy() 
        
        status = cp_refiner.cp_refinement(
            time_limit=cp_time_limit, 
            fix_noncon=False, 
            hints=False 
        ) 

        if status in (OPTIMAL, FEASIBLE):
            self.grid = cp_refiner.grid.copy() 

    def _prepare_final_results(self, start_time, total_iterations, ls_iterations, acceptance_prob, tabu_size, cp_limit, empty_factor_init, alpha):
    
        end_time = time.time()
        total_time = end_time - start_time
        solution_is_valid = self.objective_f() == 0

        results = {
            'Problem_Size': self.N, 
            'Total_Iterations': total_iterations,
            'LS_Iterations': ls_iterations,
            'Acceptance_Prob': acceptance_prob,
            'Tabu_Size': tabu_size,
            'CP_Limit_s': cp_limit,
            'Empty_Factor_Init': empty_factor_init,
            'Alpha_Decay': alpha,
        
            'Execution_Time_s': total_time,
            'Best_Cost': self.best_cost,
            'Solution_Valid': solution_is_valid,
            'CP_Calls': self.cp_call_count,              
            'CP_Successes': self.cp_success_count,
        }
        return results
         
    def solve_ils_cp(self, total_iterations: int = 1000, ls_iterations: int = 5000, acceptance_prob: float = 0.05, tabu_size: int = 10, cp_limit: float = 8.0, empty_factor_init: float = 0.1, alpha: float = 0.99) -> dict:

        start_time = time.time()

        total_ls_moves_to_solution = total_iterations * ls_iterations

        self.convergence_data = []

        if self.current_cost == 0 and self.is_valid():
            print("ILS: Rešenje je pronađeno u inicijalizaciji.")
            return self._prepare_final_results(start_time, 0, 0, acceptance_prob, tabu_size, cp_limit, empty_factor_init, alpha)

        self.best_cost = self.current_cost 
        self.best_grid = self.grid.copy()

        empty_factor = empty_factor_init

        print(f"Hibridni ILS (Početna cena: {self.best_cost})")

        for k in range(1, total_iterations + 1):

            self.total_iterations_run = k

            self.current_cost = self.objective_f()
            self._min_conflicts_with_tabu(ls_iterations, acceptance_prob, tabu_size)

            moves_used_in_ls = self._min_conflicts_with_tabu(ls_iterations, tabu_size)

            self.current_cost = self.objective_f()

            if self.current_cost == 0:
                print(f"ILS uspešno rešen u iteraciji {k} nakon LS-a.")
                total_ls_moves_to_solution = (k - 1) * ls_iterations + moves_used_in_ls
                self.best_cost = 0  
                self.best_grid = self.grid.copy() 
                
                return self._prepare_final_results(start_time, k, total_ls_moves_to_solution, acceptance_prob, tabu_size, cp_limit, empty_factor_init, alpha)
            
            if self.current_cost < self.best_cost:
                self.best_cost = self.current_cost
                self.best_grid = self.grid.copy()

            if self.current_cost > self.best_cost:
                self.grid = self.best_grid.copy()
                self.current_cost = self.best_cost 

            if self.current_cost > 0:
                self.cp_call_count += 1
    
            self.perturb(p_rate=empty_factor, cp_time_limit=cp_limit)

            self.current_cost = self.objective_f() 
        
            if self.current_cost <= self.best_cost:
            
                if self.current_cost < self.best_cost or self.current_cost == 0:
                    self.cp_success_count += 1
                
                    self.best_cost = self.current_cost
                    self.best_grid = self.grid.copy()

            if self.current_cost == 0:
                total_ls_moves_to_solution = k * ls_iterations
                print(f"ILS uspešno rešen u iteraciji {k} nakon perturbacije/CP-a.")
                return self._prepare_final_results(start_time, k, total_ls_moves_to_solution, acceptance_prob, tabu_size, cp_limit, empty_factor_init, alpha) 
                   
            current_elapsed_time = time.time() - start_time
        
            self.convergence_data.append({
                'k': k,                              
                'best_cost': self.best_cost,         
                'time_s': current_elapsed_time      
                })

            empty_factor *= alpha

            print(f"ILS Ciklus {k}/{total_iterations}: Trošak: {self.current_cost}, Najbolji: {self.best_cost}, Faktor kvarenja: {empty_factor:.3f}")

            if k % 50 == 0:
                self.display_grid(f"ILS Stanje nakon {k} ciklusa (Cena: {self.best_cost})")

        self.grid = self.best_grid.copy()

        end_time = time.time()
        total_time = end_time - start_time
        solution_is_valid = self.objective_f() == 0

        if not solution_is_valid:
             total_ls_moves_to_solution = total_iterations * ls_iterations
        else:
             total_ls_moves_to_solution = total_iterations * ls_iterations

        print(f"\nILS završio nakon {total_iterations} iteracija. Najbolja cena: {self.best_cost}")
        print("\n--- ZAVRŠNA STATISTIKA ILS-CP ---")
        print(f"Najbolja postignuta cena: {self.best_cost}")
        print(f"Rešenje validno: {solution_is_valid}")
        print(f"Vreme izvršenja (ILS-CP): {total_time:.4f} sekundi")
        print(f"CP poziva/uspeha: {self.cp_call_count} / {self.cp_success_count}")
        print("\n")

        return self._prepare_final_results(start_time, total_iterations, total_ls_moves_to_solution, acceptance_prob, tabu_size, cp_limit, empty_factor_init, alpha)
        
class SudokuCP(SudokuSolver):
    def __init__(self, puzzle, seed = None):
        super().__init__(puzzle)
        self.random = np.random.default_rng(seed)

    def _build_cp_model(self):
        self.model = cp_model.CpModel()
        N = self.N
        K = self.K

        x = []
        for i in range(N):
            row = []
            for j in range(N):
                var = self.model.NewIntVar(1, N, f"x[{i},{j}]")
                row.append(var)

            x.append(row)

        for i in range(N):
            self.model.AddAllDifferent(x[i])

        for j in range(N):
            col = []
            for i in range(N):
                col.append(x[i][j])

            self.model.AddAllDifferent(col)

        for ik in range(K):
            for jk in range(K):
                ik_start = ik * K
                ik_end = (ik + 1) * K
                jk_start = jk * K
                jk_end = (jk + 1) * K

                cells = []
                for i in range(ik_start, ik_end):
                    for j in range(jk_start, jk_end):
                        cells.append(x[i][j])

                self.model.AddAllDifferent(cells)

        self.cp_vars = x
        self.cp_solver = cp_model.CpSolver()

    def _is_cell_nonconflicting(self, i: int, j: int ) ->bool:
        val = int(self.grid[i, j])

        if val == 0:
            return False

        if np.count_nonzero(self.grid[i, :] == val) > 1:
            return False

        if np.count_nonzero(self.grid[:, j] == val ) > 1:
            return False

        ik = i // self.K
        jk = j // self.K

        ik_start = ik * self.K
        ik_end = (ik + 1) * self.K
        jk_start = jk * self.K
        jk_end = (jk + 1) * self.K

        block = self.grid[ik_start:ik_end, jk_start:jk_end]

        if np.count_nonzero(block == val) > 1:
            return False

        return True

    def cp_refinement(self, time_limit: float | None = 10.0, fix_noncon: bool = False, hints: bool = True, log_search: bool = False):
        
        self._build_cp_model() 

        model = self.model
        x = self.cp_vars
        solver = self.cp_solver
        N = self.N
        
        for i in range(N):
            for j in range(N):
                if self.fixed_mask[i, j] and self.grid[i, j] != 0:
                    model.Add(x[i][j] == int(self.grid[i][j]))
        
        if fix_noncon:
            for i in range(N):
                for j in range(N):
                    if not self.fixed_mask[i, j] and self.grid[i, j] != 0:
                        if self._is_cell_nonconflicting(i, j):
                            model.Add(x[i][j] == int(self.grid[i][j]))

        if hints:
            for i in range(N):
                for j in range(N):
                    val = int(self.grid[i, j])
                    if 1 <= val <= N:
                        model.AddHint(x[i][j], val)

        if time_limit is not None:
            solver.parameters.max_time_in_seconds = float(time_limit)
        
        solver.parameters.log_search_progress = bool(log_search)

        status = solver.Solve(model)

        if status in (OPTIMAL, FEASIBLE):
            for i in range(N):
                for j in range(N):
                    self.grid[i, j] = int(solver.Value(x[i][j])) 

        return status
