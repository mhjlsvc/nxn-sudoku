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
        
        def print_separator():
            line = "+---" * self.K 
            print(line * self.K + "+")

        
        for i in range(self.N):
            if i % self.K == 0:
                print_separator()
            
            row_str = "|"
            for j in range(self.N):
                val = str(self.grid[i, j]) if self.grid[i, j] != 0 else "."
                
                row_str += f" {val} "
                if (j + 1) % self.K == 0:
                    row_str += "|"
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

    # def get_conflicting_cells(self) -> Set[Tuple[int, int]]: #Napravljena druga fja koja radi ovaj posao, ali mozda zatreba ova
  
    #     N = self.N
    #     conflicting_cells = set()

    #     for i in range(N):
    #         for j in range(N):
    #             val = self.grid[i, j]
                
    #             if val == 0: continue
                
    #             is_conflicting = False
                                
    #             if np.count_nonzero(self.grid[i, :] == val) > 1:
    #                 is_conflicting = True
                
    #             if np.count_nonzero(self.grid[:, j] == val) > 1:
    #                 is_conflicting = True

    #             ik_start = (i // self.K) * self.K
    #             ik_end = ik_start + self.K
    #             jk_start = (j // self.K) * self.K
    #             jk_end = jk_start + self.K
    #             block = self.grid[ik_start:ik_end, jk_start:jk_end]
                
    #             if np.count_nonzero(block == val) > 1:
    #                 is_conflicting = True
                
    #             if is_conflicting and not self.fixed_mask[i, j]:
    #                 conflicting_cells.add((i, j))

    #     return conflicting_cells
    
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

    def _calculate_delta_cost(self, i: int, j: int, new_value: int) -> int: # Za LS
        
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

    def _min_conflicts_with_tabu(self, iteration_limit: int, acceptance_prob: float, tabu_size: int) -> None:

        tabu = deque([], maxlen=tabu_size)
        tabu_set = set()

        def add_tabu(a: Tuple[int, int], b: Tuple[int, int]):
            key = tuple(sorted((a, b)))
            tabu.append(key)
            tabu_set.clear()
            tabu_set.update(tabu)

        for _t in range(50):
            i = self.random.integers(0, self.N) 
            j = self.random.integers(0, self.N)
    
        for _ in range(iteration_limit):
            if self.current_cost == 0:
                return

            chosen = None
            for _t in range(50):                
                i = self.random.integers(0, self.N) 
                j = self.random.integers(0, self.N) 
                
                if not self.fixed_mask[i, j] and self._cell_in_conflict(i, j):
                    chosen = (i, j)
                    break
            
            if chosen is None:
                unfixed = [(i, j) for i in range(self.N) for j in range(self.N) if not self.fixed_mask[i, j]]
                if not unfixed: return
                chosen = self.random.choice(unfixed)
            i, j = chosen

            best_move, delta, asp = self._best_swap_in_subgrid(i, j, tabu_set, self.best_cost)
            if best_move is None: continue

            new_cost = self.current_cost + delta
            
            if new_cost < self.current_cost or self.random.random() < acceptance_prob or asp:
                i, j, ii, jj = best_move 
                
                self._apply_swap(i, j, ii, jj) 
                
                add_tabu((i, j), (ii, jj))

            if self.current_cost == 0:
                return
            
class SudokuCP(SudokuSolver):
    def __init__(self, puzzle, seed = None):
        super().__init__(puzzle)

        self.seed = seed
        self.model = None
        self.cp_vars = None
        self.cp_solver = None

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

        for i in range(N):
            for j in range(N):
                if self.grid[i, j] != 0:
                    self.model.Add(x[i][j] == int(self.grid[i, j]))

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
    
    def cp_refinement(self, time_limit: float | None = 10.0, fix_noncon: bool = True, hints: bool = True, log_search: bool = False):
        if self.model is None or self.cp_vars is None or self.cp_solver is None:
            self._build_cp_model()

        model = self.model
        x = self.cp_vars
        solver = self.cp_solver
        N = self.N

        for i in range(N):
            for j in range(N):
                if self.fixed_mask[i, j] and self.grid[i, j] != 0:
                    model.Add(x[i][j] == int(self.grid[i, j]))

        
        if fix_noncon:
            for i in range(N):
                for j in range(N):
                    if not self.fixed_mask[i, j] and self.grid[i, j] != 0:
                        if self._is_cell_nonconflicting(i, j):
                            model.Add(x[i][j] == int(self.grid[i, j]))

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
