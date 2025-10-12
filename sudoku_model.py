import numpy as np
from collections import deque
from typing import Set, Tuple, Optional
import random

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

    def get_conflicts(self):
        conlficts = []

        for i in range(self.N):
            for j in range(self.N):
                v = self.grid[i,j]
                if v == 0:
                    continue
                
                con_row = np.count_nonzero(self.grid[i, :] == v) > 1  
                con_col = np.count_nonzero(self.grid[:, j] == v) > 1

                ik = i // self.K 
                jk = j // self.K

                ik_start = ik * self.K
                ik_end = (ik + 1 ) * self.K
                jk_start = jk * self.K
                jk_end = (jk + 1 ) * self.K

                block = self.grid[ik_start:ik_end, jk_start:jk_end]
                con_block = np.count_nonzero(block == v) > 1

                if con_row or con_col or con_block:
                    conlficts.append((i, j))

        return conlficts
    
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

class ILS_CP(SudokuSolver):

    def __init__(self, puzzle, seed: int | None = None):
        super().__init__(puzzle)
        
        if seed is not None:
            self.random = random.Random(seed)
        else:
            self.random = random.Random()
        
        self.current_cost = 0
        self.best_cost = float('inf')
        self.best_grid = self.grid.copy()
        
        self.row_counts = np.zeros((self.N, self.N + 1), dtype=int)
        self.col_counts = np.zeros((self.N, self.N + 1), dtype=int)
        self.row_missing = np.zeros(self.N, dtype=int)
        self.col_missing = np.zeros(self.N, dtype=int)

        self._initialize_auxiliary_structures()

