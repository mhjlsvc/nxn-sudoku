import numpy as np

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
