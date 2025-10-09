import numpy as np

class SudokuSolver:
    
    def __init__(self, puzzle):
       
        self.grid = np.array(puzzle, dtype=int)
        self.N = self.grid.shape[0]  # Veličina Sudokua
        self.K = int(np.sqrt(self.N)) # Veličina bloka

        # Matrica za praćenje fiksiranih ćelija
        self.fixed_mask = (self.grid != 0) 

    def display_grid(self, title="Trenutna Sudoku ploča"):
        
        print(f"\n--- {title} (Veličina: {self.N}x{self.N}) ---")
        
        # Funkcija za ispis separatora
        def print_separator():
            line = "+---" * self.K 
            print(line * self.K + "+")

        # Ispisivanje ploče
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
