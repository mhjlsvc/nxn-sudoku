Tabela parametara i metrike
 
|   Kolona         |  Opis                                                                                                         |
|------------------|---------------------------------------------------------------------------------------------------------------|
| Probem_Size      |  Veličina instance problema (npr. 9×9 Sudoku)                                                                 |
| LS Iterations    |  Maksimalni broj koraka u fazi lokalne pretrage                                                               |
| Acceptance Prob  |  Verovatnoća prihvatanja lošijeg rešenja u lokalnoj pretrazi                                                  |
| Tabu Size	       |  Veličina Tabu liste u Min-Conflicts varijanti                                                                |
| CP Limit         |  Maksimalno vreme u sekundama dozvoljeno rešavaču ograničenja (CP Solver) prilikom poziva                     |
| Empty_Factor_Init|  Početna veličina "kvarenja" rešenja koje se unosi u fazi perturbacije                                        |
| Alpha Decay	     |  Faktor opadanja koji se koristi za prilagođavanje faktora kvarenja (perturbacije) tokom iteracija            |
| Execution Time   |  Ukupno vreme izvršavanja algoritma u sekundama                                                               |
| Best Cost	       |  Najbolja pronađena cena (0 označava potpuno rešen problem)                                                   |
| Solution Valid   |  Da li je najbolje pronađeno rešenje validno (True ako je Best Cost = 0)                                      |
| CP Calls         |  Broj poziva CP rešavača                                                                                      |
| CP Successes     |  Broj puta kada je CP pronašao validno rešenje ili novo najbolje rešenje                                      |

*Napomena *
Posebnu pažnju treba obratiti na eksperimente 10 i 11. Ovi eksperimenti su sprovedeni na nerešivim instancama problema, koje su namerno uključene radi validacije pouzdanosti algoritma.

Rezultati eksperimenta 1

ID  | Probem_Size  | Total_Iterations| LS_Iterations | Acceptance_Prob | Tabu_Size  | CP_Limit_s  | Empty_Factor_Init | Alpha_Decay  | Execution_Time_s | Best_Cost | Solution_Valid | CP_Calls | CP_Successes  | 
----|--------------|-----------------|---------------|-----------------|------------|-------------|-------------------|--------------|------------------|-----------|----------------|----------|---------------|  
1   | 4            | 0               | 0             | 0.15            | 10         | 15          | 0.2               | 0.995        | 0.0002           | 0         | True           | 0        | 0             | 
2   | 4            | 0               | 0             | 0.15            | 10         | 15          | 0.2               | 0.995        | 0.0001           | 0         | True           | 0        | 0             | 
3   | 4            | 0               | 0             | 0.15            | 10         | 15          | 0.2               | 0.995        | 0.0001           | 0         | True           | 0        | 0             | 
4   | 9            | 1               | 5000          | 0.15            | 10         | 15          | 0.2               | 0.995        | 18.4516          | 0         | True           | 1        | 1             | 
5   | 9            | 1               | 5000          | 0.15            | 10         | 15          | 0.2               | 0.995        | 18.5609          | 0         | True           | 1        | 1             | 
6   | 9            | 1               | 5000          | 0.15            | 10         | 15          | 0.2               | 0.995        | 18.7135          | 0         | True           | 1        | 1             | 
7   | 16           | 1               | 5000          | 0.15            | 10         | 15          | 0.2               | 0.995        | 58.0719          | 0         | True           | 1        | 1             | 
8   | 16           | 1               | 5000          | 0.15            | 10         | 15          | 0.2               | 0.995        | 58.6738          | 0         | True           | 1        | 1             | 
9   | 25           | 1               | 5000          | 0.15            | 10         | 15          | 0.2               | 0.995        | 144.056          | 0         | True           | 1        | 1             | 
10  | 9            | 200             | 10000         | 0.15            | 10         | 15          | 0.2               | 0.995        | 37.252           | 2         | False *        | 200      | 1             | 
11  | 16           | 200             | 10000         | 0.15            | 10         | 15          | 0.2               | 0.995        | 58.043           | 90        | False *        | 200      | 1             | 

Rezultati eksperimenta 2

ID  | Probem_Size  | Total_Iterations| LS_Iterations | Acceptance_Prob | Tabu_Size  | CP_Limit_s  | Empty_Factor_Init | Alpha_Decay  | Execution_Time_s | Best_Cost | Solution_Valid | CP_Calls | CP_Successes  | 
----|--------------|-----------------|---------------|-----------------|------------|-------------|-------------------|--------------|------------------|-----------|----------------|----------|---------------|  
1   | 4            | 0               | 0             | 0.15            | 10         | 15          | 0.2               | 0.2 *        | 0.0002           | 0         | True           | 0        | 0             | 
2   | 4            | 0               | 0             | 0.15            | 10         | 15          | 0.2               | 0.2 *        | 0.0002           | 0         | True           | 0        | 0             | 
3   | 4            | 0               | 0             | 0.15            | 10         | 15          | 0.2               | 0.2 *        | 0.0002           | 0         | True           | 0        | 0             | 
4   | 9            | 1               | 5000          | 0.15            | 10         | 15          | 0.2               | 0.2 *        | 22.0283          | 0         | True           | 1        | 1             | 
5   | 9            | 1               | 5000          | 0.15            | 10         | 15          | 0.2               | 0.2 *        | 21.4642          | 0         | True           | 1        | 1             | 
6   | 9            | 1               | 5000          | 0.15            | 10         | 15          | 0.2               | 0.2 *        | 21.6787          | 0         | True           | 1        | 1             | 
7   | 16           | 1               | 5000          | 0.15            | 10         | 15          | 0.2               | 0.2 *        | 65.9350          | 0         | True           | 1        | 1             | 
8   | 16           | 1               | 5000          | 0.15            | 10         | 15          | 0.2               | 0.2 *        | 110.350          | 0         | True           | 1        | 1             | 
9   | 25           | 1               | 5000          | 0.15            | 10         | 15          | 0.2               | 0.2 *        | 278.219 *        | 0         | True           | 1        | 1             | 
10  | 9            | 200             | 1000000       | 0.15            | 10         | 15          | 0.2               | 0.2 *        | 37.004           | 2         | False          | 200      | 1             | 
11  | 16           | 200             | 1000000       | 0.15            | 10         | 15          | 0.2               | 0.2 *        | 117.197 *        | 90        | False          | 200      | 1             | 

Rezultati eksperimenta 3

ID  | Probem_Size  | Total_Iterations| LS_Iterations | Acceptance_Prob | Tabu_Size  | CP_Limit_s  | Empty_Factor_Init | Alpha_Decay  | Execution_Time_s | Best_Cost | Solution_Valid | CP_Calls | CP_Successes  | 
----|--------------|-----------------|---------------|-----------------|------------|-------------|-------------------|--------------|------------------|-----------|----------------|----------|---------------|  
1   | 4            | 0               | 0             | 0.15            | 10         | 15          | 0.2               | 0.995        | 0.0002           | 0         | True           | 0        | 0             | 
2   | 4            | 0               | 0             | 0.15            | 10         | 15          | 0.2               | 0.995        | 0.0002           | 0         | True           | 0        | 0             | 
3   | 4            | 0               | 0             | 0.15            | 10         | 15          | 0.2               | 0.995        | 0.0002           | 0         | True           | 0        | 0             | 
4   | 9            | 1               | 50            | 0.15            | 10         | 15          | 0.2               | 0.995        | 0.2584           | 0         | True           | 1        | 1             | 
5   | 9            | 1               | 50            | 0.15            | 10         | 15          | 0.2               | 0.995        | 0.2975           | 0         | True           | 1        | 1             | 
6   | 9            | 1               | 50            | 0.15            | 10         | 15          | 0.2               | 0.995        | 0.2693           | 0         | True           | 1        | 1             | 
7   | 16           | 1               | 50            | 0.15            | 10         | 15          | 0.2               | 0.995        | 0.8752           | 0         | True           | 1        | 1             | 
8   | 16           | 1               | 50            | 0.15            | 10         | 15          | 0.2               | 0.995        | 1.3291           | 0         | True           | 1        | 1             | 
9   | 25           | 1               | 50            | 0.15            | 10         | 15          | 0.2               | 0.995        | 3.7345 *         | 0         | True           | 1        | 1             | 
10  | 9            | 50  *           | 2500          | 0.15            | 10         | 15          | 0.2               | 0.995        | 0.6796           | 2         | False          | 50       | 2             | 
11  | 16           | 50  *           | 2500          | 0.15            | 10         | 15          | 0.2               | 0.995        | 1.6111           | 90        | False          | 50       | 1             | 

Rezultati eksperimenta 4  - Uporedni eksperiment (Verifikacija efikasnosti)

ID  | Kategorija    | SR(%)           |Avg. LS Poteza | Avg. Vreme (s)  | Ukupni uspesi  | Ukupno pokretanja  |
----|-------------- |-----------------|---------------|-----------------|----------------|--------------------|  
1   | 5% Fiksirano  | 100.00          | 2000          | 14.04           | 4              | 4                  | 
2   | 10% Fiksirano | 100.00          | 2000          | 14.38           | 4              | 4                  |
3   | 15% Fiksirano | 100.00          | 2000          | 14.29           | 4              | 4                  |
4   | 20% Fiksirano | 100.00          | 2000          | 14.49           | 4              | 4                  |
5   | 25% Fiksirano | 100.00          | 2000          | 14.48           | 4              | 4                  | 
6   | 30% Fiksirano | 100.00          | 2000          | 14.43           | 4              | 4                  |
7   | 40% Fiksirano | 100.00          | 2000          | 14.57           | 4              | 4                  | 
8   | 50% Fiksirano | 100.00          | 2000          | 14.21           | 4              | 4                  | 
9   | 60% Fiksirano | 100.00          | 2000          | 14.17           | 4              | 4                  |  
10  | 70% Fiksirano | 100.00          | 500           | 3.54            | 4              | 4                  | 
11  | 80% Fiksirano | 100.00          | 0             | 0.00            | 4              | 4                  | 
12  | 90% Fiksirano | 100.00          | 0             | 0.00            | 4              | 4                  | 
13  | 100% Fiksirano| 100.00          | 0             | 0.00            | 4              | 4                  | 

        
