\ Model LinMcCormick
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x[1] + x[2] + x[3]
Subject To
 R0: 0.0025 x[4] + 0.0025 x[6] <= 1
 R1: - 0.0025 x[4] + 0.0025 x[5] + 0.0025 x[7] <= 1
 R2: - 0.01 x[5] + 0.01 x[8] <= 1
 qc0: 100 x[1] + 833.33252 x[4] + [ - x[1] * x[6] ] <= 83333.333
 qc1: - 1250 x[4] + 1250 x[5] + [ x[2] * x[4] - x[2] * x[7] ] <= 0
 qc2: - 2500 x[5] + [ x[3] * x[5] - x[3] * x[8] ] <= -1.25e+06
Bounds
 100 <= x[1] <= 10000
 1000 <= x[2] <= 10000
 1000 <= x[3] <= 10000
 10 <= x[4] <= 1000
 10 <= x[5] <= 1000
 10 <= x[6] <= 1000
 10 <= x[7] <= 1000
 10 <= x[8] <= 1000
End
