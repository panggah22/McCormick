\ Model Case33-Linear
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 P_Gen[0,0] + 0 Q_Gen[0,0] + 0 P_hat[0,1,0] + 0 P_hat[1,0,0]
   + 0 P_hat[1,2,0] + 0 P_hat[2,1,0] + 0 P_hat[2,3,0] + 0 P_hat[3,2,0]
   + 0 P_hat[3,4,0] + 0 P_hat[4,3,0] + 0 P_hat[4,5,0] + 0 P_hat[5,4,0]
   + 0 P_hat[5,6,0] + 0 P_hat[6,5,0] + 0 P_hat[6,7,0] + 0 P_hat[7,6,0]
   + 0 P_hat[7,8,0] + 0 P_hat[8,7,0] + 0 P_hat[8,9,0] + 0 P_hat[9,8,0]
   + 0 P_hat[9,10,0] + 0 P_hat[10,9,0] + 0 P_hat[10,11,0]
   + 0 P_hat[11,10,0] + 0 P_hat[11,12,0] + 0 P_hat[12,11,0]
   + 0 P_hat[12,13,0] + 0 P_hat[13,12,0] + 0 P_hat[13,14,0]
   + 0 P_hat[14,13,0] + 0 P_hat[14,15,0] + 0 P_hat[15,14,0]
   + 0 P_hat[15,16,0] + 0 P_hat[16,15,0] + 0 P_hat[16,17,0]
   + 0 P_hat[17,16,0] + 0 P_hat[1,18,0] + 0 P_hat[18,1,0]
   + 0 P_hat[18,19,0] + 0 P_hat[19,18,0] + 0 P_hat[19,20,0]
   + 0 P_hat[20,19,0] + 0 P_hat[20,21,0] + 0 P_hat[21,20,0]
   + 0 P_hat[2,22,0] + 0 P_hat[22,2,0] + 0 P_hat[22,23,0]
   + 0 P_hat[23,22,0] + 0 P_hat[23,24,0] + 0 P_hat[24,23,0]
   + 0 P_hat[5,25,0] + 0 P_hat[25,5,0] + 0 P_hat[25,26,0]
   + 0 P_hat[26,25,0] + 0 P_hat[26,27,0] + 0 P_hat[27,26,0]
   + 0 P_hat[27,28,0] + 0 P_hat[28,27,0] + 0 P_hat[28,29,0]
   + 0 P_hat[29,28,0] + 0 P_hat[29,30,0] + 0 P_hat[30,29,0]
   + 0 P_hat[30,31,0] + 0 P_hat[31,30,0] + 0 P_hat[31,32,0]
   + 0 P_hat[32,31,0] + 0 ploss[0]
Subject To
 P-Balance[0,0]: - P_Inj[0,0] + P_Line[0,1,0] + 0.0922 L_Line[0,1,0] = 0
 P-Balance[1,0]: - P_Inj[1,0] - P_Line[0,1,0] + P_Line[1,2,0]
   + P_Line[1,18,0] + 0.493 L_Line[1,2,0] + 0.164 L_Line[1,18,0] = 0
 P-Balance[2,0]: - P_Inj[2,0] - P_Line[1,2,0] + P_Line[2,3,0]
   + P_Line[2,22,0] + 0.366 L_Line[2,3,0] + 0.4512 L_Line[2,22,0] = 0
 P-Balance[3,0]: - P_Inj[3,0] - P_Line[2,3,0] + P_Line[3,4,0]
   + 0.3811 L_Line[3,4,0] = 0
 P-Balance[4,0]: - P_Inj[4,0] - P_Line[3,4,0] + P_Line[4,5,0]
   + 0.819 L_Line[4,5,0] = 0
 P-Balance[5,0]: - P_Inj[5,0] - P_Line[4,5,0] + P_Line[5,6,0]
   + P_Line[5,25,0] + 0.1872 L_Line[5,6,0] + 0.203 L_Line[5,25,0] = 0
 P-Balance[6,0]: - P_Inj[6,0] - P_Line[5,6,0] + P_Line[6,7,0]
   + 0.7114 L_Line[6,7,0] = 0
 P-Balance[7,0]: - P_Inj[7,0] - P_Line[6,7,0] + P_Line[7,8,0]
   + 1.03 L_Line[7,8,0] = 0
 P-Balance[8,0]: - P_Inj[8,0] - P_Line[7,8,0] + P_Line[8,9,0]
   + 1.044 L_Line[8,9,0] = 0
 P-Balance[9,0]: - P_Inj[9,0] - P_Line[8,9,0] + P_Line[9,10,0]
   + 0.1966 L_Line[9,10,0] = 0
 P-Balance[10,0]: - P_Inj[10,0] - P_Line[9,10,0] + P_Line[10,11,0]
   + 0.3744 L_Line[10,11,0] = 0
 P-Balance[11,0]: - P_Inj[11,0] - P_Line[10,11,0] + P_Line[11,12,0]
   + 1.468 L_Line[11,12,0] = 0
 P-Balance[12,0]: - P_Inj[12,0] - P_Line[11,12,0] + P_Line[12,13,0]
   + 0.5416 L_Line[12,13,0] = 0
 P-Balance[13,0]: - P_Inj[13,0] - P_Line[12,13,0] + P_Line[13,14,0]
   + 0.591 L_Line[13,14,0] = 0
 P-Balance[14,0]: - P_Inj[14,0] - P_Line[13,14,0] + P_Line[14,15,0]
   + 0.7463 L_Line[14,15,0] = 0
 P-Balance[15,0]: - P_Inj[15,0] - P_Line[14,15,0] + P_Line[15,16,0]
   + 1.289 L_Line[15,16,0] = 0
 P-Balance[16,0]: - P_Inj[16,0] - P_Line[15,16,0] + P_Line[16,17,0]
   + 0.732 L_Line[16,17,0] = 0
 P-Balance[17,0]: - P_Inj[17,0] - P_Line[16,17,0] = 0
 P-Balance[18,0]: - P_Inj[18,0] - P_Line[1,18,0] + P_Line[18,19,0]
   + 1.5042 L_Line[18,19,0] = 0
 P-Balance[19,0]: - P_Inj[19,0] - P_Line[18,19,0] + P_Line[19,20,0]
   + 0.4095 L_Line[19,20,0] = 0
 P-Balance[20,0]: - P_Inj[20,0] - P_Line[19,20,0] + P_Line[20,21,0]
   + 0.7089 L_Line[20,21,0] = 0
 P-Balance[21,0]: - P_Inj[21,0] - P_Line[20,21,0] = 0
 P-Balance[22,0]: - P_Inj[22,0] - P_Line[2,22,0] + P_Line[22,23,0]
   + 0.898 L_Line[22,23,0] = 0
 P-Balance[23,0]: - P_Inj[23,0] - P_Line[22,23,0] + P_Line[23,24,0]
   + 0.896 L_Line[23,24,0] = 0
 P-Balance[24,0]: - P_Inj[24,0] - P_Line[23,24,0] = 0
 P-Balance[25,0]: - P_Inj[25,0] - P_Line[5,25,0] + P_Line[25,26,0]
   + 0.2842 L_Line[25,26,0] = 0
 P-Balance[26,0]: - P_Inj[26,0] - P_Line[25,26,0] + P_Line[26,27,0]
   + 1.059 L_Line[26,27,0] = 0
 P-Balance[27,0]: - P_Inj[27,0] - P_Line[26,27,0] + P_Line[27,28,0]
   + 0.8042 L_Line[27,28,0] = 0
 P-Balance[28,0]: - P_Inj[28,0] - P_Line[27,28,0] + P_Line[28,29,0]
   + 0.5075 L_Line[28,29,0] = 0
 P-Balance[29,0]: - P_Inj[29,0] - P_Line[28,29,0] + P_Line[29,30,0]
   + 0.9744 L_Line[29,30,0] = 0
 P-Balance[30,0]: - P_Inj[30,0] - P_Line[29,30,0] + P_Line[30,31,0]
   + 0.3105 L_Line[30,31,0] = 0
 P-Balance[31,0]: - P_Inj[31,0] - P_Line[30,31,0] + P_Line[31,32,0]
   + 0.341 L_Line[31,32,0] = 0
 P-Balance[32,0]: - P_Inj[32,0] - P_Line[31,32,0] = 0
 Q-Balance[0,0]: - Q_Inj[0,0] + Q_Line[0,1,0] + 0.047 L_Line[0,1,0] = 0
 Q-Balance[1,0]: - Q_Inj[1,0] - Q_Line[0,1,0] + Q_Line[1,2,0]
   + Q_Line[1,18,0] + 0.2511 L_Line[1,2,0] + 0.1565 L_Line[1,18,0] = 0
 Q-Balance[2,0]: - Q_Inj[2,0] - Q_Line[1,2,0] + Q_Line[2,3,0]
   + Q_Line[2,22,0] + 0.1864 L_Line[2,3,0] + 0.3083 L_Line[2,22,0] = 0
 Q-Balance[3,0]: - Q_Inj[3,0] - Q_Line[2,3,0] + Q_Line[3,4,0]
   + 0.1941 L_Line[3,4,0] = 0
 Q-Balance[4,0]: - Q_Inj[4,0] - Q_Line[3,4,0] + Q_Line[4,5,0]
   + 0.707 L_Line[4,5,0] = 0
 Q-Balance[5,0]: - Q_Inj[5,0] - Q_Line[4,5,0] + Q_Line[5,6,0]
   + Q_Line[5,25,0] + 0.6188 L_Line[5,6,0] + 0.1034 L_Line[5,25,0] = 0
 Q-Balance[6,0]: - Q_Inj[6,0] - Q_Line[5,6,0] + Q_Line[6,7,0]
   + 0.2351 L_Line[6,7,0] = 0
 Q-Balance[7,0]: - Q_Inj[7,0] - Q_Line[6,7,0] + Q_Line[7,8,0]
   + 0.74 L_Line[7,8,0] = 0
 Q-Balance[8,0]: - Q_Inj[8,0] - Q_Line[7,8,0] + Q_Line[8,9,0]
   + 0.74 L_Line[8,9,0] = 0
 Q-Balance[9,0]: - Q_Inj[9,0] - Q_Line[8,9,0] + Q_Line[9,10,0]
   + 0.065 L_Line[9,10,0] = 0
 Q-Balance[10,0]: - Q_Inj[10,0] - Q_Line[9,10,0] + Q_Line[10,11,0]
   + 0.1238 L_Line[10,11,0] = 0
 Q-Balance[11,0]: - Q_Inj[11,0] - Q_Line[10,11,0] + Q_Line[11,12,0]
   + 1.155 L_Line[11,12,0] = 0
 Q-Balance[12,0]: - Q_Inj[12,0] - Q_Line[11,12,0] + Q_Line[12,13,0]
   + 0.7129 L_Line[12,13,0] = 0
 Q-Balance[13,0]: - Q_Inj[13,0] - Q_Line[12,13,0] + Q_Line[13,14,0]
   + 0.526 L_Line[13,14,0] = 0
 Q-Balance[14,0]: - Q_Inj[14,0] - Q_Line[13,14,0] + Q_Line[14,15,0]
   + 0.545 L_Line[14,15,0] = 0
 Q-Balance[15,0]: - Q_Inj[15,0] - Q_Line[14,15,0] + Q_Line[15,16,0]
   + 1.721 L_Line[15,16,0] = 0
 Q-Balance[16,0]: - Q_Inj[16,0] - Q_Line[15,16,0] + Q_Line[16,17,0]
   + 0.574 L_Line[16,17,0] = 0
 Q-Balance[17,0]: - Q_Inj[17,0] - Q_Line[16,17,0] = 0
 Q-Balance[18,0]: - Q_Inj[18,0] - Q_Line[1,18,0] + Q_Line[18,19,0]
   + 1.3554 L_Line[18,19,0] = 0
 Q-Balance[19,0]: - Q_Inj[19,0] - Q_Line[18,19,0] + Q_Line[19,20,0]
   + 0.4784 L_Line[19,20,0] = 0
 Q-Balance[20,0]: - Q_Inj[20,0] - Q_Line[19,20,0] + Q_Line[20,21,0]
   + 0.9373 L_Line[20,21,0] = 0
 Q-Balance[21,0]: - Q_Inj[21,0] - Q_Line[20,21,0] = 0
 Q-Balance[22,0]: - Q_Inj[22,0] - Q_Line[2,22,0] + Q_Line[22,23,0]
   + 0.7091 L_Line[22,23,0] = 0
 Q-Balance[23,0]: - Q_Inj[23,0] - Q_Line[22,23,0] + Q_Line[23,24,0]
   + 0.7011 L_Line[23,24,0] = 0
 Q-Balance[24,0]: - Q_Inj[24,0] - Q_Line[23,24,0] = 0
 Q-Balance[25,0]: - Q_Inj[25,0] - Q_Line[5,25,0] + Q_Line[25,26,0]
   + 0.1447 L_Line[25,26,0] = 0
 Q-Balance[26,0]: - Q_Inj[26,0] - Q_Line[25,26,0] + Q_Line[26,27,0]
   + 0.9337 L_Line[26,27,0] = 0
 Q-Balance[27,0]: - Q_Inj[27,0] - Q_Line[26,27,0] + Q_Line[27,28,0]
   + 0.7006 L_Line[27,28,0] = 0
 Q-Balance[28,0]: - Q_Inj[28,0] - Q_Line[27,28,0] + Q_Line[28,29,0]
   + 0.2585 L_Line[28,29,0] = 0
 Q-Balance[29,0]: - Q_Inj[29,0] - Q_Line[28,29,0] + Q_Line[29,30,0]
   + 0.963 L_Line[29,30,0] = 0
 Q-Balance[30,0]: - Q_Inj[30,0] - Q_Line[29,30,0] + Q_Line[30,31,0]
   + 0.3619 L_Line[30,31,0] = 0
 Q-Balance[31,0]: - Q_Inj[31,0] - Q_Line[30,31,0] + Q_Line[31,32,0]
   + 0.5302 L_Line[31,32,0] = 0
 Q-Balance[32,0]: - Q_Inj[32,0] - Q_Line[31,32,0] = 0
 V-Slack[0,0]: V_squared[0,0] = 1
 Ohms-Law[0,1,0]: [ V_squared[1,0] * L_Line[0,1,0] - P_Line[0,1,0] ^2
   - Q_Line[0,1,0] ^2 ] = 0
 Ohms-Law[1,2,0]: [ V_squared[2,0] * L_Line[1,2,0] - P_Line[1,2,0] ^2
   - Q_Line[1,2,0] ^2 ] = 0
 Ohms-Law[2,3,0]: [ V_squared[3,0] * L_Line[2,3,0] - P_Line[2,3,0] ^2
   - Q_Line[2,3,0] ^2 ] = 0
 Ohms-Law[3,4,0]: [ V_squared[4,0] * L_Line[3,4,0] - P_Line[3,4,0] ^2
   - Q_Line[3,4,0] ^2 ] = 0
 Ohms-Law[4,5,0]: [ V_squared[5,0] * L_Line[4,5,0] - P_Line[4,5,0] ^2
   - Q_Line[4,5,0] ^2 ] = 0
 Ohms-Law[5,6,0]: [ V_squared[6,0] * L_Line[5,6,0] - P_Line[5,6,0] ^2
   - Q_Line[5,6,0] ^2 ] = 0
 Ohms-Law[6,7,0]: [ V_squared[7,0] * L_Line[6,7,0] - P_Line[6,7,0] ^2
   - Q_Line[6,7,0] ^2 ] = 0
 Ohms-Law[7,8,0]: [ V_squared[8,0] * L_Line[7,8,0] - P_Line[7,8,0] ^2
   - Q_Line[7,8,0] ^2 ] = 0
 Ohms-Law[8,9,0]: [ V_squared[9,0] * L_Line[8,9,0] - P_Line[8,9,0] ^2
   - Q_Line[8,9,0] ^2 ] = 0
 Ohms-Law[9,10,0]: [ V_squared[10,0] * L_Line[9,10,0] - P_Line[9,10,0] ^2
   - Q_Line[9,10,0] ^2 ] = 0
 Ohms-Law[10,11,0]: [ V_squared[11,0] * L_Line[10,11,0]
   - P_Line[10,11,0] ^2 - Q_Line[10,11,0] ^2 ] = 0
 Ohms-Law[11,12,0]: [ V_squared[12,0] * L_Line[11,12,0]
   - P_Line[11,12,0] ^2 - Q_Line[11,12,0] ^2 ] = 0
 Ohms-Law[12,13,0]: [ V_squared[13,0] * L_Line[12,13,0]
   - P_Line[12,13,0] ^2 - Q_Line[12,13,0] ^2 ] = 0
 Ohms-Law[13,14,0]: [ V_squared[14,0] * L_Line[13,14,0]
   - P_Line[13,14,0] ^2 - Q_Line[13,14,0] ^2 ] = 0
 Ohms-Law[14,15,0]: [ V_squared[15,0] * L_Line[14,15,0]
   - P_Line[14,15,0] ^2 - Q_Line[14,15,0] ^2 ] = 0
 Ohms-Law[15,16,0]: [ V_squared[16,0] * L_Line[15,16,0]
   - P_Line[15,16,0] ^2 - Q_Line[15,16,0] ^2 ] = 0
 Ohms-Law[16,17,0]: [ V_squared[17,0] * L_Line[16,17,0]
   - P_Line[16,17,0] ^2 - Q_Line[16,17,0] ^2 ] = 0
 Ohms-Law[1,18,0]: [ V_squared[18,0] * L_Line[1,18,0] - P_Line[1,18,0] ^2
   - Q_Line[1,18,0] ^2 ] = 0
 Ohms-Law[18,19,0]: [ V_squared[19,0] * L_Line[18,19,0]
   - P_Line[18,19,0] ^2 - Q_Line[18,19,0] ^2 ] = 0
 Ohms-Law[19,20,0]: [ V_squared[20,0] * L_Line[19,20,0]
   - P_Line[19,20,0] ^2 - Q_Line[19,20,0] ^2 ] = 0
 Ohms-Law[20,21,0]: [ V_squared[21,0] * L_Line[20,21,0]
   - P_Line[20,21,0] ^2 - Q_Line[20,21,0] ^2 ] = 0
 Ohms-Law[2,22,0]: [ V_squared[22,0] * L_Line[2,22,0] - P_Line[2,22,0] ^2
   - Q_Line[2,22,0] ^2 ] = 0
 Ohms-Law[22,23,0]: [ V_squared[23,0] * L_Line[22,23,0]
   - P_Line[22,23,0] ^2 - Q_Line[22,23,0] ^2 ] = 0
 Ohms-Law[23,24,0]: [ V_squared[24,0] * L_Line[23,24,0]
   - P_Line[23,24,0] ^2 - Q_Line[23,24,0] ^2 ] = 0
 Ohms-Law[5,25,0]: [ V_squared[25,0] * L_Line[5,25,0] - P_Line[5,25,0] ^2
   - Q_Line[5,25,0] ^2 ] = 0
 Ohms-Law[25,26,0]: [ V_squared[26,0] * L_Line[25,26,0]
   - P_Line[25,26,0] ^2 - Q_Line[25,26,0] ^2 ] = 0
 Ohms-Law[26,27,0]: [ V_squared[27,0] * L_Line[26,27,0]
   - P_Line[26,27,0] ^2 - Q_Line[26,27,0] ^2 ] = 0
 Ohms-Law[27,28,0]: [ V_squared[28,0] * L_Line[27,28,0]
   - P_Line[27,28,0] ^2 - Q_Line[27,28,0] ^2 ] = 0
 Ohms-Law[28,29,0]: [ V_squared[29,0] * L_Line[28,29,0]
   - P_Line[28,29,0] ^2 - Q_Line[28,29,0] ^2 ] = 0
 Ohms-Law[29,30,0]: [ V_squared[30,0] * L_Line[29,30,0]
   - P_Line[29,30,0] ^2 - Q_Line[29,30,0] ^2 ] = 0
 Ohms-Law[30,31,0]: [ V_squared[31,0] * L_Line[30,31,0]
   - P_Line[30,31,0] ^2 - Q_Line[30,31,0] ^2 ] = 0
 Ohms-Law[31,32,0]: [ V_squared[32,0] * L_Line[31,32,0]
   - P_Line[31,32,0] ^2 - Q_Line[31,32,0] ^2 ] = 0
Bounds
 0.9025 <= V_squared[0,0] <= 1.1025
 0.9025 <= V_squared[1,0] <= 1.1025
 0.9025 <= V_squared[2,0] <= 1.1025
 0.9025 <= V_squared[3,0] <= 1.1025
 0.9025 <= V_squared[4,0] <= 1.1025
 0.9025 <= V_squared[5,0] <= 1.1025
 0.9025 <= V_squared[6,0] <= 1.1025
 0.9025 <= V_squared[7,0] <= 1.1025
 0.9025 <= V_squared[8,0] <= 1.1025
 0.9025 <= V_squared[9,0] <= 1.1025
 0.9025 <= V_squared[10,0] <= 1.1025
 0.9025 <= V_squared[11,0] <= 1.1025
 0.9025 <= V_squared[12,0] <= 1.1025
 0.9025 <= V_squared[13,0] <= 1.1025
 0.9025 <= V_squared[14,0] <= 1.1025
 0.9025 <= V_squared[15,0] <= 1.1025
 0.9025 <= V_squared[16,0] <= 1.1025
 0.9025 <= V_squared[17,0] <= 1.1025
 0.9025 <= V_squared[18,0] <= 1.1025
 0.9025 <= V_squared[19,0] <= 1.1025
 0.9025 <= V_squared[20,0] <= 1.1025
 0.9025 <= V_squared[21,0] <= 1.1025
 0.9025 <= V_squared[22,0] <= 1.1025
 0.9025 <= V_squared[23,0] <= 1.1025
 0.9025 <= V_squared[24,0] <= 1.1025
 0.9025 <= V_squared[25,0] <= 1.1025
 0.9025 <= V_squared[26,0] <= 1.1025
 0.9025 <= V_squared[27,0] <= 1.1025
 0.9025 <= V_squared[28,0] <= 1.1025
 0.9025 <= V_squared[29,0] <= 1.1025
 0.9025 <= V_squared[30,0] <= 1.1025
 0.9025 <= V_squared[31,0] <= 1.1025
 0.9025 <= V_squared[32,0] <= 1.1025
 P_Gen[0,0] <= 10
 -10 <= Q_Gen[0,0] <= 10
 P_Inj[0,0] free
 P_Inj[1,0] free
 P_Inj[2,0] free
 P_Inj[3,0] free
 P_Inj[4,0] free
 P_Inj[5,0] free
 P_Inj[6,0] free
 P_Inj[7,0] free
 P_Inj[8,0] free
 P_Inj[9,0] free
 P_Inj[10,0] free
 P_Inj[11,0] free
 P_Inj[12,0] free
 P_Inj[13,0] free
 P_Inj[14,0] free
 P_Inj[15,0] free
 P_Inj[16,0] free
 P_Inj[17,0] free
 P_Inj[18,0] free
 P_Inj[19,0] free
 P_Inj[20,0] free
 P_Inj[21,0] free
 P_Inj[22,0] free
 P_Inj[23,0] free
 P_Inj[24,0] free
 P_Inj[25,0] free
 P_Inj[26,0] free
 P_Inj[27,0] free
 P_Inj[28,0] free
 P_Inj[29,0] free
 P_Inj[30,0] free
 P_Inj[31,0] free
 P_Inj[32,0] free
 Q_Inj[0,0] free
 Q_Inj[1,0] free
 Q_Inj[2,0] free
 Q_Inj[3,0] free
 Q_Inj[4,0] free
 Q_Inj[5,0] free
 Q_Inj[6,0] free
 Q_Inj[7,0] free
 Q_Inj[8,0] free
 Q_Inj[9,0] free
 Q_Inj[10,0] free
 Q_Inj[11,0] free
 Q_Inj[12,0] free
 Q_Inj[13,0] free
 Q_Inj[14,0] free
 Q_Inj[15,0] free
 Q_Inj[16,0] free
 Q_Inj[17,0] free
 Q_Inj[18,0] free
 Q_Inj[19,0] free
 Q_Inj[20,0] free
 Q_Inj[21,0] free
 Q_Inj[22,0] free
 Q_Inj[23,0] free
 Q_Inj[24,0] free
 Q_Inj[25,0] free
 Q_Inj[26,0] free
 Q_Inj[27,0] free
 Q_Inj[28,0] free
 Q_Inj[29,0] free
 Q_Inj[30,0] free
 Q_Inj[31,0] free
 Q_Inj[32,0] free
 -5 <= P_Line[0,1,0] <= 5
 -5 <= P_Line[1,2,0] <= 5
 -5 <= P_Line[2,3,0] <= 5
 -5 <= P_Line[3,4,0] <= 5
 -5 <= P_Line[4,5,0] <= 5
 -5 <= P_Line[5,6,0] <= 5
 -5 <= P_Line[6,7,0] <= 5
 -5 <= P_Line[7,8,0] <= 5
 -5 <= P_Line[8,9,0] <= 5
 -5 <= P_Line[9,10,0] <= 5
 -5 <= P_Line[10,11,0] <= 5
 -5 <= P_Line[11,12,0] <= 5
 -5 <= P_Line[12,13,0] <= 5
 -5 <= P_Line[13,14,0] <= 5
 -5 <= P_Line[14,15,0] <= 5
 -5 <= P_Line[15,16,0] <= 5
 -5 <= P_Line[16,17,0] <= 5
 -5 <= P_Line[1,18,0] <= 5
 -5 <= P_Line[18,19,0] <= 5
 -5 <= P_Line[19,20,0] <= 5
 -5 <= P_Line[20,21,0] <= 5
 -5 <= P_Line[2,22,0] <= 5
 -5 <= P_Line[22,23,0] <= 5
 -5 <= P_Line[23,24,0] <= 5
 -5 <= P_Line[5,25,0] <= 5
 -5 <= P_Line[25,26,0] <= 5
 -5 <= P_Line[26,27,0] <= 5
 -5 <= P_Line[27,28,0] <= 5
 -5 <= P_Line[28,29,0] <= 5
 -5 <= P_Line[29,30,0] <= 5
 -5 <= P_Line[30,31,0] <= 5
 -5 <= P_Line[31,32,0] <= 5
 Q_Line[0,1,0] free
 Q_Line[1,2,0] free
 Q_Line[2,3,0] free
 Q_Line[3,4,0] free
 Q_Line[4,5,0] free
 Q_Line[5,6,0] free
 Q_Line[6,7,0] free
 Q_Line[7,8,0] free
 Q_Line[8,9,0] free
 Q_Line[9,10,0] free
 Q_Line[10,11,0] free
 Q_Line[11,12,0] free
 Q_Line[12,13,0] free
 Q_Line[13,14,0] free
 Q_Line[14,15,0] free
 Q_Line[15,16,0] free
 Q_Line[16,17,0] free
 Q_Line[1,18,0] free
 Q_Line[18,19,0] free
 Q_Line[19,20,0] free
 Q_Line[20,21,0] free
 Q_Line[2,22,0] free
 Q_Line[22,23,0] free
 Q_Line[23,24,0] free
 Q_Line[5,25,0] free
 Q_Line[25,26,0] free
 Q_Line[26,27,0] free
 Q_Line[27,28,0] free
 Q_Line[28,29,0] free
 Q_Line[29,30,0] free
 Q_Line[30,31,0] free
 Q_Line[31,32,0] free
 P_hat[0,1,0] <= 5
 P_hat[1,0,0] <= 5
 P_hat[1,2,0] <= 5
 P_hat[2,1,0] <= 5
 P_hat[2,3,0] <= 5
 P_hat[3,2,0] <= 5
 P_hat[3,4,0] <= 5
 P_hat[4,3,0] <= 5
 P_hat[4,5,0] <= 5
 P_hat[5,4,0] <= 5
 P_hat[5,6,0] <= 5
 P_hat[6,5,0] <= 5
 P_hat[6,7,0] <= 5
 P_hat[7,6,0] <= 5
 P_hat[7,8,0] <= 5
 P_hat[8,7,0] <= 5
 P_hat[8,9,0] <= 5
 P_hat[9,8,0] <= 5
 P_hat[9,10,0] <= 5
 P_hat[10,9,0] <= 5
 P_hat[10,11,0] <= 5
 P_hat[11,10,0] <= 5
 P_hat[11,12,0] <= 5
 P_hat[12,11,0] <= 5
 P_hat[12,13,0] <= 5
 P_hat[13,12,0] <= 5
 P_hat[13,14,0] <= 5
 P_hat[14,13,0] <= 5
 P_hat[14,15,0] <= 5
 P_hat[15,14,0] <= 5
 P_hat[15,16,0] <= 5
 P_hat[16,15,0] <= 5
 P_hat[16,17,0] <= 5
 P_hat[17,16,0] <= 5
 P_hat[1,18,0] <= 5
 P_hat[18,1,0] <= 5
 P_hat[18,19,0] <= 5
 P_hat[19,18,0] <= 5
 P_hat[19,20,0] <= 5
 P_hat[20,19,0] <= 5
 P_hat[20,21,0] <= 5
 P_hat[21,20,0] <= 5
 P_hat[2,22,0] <= 5
 P_hat[22,2,0] <= 5
 P_hat[22,23,0] <= 5
 P_hat[23,22,0] <= 5
 P_hat[23,24,0] <= 5
 P_hat[24,23,0] <= 5
 P_hat[5,25,0] <= 5
 P_hat[25,5,0] <= 5
 P_hat[25,26,0] <= 5
 P_hat[26,25,0] <= 5
 P_hat[26,27,0] <= 5
 P_hat[27,26,0] <= 5
 P_hat[27,28,0] <= 5
 P_hat[28,27,0] <= 5
 P_hat[28,29,0] <= 5
 P_hat[29,28,0] <= 5
 P_hat[29,30,0] <= 5
 P_hat[30,29,0] <= 5
 P_hat[30,31,0] <= 5
 P_hat[31,30,0] <= 5
 P_hat[31,32,0] <= 5
 P_hat[32,31,0] <= 5
End
