\ Model bus69_convex
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  L_Line[0,1,0] + L_Line[1,2,0] + L_Line[2,3,0] + L_Line[3,4,0]
   + L_Line[4,5,0] + L_Line[5,6,0] + L_Line[6,7,0] + L_Line[7,8,0]
   + L_Line[8,9,0] + L_Line[9,10,0] + L_Line[10,11,0] + L_Line[11,12,0]
   + L_Line[12,13,0] + L_Line[13,14,0] + L_Line[14,15,0] + L_Line[15,16,0]
   + L_Line[16,17,0] + L_Line[17,18,0] + L_Line[18,19,0] + L_Line[19,20,0]
   + L_Line[20,21,0] + L_Line[21,22,0] + L_Line[22,23,0] + L_Line[23,24,0]
   + L_Line[24,25,0] + L_Line[25,26,0] + L_Line[2,27,0] + L_Line[27,28,0]
   + L_Line[28,29,0] + L_Line[29,30,0] + L_Line[30,31,0] + L_Line[31,32,0]
   + L_Line[32,33,0] + L_Line[33,34,0] + L_Line[2,35,0] + L_Line[35,36,0]
   + L_Line[36,37,0] + L_Line[37,38,0] + L_Line[38,39,0] + L_Line[39,40,0]
   + L_Line[40,41,0] + L_Line[41,42,0] + L_Line[42,43,0] + L_Line[43,44,0]
   + L_Line[44,45,0] + L_Line[3,46,0] + L_Line[46,47,0] + L_Line[47,48,0]
   + L_Line[48,49,0] + L_Line[7,50,0] + L_Line[50,51,0] + L_Line[8,52,0]
   + L_Line[52,53,0] + L_Line[53,54,0] + L_Line[54,55,0] + L_Line[55,56,0]
   + L_Line[56,57,0] + L_Line[57,58,0] + L_Line[58,59,0] + L_Line[59,60,0]
   + L_Line[60,61,0] + L_Line[61,62,0] + L_Line[62,63,0] + L_Line[63,64,0]
   + L_Line[10,65,0] + L_Line[65,66,0] + L_Line[11,67,0] + L_Line[67,68,0]
Subject To
 P-Balance[0,0]: - P_Inj[0,0] + P_Line[0,1,0] + 3.11963e-06 L_Line[0,1,0]
   = 0
 P-Balance[1,0]: - P_Inj[1,0] - P_Line[0,1,0] + P_Line[1,2,0]
   + 3.11963e-06 L_Line[1,2,0] = 0
 P-Balance[2,0]: - P_Inj[2,0] - P_Line[1,2,0] + P_Line[2,3,0]
   + P_Line[2,27,0] + P_Line[2,35,0] + 9.35888e-06 L_Line[2,3,0]
   + 2.74527e-05 L_Line[2,27,0] + 2.74527e-05 L_Line[2,35,0] = 0
 P-Balance[3,0]: - P_Inj[3,0] - P_Line[2,3,0] + P_Line[3,4,0]
   + P_Line[3,46,0] + 1.5660524746124801e-04 L_Line[3,4,0]
   + 2.12135e-05 L_Line[3,46,0] = 0
 P-Balance[4,0]: - P_Inj[4,0] - P_Line[3,4,0] + P_Line[4,5,0]
   + 0.00228356655660625 L_Line[4,5,0] = 0
 P-Balance[5,0]: - P_Inj[5,0] - P_Line[4,5,0] + P_Line[5,6,0]
   + 0.00237715534990978 L_Line[5,6,0] = 0
 P-Balance[6,0]: - P_Inj[6,0] - P_Line[5,6,0] + P_Line[6,7,0]
   + 5.7525911617239305e-04 L_Line[6,7,0] = 0
 P-Balance[7,0]: - P_Inj[7,0] - P_Line[6,7,0] + P_Line[7,8,0]
   + P_Line[7,50,0] + 3.075951673242839e-04 L_Line[7,8,0]
   + 5.7900266790453442e-04 L_Line[7,50,0] = 0
 P-Balance[8,0]: - P_Inj[8,0] - P_Line[7,8,0] + P_Line[8,9,0]
   + P_Line[8,52,0] + 0.00510994811437299 L_Line[8,9,0]
   + 0.00108563 L_Line[8,52,0] = 0
 P-Balance[9,0]: - P_Inj[9,0] - P_Line[8,9,0] + P_Line[9,10,0]
   + 0.00116798814042811 L_Line[9,10,0] = 0
 P-Balance[10,0]: - P_Inj[10,0] - P_Line[9,10,0] + P_Line[10,11,0]
   + P_Line[10,65,0] + 0.0044386045037423 L_Line[10,11,0]
   + 0.00125533768084474 L_Line[10,65,0] = 0
 P-Balance[11,0]: - P_Inj[11,0] - P_Line[10,11,0] + P_Line[11,12,0]
   + P_Line[11,67,0] + 0.00642643047350938 L_Line[11,12,0]
   + 0.00461330358457557 L_Line[11,67,0] = 0
 P-Balance[12,0]: - P_Inj[12,0] - P_Line[11,12,0] + P_Line[12,13,0]
   + 0.00651378 L_Line[12,13,0] = 0
 P-Balance[13,0]: - P_Inj[13,0] - P_Line[12,13,0] + P_Line[13,14,0]
   + 0.00660112955434265 L_Line[13,14,0] = 0
 P-Balance[14,0]: - P_Inj[14,0] - P_Line[13,14,0] + P_Line[14,15,0]
   + 0.00122663711756499 L_Line[14,15,0] = 0
 P-Balance[15,0]: - P_Inj[15,0] - P_Line[14,15,0] + P_Line[15,16,0]
   + 0.00233597628085623 L_Line[15,16,0] = 0
 P-Balance[16,0]: - P_Inj[16,0] - P_Line[15,16,0] + P_Line[16,17,0]
   + 2.93245e-05 L_Line[16,17,0] = 0
 P-Balance[17,0]: - P_Inj[17,0] - P_Line[16,17,0] + P_Line[17,18,0]
   + 0.0020439792457492 L_Line[17,18,0] = 0
 P-Balance[18,0]: - P_Inj[18,0] - P_Line[17,18,0] + P_Line[18,19,0]
   + 0.00131398665798163 L_Line[18,19,0] = 0
 P-Balance[19,0]: - P_Inj[19,0] - P_Line[18,19,0] + P_Line[19,20,0]
   + 0.00213132878616583 L_Line[19,20,0] = 0
 P-Balance[20,0]: - P_Inj[20,0] - P_Line[19,20,0] + P_Line[20,21,0]
   + 8.73495e-05 L_Line[20,21,0] = 0
 P-Balance[21,0]: - P_Inj[21,0] - P_Line[20,21,0] + P_Line[21,22,0]
   + 9.9266513430615756e-04 L_Line[21,22,0] = 0
 P-Balance[22,0]: - P_Inj[22,0] - P_Line[21,22,0] + P_Line[22,23,0]
   + 0.00216065327473427 L_Line[22,23,0] = 0
 P-Balance[23,0]: - P_Inj[23,0] - P_Line[22,23,0] + P_Line[23,24,0]
   + 0.00467195256171245 L_Line[23,24,0] = 0
 P-Balance[24,0]: - P_Inj[24,0] - P_Line[23,24,0] + P_Line[24,25,0]
   + 0.00192730521676412 L_Line[24,25,0] = 0
 P-Balance[25,0]: - P_Inj[25,0] - P_Line[24,25,0] + P_Line[25,26,0]
   + 0.00108063860001148 L_Line[25,26,0] = 0
 P-Balance[26,0]: - P_Inj[26,0] - P_Line[25,26,0] = 0
 P-Balance[27,0]: - P_Inj[27,0] - P_Line[2,27,0] + P_Line[27,28,0]
   + 3.9931218476174792e-04 L_Line[27,28,0] = 0
 P-Balance[28,0]: - P_Inj[28,0] - P_Line[27,28,0] + P_Line[28,29,0]
   + 0.00248197479840974 L_Line[28,29,0] = 0
 P-Balance[29,0]: - P_Inj[29,0] - P_Line[28,29,0] + P_Line[29,30,0]
   + 4.3799555266054222e-04 L_Line[29,30,0] = 0
 P-Balance[30,0]: - P_Inj[30,0] - P_Line[29,30,0] + P_Line[30,31,0]
   + 0.00218997776330271 L_Line[30,31,0] = 0
 P-Balance[31,0]: - P_Inj[31,0] - P_Line[30,31,0] + P_Line[31,32,0]
   + 0.00523473317211104 L_Line[31,32,0] = 0
 P-Balance[32,0]: - P_Inj[32,0] - P_Line[31,32,0] + P_Line[32,33,0]
   + 0.0106566439308291 L_Line[32,33,0] = 0
 P-Balance[33,0]: - P_Inj[33,0] - P_Line[32,33,0] + P_Line[33,34,0]
   + 0.00919665875529401 L_Line[33,34,0] = 0
 P-Balance[34,0]: - P_Inj[34,0] - P_Line[33,34,0] = 0
 P-Balance[35,0]: - P_Inj[35,0] - P_Line[2,35,0] + P_Line[35,36,0]
   + 3.9931218476174792e-04 L_Line[35,36,0] = 0
 P-Balance[36,0]: - P_Inj[36,0] - P_Line[35,36,0] + P_Line[36,37,0]
   + 6.5699332899081335e-04 L_Line[36,37,0] = 0
 P-Balance[37,0]: - P_Inj[37,0] - P_Line[36,37,0] + P_Line[37,38,0]
   + 1.8967328776183025e-04 L_Line[37,38,0] = 0
 P-Balance[38,0]: - P_Inj[38,0] - P_Line[37,38,0] + P_Line[38,39,0]
   + 1.12307e-05 L_Line[38,39,0] = 0
 P-Balance[39,0]: - P_Inj[39,0] - P_Line[38,39,0] + P_Line[39,40,0]
   + 0.00454404787753095 L_Line[39,40,0] = 0
 P-Balance[40,0]: - P_Inj[40,0] - P_Line[39,40,0] + P_Line[40,41,0]
   + 0.00193416839493972 L_Line[40,41,0] = 0
 P-Balance[41,0]: - P_Inj[41,0] - P_Line[40,41,0] + P_Line[41,42,0]
   + 2.5580936836299475e-04 L_Line[41,42,0] = 0
 P-Balance[42,0]: - P_Inj[42,0] - P_Line[41,42,0] + P_Line[42,43,0]
   + 5.74011e-05 L_Line[42,43,0] = 0
 P-Balance[43,0]: - P_Inj[43,0] - P_Line[42,43,0] + P_Line[43,44,0]
   + 6.7945463938366159e-04 L_Line[43,44,0] = 0
 P-Balance[44,0]: - P_Inj[44,0] - P_Line[43,44,0] + P_Line[44,45,0]
   + 5.61533e-06 L_Line[44,45,0] = 0
 P-Balance[45,0]: - P_Inj[45,0] - P_Line[44,45,0] = 0
 P-Balance[46,0]: - P_Inj[46,0] - P_Line[3,46,0] + P_Line[46,47,0]
   + 5.3096042067538659e-04 L_Line[46,47,0] = 0
 P-Balance[47,0]: - P_Inj[47,0] - P_Line[46,47,0] + P_Line[47,48,0]
   + 0.00180813548662429 L_Line[47,48,0] = 0
 P-Balance[48,0]: - P_Inj[48,0] - P_Line[47,48,0] + P_Line[48,49,0]
   + 5.1286658730336995e-04 L_Line[48,49,0] = 0
 P-Balance[49,0]: - P_Inj[49,0] - P_Line[48,49,0] = 0
 P-Balance[50,0]: - P_Inj[50,0] - P_Line[7,50,0] + P_Line[50,51,0]
   + 0.00206519270556466 L_Line[50,51,0] = 0
 P-Balance[51,0]: - P_Inj[51,0] - P_Line[50,51,0] = 0
 P-Balance[52,0]: - P_Inj[52,0] - P_Line[8,52,0] + P_Line[52,53,0]
   + 0.00126656833604117 L_Line[52,53,0] = 0
 P-Balance[53,0]: - P_Inj[53,0] - P_Line[52,53,0] + P_Line[53,54,0]
   + 0.00177319567045764 L_Line[53,54,0] = 0
 P-Balance[54,0]: - P_Inj[54,0] - P_Line[53,54,0] + P_Line[54,55,0]
   + 0.00175510183708562 L_Line[54,55,0] = 0
 P-Balance[55,0]: - P_Inj[55,0] - P_Line[54,55,0] + P_Line[55,56,0]
   + 0.00992041209017467 L_Line[55,56,0] = 0
 P-Balance[56,0]: - P_Inj[56,0] - P_Line[55,56,0] + P_Line[56,57,0]
   + 0.00488970248746534 L_Line[56,57,0] = 0
 P-Balance[57,0]: - P_Inj[57,0] - P_Line[56,57,0] + P_Line[57,58,0]
   + 0.00189798072819568 L_Line[57,58,0] = 0
 P-Balance[58,0]: - P_Inj[58,0] - P_Line[57,58,0] + P_Line[58,59,0]
   + 0.00240897553963298 L_Line[58,59,0] = 0
 P-Balance[59,0]: - P_Inj[59,0] - P_Line[58,59,0] + P_Line[59,60,0]
   + 0.00316642084010292 L_Line[59,60,0] = 0
 P-Balance[60,0]: - P_Inj[60,0] - P_Line[59,60,0] + P_Line[60,61,0]
   + 6.0770323118428506e-04 L_Line[60,61,0] = 0
 P-Balance[61,0]: - P_Inj[61,0] - P_Line[60,61,0] + P_Line[61,62,0]
   + 9.0469166860083496e-04 L_Line[61,62,0] = 0
 P-Balance[62,0]: - P_Inj[62,0] - P_Line[61,62,0] + P_Line[62,63,0]
   + 0.00443298917614409 L_Line[62,63,0] = 0
 P-Balance[63,0]: - P_Inj[63,0] - P_Line[62,63,0] + P_Line[63,64,0]
   + 0.0064950622552653 L_Line[63,64,0] = 0
 P-Balance[64,0]: - P_Inj[64,0] - P_Line[63,64,0] = 0
 P-Balance[65,0]: - P_Inj[65,0] - P_Line[10,65,0] + P_Line[65,66,0]
   + 2.93245e-05 L_Line[65,66,0] = 0
 P-Balance[66,0]: - P_Inj[66,0] - P_Line[65,66,0] = 0
 P-Balance[67,0]: - P_Inj[67,0] - P_Line[11,67,0] + P_Line[67,68,0]
   + 2.93245e-05 L_Line[67,68,0] = 0
 P-Balance[68,0]: - P_Inj[68,0] - P_Line[67,68,0] = 0
 Q-Balance[0,0]: - Q_Inj[0,0] + Q_Line[0,1,0] + 7.4871e-06 L_Line[0,1,0]
   = 0
 Q-Balance[1,0]: - Q_Inj[1,0] - Q_Line[0,1,0] + Q_Line[1,2,0]
   + 7.4871e-06 L_Line[1,2,0] = 0
 Q-Balance[2,0]: - Q_Inj[2,0] - Q_Line[1,2,0] + Q_Line[2,3,0]
   + Q_Line[2,27,0] + Q_Line[2,35,0] + 2.24613e-05 L_Line[2,3,0]
   + 6.73839e-05 L_Line[2,27,0] + 6.73839e-05 L_Line[2,35,0] = 0
 Q-Balance[3,0]: - Q_Inj[3,0] - Q_Line[2,3,0] + Q_Line[3,4,0]
   + Q_Line[3,46,0] + 0.000183434 L_Line[3,4,0]
   + 5.24097e-05 L_Line[3,46,0] = 0
 Q-Balance[4,0]: - Q_Inj[4,0] - Q_Line[3,4,0] + Q_Line[4,5,0]
   + 0.00116299673811859 L_Line[4,5,0] = 0
 Q-Balance[5,0]: - Q_Inj[5,0] - Q_Line[4,5,0] + Q_Line[5,6,0]
   + 0.00121103898534774 L_Line[5,6,0] = 0
 Q-Balance[6,0]: - Q_Inj[6,0] - Q_Line[5,6,0] + Q_Line[6,7,0]
   + 2.9324488568440859e-04 L_Line[6,7,0] = 0
 Q-Balance[7,0]: - Q_Inj[7,0] - Q_Line[6,7,0] + Q_Line[7,8,0]
   + Q_Line[7,50,0] + 1.5660524746124801e-04 L_Line[7,8,0]
   + 2.9511666155047933e-04 L_Line[7,50,0] = 0
 Q-Balance[8,0]: - Q_Inj[8,0] - Q_Line[7,8,0] + Q_Line[8,9,0]
   + Q_Line[8,52,0] + 0.00168896575648446 L_Line[8,9,0]
   + 5.527978057795447e-04 L_Line[8,52,0] = 0
 Q-Balance[9,0]: - Q_Inj[9,0] - Q_Line[8,9,0] + Q_Line[9,10,0]
   + 3.8620975369925301e-04 L_Line[9,10,0] = 0
 Q-Balance[10,0]: - Q_Inj[10,0] - Q_Line[9,10,0] + Q_Line[10,11,0]
   + Q_Line[10,65,0] + 0.00146684835371073 L_Line[10,11,0]
   + 3.8121835138973123e-04 L_Line[10,65,0] = 0
 Q-Balance[11,0]: - Q_Inj[11,0] - Q_Line[10,11,0] + Q_Line[11,12,0]
   + Q_Line[11,67,0] + 0.00212134598154679 L_Line[11,12,0]
   + 0.00152487340555892 L_Line[11,67,0] = 0
 Q-Balance[12,0]: - Q_Inj[12,0] - Q_Line[11,12,0] + Q_Line[12,13,0]
   + 0.0021525422459813 L_Line[12,13,0] = 0
 Q-Balance[13,0]: - Q_Inj[13,0] - Q_Line[12,13,0] + Q_Line[13,14,0]
   + 0.00218124280926105 L_Line[13,14,0] = 0
 Q-Balance[14,0]: - Q_Inj[14,0] - Q_Line[13,14,0] + Q_Line[14,15,0]
   + 4.0555143764865021e-04 L_Line[14,15,0] = 0
 Q-Balance[15,0]: - Q_Inj[15,0] - Q_Line[14,15,0] + Q_Line[15,16,0]
   + 7.7241950739850602e-04 L_Line[15,16,0] = 0
 Q-Balance[16,0]: - Q_Inj[16,0] - Q_Line[15,16,0] + Q_Line[16,17,0]
   + 9.9828e-06 L_Line[16,17,0] = 0
 Q-Balance[17,0]: - Q_Inj[17,0] - Q_Line[16,17,0] + Q_Line[17,18,0]
   + 0.000675711 L_Line[17,18,0] = 0
 Q-Balance[18,0]: - Q_Inj[18,0] - Q_Line[17,18,0] + Q_Line[18,19,0]
   + 4.3050844919625947e-04 L_Line[18,19,0] = 0
 Q-Balance[19,0]: - Q_Inj[19,0] - Q_Line[18,19,0] + Q_Line[19,20,0]
   + 7.0441165093127085e-04 L_Line[19,20,0] = 0
 Q-Balance[20,0]: - Q_Inj[20,0] - Q_Line[19,20,0] + Q_Line[20,21,0]
   + 2.87006e-05 L_Line[20,21,0] = 0
 Q-Balance[21,0]: - Q_Inj[21,0] - Q_Line[20,21,0] + Q_Line[21,22,0]
   + 3.2818470185106156e-04 L_Line[21,22,0] = 0
 Q-Balance[22,0]: - Q_Inj[22,0] - Q_Line[21,22,0] + Q_Line[22,23,0]
   + 7.1439445555031462e-04 L_Line[22,23,0] = 0
 Q-Balance[23,0]: - Q_Inj[23,0] - Q_Line[22,23,0] + Q_Line[23,24,0]
   + 0.00154421508950832 L_Line[23,24,0] = 0
 Q-Balance[24,0]: - Q_Inj[24,0] - Q_Line[23,24,0] + Q_Line[24,25,0]
   + 6.3702771975272592e-04 L_Line[24,25,0] = 0
 Q-Balance[25,0]: - Q_Inj[25,0] - Q_Line[24,25,0] + Q_Line[25,26,0]
   + 3.568852651308122e-04 L_Line[25,26,0] = 0
 Q-Balance[26,0]: - Q_Inj[26,0] - Q_Line[25,26,0] = 0
 Q-Balance[27,0]: - Q_Inj[27,0] - Q_Line[2,27,0] + Q_Line[27,28,0]
   + 0.000976443 L_Line[27,28,0] = 0
 Q-Balance[28,0]: - Q_Inj[28,0] - Q_Line[27,28,0] + Q_Line[28,29,0]
   + 8.2046175462765396e-04 L_Line[28,29,0] = 0
 Q-Balance[29,0]: - Q_Inj[29,0] - Q_Line[28,29,0] + Q_Line[29,30,0]
   + 1.4475066697613361e-04 L_Line[29,30,0] = 0
 Q-Balance[30,0]: - Q_Inj[30,0] - Q_Line[29,30,0] + Q_Line[30,31,0]
   + 7.2375333488066806e-04 L_Line[30,31,0] = 0
 Q-Balance[31,0]: - Q_Inj[31,0] - Q_Line[30,31,0] + Q_Line[31,32,0]
   + 0.00175697361295169 L_Line[31,32,0] = 0
 Q-Balance[32,0]: - Q_Inj[32,0] - Q_Line[31,32,0] + Q_Line[32,33,0]
   + 0.00352268217994504 L_Line[32,33,0] = 0
 Q-Balance[33,0]: - Q_Inj[33,0] - Q_Line[32,33,0] + Q_Line[33,34,0]
   + 0.0030403879317875 L_Line[33,34,0] = 0
 Q-Balance[34,0]: - Q_Inj[34,0] - Q_Line[33,34,0] = 0
 Q-Balance[35,0]: - Q_Inj[35,0] - Q_Line[2,35,0] + Q_Line[35,36,0]
   + 0.000976443 L_Line[35,36,0] = 0
 Q-Balance[36,0]: - Q_Inj[36,0] - Q_Line[35,36,0] + Q_Line[36,37,0]
   + 7.6742810508898418e-04 L_Line[36,37,0] = 0
 Q-Balance[37,0]: - Q_Inj[37,0] - Q_Line[36,37,0] + Q_Line[37,38,0]
   + 2.2149347748503202e-04 L_Line[37,38,0] = 0
 Q-Balance[38,0]: - Q_Inj[38,0] - Q_Line[37,38,0] + Q_Line[38,39,0]
   + 1.31024e-05 L_Line[38,39,0] = 0
 Q-Balance[39,0]: - Q_Inj[39,0] - Q_Line[38,39,0] + Q_Line[39,40,0]
   + 0.00530898028146518 L_Line[39,40,0] = 0
 Q-Balance[40,0]: - Q_Inj[40,0] - Q_Line[39,40,0] + Q_Line[40,41,0]
   + 0.00226048132092471 L_Line[40,41,0] = 0
 Q-Balance[41,0]: - Q_Inj[41,0] - Q_Line[40,41,0] + Q_Line[41,42,0]
   + 2.9823628799393047e-04 L_Line[41,42,0] = 0
 Q-Balance[42,0]: - Q_Inj[42,0] - Q_Line[41,42,0] + Q_Line[42,43,0]
   + 7.23753e-05 L_Line[42,43,0] = 0
 Q-Balance[43,0]: - Q_Inj[43,0] - Q_Line[42,43,0] + Q_Line[43,44,0]
   + 8.5664942137168734e-04 L_Line[43,44,0] = 0
 Q-Balance[44,0]: - Q_Inj[44,0] - Q_Line[43,44,0] + Q_Line[44,45,0]
   + 7.4871e-06 L_Line[44,45,0] = 0
 Q-Balance[45,0]: - Q_Inj[45,0] - Q_Line[44,45,0] = 0
 Q-Balance[46,0]: - Q_Inj[46,0] - Q_Line[3,46,0] + Q_Line[46,47,0]
   + 0.00129963637634175 L_Line[46,47,0] = 0
 Q-Balance[47,0]: - Q_Inj[47,0] - Q_Line[46,47,0] + Q_Line[47,48,0]
   + 0.00442425422210243 L_Line[47,48,0] = 0
 Q-Balance[48,0]: - Q_Inj[48,0] - Q_Line[47,48,0] + Q_Line[48,49,0]
   + 0.00125471375555605 L_Line[48,49,0] = 0
 Q-Balance[49,0]: - Q_Inj[49,0] - Q_Line[48,49,0] = 0
 Q-Balance[50,0]: - Q_Inj[50,0] - Q_Line[7,50,0] + Q_Line[50,51,0]
   + 6.9505277160091742e-04 L_Line[50,51,0] = 0
 Q-Balance[51,0]: - Q_Inj[51,0] - Q_Line[50,51,0] = 0
 Q-Balance[52,0]: - Q_Inj[52,0] - Q_Line[8,52,0] + Q_Line[52,53,0]
   + 6.4513874850569901e-04 L_Line[52,53,0] = 0
 Q-Balance[53,0]: - Q_Inj[53,0] - Q_Line[52,53,0] + Q_Line[53,54,0]
   + 9.0281989273476439e-04 L_Line[53,54,0] = 0
 Q-Balance[54,0]: - Q_Inj[54,0] - Q_Line[53,54,0] + Q_Line[54,55,0]
   + 0.000894085 L_Line[54,55,0] = 0
 Q-Balance[55,0]: - Q_Inj[55,0] - Q_Line[54,55,0] + Q_Line[55,56,0]
   + 0.00332988926573976 L_Line[55,56,0] = 0
 Q-Balance[56,0]: - Q_Inj[56,0] - Q_Line[55,56,0] + Q_Line[56,57,0]
   + 0.00164092350925531 L_Line[56,57,0] = 0
 Q-Balance[57,0]: - Q_Inj[57,0] - Q_Line[56,57,0] + Q_Line[57,58,0]
   + 6.2766884042237249e-04 L_Line[57,58,0] = 0
 Q-Balance[58,0]: - Q_Inj[58,0] - Q_Line[57,58,0] + Q_Line[58,59,0]
   + 7.312404383449508e-04 L_Line[58,59,0] = 0
 Q-Balance[59,0]: - Q_Inj[59,0] - Q_Line[58,59,0] + Q_Line[59,60,0]
   + 0.00161284687126425 L_Line[59,60,0] = 0
 Q-Balance[60,0]: - Q_Inj[60,0] - Q_Line[59,60,0] + Q_Line[60,61,0]
   + 0.000309467 L_Line[60,61,0] = 0
 Q-Balance[61,0]: - Q_Inj[61,0] - Q_Line[60,61,0] + Q_Line[61,62,0]
   + 4.6045686305339057e-04 L_Line[61,62,0] = 0
 Q-Balance[62,0]: - Q_Inj[62,0] - Q_Line[61,62,0] + Q_Line[62,63,0]
   + 0.00225798561976995 L_Line[62,63,0] = 0
 Q-Balance[63,0]: - Q_Inj[63,0] - Q_Line[62,63,0] + Q_Line[63,64,0]
   + 0.00330805188063561 L_Line[63,64,0] = 0
 Q-Balance[64,0]: - Q_Inj[64,0] - Q_Line[63,64,0] = 0
 Q-Balance[65,0]: - Q_Inj[65,0] - Q_Line[10,65,0] + Q_Line[65,66,0]
   + 8.73495e-06 L_Line[65,66,0] = 0
 Q-Balance[66,0]: - Q_Inj[66,0] - Q_Line[65,66,0] = 0
 Q-Balance[67,0]: - Q_Inj[67,0] - Q_Line[11,67,0] + Q_Line[67,68,0]
   + 9.9828e-06 L_Line[67,68,0] = 0
 Q-Balance[68,0]: - Q_Inj[68,0] - Q_Line[67,68,0] = 0
 V-Drop[0,1,0]: V_squared[0,0] - V_squared[1,0] - 6.23925e-06 P_Line[0,1,0]
   - 1.49742e-05 Q_Line[0,1,0] + 6.57888e-11 L_Line[0,1,0] = 0
 V-Drop[1,2,0]: V_squared[1,0] - V_squared[2,0] - 6.23925e-06 P_Line[1,2,0]
   - 1.49742e-05 Q_Line[1,2,0] + 6.57888e-11 L_Line[1,2,0] = 0
 V-Drop[2,3,0]: V_squared[2,0] - V_squared[3,0] - 1.87178e-05 P_Line[2,3,0]
   - 4.49226e-05 Q_Line[2,3,0] + 5.92099e-10 L_Line[2,3,0] = 0
 V-Drop[3,4,0]: V_squared[3,0] - V_squared[4,0]
   - 3.1321049492249602e-04 P_Line[3,4,0] - 0.000366868 Q_Line[3,4,0]
   + 5.81732e-08 L_Line[3,4,0] = 0
 V-Drop[4,5,0]: V_squared[4,0] - V_squared[5,0]
   - 0.00456713311321249 P_Line[4,5,0] - 0.00232599347623718 Q_Line[4,5,0]
   + 6.56724e-06 L_Line[4,5,0] = 0
 V-Drop[5,6,0]: V_squared[5,0] - V_squared[6,0]
   - 0.00475431069981956 P_Line[5,6,0] - 0.00242207797069548 Q_Line[5,6,0]
   + 7.11748e-06 L_Line[5,6,0] = 0
 V-Drop[6,7,0]: V_squared[6,0] - V_squared[7,0]
   - 0.00115051823234479 P_Line[6,7,0]
   - 5.8648977136881717e-04 Q_Line[6,7,0] + 4.16916e-07 L_Line[6,7,0] = 0
 V-Drop[7,8,0]: V_squared[7,0] - V_squared[8,0]
   - 6.1519033464856781e-04 P_Line[7,8,0]
   - 3.1321049492249602e-04 Q_Line[7,8,0] + 1.1914e-07 L_Line[7,8,0] = 0
 V-Drop[8,9,0]: V_squared[8,0] - V_squared[9,0]
   - 0.010219896228746 P_Line[8,9,0] - 0.00337793151296891 Q_Line[8,9,0]
   + 2.89642e-05 L_Line[8,9,0] = 0
 V-Drop[9,10,0]: V_squared[9,0] - V_squared[10,0]
   - 0.00233597628085623 P_Line[9,10,0]
   - 7.7241950739850602e-04 Q_Line[9,10,0] + 1.51335e-06 L_Line[9,10,0]
   = 0
 V-Drop[10,11,0]: V_squared[10,0] - V_squared[11,0]
   - 0.00887720900748461 P_Line[10,11,0]
   - 0.00293369670742147 Q_Line[10,11,0] + 2.18529e-05 L_Line[10,11,0] = 0
 V-Drop[11,12,0]: V_squared[11,0] - V_squared[12,0]
   - 0.0128528609470188 P_Line[11,12,0]
   - 0.00424269196309357 Q_Line[11,12,0] + 4.57991e-05 L_Line[11,12,0] = 0
 V-Drop[12,13,0]: V_squared[12,0] - V_squared[13,0]
   - 0.013027560027852 P_Line[12,13,0]
   - 0.00430508449196259 Q_Line[12,13,0] + 4.70628e-05 L_Line[12,13,0] = 0
 V-Drop[13,14,0]: V_squared[13,0] - V_squared[14,0]
   - 0.0132022591086853 P_Line[13,14,0]
   - 0.0043624856185221 Q_Line[13,14,0] + 4.83327e-05 L_Line[13,14,0] = 0
 V-Drop[14,15,0]: V_squared[14,0] - V_squared[15,0]
   - 0.00245327423512999 P_Line[14,15,0]
   - 8.1110287529730042e-04 Q_Line[14,15,0] + 1.66911e-06 L_Line[14,15,0]
   = 0
 V-Drop[15,16,0]: V_squared[15,0] - V_squared[16,0]
   - 0.00467195256171245 P_Line[15,16,0]
   - 0.00154483901479701 Q_Line[15,16,0] + 6.05342e-06 L_Line[15,16,0] = 0
 V-Drop[16,17,0]: V_squared[16,0] - V_squared[17,0]
   - 5.8649e-05 P_Line[16,17,0] - 1.99656e-05 Q_Line[16,17,0]
   + 9.59582e-10 L_Line[16,17,0] = 0
 V-Drop[17,18,0]: V_squared[17,0] - V_squared[18,0]
   - 0.00408795849149839 P_Line[17,18,0]
   - 0.00135142217530304 Q_Line[17,18,0] + 4.63444e-06 L_Line[17,18,0] = 0
 V-Drop[18,19,0]: V_squared[18,0] - V_squared[19,0]
   - 0.00262797331596325 P_Line[18,19,0]
   - 8.6101689839251894e-04 Q_Line[18,19,0] + 1.9119e-06 L_Line[18,19,0]
   = 0
 V-Drop[19,20,0]: V_squared[19,0] - V_squared[20,0]
   - 0.00426265757233166 P_Line[19,20,0]
   - 0.00140882330186254 Q_Line[19,20,0] + 5.03876e-06 L_Line[19,20,0] = 0
 V-Drop[20,21,0]: V_squared[20,0] - V_squared[21,0]
   - 0.000174699 P_Line[20,21,0] - 5.74011e-05 Q_Line[20,21,0]
   + 8.45366e-09 L_Line[20,21,0] = 0
 V-Drop[21,22,0]: V_squared[21,0] - V_squared[22,0]
   - 0.00198533026861232 P_Line[21,22,0]
   - 6.5636940370212313e-04 Q_Line[21,22,0] + 1.09309e-06 L_Line[21,22,0]
   = 0
 V-Drop[22,23,0]: V_squared[22,0] - V_squared[23,0]
   - 0.00432130654946854 P_Line[22,23,0]
   - 0.00142878891110063 Q_Line[22,23,0] + 5.17878e-06 L_Line[22,23,0] = 0
 V-Drop[23,24,0]: V_squared[23,0] - V_squared[24,0]
   - 0.0093439051234249 P_Line[23,24,0]
   - 0.00308843017901664 Q_Line[23,24,0] + 2.42117e-05 L_Line[23,24,0] = 0
 V-Drop[24,25,0]: V_squared[24,0] - V_squared[25,0]
   - 0.00385461043352825 P_Line[24,25,0]
   - 0.00127405543950545 Q_Line[24,25,0] + 4.12031e-06 L_Line[24,25,0] = 0
 V-Drop[25,26,0]: V_squared[25,0] - V_squared[26,0]
   - 0.00216127720002296 P_Line[25,26,0]
   - 7.137705302616244e-04 Q_Line[25,26,0] + 1.29515e-06 L_Line[25,26,0]
   = 0
 V-Drop[2,27,0]: V_squared[2,0] - V_squared[27,0]
   - 5.49054e-05 P_Line[2,27,0] - 1.3476786235708992e-04 Q_Line[2,27,0]
   + 5.29425e-09 L_Line[2,27,0] = 0
 V-Drop[27,28,0]: V_squared[27,0] - V_squared[28,0]
   - 7.9862436952349584e-04 P_Line[27,28,0]
   - 0.00195288615360042 Q_Line[27,28,0] + 1.11289e-06 L_Line[27,28,0] = 0
 V-Drop[28,29,0]: V_squared[28,0] - V_squared[29,0]
   - 0.00496394959681948 P_Line[28,29,0]
   - 0.00164092350925531 Q_Line[28,29,0] + 6.83336e-06 L_Line[28,29,0] = 0
 V-Drop[29,30,0]: V_squared[29,0] - V_squared[30,0]
   - 8.7599110532108444e-04 P_Line[29,30,0]
   - 2.8950133395226721e-04 Q_Line[29,30,0] + 2.12793e-07 L_Line[29,30,0]
   = 0
 V-Drop[30,31,0]: V_squared[30,0] - V_squared[31,0]
   - 0.00437995552660542 P_Line[30,31,0]
   - 0.00144750666976134 Q_Line[30,31,0] + 5.31982e-06 L_Line[30,31,0] = 0
 V-Drop[31,32,0]: V_squared[31,0] - V_squared[32,0]
   - 0.0104694663442221 P_Line[31,32,0]
   - 0.00351394722590338 Q_Line[31,32,0] + 3.04894e-05 L_Line[31,32,0] = 0
 V-Drop[32,33,0]: V_squared[32,0] - V_squared[33,0]
   - 0.0213132878616583 P_Line[32,33,0]
   - 0.00704536435989009 Q_Line[32,33,0]
   + 1.2597334960938006e-04 L_Line[32,33,0] = 0
 V-Drop[33,34,0]: V_squared[33,0] - V_squared[34,0]
   - 0.018393317510588 P_Line[33,34,0]
   - 0.00608077586357499 Q_Line[33,34,0] + 9.38225e-05 L_Line[33,34,0] = 0
 V-Drop[2,35,0]: V_squared[2,0] - V_squared[35,0]
   - 5.49054e-05 P_Line[2,35,0] - 1.3476786235708992e-04 Q_Line[2,35,0]
   + 5.29425e-09 L_Line[2,35,0] = 0
 V-Drop[35,36,0]: V_squared[35,0] - V_squared[36,0]
   - 7.9862436952349584e-04 P_Line[35,36,0]
   - 0.00195288615360042 Q_Line[35,36,0] + 1.11289e-06 L_Line[35,36,0] = 0
 V-Drop[36,37,0]: V_squared[36,0] - V_squared[37,0]
   - 0.00131398665798163 P_Line[36,37,0]
   - 0.00153485621017797 Q_Line[36,37,0] + 1.02059e-06 L_Line[36,37,0] = 0
 V-Drop[37,38,0]: V_squared[37,0] - V_squared[38,0]
   - 3.7934657552366049e-04 P_Line[37,38,0] - 0.000442987 Q_Line[37,38,0]
   + 8.50353e-08 L_Line[37,38,0] = 0
 V-Drop[38,39,0]: V_squared[38,0] - V_squared[39,0]
   - 2.24613e-05 P_Line[38,39,0] - 2.62049e-05 Q_Line[38,39,0]
   + 2.97801e-10 L_Line[38,39,0] = 0
 V-Drop[39,40,0]: V_squared[39,0] - V_squared[40,0]
   - 0.0090880957550619 P_Line[39,40,0]
   - 0.0106179605629304 Q_Line[39,40,0] + 4.88336e-05 L_Line[39,40,0] = 0
 V-Drop[40,41,0]: V_squared[40,0] - V_squared[41,0]
   - 0.00386833678987943 P_Line[40,41,0]
   - 0.00452096264184941 Q_Line[40,41,0] + 8.85078e-06 L_Line[40,41,0] = 0
 V-Drop[41,42,0]: V_squared[41,0] - V_squared[42,0]
   - 5.1161873672598949e-04 P_Line[41,42,0]
   - 5.9647257598786094e-04 Q_Line[41,42,0] + 1.54383e-07 L_Line[41,42,0]
   = 0
 V-Drop[42,43,0]: V_squared[42,0] - V_squared[43,0]
   - 1.1480225311900251e-04 P_Line[42,43,0]
   - 1.4475066697613361e-04 Q_Line[42,43,0] + 8.53308e-09 L_Line[42,43,0]
   = 0
 V-Drop[43,44,0]: V_squared[43,0] - V_squared[44,0]
   - 0.00135890927876732 P_Line[43,44,0]
   - 0.00171329884274337 Q_Line[43,44,0] + 1.19551e-06 L_Line[43,44,0] = 0
 V-Drop[44,45,0]: V_squared[44,0] - V_squared[45,0]
   - 1.12307e-05 P_Line[44,45,0] - 1.49742e-05 Q_Line[44,45,0]
   + 8.75886e-11 L_Line[44,45,0] = 0
 V-Drop[3,46,0]: V_squared[3,0] - V_squared[46,0]
   - 4.24269e-05 P_Line[3,46,0] - 1.0481944849995881e-04 Q_Line[3,46,0]
   + 3.19679e-09 L_Line[3,46,0] = 0
 V-Drop[46,47,0]: V_squared[46,0] - V_squared[47,0]
   - 0.00106192084135077 P_Line[46,47,0]
   - 0.0025992727526835 Q_Line[46,47,0] + 1.97097e-06 L_Line[46,47,0] = 0
 V-Drop[47,48,0]: V_squared[47,0] - V_squared[48,0]
   - 0.00361627097324858 P_Line[47,48,0]
   - 0.00884850844420486 Q_Line[47,48,0] + 2.28434e-05 L_Line[47,48,0] = 0
 V-Drop[48,49,0]: V_squared[48,0] - V_squared[49,0]
   - 0.00102573317460674 P_Line[48,49,0]
   - 0.00250942751111211 Q_Line[48,49,0] + 1.83734e-06 L_Line[48,49,0] = 0
 V-Drop[7,50,0]: V_squared[7,0] - V_squared[50,0]
   - 0.00115800533580907 P_Line[7,50,0]
   - 5.9023332310095865e-04 Q_Line[7,50,0] + 4.22338e-07 L_Line[7,50,0]
   = 0
 V-Drop[50,51,0]: V_squared[50,0] - V_squared[51,0]
   - 0.00413038541112933 P_Line[50,51,0]
   - 0.00139010554320183 Q_Line[50,51,0] + 4.74812e-06 L_Line[50,51,0] = 0
 V-Drop[8,52,0]: V_squared[8,0] - V_squared[52,0]
   - 0.00217126 P_Line[8,52,0] - 0.00110559561155909 Q_Line[8,52,0]
   + 1.48418e-06 L_Line[8,52,0] = 0
 V-Drop[52,53,0]: V_squared[52,0] - V_squared[53,0]
   - 0.00253313667208234 P_Line[52,53,0]
   - 0.0012902774970114 Q_Line[52,53,0] + 2.0204e-06 L_Line[52,53,0] = 0
 V-Drop[53,54,0]: V_squared[53,0] - V_squared[54,0]
   - 0.00354639134091527 P_Line[53,54,0]
   - 0.00180563978546953 Q_Line[53,54,0] + 3.95931e-06 L_Line[53,54,0] = 0
 V-Drop[54,55,0]: V_squared[54,0] - V_squared[55,0]
   - 0.00351020367417124 P_Line[54,55,0]
   - 0.0017881698773862 Q_Line[54,55,0] + 3.87977e-06 L_Line[54,55,0] = 0
 V-Drop[55,56,0]: V_squared[55,0] - V_squared[56,0]
   - 0.0198408241803493 P_Line[55,56,0]
   - 0.00665977853147953 Q_Line[55,56,0]
   + 1.0950273856097277e-04 L_Line[55,56,0] = 0
 V-Drop[56,57,0]: V_squared[56,0] - V_squared[57,0]
   - 0.00977940497493068 P_Line[56,57,0]
   - 0.00328184701851062 Q_Line[56,57,0] + 2.66018e-05 L_Line[56,57,0] = 0
 V-Drop[57,58,0]: V_squared[57,0] - V_squared[58,0]
   - 0.00379596145639137 P_Line[57,58,0]
   - 0.00125533768084474 Q_Line[57,58,0] + 3.9963e-06 L_Line[57,58,0] = 0
 V-Drop[58,59,0]: V_squared[58,0] - V_squared[59,0]
   - 0.00481795107926596 P_Line[58,59,0]
   - 0.0014624808766899 Q_Line[58,59,0] + 6.33788e-06 L_Line[58,59,0] = 0
 V-Drop[59,60,0]: V_squared[59,0] - V_squared[60,0]
   - 0.00633284168020585 P_Line[59,60,0]
   - 0.00322569374252849 Q_Line[59,60,0] + 1.26275e-05 L_Line[59,60,0] = 0
 V-Drop[60,61,0]: V_squared[60,0] - V_squared[61,0]
   - 0.00121540646236857 P_Line[60,61,0]
   - 6.1893388638070918e-04 Q_Line[60,61,0] + 4.65073e-07 L_Line[60,61,0]
   = 0
 V-Drop[61,62,0]: V_squared[61,0] - V_squared[62,0]
   - 0.00180938333720167 P_Line[61,62,0]
   - 9.2091372610678113e-04 Q_Line[61,62,0] + 1.03049e-06 L_Line[61,62,0]
   = 0
 V-Drop[62,63,0]: V_squared[62,0] - V_squared[63,0]
   - 0.00886597835228818 P_Line[62,63,0]
   - 0.00451597123953989 Q_Line[62,63,0] + 2.47499e-05 L_Line[62,63,0] = 0
 V-Drop[63,64,0]: V_squared[63,0] - V_squared[64,0]
   - 0.0129901245105306 P_Line[63,64,0]
   - 0.00661610376127121 Q_Line[63,64,0] + 5.3129e-05 L_Line[63,64,0] = 0
 V-Drop[10,65,0]: V_squared[10,0] - V_squared[65,0]
   - 0.00251067536168949 P_Line[10,65,0]
   - 7.6243670277946246e-04 Q_Line[10,65,0] + 1.7212e-06 L_Line[10,65,0]
   = 0
 V-Drop[65,66,0]: V_squared[65,0] - V_squared[66,0]
   - 5.8649e-05 P_Line[65,66,0] - 1.74699e-05 Q_Line[65,66,0]
   + 9.36225e-10 L_Line[65,66,0] = 0
 V-Drop[11,67,0]: V_squared[11,0] - V_squared[67,0]
   - 0.00922660716915114 P_Line[11,67,0]
   - 0.00304974681111785 Q_Line[11,67,0] + 2.36078e-05 L_Line[11,67,0] = 0
 V-Drop[67,68,0]: V_squared[67,0] - V_squared[68,0]
   - 5.8649e-05 P_Line[67,68,0] - 1.99656e-05 Q_Line[67,68,0]
   + 9.59582e-10 L_Line[67,68,0] = 0
 V-Slack[0,0]: V_squared[0,0] = 1
 P-Injection[0,0]: - P_Gen[0,0] + P_Inj[0,0] = 0
 P-Injection[1,0]: P_Inj[1,0] = 0
 P-Injection[2,0]: P_Inj[2,0] = 0
 P-Injection[3,0]: P_Inj[3,0] = 0
 P-Injection[4,0]: P_Inj[4,0] = 0
 P-Injection[5,0]: P_Inj[5,0] = -0.0026
 P-Injection[6,0]: P_Inj[6,0] = -0.0404
 P-Injection[7,0]: P_Inj[7,0] = -0.075
 P-Injection[8,0]: P_Inj[8,0] = -0.03
 P-Injection[9,0]: P_Inj[9,0] = -0.028
 P-Injection[10,0]: P_Inj[10,0] = -0.145
 P-Injection[11,0]: P_Inj[11,0] = -0.145
 P-Injection[12,0]: P_Inj[12,0] = -0.008
 P-Injection[13,0]: P_Inj[13,0] = -0.008
 P-Injection[14,0]: P_Inj[14,0] = 0
 P-Injection[15,0]: P_Inj[15,0] = -0.0455
 P-Injection[16,0]: P_Inj[16,0] = -0.06
 P-Injection[17,0]: P_Inj[17,0] = -0.06
 P-Injection[18,0]: P_Inj[18,0] = 0
 P-Injection[19,0]: P_Inj[19,0] = -0.001
 P-Injection[20,0]: P_Inj[20,0] = -0.114
 P-Injection[21,0]: P_Inj[21,0] = -0.005
 P-Injection[22,0]: P_Inj[22,0] = 0
 P-Injection[23,0]: P_Inj[23,0] = -0.028
 P-Injection[24,0]: P_Inj[24,0] = 0
 P-Injection[25,0]: P_Inj[25,0] = -0.014
 P-Injection[26,0]: P_Inj[26,0] = -0.014
 P-Injection[27,0]: P_Inj[27,0] = -0.026
 P-Injection[28,0]: P_Inj[28,0] = -0.026
 P-Injection[29,0]: P_Inj[29,0] = 0
 P-Injection[30,0]: P_Inj[30,0] = 0
 P-Injection[31,0]: P_Inj[31,0] = 0
 P-Injection[32,0]: P_Inj[32,0] = -0.014
 P-Injection[33,0]: P_Inj[33,0] = -0.0195
 P-Injection[34,0]: P_Inj[34,0] = -0.006
 P-Injection[35,0]: P_Inj[35,0] = -0.026
 P-Injection[36,0]: P_Inj[36,0] = -0.026
 P-Injection[37,0]: P_Inj[37,0] = 0
 P-Injection[38,0]: P_Inj[38,0] = -0.024
 P-Injection[39,0]: P_Inj[39,0] = -0.024
 P-Injection[40,0]: P_Inj[40,0] = -0.0012
 P-Injection[41,0]: P_Inj[41,0] = 0
 P-Injection[42,0]: P_Inj[42,0] = -0.006
 P-Injection[43,0]: P_Inj[43,0] = 0
 P-Injection[44,0]: P_Inj[44,0] = -0.03922
 P-Injection[45,0]: P_Inj[45,0] = -0.03922
 P-Injection[46,0]: P_Inj[46,0] = 0
 P-Injection[47,0]: P_Inj[47,0] = -0.079
 P-Injection[48,0]: P_Inj[48,0] = -0.3847
 P-Injection[49,0]: P_Inj[49,0] = -0.3847
 P-Injection[50,0]: P_Inj[50,0] = -0.0405
 P-Injection[51,0]: P_Inj[51,0] = -0.0036
 P-Injection[52,0]: P_Inj[52,0] = -0.00435
 P-Injection[53,0]: P_Inj[53,0] = -0.0264
 P-Injection[54,0]: P_Inj[54,0] = -0.024
 P-Injection[55,0]: P_Inj[55,0] = 0
 P-Injection[56,0]: P_Inj[56,0] = 0
 P-Injection[57,0]: P_Inj[57,0] = 0
 P-Injection[58,0]: P_Inj[58,0] = -0.1
 P-Injection[59,0]: P_Inj[59,0] = 0
 P-Injection[60,0]: P_Inj[60,0] = -1.244
 P-Injection[61,0]: P_Inj[61,0] = -0.032
 P-Injection[62,0]: P_Inj[62,0] = 0
 P-Injection[63,0]: P_Inj[63,0] = -0.227
 P-Injection[64,0]: P_Inj[64,0] = -0.059
 P-Injection[65,0]: P_Inj[65,0] = -0.018
 P-Injection[66,0]: P_Inj[66,0] = -0.018
 P-Injection[67,0]: P_Inj[67,0] = -0.028
 P-Injection[68,0]: P_Inj[68,0] = -0.028
 Q-Injection[0,0]: - Q_Gen[0,0] + Q_Inj[0,0] = 0
 Q-Injection[1,0]: Q_Inj[1,0] = 0
 Q-Injection[2,0]: Q_Inj[2,0] = 0
 Q-Injection[3,0]: Q_Inj[3,0] = 0
 Q-Injection[4,0]: Q_Inj[4,0] = 0
 Q-Injection[5,0]: Q_Inj[5,0] = -0.0022
 Q-Injection[6,0]: Q_Inj[6,0] = -0.03
 Q-Injection[7,0]: Q_Inj[7,0] = -0.054
 Q-Injection[8,0]: Q_Inj[8,0] = -0.022
 Q-Injection[9,0]: Q_Inj[9,0] = -0.019
 Q-Injection[10,0]: Q_Inj[10,0] = -0.104
 Q-Injection[11,0]: Q_Inj[11,0] = -0.104
 Q-Injection[12,0]: Q_Inj[12,0] = -0.005
 Q-Injection[13,0]: Q_Inj[13,0] = -0.0055
 Q-Injection[14,0]: Q_Inj[14,0] = 0
 Q-Injection[15,0]: Q_Inj[15,0] = -0.03
 Q-Injection[16,0]: Q_Inj[16,0] = -0.035
 Q-Injection[17,0]: Q_Inj[17,0] = -0.035
 Q-Injection[18,0]: Q_Inj[18,0] = 0
 Q-Injection[19,0]: Q_Inj[19,0] = -0.0006
 Q-Injection[20,0]: Q_Inj[20,0] = -0.081
 Q-Injection[21,0]: Q_Inj[21,0] = -0.0035
 Q-Injection[22,0]: Q_Inj[22,0] = 0
 Q-Injection[23,0]: Q_Inj[23,0] = -0.02
 Q-Injection[24,0]: Q_Inj[24,0] = 0
 Q-Injection[25,0]: Q_Inj[25,0] = -0.01
 Q-Injection[26,0]: Q_Inj[26,0] = -0.01
 Q-Injection[27,0]: Q_Inj[27,0] = -0.0186
 Q-Injection[28,0]: Q_Inj[28,0] = -0.0186
 Q-Injection[29,0]: Q_Inj[29,0] = 0
 Q-Injection[30,0]: Q_Inj[30,0] = 0
 Q-Injection[31,0]: Q_Inj[31,0] = 0
 Q-Injection[32,0]: Q_Inj[32,0] = -0.01
 Q-Injection[33,0]: Q_Inj[33,0] = -0.014
 Q-Injection[34,0]: Q_Inj[34,0] = -0.004
 Q-Injection[35,0]: Q_Inj[35,0] = -0.01855
 Q-Injection[36,0]: Q_Inj[36,0] = -0.01855
 Q-Injection[37,0]: Q_Inj[37,0] = 0
 Q-Injection[38,0]: Q_Inj[38,0] = -0.017
 Q-Injection[39,0]: Q_Inj[39,0] = -0.017
 Q-Injection[40,0]: Q_Inj[40,0] = -0.001
 Q-Injection[41,0]: Q_Inj[41,0] = 0
 Q-Injection[42,0]: Q_Inj[42,0] = -0.0043
 Q-Injection[43,0]: Q_Inj[43,0] = 0
 Q-Injection[44,0]: Q_Inj[44,0] = -0.0263
 Q-Injection[45,0]: Q_Inj[45,0] = -0.0263
 Q-Injection[46,0]: Q_Inj[46,0] = 0
 Q-Injection[47,0]: Q_Inj[47,0] = -0.0564
 Q-Injection[48,0]: Q_Inj[48,0] = -0.2745
 Q-Injection[49,0]: Q_Inj[49,0] = -0.2745
 Q-Injection[50,0]: Q_Inj[50,0] = -0.0283
 Q-Injection[51,0]: Q_Inj[51,0] = -0.0027
 Q-Injection[52,0]: Q_Inj[52,0] = -0.0035
 Q-Injection[53,0]: Q_Inj[53,0] = -0.019
 Q-Injection[54,0]: Q_Inj[54,0] = -0.0172
 Q-Injection[55,0]: Q_Inj[55,0] = 0
 Q-Injection[56,0]: Q_Inj[56,0] = 0
 Q-Injection[57,0]: Q_Inj[57,0] = 0
 Q-Injection[58,0]: Q_Inj[58,0] = -0.072
 Q-Injection[59,0]: Q_Inj[59,0] = 0
 Q-Injection[60,0]: Q_Inj[60,0] = -0.888
 Q-Injection[61,0]: Q_Inj[61,0] = -0.023
 Q-Injection[62,0]: Q_Inj[62,0] = 0
 Q-Injection[63,0]: Q_Inj[63,0] = -0.162
 Q-Injection[64,0]: Q_Inj[64,0] = -0.042
 Q-Injection[65,0]: Q_Inj[65,0] = -0.013
 Q-Injection[66,0]: Q_Inj[66,0] = -0.013
 Q-Injection[67,0]: Q_Inj[67,0] = -0.02
 Q-Injection[68,0]: Q_Inj[68,0] = -0.02
 Ohms-Law[0,1,0]: [ V_squared[1,0] * L_Line[0,1,0] - P_Line[0,1,0] ^2
   - Q_Line[0,1,0] ^2 ] >= 0
 Ohms-Law[1,2,0]: [ V_squared[2,0] * L_Line[1,2,0] - P_Line[1,2,0] ^2
   - Q_Line[1,2,0] ^2 ] >= 0
 Ohms-Law[2,3,0]: [ V_squared[3,0] * L_Line[2,3,0] - P_Line[2,3,0] ^2
   - Q_Line[2,3,0] ^2 ] >= 0
 Ohms-Law[3,4,0]: [ V_squared[4,0] * L_Line[3,4,0] - P_Line[3,4,0] ^2
   - Q_Line[3,4,0] ^2 ] >= 0
 Ohms-Law[4,5,0]: [ V_squared[5,0] * L_Line[4,5,0] - P_Line[4,5,0] ^2
   - Q_Line[4,5,0] ^2 ] >= 0
 Ohms-Law[5,6,0]: [ V_squared[6,0] * L_Line[5,6,0] - P_Line[5,6,0] ^2
   - Q_Line[5,6,0] ^2 ] >= 0
 Ohms-Law[6,7,0]: [ V_squared[7,0] * L_Line[6,7,0] - P_Line[6,7,0] ^2
   - Q_Line[6,7,0] ^2 ] >= 0
 Ohms-Law[7,8,0]: [ V_squared[8,0] * L_Line[7,8,0] - P_Line[7,8,0] ^2
   - Q_Line[7,8,0] ^2 ] >= 0
 Ohms-Law[8,9,0]: [ V_squared[9,0] * L_Line[8,9,0] - P_Line[8,9,0] ^2
   - Q_Line[8,9,0] ^2 ] >= 0
 Ohms-Law[9,10,0]: [ V_squared[10,0] * L_Line[9,10,0] - P_Line[9,10,0] ^2
   - Q_Line[9,10,0] ^2 ] >= 0
 Ohms-Law[10,11,0]: [ V_squared[11,0] * L_Line[10,11,0]
   - P_Line[10,11,0] ^2 - Q_Line[10,11,0] ^2 ] >= 0
 Ohms-Law[11,12,0]: [ V_squared[12,0] * L_Line[11,12,0]
   - P_Line[11,12,0] ^2 - Q_Line[11,12,0] ^2 ] >= 0
 Ohms-Law[12,13,0]: [ V_squared[13,0] * L_Line[12,13,0]
   - P_Line[12,13,0] ^2 - Q_Line[12,13,0] ^2 ] >= 0
 Ohms-Law[13,14,0]: [ V_squared[14,0] * L_Line[13,14,0]
   - P_Line[13,14,0] ^2 - Q_Line[13,14,0] ^2 ] >= 0
 Ohms-Law[14,15,0]: [ V_squared[15,0] * L_Line[14,15,0]
   - P_Line[14,15,0] ^2 - Q_Line[14,15,0] ^2 ] >= 0
 Ohms-Law[15,16,0]: [ V_squared[16,0] * L_Line[15,16,0]
   - P_Line[15,16,0] ^2 - Q_Line[15,16,0] ^2 ] >= 0
 Ohms-Law[16,17,0]: [ V_squared[17,0] * L_Line[16,17,0]
   - P_Line[16,17,0] ^2 - Q_Line[16,17,0] ^2 ] >= 0
 Ohms-Law[17,18,0]: [ V_squared[18,0] * L_Line[17,18,0]
   - P_Line[17,18,0] ^2 - Q_Line[17,18,0] ^2 ] >= 0
 Ohms-Law[18,19,0]: [ V_squared[19,0] * L_Line[18,19,0]
   - P_Line[18,19,0] ^2 - Q_Line[18,19,0] ^2 ] >= 0
 Ohms-Law[19,20,0]: [ V_squared[20,0] * L_Line[19,20,0]
   - P_Line[19,20,0] ^2 - Q_Line[19,20,0] ^2 ] >= 0
 Ohms-Law[20,21,0]: [ V_squared[21,0] * L_Line[20,21,0]
   - P_Line[20,21,0] ^2 - Q_Line[20,21,0] ^2 ] >= 0
 Ohms-Law[21,22,0]: [ V_squared[22,0] * L_Line[21,22,0]
   - P_Line[21,22,0] ^2 - Q_Line[21,22,0] ^2 ] >= 0
 Ohms-Law[22,23,0]: [ V_squared[23,0] * L_Line[22,23,0]
   - P_Line[22,23,0] ^2 - Q_Line[22,23,0] ^2 ] >= 0
 Ohms-Law[23,24,0]: [ V_squared[24,0] * L_Line[23,24,0]
   - P_Line[23,24,0] ^2 - Q_Line[23,24,0] ^2 ] >= 0
 Ohms-Law[24,25,0]: [ V_squared[25,0] * L_Line[24,25,0]
   - P_Line[24,25,0] ^2 - Q_Line[24,25,0] ^2 ] >= 0
 Ohms-Law[25,26,0]: [ V_squared[26,0] * L_Line[25,26,0]
   - P_Line[25,26,0] ^2 - Q_Line[25,26,0] ^2 ] >= 0
 Ohms-Law[2,27,0]: [ V_squared[27,0] * L_Line[2,27,0] - P_Line[2,27,0] ^2
   - Q_Line[2,27,0] ^2 ] >= 0
 Ohms-Law[27,28,0]: [ V_squared[28,0] * L_Line[27,28,0]
   - P_Line[27,28,0] ^2 - Q_Line[27,28,0] ^2 ] >= 0
 Ohms-Law[28,29,0]: [ V_squared[29,0] * L_Line[28,29,0]
   - P_Line[28,29,0] ^2 - Q_Line[28,29,0] ^2 ] >= 0
 Ohms-Law[29,30,0]: [ V_squared[30,0] * L_Line[29,30,0]
   - P_Line[29,30,0] ^2 - Q_Line[29,30,0] ^2 ] >= 0
 Ohms-Law[30,31,0]: [ V_squared[31,0] * L_Line[30,31,0]
   - P_Line[30,31,0] ^2 - Q_Line[30,31,0] ^2 ] >= 0
 Ohms-Law[31,32,0]: [ V_squared[32,0] * L_Line[31,32,0]
   - P_Line[31,32,0] ^2 - Q_Line[31,32,0] ^2 ] >= 0
 Ohms-Law[32,33,0]: [ V_squared[33,0] * L_Line[32,33,0]
   - P_Line[32,33,0] ^2 - Q_Line[32,33,0] ^2 ] >= 0
 Ohms-Law[33,34,0]: [ V_squared[34,0] * L_Line[33,34,0]
   - P_Line[33,34,0] ^2 - Q_Line[33,34,0] ^2 ] >= 0
 Ohms-Law[2,35,0]: [ V_squared[35,0] * L_Line[2,35,0] - P_Line[2,35,0] ^2
   - Q_Line[2,35,0] ^2 ] >= 0
 Ohms-Law[35,36,0]: [ V_squared[36,0] * L_Line[35,36,0]
   - P_Line[35,36,0] ^2 - Q_Line[35,36,0] ^2 ] >= 0
 Ohms-Law[36,37,0]: [ V_squared[37,0] * L_Line[36,37,0]
   - P_Line[36,37,0] ^2 - Q_Line[36,37,0] ^2 ] >= 0
 Ohms-Law[37,38,0]: [ V_squared[38,0] * L_Line[37,38,0]
   - P_Line[37,38,0] ^2 - Q_Line[37,38,0] ^2 ] >= 0
 Ohms-Law[38,39,0]: [ V_squared[39,0] * L_Line[38,39,0]
   - P_Line[38,39,0] ^2 - Q_Line[38,39,0] ^2 ] >= 0
 Ohms-Law[39,40,0]: [ V_squared[40,0] * L_Line[39,40,0]
   - P_Line[39,40,0] ^2 - Q_Line[39,40,0] ^2 ] >= 0
 Ohms-Law[40,41,0]: [ V_squared[41,0] * L_Line[40,41,0]
   - P_Line[40,41,0] ^2 - Q_Line[40,41,0] ^2 ] >= 0
 Ohms-Law[41,42,0]: [ V_squared[42,0] * L_Line[41,42,0]
   - P_Line[41,42,0] ^2 - Q_Line[41,42,0] ^2 ] >= 0
 Ohms-Law[42,43,0]: [ V_squared[43,0] * L_Line[42,43,0]
   - P_Line[42,43,0] ^2 - Q_Line[42,43,0] ^2 ] >= 0
 Ohms-Law[43,44,0]: [ V_squared[44,0] * L_Line[43,44,0]
   - P_Line[43,44,0] ^2 - Q_Line[43,44,0] ^2 ] >= 0
 Ohms-Law[44,45,0]: [ V_squared[45,0] * L_Line[44,45,0]
   - P_Line[44,45,0] ^2 - Q_Line[44,45,0] ^2 ] >= 0
 Ohms-Law[3,46,0]: [ V_squared[46,0] * L_Line[3,46,0] - P_Line[3,46,0] ^2
   - Q_Line[3,46,0] ^2 ] >= 0
 Ohms-Law[46,47,0]: [ V_squared[47,0] * L_Line[46,47,0]
   - P_Line[46,47,0] ^2 - Q_Line[46,47,0] ^2 ] >= 0
 Ohms-Law[47,48,0]: [ V_squared[48,0] * L_Line[47,48,0]
   - P_Line[47,48,0] ^2 - Q_Line[47,48,0] ^2 ] >= 0
 Ohms-Law[48,49,0]: [ V_squared[49,0] * L_Line[48,49,0]
   - P_Line[48,49,0] ^2 - Q_Line[48,49,0] ^2 ] >= 0
 Ohms-Law[7,50,0]: [ V_squared[50,0] * L_Line[7,50,0] - P_Line[7,50,0] ^2
   - Q_Line[7,50,0] ^2 ] >= 0
 Ohms-Law[50,51,0]: [ V_squared[51,0] * L_Line[50,51,0]
   - P_Line[50,51,0] ^2 - Q_Line[50,51,0] ^2 ] >= 0
 Ohms-Law[8,52,0]: [ V_squared[52,0] * L_Line[8,52,0] - P_Line[8,52,0] ^2
   - Q_Line[8,52,0] ^2 ] >= 0
 Ohms-Law[52,53,0]: [ V_squared[53,0] * L_Line[52,53,0]
   - P_Line[52,53,0] ^2 - Q_Line[52,53,0] ^2 ] >= 0
 Ohms-Law[53,54,0]: [ V_squared[54,0] * L_Line[53,54,0]
   - P_Line[53,54,0] ^2 - Q_Line[53,54,0] ^2 ] >= 0
 Ohms-Law[54,55,0]: [ V_squared[55,0] * L_Line[54,55,0]
   - P_Line[54,55,0] ^2 - Q_Line[54,55,0] ^2 ] >= 0
 Ohms-Law[55,56,0]: [ V_squared[56,0] * L_Line[55,56,0]
   - P_Line[55,56,0] ^2 - Q_Line[55,56,0] ^2 ] >= 0
 Ohms-Law[56,57,0]: [ V_squared[57,0] * L_Line[56,57,0]
   - P_Line[56,57,0] ^2 - Q_Line[56,57,0] ^2 ] >= 0
 Ohms-Law[57,58,0]: [ V_squared[58,0] * L_Line[57,58,0]
   - P_Line[57,58,0] ^2 - Q_Line[57,58,0] ^2 ] >= 0
 Ohms-Law[58,59,0]: [ V_squared[59,0] * L_Line[58,59,0]
   - P_Line[58,59,0] ^2 - Q_Line[58,59,0] ^2 ] >= 0
 Ohms-Law[59,60,0]: [ V_squared[60,0] * L_Line[59,60,0]
   - P_Line[59,60,0] ^2 - Q_Line[59,60,0] ^2 ] >= 0
 Ohms-Law[60,61,0]: [ V_squared[61,0] * L_Line[60,61,0]
   - P_Line[60,61,0] ^2 - Q_Line[60,61,0] ^2 ] >= 0
 Ohms-Law[61,62,0]: [ V_squared[62,0] * L_Line[61,62,0]
   - P_Line[61,62,0] ^2 - Q_Line[61,62,0] ^2 ] >= 0
 Ohms-Law[62,63,0]: [ V_squared[63,0] * L_Line[62,63,0]
   - P_Line[62,63,0] ^2 - Q_Line[62,63,0] ^2 ] >= 0
 Ohms-Law[63,64,0]: [ V_squared[64,0] * L_Line[63,64,0]
   - P_Line[63,64,0] ^2 - Q_Line[63,64,0] ^2 ] >= 0
 Ohms-Law[10,65,0]: [ V_squared[65,0] * L_Line[10,65,0]
   - P_Line[10,65,0] ^2 - Q_Line[10,65,0] ^2 ] >= 0
 Ohms-Law[65,66,0]: [ V_squared[66,0] * L_Line[65,66,0]
   - P_Line[65,66,0] ^2 - Q_Line[65,66,0] ^2 ] >= 0
 Ohms-Law[11,67,0]: [ V_squared[67,0] * L_Line[11,67,0]
   - P_Line[11,67,0] ^2 - Q_Line[11,67,0] ^2 ] >= 0
 Ohms-Law[67,68,0]: [ V_squared[68,0] * L_Line[67,68,0]
   - P_Line[67,68,0] ^2 - Q_Line[67,68,0] ^2 ] >= 0
Bounds
 0.81 <= V_squared[0,0] <= 1.21
 0.81 <= V_squared[1,0] <= 1.21
 0.81 <= V_squared[2,0] <= 1.21
 0.81 <= V_squared[3,0] <= 1.21
 0.81 <= V_squared[4,0] <= 1.21
 0.81 <= V_squared[5,0] <= 1.21
 0.81 <= V_squared[6,0] <= 1.21
 0.81 <= V_squared[7,0] <= 1.21
 0.81 <= V_squared[8,0] <= 1.21
 0.81 <= V_squared[9,0] <= 1.21
 0.81 <= V_squared[10,0] <= 1.21
 0.81 <= V_squared[11,0] <= 1.21
 0.81 <= V_squared[12,0] <= 1.21
 0.81 <= V_squared[13,0] <= 1.21
 0.81 <= V_squared[14,0] <= 1.21
 0.81 <= V_squared[15,0] <= 1.21
 0.81 <= V_squared[16,0] <= 1.21
 0.81 <= V_squared[17,0] <= 1.21
 0.81 <= V_squared[18,0] <= 1.21
 0.81 <= V_squared[19,0] <= 1.21
 0.81 <= V_squared[20,0] <= 1.21
 0.81 <= V_squared[21,0] <= 1.21
 0.81 <= V_squared[22,0] <= 1.21
 0.81 <= V_squared[23,0] <= 1.21
 0.81 <= V_squared[24,0] <= 1.21
 0.81 <= V_squared[25,0] <= 1.21
 0.81 <= V_squared[26,0] <= 1.21
 0.81 <= V_squared[27,0] <= 1.21
 0.81 <= V_squared[28,0] <= 1.21
 0.81 <= V_squared[29,0] <= 1.21
 0.81 <= V_squared[30,0] <= 1.21
 0.81 <= V_squared[31,0] <= 1.21
 0.81 <= V_squared[32,0] <= 1.21
 0.81 <= V_squared[33,0] <= 1.21
 0.81 <= V_squared[34,0] <= 1.21
 0.81 <= V_squared[35,0] <= 1.21
 0.81 <= V_squared[36,0] <= 1.21
 0.81 <= V_squared[37,0] <= 1.21
 0.81 <= V_squared[38,0] <= 1.21
 0.81 <= V_squared[39,0] <= 1.21
 0.81 <= V_squared[40,0] <= 1.21
 0.81 <= V_squared[41,0] <= 1.21
 0.81 <= V_squared[42,0] <= 1.21
 0.81 <= V_squared[43,0] <= 1.21
 0.81 <= V_squared[44,0] <= 1.21
 0.81 <= V_squared[45,0] <= 1.21
 0.81 <= V_squared[46,0] <= 1.21
 0.81 <= V_squared[47,0] <= 1.21
 0.81 <= V_squared[48,0] <= 1.21
 0.81 <= V_squared[49,0] <= 1.21
 0.81 <= V_squared[50,0] <= 1.21
 0.81 <= V_squared[51,0] <= 1.21
 0.81 <= V_squared[52,0] <= 1.21
 0.81 <= V_squared[53,0] <= 1.21
 0.81 <= V_squared[54,0] <= 1.21
 0.81 <= V_squared[55,0] <= 1.21
 0.81 <= V_squared[56,0] <= 1.21
 0.81 <= V_squared[57,0] <= 1.21
 0.81 <= V_squared[58,0] <= 1.21
 0.81 <= V_squared[59,0] <= 1.21
 0.81 <= V_squared[60,0] <= 1.21
 0.81 <= V_squared[61,0] <= 1.21
 0.81 <= V_squared[62,0] <= 1.21
 0.81 <= V_squared[63,0] <= 1.21
 0.81 <= V_squared[64,0] <= 1.21
 0.81 <= V_squared[65,0] <= 1.21
 0.81 <= V_squared[66,0] <= 1.21
 0.81 <= V_squared[67,0] <= 1.21
 0.81 <= V_squared[68,0] <= 1.21
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
 P_Inj[33,0] free
 P_Inj[34,0] free
 P_Inj[35,0] free
 P_Inj[36,0] free
 P_Inj[37,0] free
 P_Inj[38,0] free
 P_Inj[39,0] free
 P_Inj[40,0] free
 P_Inj[41,0] free
 P_Inj[42,0] free
 P_Inj[43,0] free
 P_Inj[44,0] free
 P_Inj[45,0] free
 P_Inj[46,0] free
 P_Inj[47,0] free
 P_Inj[48,0] free
 P_Inj[49,0] free
 P_Inj[50,0] free
 P_Inj[51,0] free
 P_Inj[52,0] free
 P_Inj[53,0] free
 P_Inj[54,0] free
 P_Inj[55,0] free
 P_Inj[56,0] free
 P_Inj[57,0] free
 P_Inj[58,0] free
 P_Inj[59,0] free
 P_Inj[60,0] free
 P_Inj[61,0] free
 P_Inj[62,0] free
 P_Inj[63,0] free
 P_Inj[64,0] free
 P_Inj[65,0] free
 P_Inj[66,0] free
 P_Inj[67,0] free
 P_Inj[68,0] free
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
 Q_Inj[33,0] free
 Q_Inj[34,0] free
 Q_Inj[35,0] free
 Q_Inj[36,0] free
 Q_Inj[37,0] free
 Q_Inj[38,0] free
 Q_Inj[39,0] free
 Q_Inj[40,0] free
 Q_Inj[41,0] free
 Q_Inj[42,0] free
 Q_Inj[43,0] free
 Q_Inj[44,0] free
 Q_Inj[45,0] free
 Q_Inj[46,0] free
 Q_Inj[47,0] free
 Q_Inj[48,0] free
 Q_Inj[49,0] free
 Q_Inj[50,0] free
 Q_Inj[51,0] free
 Q_Inj[52,0] free
 Q_Inj[53,0] free
 Q_Inj[54,0] free
 Q_Inj[55,0] free
 Q_Inj[56,0] free
 Q_Inj[57,0] free
 Q_Inj[58,0] free
 Q_Inj[59,0] free
 Q_Inj[60,0] free
 Q_Inj[61,0] free
 Q_Inj[62,0] free
 Q_Inj[63,0] free
 Q_Inj[64,0] free
 Q_Inj[65,0] free
 Q_Inj[66,0] free
 Q_Inj[67,0] free
 Q_Inj[68,0] free
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
 -5 <= P_Line[17,18,0] <= 5
 -5 <= P_Line[18,19,0] <= 5
 -5 <= P_Line[19,20,0] <= 5
 -5 <= P_Line[20,21,0] <= 5
 -5 <= P_Line[21,22,0] <= 5
 -5 <= P_Line[22,23,0] <= 5
 -5 <= P_Line[23,24,0] <= 5
 -5 <= P_Line[24,25,0] <= 5
 -5 <= P_Line[25,26,0] <= 5
 -5 <= P_Line[2,27,0] <= 5
 -5 <= P_Line[27,28,0] <= 5
 -5 <= P_Line[28,29,0] <= 5
 -5 <= P_Line[29,30,0] <= 5
 -5 <= P_Line[30,31,0] <= 5
 -5 <= P_Line[31,32,0] <= 5
 -5 <= P_Line[32,33,0] <= 5
 -5 <= P_Line[33,34,0] <= 5
 -5 <= P_Line[2,35,0] <= 5
 -5 <= P_Line[35,36,0] <= 5
 -5 <= P_Line[36,37,0] <= 5
 -5 <= P_Line[37,38,0] <= 5
 -5 <= P_Line[38,39,0] <= 5
 -5 <= P_Line[39,40,0] <= 5
 -5 <= P_Line[40,41,0] <= 5
 -5 <= P_Line[41,42,0] <= 5
 -5 <= P_Line[42,43,0] <= 5
 -5 <= P_Line[43,44,0] <= 5
 -5 <= P_Line[44,45,0] <= 5
 -5 <= P_Line[3,46,0] <= 5
 -5 <= P_Line[46,47,0] <= 5
 -5 <= P_Line[47,48,0] <= 5
 -5 <= P_Line[48,49,0] <= 5
 -5 <= P_Line[7,50,0] <= 5
 -5 <= P_Line[50,51,0] <= 5
 -5 <= P_Line[8,52,0] <= 5
 -5 <= P_Line[52,53,0] <= 5
 -5 <= P_Line[53,54,0] <= 5
 -5 <= P_Line[54,55,0] <= 5
 -5 <= P_Line[55,56,0] <= 5
 -5 <= P_Line[56,57,0] <= 5
 -5 <= P_Line[57,58,0] <= 5
 -5 <= P_Line[58,59,0] <= 5
 -5 <= P_Line[59,60,0] <= 5
 -5 <= P_Line[60,61,0] <= 5
 -5 <= P_Line[61,62,0] <= 5
 -5 <= P_Line[62,63,0] <= 5
 -5 <= P_Line[63,64,0] <= 5
 -5 <= P_Line[10,65,0] <= 5
 -5 <= P_Line[65,66,0] <= 5
 -5 <= P_Line[11,67,0] <= 5
 -5 <= P_Line[67,68,0] <= 5
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
 Q_Line[17,18,0] free
 Q_Line[18,19,0] free
 Q_Line[19,20,0] free
 Q_Line[20,21,0] free
 Q_Line[21,22,0] free
 Q_Line[22,23,0] free
 Q_Line[23,24,0] free
 Q_Line[24,25,0] free
 Q_Line[25,26,0] free
 Q_Line[2,27,0] free
 Q_Line[27,28,0] free
 Q_Line[28,29,0] free
 Q_Line[29,30,0] free
 Q_Line[30,31,0] free
 Q_Line[31,32,0] free
 Q_Line[32,33,0] free
 Q_Line[33,34,0] free
 Q_Line[2,35,0] free
 Q_Line[35,36,0] free
 Q_Line[36,37,0] free
 Q_Line[37,38,0] free
 Q_Line[38,39,0] free
 Q_Line[39,40,0] free
 Q_Line[40,41,0] free
 Q_Line[41,42,0] free
 Q_Line[42,43,0] free
 Q_Line[43,44,0] free
 Q_Line[44,45,0] free
 Q_Line[3,46,0] free
 Q_Line[46,47,0] free
 Q_Line[47,48,0] free
 Q_Line[48,49,0] free
 Q_Line[7,50,0] free
 Q_Line[50,51,0] free
 Q_Line[8,52,0] free
 Q_Line[52,53,0] free
 Q_Line[53,54,0] free
 Q_Line[54,55,0] free
 Q_Line[55,56,0] free
 Q_Line[56,57,0] free
 Q_Line[57,58,0] free
 Q_Line[58,59,0] free
 Q_Line[59,60,0] free
 Q_Line[60,61,0] free
 Q_Line[61,62,0] free
 Q_Line[62,63,0] free
 Q_Line[63,64,0] free
 Q_Line[10,65,0] free
 Q_Line[65,66,0] free
 Q_Line[11,67,0] free
 Q_Line[67,68,0] free
End
