\ Model Case33-Linear
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 
Subject To
 P-Balance[0,0]: - P_Inj[0,0] + P_Line[0,1,0]
   + 5.7525911617239305e-04 L_Line[0,1,0] = 0
 P-Balance[1,0]: - P_Inj[1,0] - P_Line[0,1,0] + P_Line[1,2,0]
   + P_Line[1,18,0] + 0.00307595167324284 L_Line[1,2,0]
   + 0.00102323747345198 L_Line[1,18,0] = 0
 P-Balance[2,0]: - P_Inj[2,0] - P_Line[1,2,0] + P_Line[2,3,0]
   + P_Line[2,22,0] + 0.00228356655660625 L_Line[2,3,0]
   + 0.00281515090257032 L_Line[2,22,0] = 0
 P-Balance[3,0]: - P_Inj[3,0] - P_Line[2,3,0] + P_Line[3,4,0]
   + 0.00237777927519847 L_Line[3,4,0] = 0
 P-Balance[4,0]: - P_Inj[4,0] - P_Line[3,4,0] + P_Line[4,5,0]
   + 0.00510994811437299 L_Line[4,5,0] = 0
 P-Balance[5,0]: - P_Inj[5,0] - P_Line[4,5,0] + P_Line[5,6,0]
   + P_Line[5,25,0] + 0.00116798814042811 L_Line[5,6,0]
   + 0.00126656833604117 L_Line[5,25,0] = 0
 P-Balance[6,0]: - P_Inj[6,0] - P_Line[5,6,0] + P_Line[6,7,0]
   + 0.0044386045037423 L_Line[6,7,0] = 0
 P-Balance[7,0]: - P_Inj[7,0] - P_Line[6,7,0] + P_Line[7,8,0]
   + 0.00642643047350938 L_Line[7,8,0] = 0
 P-Balance[8,0]: - P_Inj[8,0] - P_Line[7,8,0] + P_Line[8,9,0]
   + 0.00651378 L_Line[8,9,0] = 0
 P-Balance[9,0]: - P_Inj[9,0] - P_Line[8,9,0] + P_Line[9,10,0]
   + 0.00122663711756499 L_Line[9,10,0] = 0
 P-Balance[10,0]: - P_Inj[10,0] - P_Line[9,10,0] + P_Line[10,11,0]
   + 0.00233597628085623 L_Line[10,11,0] = 0
 P-Balance[11,0]: - P_Inj[11,0] - P_Line[10,11,0] + P_Line[11,12,0]
   + 0.00915922323797259 L_Line[11,12,0] = 0
 P-Balance[12,0]: - P_Inj[12,0] - P_Line[11,12,0] + P_Line[12,13,0]
   + 0.00337917936354629 L_Line[12,13,0] = 0
 P-Balance[13,0]: - P_Inj[13,0] - P_Line[12,13,0] + P_Line[13,14,0]
   + 0.00368739845615927 L_Line[13,14,0] = 0
 P-Balance[14,0]: - P_Inj[14,0] - P_Line[13,14,0] + P_Line[14,15,0]
   + 0.00465635442949519 L_Line[14,15,0] = 0
 P-Balance[15,0]: - P_Inj[15,0] - P_Line[14,15,0] + P_Line[15,16,0]
   + 0.00804239697121708 L_Line[15,16,0] = 0
 P-Balance[16,0]: - P_Inj[16,0] - P_Line[15,16,0] + P_Line[16,17,0]
   + 0.00456713311321249 L_Line[16,17,0] = 0
 P-Balance[17,0]: - P_Inj[17,0] - P_Line[16,17,0] = 0
 P-Balance[18,0]: - P_Inj[18,0] - P_Line[1,18,0] + P_Line[18,19,0]
   + 0.00938508419247846 L_Line[18,19,0] = 0
 P-Balance[19,0]: - P_Inj[19,0] - P_Line[18,19,0] + P_Line[19,20,0]
   + 0.0025549740571865 L_Line[19,20,0] = 0
 P-Balance[20,0]: - P_Inj[20,0] - P_Line[19,20,0] + P_Line[20,21,0]
   + 0.00442300637152505 L_Line[20,21,0] = 0
 P-Balance[21,0]: - P_Inj[21,0] - P_Line[20,21,0] = 0
 P-Balance[22,0]: - P_Inj[22,0] - P_Line[2,22,0] + P_Line[22,23,0]
   + 0.00560284909243827 L_Line[22,23,0] = 0
 P-Balance[23,0]: - P_Inj[23,0] - P_Line[22,23,0] + P_Line[23,24,0]
   + 0.00559037058666447 L_Line[23,24,0] = 0
 P-Balance[24,0]: - P_Inj[24,0] - P_Line[23,24,0] = 0
 P-Balance[25,0]: - P_Inj[25,0] - P_Line[5,25,0] + P_Line[25,26,0]
   + 0.00177319567045764 L_Line[25,26,0] = 0
 P-Balance[26,0]: - P_Inj[26,0] - P_Line[25,26,0] + P_Line[26,27,0]
   + 0.00660736880722955 L_Line[26,27,0] = 0
 P-Balance[27,0]: - P_Inj[27,0] - P_Line[26,27,0] + P_Line[27,28,0]
   + 0.00501760717164684 L_Line[27,28,0] = 0
 P-Balance[28,0]: - P_Inj[28,0] - P_Line[27,28,0] + P_Line[28,29,0]
   + 0.00316642084010292 L_Line[28,29,0] = 0
 P-Balance[29,0]: - P_Inj[29,0] - P_Line[28,29,0] + P_Line[29,30,0]
   + 0.00607952801299761 L_Line[29,30,0] = 0
 P-Balance[30,0]: - P_Inj[30,0] - P_Line[29,30,0] + P_Line[30,31,0]
   + 0.00193728802138317 L_Line[30,31,0] = 0
 P-Balance[31,0]: - P_Inj[31,0] - P_Line[30,31,0] + P_Line[31,32,0]
   + 0.00212758523443369 L_Line[31,32,0] = 0
 P-Balance[32,0]: - P_Inj[32,0] - P_Line[31,32,0] = 0
 Q-Balance[0,0]: - Q_Inj[0,0] + Q_Line[0,1,0]
   + 2.9324488568440859e-04 L_Line[0,1,0] = 0
 Q-Balance[1,0]: - Q_Inj[1,0] - Q_Line[0,1,0] + Q_Line[1,2,0]
   + Q_Line[1,18,0] + 0.00156667639990117 L_Line[1,2,0]
   + 0.000976443 L_Line[1,18,0] = 0
 Q-Balance[2,0]: - Q_Inj[2,0] - Q_Line[1,2,0] + Q_Line[2,3,0]
   + Q_Line[2,22,0] + 0.00116299673811859 L_Line[2,3,0]
   + 0.00192356166503198 L_Line[2,22,0] = 0
 Q-Balance[3,0]: - Q_Inj[3,0] - Q_Line[2,3,0] + Q_Line[3,4,0]
   + 0.00121103898534774 L_Line[3,4,0] = 0
 Q-Balance[4,0]: - Q_Inj[4,0] - Q_Line[3,4,0] + Q_Line[4,5,0]
   + 0.00441115179103993 L_Line[4,5,0] = 0
 Q-Balance[5,0]: - Q_Inj[5,0] - Q_Line[4,5,0] + Q_Line[5,6,0]
   + Q_Line[5,25,0] + 0.00386084968641515 L_Line[5,6,0]
   + 6.4513874850569901e-04 L_Line[5,25,0] = 0
 Q-Balance[6,0]: - Q_Inj[6,0] - Q_Line[5,6,0] + Q_Line[6,7,0]
   + 0.00146684835371073 L_Line[6,7,0] = 0
 Q-Balance[7,0]: - Q_Inj[7,0] - Q_Line[6,7,0] + Q_Line[7,8,0]
   + 0.00461704713630771 L_Line[7,8,0] = 0
 Q-Balance[8,0]: - Q_Inj[8,0] - Q_Line[7,8,0] + Q_Line[8,9,0]
   + 0.00461704713630771 L_Line[8,9,0] = 0
 Q-Balance[9,0]: - Q_Inj[9,0] - Q_Line[8,9,0] + Q_Line[9,10,0]
   + 4.0555143764865021e-04 L_Line[9,10,0] = 0
 Q-Balance[10,0]: - Q_Inj[10,0] - Q_Line[9,10,0] + Q_Line[10,11,0]
   + 7.7241950739850602e-04 L_Line[10,11,0] = 0
 Q-Balance[11,0]: - Q_Inj[11,0] - Q_Line[10,11,0] + Q_Line[11,12,0]
   + 0.00720633708437217 L_Line[11,12,0] = 0
 Q-Balance[12,0]: - Q_Inj[12,0] - Q_Line[11,12,0] + Q_Line[12,13,0]
   + 0.00444796338307266 L_Line[12,13,0] = 0
 Q-Balance[13,0]: - Q_Inj[13,0] - Q_Line[12,13,0] + Q_Line[13,14,0]
   + 0.00328184701851062 L_Line[13,14,0] = 0
 Q-Balance[14,0]: - Q_Inj[14,0] - Q_Line[13,14,0] + Q_Line[14,15,0]
   + 0.00340039282336176 L_Line[14,15,0] = 0
 Q-Balance[15,0]: - Q_Inj[15,0] - Q_Line[14,15,0] + Q_Line[15,16,0]
   + 0.0107377542183589 L_Line[15,16,0] = 0
 Q-Balance[16,0]: - Q_Inj[16,0] - Q_Line[15,16,0] + Q_Line[16,17,0]
   + 0.00358133115708193 L_Line[16,17,0] = 0
 Q-Balance[17,0]: - Q_Inj[17,0] - Q_Line[16,17,0] = 0
 Q-Balance[18,0]: - Q_Inj[18,0] - Q_Line[1,18,0] + Q_Line[18,19,0]
   + 0.00845668336290739 L_Line[18,19,0] = 0
 Q-Balance[19,0]: - Q_Inj[19,0] - Q_Line[18,19,0] + Q_Line[19,20,0]
   + 0.00298485858109407 L_Line[19,20,0] = 0
 Q-Balance[20,0]: - Q_Inj[20,0] - Q_Line[19,20,0] + Q_Line[20,21,0]
   + 0.00584805173089354 L_Line[20,21,0] = 0
 Q-Balance[21,0]: - Q_Inj[21,0] - Q_Line[20,21,0] = 0
 Q-Balance[22,0]: - Q_Inj[22,0] - Q_Line[2,22,0] + Q_Line[22,23,0]
   + 0.00442425422210243 L_Line[22,23,0] = 0
 Q-Balance[23,0]: - Q_Inj[23,0] - Q_Line[22,23,0] + Q_Line[23,24,0]
   + 0.00437434019900721 L_Line[23,24,0] = 0
 Q-Balance[24,0]: - Q_Inj[24,0] - Q_Line[23,24,0] = 0
 Q-Balance[25,0]: - Q_Inj[25,0] - Q_Line[5,25,0] + Q_Line[25,26,0]
   + 9.0281989273476439e-04 L_Line[25,26,0] = 0
 Q-Balance[26,0]: - Q_Inj[26,0] - Q_Line[25,26,0] + Q_Line[26,27,0]
   + 0.00582559042050069 L_Line[26,27,0] = 0
 Q-Balance[27,0]: - Q_Inj[27,0] - Q_Line[26,27,0] + Q_Line[27,28,0]
   + 0.00437122057256376 L_Line[27,28,0] = 0
 Q-Balance[28,0]: - Q_Inj[28,0] - Q_Line[27,28,0] + Q_Line[28,29,0]
   + 0.00161284687126425 L_Line[28,29,0] = 0
 Q-Balance[29,0]: - Q_Inj[29,0] - Q_Line[28,29,0] + Q_Line[29,30,0]
   + 0.00600840053008693 L_Line[29,30,0] = 0
 Q-Balance[30,0]: - Q_Inj[30,0] - Q_Line[29,30,0] + Q_Line[30,31,0]
   + 0.00225798561976995 L_Line[30,31,0] = 0
 Q-Balance[31,0]: - Q_Inj[31,0] - Q_Line[30,31,0] + Q_Line[31,32,0]
   + 0.00330805188063561 L_Line[31,32,0] = 0
 Q-Balance[32,0]: - Q_Inj[32,0] - Q_Line[31,32,0] = 0
 V-Drop[0,1,0]: V_squared[0,0] - V_squared[1,0]
   - 0.00115051823234479 P_Line[0,1,0]
   - 5.8648977136881717e-04 Q_Line[0,1,0] + 4.16916e-07 L_Line[0,1,0] = 0
 V-Drop[1,2,0]: V_squared[1,0] - V_squared[2,0]
   - 0.00615190334648568 P_Line[1,2,0] - 0.00313335279980234 Q_Line[1,2,0]
   + 1.1916e-05 L_Line[1,2,0] = 0
 V-Drop[2,3,0]: V_squared[2,0] - V_squared[3,0]
   - 0.00456713311321249 P_Line[2,3,0] - 0.00232599347623718 Q_Line[2,3,0]
   + 6.56724e-06 L_Line[2,3,0] = 0
 V-Drop[3,4,0]: V_squared[3,0] - V_squared[4,0]
   - 0.00475555855039694 P_Line[3,4,0] - 0.00242207797069548 Q_Line[3,4,0]
   + 7.12045e-06 L_Line[3,4,0] = 0
 V-Drop[4,5,0]: V_squared[4,0] - V_squared[5,0]
   - 0.010219896228746 P_Line[4,5,0] - 0.00882230358207987 Q_Line[4,5,0]
   + 4.55698e-05 L_Line[4,5,0] = 0
 V-Drop[5,6,0]: V_squared[5,0] - V_squared[6,0]
   - 0.00233597628085623 P_Line[5,6,0] - 0.0077216993728303 Q_Line[5,6,0]
   + 1.62704e-05 L_Line[5,6,0] = 0
 V-Drop[6,7,0]: V_squared[6,0] - V_squared[7,0]
   - 0.00887720900748461 P_Line[6,7,0] - 0.00293369670742147 Q_Line[6,7,0]
   + 2.18529e-05 L_Line[6,7,0] = 0
 V-Drop[7,8,0]: V_squared[7,0] - V_squared[8,0]
   - 0.0128528609470188 P_Line[7,8,0] - 0.00923409427261542 Q_Line[7,8,0]
   + 6.26161e-05 L_Line[7,8,0] = 0
 V-Drop[8,9,0]: V_squared[8,0] - V_squared[9,0]
   - 0.013027560027852 P_Line[8,9,0] - 0.00923409427261542 Q_Line[8,9,0]
   + 6.37465e-05 L_Line[8,9,0] = 0
 V-Drop[9,10,0]: V_squared[9,0] - V_squared[10,0]
   - 0.00245327423512999 P_Line[9,10,0]
   - 8.1110287529730042e-04 Q_Line[9,10,0] + 1.66911e-06 L_Line[9,10,0]
   = 0
 V-Drop[10,11,0]: V_squared[10,0] - V_squared[11,0]
   - 0.00467195256171245 P_Line[10,11,0]
   - 0.00154483901479701 Q_Line[10,11,0] + 6.05342e-06 L_Line[10,11,0] = 0
 V-Drop[11,12,0]: V_squared[11,0] - V_squared[12,0]
   - 0.0183184464759452 P_Line[11,12,0]
   - 0.0144126741687443 Q_Line[11,12,0]
   + 1.358226644966147e-04 L_Line[11,12,0] = 0
 V-Drop[12,13,0]: V_squared[12,0] - V_squared[13,0]
   - 0.00675835872709258 P_Line[12,13,0]
   - 0.00889592676614531 Q_Line[12,13,0] + 3.12032e-05 L_Line[12,13,0] = 0
 V-Drop[13,14,0]: V_squared[13,0] - V_squared[14,0]
   - 0.00737479691231853 P_Line[13,14,0]
   - 0.00656369403702123 Q_Line[13,14,0] + 2.43674e-05 L_Line[13,14,0] = 0
 V-Drop[14,15,0]: V_squared[14,0] - V_squared[15,0]
   - 0.00931270885899039 P_Line[14,15,0]
   - 0.00680078564672352 Q_Line[14,15,0] + 3.32443e-05 L_Line[14,15,0] = 0
 V-Drop[15,16,0]: V_squared[15,0] - V_squared[16,0]
   - 0.0160847939424342 P_Line[15,16,0]
   - 0.0214755084367178 Q_Line[15,16,0]
   + 1.799795146965255e-04 L_Line[15,16,0] = 0
 V-Drop[16,17,0]: V_squared[16,0] - V_squared[17,0]
   - 0.00913426622642498 P_Line[16,17,0]
   - 0.00716266231416385 Q_Line[16,17,0] + 3.36846e-05 L_Line[16,17,0] = 0
 V-Drop[1,18,0]: V_squared[1,0] - V_squared[18,0]
   - 0.00204647494690396 P_Line[1,18,0]
   - 0.00195288615360042 Q_Line[1,18,0] + 2.00046e-06 L_Line[1,18,0] = 0
 V-Drop[18,19,0]: V_squared[18,0] - V_squared[19,0]
   - 0.0187701683849569 P_Line[18,19,0]
   - 0.0169133667258148 Q_Line[18,19,0]
   + 1.5959529880038364e-04 L_Line[18,19,0] = 0
 V-Drop[19,20,0]: V_squared[19,0] - V_squared[20,0]
   - 0.00510994811437299 P_Line[19,20,0]
   - 0.00596971716218813 Q_Line[19,20,0] + 1.54373e-05 L_Line[19,20,0] = 0
 V-Drop[20,21,0]: V_squared[20,0] - V_squared[21,0]
   - 0.0088460127430501 P_Line[20,21,0]
   - 0.0116961034617871 Q_Line[20,21,0] + 5.37627e-05 L_Line[20,21,0] = 0
 V-Drop[2,22,0]: V_squared[2,0] - V_squared[22,0]
   - 0.00563030180514065 P_Line[2,22,0]
   - 0.00384712333006396 Q_Line[2,22,0] + 1.16252e-05 L_Line[2,22,0] = 0
 V-Drop[22,23,0]: V_squared[22,0] - V_squared[23,0]
   - 0.0112056981848765 P_Line[22,23,0]
   - 0.00884850844420486 Q_Line[22,23,0] + 5.09659e-05 L_Line[22,23,0] = 0
 V-Drop[23,24,0]: V_squared[23,0] - V_squared[24,0]
   - 0.0111807411733289 P_Line[23,24,0]
   - 0.00874868039801442 Q_Line[23,24,0] + 5.03871e-05 L_Line[23,24,0] = 0
 V-Drop[5,25,0]: V_squared[5,0] - V_squared[25,0]
   - 0.00253313667208234 P_Line[5,25,0] - 0.0012902774970114 Q_Line[5,25,0]
   + 2.0204e-06 L_Line[5,25,0] = 0
 V-Drop[25,26,0]: V_squared[25,0] - V_squared[26,0]
   - 0.00354639134091527 P_Line[25,26,0]
   - 0.00180563978546953 Q_Line[25,26,0] + 3.95931e-06 L_Line[25,26,0] = 0
 V-Drop[26,27,0]: V_squared[26,0] - V_squared[27,0]
   - 0.0132147376144591 P_Line[26,27,0]
   - 0.0116511808410014 Q_Line[26,27,0] + 7.75948e-05 L_Line[26,27,0] = 0
 V-Drop[27,28,0]: V_squared[27,0] - V_squared[28,0]
   - 0.0100352143432937 P_Line[27,28,0]
   - 0.00874244114512752 Q_Line[27,28,0] + 4.4284e-05 L_Line[27,28,0] = 0
 V-Drop[28,29,0]: V_squared[28,0] - V_squared[29,0]
   - 0.00633284168020585 P_Line[28,29,0]
   - 0.00322569374252849 Q_Line[28,29,0] + 1.26275e-05 L_Line[28,29,0] = 0
 V-Drop[29,30,0]: V_squared[29,0] - V_squared[30,0]
   - 0.0121590560259952 P_Line[29,30,0]
   - 0.0120168010601739 Q_Line[29,30,0] + 7.30615e-05 L_Line[29,30,0] = 0
 V-Drop[30,31,0]: V_squared[30,0] - V_squared[31,0]
   - 0.00387457604276633 P_Line[30,31,0]
   - 0.00451597123953989 Q_Line[30,31,0] + 8.85158e-06 L_Line[30,31,0] = 0
 V-Drop[31,32,0]: V_squared[31,0] - V_squared[32,0]
   - 0.00425517046886738 P_Line[31,32,0]
   - 0.00661610376127121 Q_Line[31,32,0] + 1.54698e-05 L_Line[31,32,0] = 0
 V-Slack[0,0]: V_squared[0,0] = 1
 P-Injection[0,0]: - P_Gen[0,0] + P_Inj[0,0] = 0
 P-Injection[1,0]: P_Inj[1,0] = -0.1
 P-Injection[2,0]: P_Inj[2,0] = -0.09
 P-Injection[3,0]: P_Inj[3,0] = -0.12
 P-Injection[4,0]: P_Inj[4,0] = -0.06
 P-Injection[5,0]: P_Inj[5,0] = -0.06
 P-Injection[6,0]: P_Inj[6,0] = -0.2
 P-Injection[7,0]: P_Inj[7,0] = -0.2
 P-Injection[8,0]: P_Inj[8,0] = -0.06
 P-Injection[9,0]: P_Inj[9,0] = -0.06
 P-Injection[10,0]: P_Inj[10,0] = -0.045
 P-Injection[11,0]: P_Inj[11,0] = -0.06
 P-Injection[12,0]: P_Inj[12,0] = -0.06
 P-Injection[13,0]: P_Inj[13,0] = -0.12
 P-Injection[14,0]: P_Inj[14,0] = -0.06
 P-Injection[15,0]: P_Inj[15,0] = -0.06
 P-Injection[16,0]: P_Inj[16,0] = -0.06
 P-Injection[17,0]: P_Inj[17,0] = -0.09
 P-Injection[18,0]: P_Inj[18,0] = -0.09
 P-Injection[19,0]: P_Inj[19,0] = -0.09
 P-Injection[20,0]: P_Inj[20,0] = -0.09
 P-Injection[21,0]: P_Inj[21,0] = -0.09
 P-Injection[22,0]: P_Inj[22,0] = -0.09
 P-Injection[23,0]: P_Inj[23,0] = -0.42
 P-Injection[24,0]: P_Inj[24,0] = -0.42
 P-Injection[25,0]: P_Inj[25,0] = -0.06
 P-Injection[26,0]: P_Inj[26,0] = -0.06
 P-Injection[27,0]: P_Inj[27,0] = -0.06
 P-Injection[28,0]: P_Inj[28,0] = -0.12
 P-Injection[29,0]: P_Inj[29,0] = -0.2
 P-Injection[30,0]: P_Inj[30,0] = -0.15
 P-Injection[31,0]: P_Inj[31,0] = -0.21
 P-Injection[32,0]: P_Inj[32,0] = -0.06
 Q-Injection[0,0]: - Q_Gen[0,0] + Q_Inj[0,0] = 0
 Q-Injection[1,0]: Q_Inj[1,0] = -0.06
 Q-Injection[2,0]: Q_Inj[2,0] = -0.04
 Q-Injection[3,0]: Q_Inj[3,0] = -0.08
 Q-Injection[4,0]: Q_Inj[4,0] = -0.03
 Q-Injection[5,0]: Q_Inj[5,0] = -0.02
 Q-Injection[6,0]: Q_Inj[6,0] = -0.1
 Q-Injection[7,0]: Q_Inj[7,0] = -0.1
 Q-Injection[8,0]: Q_Inj[8,0] = -0.02
 Q-Injection[9,0]: Q_Inj[9,0] = -0.02
 Q-Injection[10,0]: Q_Inj[10,0] = -0.03
 Q-Injection[11,0]: Q_Inj[11,0] = -0.035
 Q-Injection[12,0]: Q_Inj[12,0] = -0.035
 Q-Injection[13,0]: Q_Inj[13,0] = -0.08
 Q-Injection[14,0]: Q_Inj[14,0] = -0.01
 Q-Injection[15,0]: Q_Inj[15,0] = -0.02
 Q-Injection[16,0]: Q_Inj[16,0] = -0.02
 Q-Injection[17,0]: Q_Inj[17,0] = -0.04
 Q-Injection[18,0]: Q_Inj[18,0] = -0.04
 Q-Injection[19,0]: Q_Inj[19,0] = -0.04
 Q-Injection[20,0]: Q_Inj[20,0] = -0.04
 Q-Injection[21,0]: Q_Inj[21,0] = -0.04
 Q-Injection[22,0]: Q_Inj[22,0] = -0.05
 Q-Injection[23,0]: Q_Inj[23,0] = -0.2
 Q-Injection[24,0]: Q_Inj[24,0] = -0.2
 Q-Injection[25,0]: Q_Inj[25,0] = -0.025
 Q-Injection[26,0]: Q_Inj[26,0] = -0.025
 Q-Injection[27,0]: Q_Inj[27,0] = -0.02
 Q-Injection[28,0]: Q_Inj[28,0] = -0.07
 Q-Injection[29,0]: Q_Inj[29,0] = -0.6
 Q-Injection[30,0]: Q_Inj[30,0] = -0.07
 Q-Injection[31,0]: Q_Inj[31,0] = -0.1
 Q-Injection[32,0]: Q_Inj[32,0] = -0.04
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
End
