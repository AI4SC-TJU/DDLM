#!/bin/bash
mkdir -p 'Results/2_4Prob-2D/DNLM(PINN)/G1e_2-N2e4-baseline/simulation-test-1/'
python3 -u DNLM-4prob-2D-Compensent.py  --result 'Results/2_4Prob-2D/DNLM(PINN)/G1e_2-N2e4-baseline/simulation-test-1' --beta 1000 > 'Results/2_4Prob-2D/DNLM(PINN)/G1e_2-N2e4-baseline/simulation-test-1/logger.txt'