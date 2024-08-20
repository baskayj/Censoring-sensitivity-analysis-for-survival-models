#!/bin/bash
. venv/bin/activate
python3 optimize_veteran_cox.py model="SurvivalSVM" 2>&1 | tee Logs/veteran_svm_log.txt
deactivate