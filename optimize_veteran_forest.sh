#!/bin/bash
. venv/bin/activate
python3 optimize_veteran_cox.py model="RandomSurvivalForest" 2>&1 | tee Logs/veteran_forest_log.txt
deactivate