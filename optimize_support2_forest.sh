#!/bin/bash
. venv/bin/activate
python3 optimize_support2_cox.py model="RandomSurvivalForest" 2>&1 | tee Logs/support2_forest_log.txt
deactivate