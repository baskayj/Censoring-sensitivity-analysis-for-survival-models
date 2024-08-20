#!/bin/bash
. venv/bin/activate
python3 optimize_spinemet_cox.py model="RandomSurvivalForest" 2>&1 | tee Logs/spinemet_forest_log.txt
deactivate