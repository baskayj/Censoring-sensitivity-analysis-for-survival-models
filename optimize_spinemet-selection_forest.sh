#!/bin/bash
. venv/bin/activate
python3 optimize_spinemet-selection_cox.py model="RandomSurvivalForest" 2>&1 | tee Logs/spinemet_selection_forest_log.txt
deactivate