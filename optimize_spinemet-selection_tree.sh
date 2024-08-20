#!/bin/bash
. venv/bin/activate
python3 optimize_spinemet-selection_cox.py model="SurvivalTree" 2>&1 | tee Logs/spinemet_selection_tree_log.txt
deactivate