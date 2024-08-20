#!/bin/bash
. venv/bin/activate
python3 optimize_spinemet_cox.py model="SurvivalTree" 2>&1 | tee Logs/spinemet_tree_log.txt
deactivate