#!/bin/bash
. venv/bin/activate
python3 optimize_support2_cox.py model="SurvivalTree" 2>&1 | tee Logs/support2_tree_log.txt
deactivate