#!/bin/bash
. venv/bin/activate
python3 optimize_veteran_cox.py model="SurvivalTree" 2>&1 | tee Logs/veteran_tree_log.txt
deactivate