#!/bin/bash
. venv/bin/activate
python3 optimize_metabric_cox.py model="SurvivalTree" 2>&1 | tee Logs/metabric_tree_log.txt
deactivate