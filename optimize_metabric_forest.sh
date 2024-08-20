#!/bin/bash
. venv/bin/activate
python3 optimize_metabric_cox.py model="RandomSurvivalForest" 2>&1 | tee Logs/metabric_forest_log.txt
deactivate