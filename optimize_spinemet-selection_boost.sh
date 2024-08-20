#!/bin/bash
. venv/bin/activate
python3 optimize_spinemet-selection_cox.py model="BoostingSurvival" 2>&1 | tee Logs/spinemet_selection_boost_log.txt
deactivate