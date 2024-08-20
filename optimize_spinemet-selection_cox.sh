#!/bin/bash
. venv/bin/activate
python3 optimize_spinemet-selection_cox.py model="CoxPH" 2>&1 | tee Logs/spinemet_selection_cox_log.txt
deactivate