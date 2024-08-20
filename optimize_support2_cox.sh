#!/bin/bash
. venv/bin/activate
python3 optimize_support2_cox.py model="CoxPH" 2>&1 | tee Logs/support2_cox_log.txt
deactivate