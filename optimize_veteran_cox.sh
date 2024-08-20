#!/bin/bash
. venv/bin/activate
python3 optimize_veteran_cox.py model="CoxPH" 2>&1 | tee Logs/veteran_cox_log.txt
deactivate