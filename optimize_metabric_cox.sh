#!/bin/bash
. venv/bin/activate
python3 optimize_metabric_cox.py model="CoxPH" 2>&1 | tee Logs/metabric_cox_log.txt
deactivate