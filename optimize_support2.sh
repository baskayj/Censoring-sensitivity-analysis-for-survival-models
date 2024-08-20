#!/bin/bash
. venv/bin/activate
python3  optimize_support2.py 2>&1 | tee Logs/support2_mdn_log.txt
deactivate