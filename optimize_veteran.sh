#!/bin/bash
. venv/bin/activate
python3  optimize_veteran.py 2>&1 | tee Logs/veteran_mdn_log.txt
deactivate