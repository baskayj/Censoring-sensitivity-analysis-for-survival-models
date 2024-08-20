#!/bin/bash
. venv/bin/activate
python3  optimize_spinemet.py 2>&1 | tee Logs/spinemet_mdn_log.txt
deactivate