#!/bin/bash
. venv/bin/activate
python3  optimize_spinemet-selection.py 2>&1 | tee Logs/spinemet_selection_mdn_log.txt
deactivate