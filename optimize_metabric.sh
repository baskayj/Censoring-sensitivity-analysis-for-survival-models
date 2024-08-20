#!/bin/bash
. venv/bin/activate
python3  optimize_metabric.py 2>&1 | tee Logs/metabric_mdn_log.txt
deactivate