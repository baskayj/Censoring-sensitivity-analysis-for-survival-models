#!/bin/bash
. venv/bin/activate
python3 optimize_metabric_cox.py model="BoostingSurvival" 2>&1 | tee Logs/metabric_boost_log.txt
deactivate