#!/bin/bash
. venv/bin/activate
python3 optimize_support2_cox.py model="BoostingSurvival" 2>&1 | tee Logs/support2_boost_log.txt
deactivate