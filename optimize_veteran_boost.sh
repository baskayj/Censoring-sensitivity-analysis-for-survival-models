#!/bin/bash
. venv/bin/activate
python3 optimize_veteran_cox.py model="BoostingSurvival" 2>&1 | tee Logs/veteran_boost_log.txt
deactivate