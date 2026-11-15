#!/usr/bin/env bash
# One-command setup + launch for macOS / Linux:  bash run.sh
cd "$(dirname "$0")"
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
[ -f pharmacy.db ] || python train.py
streamlit run app.py
