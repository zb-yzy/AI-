@echo off
chcp 65001 > nul
cd /d %~dp0
streamlit run "code\挑战任务\Demo.py"
pause