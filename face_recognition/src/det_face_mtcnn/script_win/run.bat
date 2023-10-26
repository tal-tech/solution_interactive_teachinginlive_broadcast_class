@echo off

call gen.bat

cd script_win
call upload_toftp.bat
