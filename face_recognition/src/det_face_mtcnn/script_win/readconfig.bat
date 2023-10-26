@echo off

CALL :readconfig %1 %2 %3

:readconfig
  for /f "skip=1 tokens=1,2 delims==" %%a IN (%1) Do if %2==%%a set %3=%%b & @echo readconfig get %%a, value is %%b
  goto :eof