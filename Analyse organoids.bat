@echo off
rem ---------------------------------------------------------------------------
rem  Analyse a folder of well photos, then review and correct the masks.
rem  Double-click this file, or drag a folder of photos onto it.
rem ---------------------------------------------------------------------------
title Organoid analysis
cd /d "%~dp0"
set "PY=%~dp0_runtime\python\python.exe"
if not exist "%PY%" (
  echo.
  echo   ERROR: the Python runtime is missing.
  echo   Expected it at: %PY%
  echo   The _runtime folder must sit next to this file. See README.md.
  echo.
  pause
  exit /b 1
)
"%PY%" -u "%~dp0scripts\batch_review.py" %*
echo.
if errorlevel 1 echo   The run ended with an error ^(see the message above^).
echo   Finished. This window can be closed.
pause