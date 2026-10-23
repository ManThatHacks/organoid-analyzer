@echo off
rem ---------------------------------------------------------------------------
rem  Re-analyse photos that ALREADY have results (a normal run skips them).
rem  Previous results and hand corrections are moved into _previous_runs\ first.
rem ---------------------------------------------------------------------------
title Re-analyse all photos
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
"%PY%" -u "%~dp0scripts\batch_review.py" --force %*
echo.
if errorlevel 1 echo   The run ended with an error ^(see the message above^).
echo   Finished. This window can be closed.
pause