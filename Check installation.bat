@echo off
rem  Self-test: is everything present and working?
title Check installation
cd /d "%~dp0"
set "PY=%~dp0_runtime\python\python.exe"
echo.
echo   CHECKING THE INSTALLATION
echo   -------------------------
echo.
if not exist "%PY%" (
  echo   [FAIL] Python runtime not found at:
  echo          %PY%
  echo   The _runtime folder must sit next to this file.
  echo.
  pause
  exit /b 1
)
echo   [ok]   Python runtime found
"%PY%" -c "import numpy, cv2, scipy, skimage, matplotlib" 2>nul
if errorlevel 1 (
  echo   [FAIL] Some packages are missing or broken. See README.md to rebuild.
  echo.
  pause
  exit /b 1
)
echo   [ok]   All packages load
for %%F in (scripts\batch_review.py scripts\organoid_quant.py scripts\well_roi.py scripts\organoid_editor.html) do (
  if not exist "%%F" (
    echo   [FAIL] Missing file: %%F
    echo.
    pause
    exit /b 1
  )
)
echo   [ok]   All scripts present
echo.
"%PY%" -c "import sys, numpy, cv2, scipy, skimage; print('   python       ', sys.version.split()[0]); print('   numpy        ', numpy.__version__); print('   opencv       ', cv2.__version__); print('   scipy        ', scipy.__version__); print('   scikit-image ', skimage.__version__)"
echo.
echo   Everything is ready. Close this window and run "Analyse organoids.bat".
echo.
pause