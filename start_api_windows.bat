@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" api\main.py %*
) else (
  py -3 api\main.py %*
)

popd
endlocal
