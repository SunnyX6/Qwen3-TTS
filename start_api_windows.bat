@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"
set "EXIT_CODE=0"
set "ENV_DIR=%SCRIPT_DIR%py312"
set "FLASH_MODE=prompt"
set "FLASH_ARG_PRESENT=0"
set "FLASH_ARGS="

if exist "%ENV_DIR%\python.exe" (
  for %%A in (%*) do (
    if /I "%%~A"=="--flash-attn" (
      set "FLASH_MODE=on"
      set "FLASH_ARG_PRESENT=1"
    )
    if /I "%%~A"=="--no-flash-attn" (
      set "FLASH_MODE=off"
      set "FLASH_ARG_PRESENT=1"
    )
  )

  if /I "!FLASH_MODE!"=="prompt" (
    echo.
    set /p "FLASH_CHOICE=Enable FlashAttention? [y/N, N uses --no-flash-attn]: "
    if /I "!FLASH_CHOICE!"=="y" set "FLASH_MODE=on"
    if /I "!FLASH_CHOICE!"=="yes" set "FLASH_MODE=on"
    if /I not "!FLASH_MODE!"=="on" set "FLASH_MODE=off"
  )

  if /I "!FLASH_MODE!"=="on" (
    "%ENV_DIR%\python.exe" -c "import flash_attn" >nul 2>&1
    set "FLASH_IMPORT_EXIT=!ERRORLEVEL!"
    if not "!FLASH_IMPORT_EXIT!"=="0" (
      echo `flash_attn` is not installed in "%ENV_DIR%". 1>&2
      echo Install it in the current environment with: 1>&2
      echo   "%ENV_DIR%\python.exe" -m pip install flash-attn --no-build-isolation 1>&2
      echo If you do not want to install it, run this script again and choose N. 1>&2
      set "EXIT_CODE=1"
    ) else (
      if "!FLASH_ARG_PRESENT!"=="0" set "FLASH_ARGS=--flash-attn"
    )
  ) else (
    if "!FLASH_ARG_PRESENT!"=="0" set "FLASH_ARGS=--no-flash-attn"
  )

  if "!EXIT_CODE!"=="0" (
    if defined FLASH_ARGS (
      "%ENV_DIR%\python.exe" api\main.py !FLASH_ARGS! %*
    ) else (
      "%ENV_DIR%\python.exe" api\main.py %*
    )
    set "EXIT_CODE=!ERRORLEVEL!"
  )
) else (
  echo Error: missing Conda environment at "%ENV_DIR%" 1>&2
  echo Create it in the current project directory with: 1>&2
  echo   cd /d "%SCRIPT_DIR%" 1>&2
  echo   conda create --prefix "%ENV_DIR%" python=3.12 -y 1>&2
  echo   conda activate "%ENV_DIR%" 1>&2
  echo   REM install a matching PyTorch build from https://pytorch.org/get-started/locally/ 1>&2
  echo   pip install -e ".[runtime,api]" 1>&2
  echo   REM optional: if you want FlashAttention, also install: 1>&2
  echo   "%ENV_DIR%\python.exe" -m pip install flash-attn --no-build-isolation 1>&2
  set "EXIT_CODE=1"
)

if not "!EXIT_CODE!"=="0" (
  if "%~1"=="" (
    echo.
    pause
  )
)

popd
endlocal & exit /b %EXIT_CODE%
