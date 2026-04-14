@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

set "EXIT_CODE=0"
set "ENV_DIR=%SCRIPT_DIR%py312"
set "DEFAULT_CHECKPOINT=Qwen3-TTS-12Hz-1.7B-Base"
set "CHECKPOINT=%DEFAULT_CHECKPOINT%"
set "DISPLAY_IP=127.0.0.1"
set "DISPLAY_PORT=7860"
set "FLASH_MODE=prompt"
set "FLASH_ARG_PRESENT=0"
set "FLASH_ARGS="
set "PORT_ARG_PRESENT=0"
set "IP_ARG_PRESENT=0"
set "POSITIONAL_CHECKPOINT_PRESENT=0"
set "EXTRA_ARGS="

if not exist "%ENV_DIR%\python.exe" (
  echo Error: missing Conda environment at "%ENV_DIR%" 1>&2
  echo Create it in the current project directory with: 1>&2
  echo   cd /d "%SCRIPT_DIR%" 1>&2
  echo   conda create --prefix "%ENV_DIR%" python=3.12 -y 1>&2
  echo   conda activate "%ENV_DIR%" 1>&2
  echo   REM install a matching PyTorch build from https://pytorch.org/get-started/locally/ 1>&2
  echo   pip install -e ".[runtime]" 1>&2
  echo   REM optional: if you want FlashAttention, also install: 1>&2
  echo   "%ENV_DIR%\python.exe" -m pip install flash-attn --no-build-isolation 1>&2
  set "EXIT_CODE=1"
  goto :finish
)

if exist "%SCRIPT_DIR%models\%DEFAULT_CHECKPOINT%" (
  set "CHECKPOINT=%SCRIPT_DIR%models\%DEFAULT_CHECKPOINT%"
)

set "EXPECT_VALUE="
for %%A in (%*) do (
  if defined EXPECT_VALUE (
    if /I "!EXPECT_VALUE!"=="checkpoint" set "CHECKPOINT=%%~A"
    if /I "!EXPECT_VALUE!"=="port" set "DISPLAY_PORT=%%~A"
    if /I "!EXPECT_VALUE!"=="ip" set "DISPLAY_IP=%%~A"
    set "EXPECT_VALUE="
  ) else (
    if "!POSITIONAL_CHECKPOINT_PRESENT!"=="0" if not "%%~A"=="" if not "%%~A"=="--checkpoint" if not "%%~A"=="-c" if not "%%~A"=="--port" if not "%%~A"=="--ip" if not "%%~A"=="--flash-attn" if not "%%~A"=="--no-flash-attn" if /I not "%%~A:~0,1"=="-" (
      set "CHECKPOINT=%%~A"
      set "POSITIONAL_CHECKPOINT_PRESENT=1"
    )
    if /I "%%~A"=="--checkpoint" set "EXPECT_VALUE=checkpoint"
    if /I "%%~A"=="-c" set "EXPECT_VALUE=checkpoint"
    if /I "%%~A"=="--flash-attn" (
      set "FLASH_MODE=on"
      set "FLASH_ARG_PRESENT=1"
    )
    if /I "%%~A"=="--no-flash-attn" (
      set "FLASH_MODE=off"
      set "FLASH_ARG_PRESENT=1"
    )
    if /I "%%~A"=="--port" (
      set "PORT_ARG_PRESENT=1"
      set "EXPECT_VALUE=port"
    )
    if /I "%%~A"=="--ip" (
      set "IP_ARG_PRESENT=1"
      set "EXPECT_VALUE=ip"
    )
    if /I "%%~A"=="--device" set "EXPECT_VALUE=skip"
    if /I "%%~A"=="--dtype" set "EXPECT_VALUE=skip"
    if /I "%%~A"=="--concurrency" set "EXPECT_VALUE=skip"
    if /I "%%~A"=="--ssl-certfile" set "EXPECT_VALUE=skip"
    if /I "%%~A"=="--ssl-keyfile" set "EXPECT_VALUE=skip"
  )
)

if /I "!FLASH_MODE!"=="prompt" (
  echo.
  set /p "FLASH_CHOICE=Enable FlashAttention for Gradio demo? [y/N, N uses --no-flash-attn]: "
  if /I "!FLASH_CHOICE!"=="y" set "FLASH_MODE=on"
  if /I "!FLASH_CHOICE!"=="yes" set "FLASH_MODE=on"
  if /I not "!FLASH_MODE!"=="on" set "FLASH_MODE=off"
)

if /I "!FLASH_MODE!"=="on" (
  if "!FLASH_ARG_PRESENT!"=="0" set "FLASH_ARGS=--flash-attn"
) else (
  if "!FLASH_ARG_PRESENT!"=="0" set "FLASH_ARGS=--no-flash-attn"
)

if "!PORT_ARG_PRESENT!"=="0" set "EXTRA_ARGS=!EXTRA_ARGS! --port 7860"
if "!IP_ARG_PRESENT!"=="0" set "EXTRA_ARGS=!EXTRA_ARGS! --ip 127.0.0.1"

echo Starting Gradio demo with checkpoint:
echo   !CHECKPOINT!
echo Open http://!DISPLAY_IP!:!DISPLAY_PORT! after startup.
echo.

"%ENV_DIR%\python.exe" -m qwen_tts.cli.demo "!CHECKPOINT!" !FLASH_ARGS! !EXTRA_ARGS! %*
set "EXIT_CODE=!ERRORLEVEL!"

:finish
if not "!EXIT_CODE!"=="0" (
  if "%~1"=="" (
    echo.
    pause
  )
)

popd
endlocal & exit /b %EXIT_CODE%
