:: -- ----------------------------------------------------------------------
:: -- 
:: -- File: "pyCreateDefaultVenv.cmd"
:: --
:: -- Args: none
:: -- 
:: -- Description: 
:: --   Create a dedicated python virtual environment for this project.
:: -- 
:: -- History:
:: -- 2024/12/12:TomislavMatas: main 1.0.0
:: -- * Initial version.
:: -- ----------------------------------------------------------------------

@echo off
setlocal
set "VENV_ROOT=%~dp0.."
set "VENV_PATH=%VENV_ROOT%\.venv"
@echo on
python -m venv "%VENV_PATH%
@if errorlevel 1 (
    @echo "FAILURE"
    @pause 
) else (
    @echo "OK"
)
