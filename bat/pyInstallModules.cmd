:: -- ----------------------------------------------------------------------
:: -- 
:: -- File: "pyInstallModules.cmd"
:: --
:: -- Args: none
:: -- 
:: -- Description: 
:: --   Install required python modules using pip.
:: -- 
:: -- History:
:: -- 2024/12/12:TomislavMatas: main 1.0.0
:: -- * Initial version.
:: -- ----------------------------------------------------------------------

@echo off
setlocal
set "VENV=%~dp0..\.venv"
call "%VENV%\Scripts\activate.bat"
call :InstallPythonModule numpy==1.26.4
call :InstallPythonModule matplotlib
call :InstallPythonModule scipy
call :InstallPythonModule gymnasium 
call :InstallPythonModule torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
call :InstallPythonModule stable_baselines3[extra]
call :InstallPythonModule pyyaml
call :InstallPythonModule loguru
call :InstallPythonModule loguru-config

goto done

:: -- sub begin
:InstallPythonModule
set args=%*
echo pip install %args% ...
pip install %args%
if errorlevel 1 (
    echo "FAIL"
    pause
) else (
    echo "OK"
) 
goto :EOF
:: -- sub end

:done
