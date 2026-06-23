@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0run_workflow.ps1"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Errore! Premi un tasto per chiudere.
    pause
)
