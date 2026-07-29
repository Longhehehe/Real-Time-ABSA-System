@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_crawl_automatic.ps1" %*
exit /b %ERRORLEVEL%
