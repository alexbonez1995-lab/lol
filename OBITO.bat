@echo off
rem ============================================================
rem  OBITO v4.0 – Starter für Windows 11
rem  Startet Ollama (falls nötig), prüft das System und startet OBITO
rem  als App (Server im Hintergrund + HUD als eigenes Fenster).
rem ============================================================
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title OBITO v4.0 – AI Engineering Nexus

set "OBITO_URL=http://127.0.0.1:8765"
set "OLLAMA_URL=http://127.0.0.1:11434"

rem ---- Python finden -------------------------------------------------
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY ( where py >nul 2>nul && set "PY=py -3" )
if not defined PY (
  echo [FEHLER] Python wurde nicht gefunden. Bitte Python 3.10 oder neuer installieren:
  echo          https://www.python.org/downloads/windows/  ^(Haken bei "Add python.exe to PATH" setzen^)
  pause
  exit /b 1
)
for /f "tokens=2 delims= " %%v in ('%PY% --version 2^>^&1') do set "PYVER=%%v"
echo [OK]     Python %PYVER%

rem ---- Ollama finden und starten ------------------------------------
where ollama >nul 2>nul
if errorlevel 1 (
  echo [FEHLER] Ollama wurde nicht gefunden. Bitte installieren: https://ollama.com/download/windows
  echo          Danach Modelle laden:  ollama pull qwen2.5:7b   und   ollama pull nomic-embed-text
  pause
  exit /b 1
)
%PY% -c "import urllib.request,sys; urllib.request.urlopen('%OLLAMA_URL%/api/tags', timeout=2); sys.exit(0)" >nul 2>nul
if errorlevel 1 (
  echo [INFO]   Ollama läuft noch nicht – wird gestartet ...
  start "Ollama" /min cmd /c "ollama serve"
  set /a TRIES=0
  :warte_ollama
  %PY% -c "import urllib.request,sys; urllib.request.urlopen('%OLLAMA_URL%/api/tags', timeout=2); sys.exit(0)" >nul 2>nul
  if errorlevel 1 (
    set /a TRIES+=1
    if !TRIES! geq 20 (
      echo [FEHLER] Ollama antwortet nicht unter %OLLAMA_URL%. Bitte "ollama serve" manuell starten.
      pause
      exit /b 1
    )
    timeout /t 1 /nobreak >nul
    goto warte_ollama
  )
)
echo [OK]     Ollama erreichbar

rem ---- Systemcheck ---------------------------------------------------
%PY% -m obito doctor
if errorlevel 1 (
  echo.
  echo [HINWEIS] Der Systemcheck meldet Probleme ^(siehe oben^). Fehlende Modelle laden, z. B.:
  echo           ollama pull qwen2.5:7b
  echo           ollama pull nomic-embed-text
  echo           Konfiguration passend zur Grafikkarte:  %PY% -m obito config --empfehlen 8
  echo.
  choice /c JN /n /m "Trotzdem starten? [J/N] "
  if errorlevel 2 exit /b 1
)

rem ---- Server + HUD-Fenster in einem Prozess ---------------------------
rem  "python -m obito app" startet den Server im Hintergrund und oeffnet das HUD
rem  als App-Fenster (Edge/Chrome ohne Browserleisten). Fenster schliessen beendet OBITO.
%PY% -c "import urllib.request,sys; urllib.request.urlopen('%OBITO_URL%/api/status', timeout=2); sys.exit(0)" >nul 2>nul
if not errorlevel 1 echo [INFO]   Ein OBITO-Server laeuft bereits - es wird nur das Fenster geoeffnet.
echo [OK]     Starte OBITO: %OBITO_URL%
%PY% -m obito app
if errorlevel 1 (
  echo.
  echo [FEHLER] OBITO wurde mit Fehler beendet. Manuell starten: %PY% -m obito app   (oder: serve)
  pause
  exit /b 1
)
endlocal
exit /b 0
