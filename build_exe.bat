@echo off
rem ============================================================
rem  OBITO v4.0 – Windows-EXE bauen (PyInstaller, Ordner dist\OBITO)
rem  Voraussetzung: Python 3.10+ ; Ollama bleibt ein eigenes Programm.
rem ============================================================
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY ( where py >nul 2>nul && set "PY=py -3" )
if not defined PY (
  echo [FEHLER] Python nicht gefunden.
  exit /b 1
)

echo [INFO]   Tests ausfuehren ...
%PY% -m unittest discover -s tests -q
if errorlevel 1 (
  echo [FEHLER] Tests schlagen fehl - Build abgebrochen.
  exit /b 1
)

%PY% -c "import PyInstaller" >nul 2>nul || (
  echo [INFO]   Installiere PyInstaller ...
  %PY% -m pip install --upgrade pyinstaller || exit /b 1
)

echo [INFO]   Baue dist\OBITO\OBITO.exe ...
%PY% -m PyInstaller packaging\obito.spec --noconfirm --clean
if errorlevel 1 (
  echo [FEHLER] Build fehlgeschlagen.
  exit /b 1
)
copy /y OBITO.bat dist\OBITO\ >nul
echo.
echo [OK]     Fertig: dist\OBITO\OBITO.exe   (Doppelklick startet Server + HUD-Fenster)
echo          Ollama muss installiert sein: https://ollama.com/download/windows
endlocal
