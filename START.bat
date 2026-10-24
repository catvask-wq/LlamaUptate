@echo off
chcp 65001 >nul

setlocal EnableExtensions EnableDelayedExpansion

title Llama Manager Updater

cd /d "%~dp0"

set "WORKDIR=%~dp0"
set "LOGFILE=%~dp0install_log.txt"
set "MANAGER_URL=http://127.0.0.1:8000/"

if exist "%LOGFILE%" (
    del /q "%LOGFILE%" >nul 2>&1
)

goto :main


:log
>>"%LOGFILE%" echo([%~1] %~2
echo([%~1] %~2
exit /b 0


:scan
set "count=0"

call :log SYS "Scanning C: drive for llama-server.exe..."

for /f "delims=" %%i in (
    'dir /s /b /a-d "C:\llama-server.exe" 2^>nul'
) do (
    if exist "%%i" (
        set /a count+=1
        set "path[!count!]=%%~dpi"

        call :log SYS "Found [!count!]: %%~dpi"
    )
)

if "!count!"=="0" (
    call :log ERR "llama-server.exe not found on C: drive."
    echo.
    echo llama-server.exe не найден на диске C:
    echo.
    pause
    exit /b 1
)

exit /b 0


:show_paths
echo.
echo Найденные установки llama-server:
echo.

for /l %%n in (1,1,!count!) do (
    echo [%%n] !path[%%n]!
)

echo.
exit /b 0


:prompt
call :show_paths

:prompt_again
set "choice="
set "choice_num="
set "SELECTED_PATH="

set /p "choice=Введите номер установки (1-!count!): "

if not defined choice (
    echo Номер не может быть пустым.
    goto :prompt_again
)

echo(!choice!| findstr /r /x "[0-9][0-9]*" >nul

if errorlevel 1 (
    echo Введите только число.
    goto :prompt_again
)

set /a "choice_num=choice"

if !choice_num! lss 1 (
    echo Номер меньше допустимого.
    goto :prompt_again
)

if !choice_num! gtr !count! (
    echo Номер вне диапазона.
    goto :prompt_again
)

for %%N in (!choice_num!) do (
    set "SELECTED_PATH=!path[%%N]!"
)

if not defined SELECTED_PATH (
    echo Не удалось получить выбранный путь.
    goto :prompt_again
)

call :log SYS "Selected number: !choice_num!"
call :log SYS "Selected path: !SELECTED_PATH!"

exit /b 0


:check_python
call :log SYS "Checking Python..."

python --version >>"%LOGFILE%" 2>&1

if errorlevel 1 (
    call :log ERR "Python was not found in PATH."
    echo.
    echo Python не найден.
    echo Установите Python или добавьте его в PATH.
    echo.
    pause
    exit /b 1
)

exit /b 0


:check_packages
call :log SYS "Checking Python packages..."

python -c "import fastapi, uvicorn" >>"%LOGFILE%" 2>&1

if not errorlevel 1 (
    call :log SYS "LIBRARIES: OK"
    exit /b 0
)

call :log SYS "Installing Python packages..."

python -m pip install --no-cache-dir --quiet fastapi uvicorn >>"%LOGFILE%" 2>&1

if errorlevel 1 (
    call :log ERR "PIP FAILED. See install_log.txt."
    echo.
    echo Не удалось установить Python-библиотеки.
    echo Подробности находятся в install_log.txt
    echo.
    pause
    exit /b 1
)

call :log SYS "LIBRARIES: INSTALLED"

exit /b 0


:manager_is_ready
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass ^
    -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri '%MANAGER_URL%' -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" ^
    >nul 2>&1

exit /b %errorlevel%


:start_manager
call :log SYS "Starting llama_manager.py..."

rem Запускаем именно Python-приложение.
rem Оно поднимает красивый web-интерфейс на порту 8000.
start "Llama Manager" /min /b "%ComSpec%" /d /c ^
python -u "%WORKDIR%llama_manager.py" ^>^>"%LOGFILE%" 2^>^&1

exit /b 0


:wait_manager
set "MANAGER_READY="

call :log SYS "Waiting for web interface..."

for /l %%n in (1,1,30) do (
    call :manager_is_ready

    if not errorlevel 1 (
        set "MANAGER_READY=1"
        goto :wait_manager_done
    )

    call :log SYS "Waiting... %%n/30"
    timeout /t 1 /nobreak >nul
)

:wait_manager_done

if defined MANAGER_READY (
    call :log SYS "Web interface is ready."
    exit /b 0
)

call :log ERR "Web interface did not start."
exit /b 1


:open_manager
call :log SYS "Opening web interface..."

rem Открываем браузер только после ответа от FastAPI.
start "" "%MANAGER_URL%"

exit /b 0


:main
call :scan

if errorlevel 1 (
    exit /b 1
)

call :prompt

if not defined SELECTED_PATH (
    call :log ERR "Selected path is empty."
    pause
    exit /b 1
)

rem Сохраняем выбранную папку.
> "%WORKDIR%llama_path.txt" echo(!SELECTED_PATH!

if not exist "%WORKDIR%llama_path.txt" (
    call :log ERR "llama_path.txt was not created."
    pause
    exit /b 1
)

call :log SYS "Saved path to llama_path.txt"

call :check_python

if errorlevel 1 (
    exit /b 1
)

call :check_packages

if errorlevel 1 (
    exit /b 1
)

rem Если интерфейс уже работает, новый Python-процесс
rem не запускаем повторно.
call :manager_is_ready

if errorlevel 1 (
    call :start_manager
) else (
    call :log SYS "Manager is already running."
)

call :wait_manager

if errorlevel 1 (
    echo.
    echo Web-интерфейс не запустился.
    echo Проверьте файл:
    echo %LOGFILE%
    echo.
    pause
    exit /b 1
)

cls

echo.
echo =========================================================
echo =              LLAMA MANAGER STARTED                   =
echo =========================================================
echo.
echo [INFO] Selected path: !SELECTED_PATH!
echo [INFO] Saved path: %WORKDIR%llama_path.txt
echo [INFO] Manager: %MANAGER_URL%
echo [INFO] Log file: %LOGFILE%
echo.
echo [ACTION] Opening browser...
echo.
echo =========================================================
echo.

call :open_manager

pause
