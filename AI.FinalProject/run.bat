@echo off
title Canteen Auto-Billing

echo ================================
echo   CANTEEN AUTO-BILLING SYSTEM
echo ================================
echo.

:: Kiem tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay Python. Hay cai Python truoc.
    pause
    exit /b
)

:: Cai thu vien neu chua co
echo [1/2] Kiem tra thu vien...
pip install -r requirements.txt -q

:: Copy model vao folder neu chua co
if exist "best_food_model6.h5" (
    if not exist "%USERPROFILE%\Downloads\best_food_model6.h5" (
        echo [2/2] Copy model vao Downloads...
        copy "best_food_model6.h5" "%USERPROFILE%\Downloads\" >nul
    )
)

echo [2/2] Khoi dong ung dung...
echo.
python canteen_gui_manual.py

if errorlevel 1 (
    echo.
    echo [LOI] Ung dung bi loi. Xem thong bao phia tren.
    pause
)
