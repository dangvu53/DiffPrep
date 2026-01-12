@echo off
REM Complete pipeline: Train -> Extract -> Test with AutoGluon

echo ========================================
echo Step 1: Train DiffPrep and save best pipeline
echo ========================================
python main.py --dataset abalone --method diffprep_fix
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ========================================
echo Step 2: Verify pipeline was saved
echo ========================================
python check_pipeline_saving.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ========================================
echo Step 3: Extract and save as Preprocessor
echo ========================================
python extract_and_save_pipeline.py --dataset abalone --method diffprep_fix
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ========================================
echo Step 4: Test with AutoGluon
echo ========================================
python evaluate_with_autogluon_v2.py --dataset abalone --method diffprep_fix --time_limit 300
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ========================================
echo COMPLETE! Check results in autogluon_results/diffprep_fix/abalone/
echo ========================================
