@echo off
rem ============================================================================================
rem  check.cmd  -  the ONE command an agent runs after every change.
rem    1. build (all targets, incl. GPU tests + benchmarks)
rem    2. nvn_tests          (pure CPU: layouts, math)
rem    3. nvn_vulkan_tests   (real GPU behaviour, validation layer if installed)
rem    4. nvn_bench          (10 repetitions, medians)  ->  bench\results\current.json
rem    5. compare.py         (vs bench\baseline.json, fails on >5%% regression)
rem  Usage:  check.cmd            full run (~2-4 min)
rem          check.cmd quick      skip benchmarks (correctness only, ~30 s)
rem  Exit code 0 = everything passed.
rem ============================================================================================
setlocal
cd /d "%~dp0"
if not defined VCINSTALLDIR call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DNVN_ENABLE_VULKAN_TESTS=ON -DNVN_ENABLE_BENCHMARKS=ON >nul || goto :fail
cmake --build build || goto :fail

echo.& echo ===== nvn_tests =====
build\nvn_tests.exe --gtest_brief=1 || goto :fail
echo.& echo ===== nvn_vulkan_tests =====
build\nvn_vulkan_tests.exe --gtest_brief=1 || goto :fail

if /i "%~1"=="quick" goto :ok
echo.& echo ===== nvn_bench =====
if not exist bench\results mkdir bench\results
set NVN_NO_VALIDATION=1
build\nvn_bench.exe --benchmark_repetitions=10 --benchmark_report_aggregates_only=true --benchmark_out=bench\results\current.json --benchmark_out_format=json >bench\results\current.txt || goto :fail
python bench\compare.py bench\baseline.json bench\results\current.json || goto :fail

:ok
echo.& echo ALL CHECKS PASSED
exit /b 0
:fail
echo.& echo CHECK FAILED
exit /b 1
