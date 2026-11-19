@echo off
rem Build helper: loads MSVC env then configures (if needed) + builds.
rem Usage: build.cmd [extra cmake -D options...]
if not defined VCINSTALLDIR call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if not exist build\build.ninja cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo %*
if not "%~1"=="" cmake -B build %*
cmake --build build
