@if "%1" == "" goto start
@setlocal
@set userinput=%1
@if not "%1"=="store" @if not "%1"=="8.1" @if not "%userinput:~0,3%"=="10." goto usage
@endlocal

:start
@call :GetVSCommonToolsDir
@if "%VS140COMNTOOLS%"=="" goto error_no_VS140COMNTOOLSDIR

@call "%VS140COMNTOOLS%VCVarsQueryRegistry.bat" 32bit No64bit %1 %2

@if "%VSINSTALLDIR%"=="" goto error_no_VSINSTALLDIR
@if "%VCINSTALLDIR%"=="" goto error_no_VCINSTALLDIR
@if "%FrameworkDir32%"=="" goto error_no_FrameworkDIR32
@if "%FrameworkVersion32%"=="" goto error_no_FrameworkVer32
@if "%Framework40Version%"=="" goto error_no_Framework40Version

@set FrameworkDir=%FrameworkDir32%
@set FrameworkVersion=%FrameworkVersion32%

@if not "%WindowsSDK_ExecutablePath_x86%" == "" @set PATH=%WindowsSDK_ExecutablePath_x86%;%PATH%

@rem
@rem Set Windows SDK include/lib path
@rem
@if not "%WindowsSdkDir%" == "" @set PATH=%WindowsSdkDir%bin\x86;%PATH%
@if not "%WindowsSdkDir%" == "" @set INCLUDE=%WindowsSdkDir%include\%WindowsSDKVersion%shared;%WindowsSdkDir%include\%WindowsSDKVersion%um;%WindowsSdkDir%include\%WindowsSDKVersion%winrt;%INCLUDE%
@if not "%WindowsSdkDir%" == "" @set LIB=%WindowsSdkDir%lib\%WindowsSDKLibVersion%um\x86;%LIB%
@if not "%WindowsSdkDir%" == "" @set LIBPATH=%WindowsLibPath%;%ExtensionSDKDir%\Microsoft.VCLibs\14.0\References\CommonConfiguration\neutral;%LIBPATH%

@REM Set NETFXSDK include/lib path
@if not "%NETFXSDKDir%" == "" @set INCLUDE=%NETFXSDKDir%include\um;%INCLUDE%
@if not "%NETFXSDKDir%" == "" @set LIB=%NETFXSDKDir%lib\um\x86;%LIB%

@rem
@rem Set UniversalCRT include/lib path, the default is the latest installed version.
@rem
@if not "%UCRTVersion%" == "" @set INCLUDE=%UniversalCRTSdkDir%include\%UCRTVersion%\ucrt;%INCLUDE%
@if not "%UCRTVersion%" == "" @set LIB=%UniversalCRTSdkDir%lib\%UCRTVersion%\ucrt\x86;%LIB%

@rem
@rem Root of Visual Studio IDE installed files.
@rem
@set DevEnvDir=%VSINSTALLDIR%Common7\IDE\

@rem PATH
@rem ----
@if exist "%VSINSTALLDIR%Team Tools\Performance Tools" @set PATH=%VSINSTALLDIR%Team Tools\Performance Tools;%PATH%

@if exist "%ProgramFiles%\HTML Help Workshop" set PATH=%ProgramFiles%\HTML Help Workshop;%PATH%
@if exist "%ProgramFiles(x86)%\HTML Help Workshop" set PATH=%ProgramFiles(x86)%\HTML Help Workshop;%PATH%
@if exist "%VCINSTALLDIR%VCPackages" set PATH=%VCINSTALLDIR%VCPackages;%PATH%
@if exist "%FrameworkDir%%Framework40Version%" set PATH=%FrameworkDir%%Framework40Version%;%PATH%
@if exist "%FrameworkDir%%FrameworkVersion%" set PATH=%FrameworkDir%%FrameworkVersion%;%PATH%
@if exist "%VSINSTALLDIR%Common7\Tools" set PATH=%VSINSTALLDIR%Common7\Tools;%PATH%
@if exist "%VCINSTALLDIR%BIN" set PATH=%VCINSTALLDIR%BIN;%PATH%
@set PATH=%DevEnvDir%;%PATH%

@rem Add path to MSBuild Binaries
@if exist "%ProgramFiles%\MSBuild\14.0\bin" set PATH=%ProgramFiles%\MSBuild\14.0\bin;%PATH%
@if exist "%ProgramFiles(x86)%\MSBuild\14.0\bin" set PATH=%ProgramFiles(x86)%\MSBuild\14.0\bin;%PATH%


@if exist "%VSINSTALLDIR%VSTSDB\Deploy" @set PATH=%VSINSTALLDIR%VSTSDB\Deploy;%PATH%

@if not "%FSHARPINSTALLDIR%" == "" @set PATH=%FSHARPINSTALLDIR%;%PATH%

@if exist "%DevEnvDir%CommonExtensions\Microsoft\TestWindow" @set PATH=%DevEnvDir%CommonExtensions\Microsoft\TestWindow;%PATH%

@rem INCLUDE
@rem -------
@if exist "%VCINSTALLDIR%ATLMFC\INCLUDE" set INCLUDE=%VCINSTALLDIR%ATLMFC\INCLUDE;%INCLUDE%
@if exist "%VCINSTALLDIR%INCLUDE" set INCLUDE=%VCINSTALLDIR%INCLUDE;%INCLUDE%

@rem LIB
@rem ---
@if "%1" == "store" goto setstorelib
@if exist "%VCINSTALLDIR%ATLMFC\LIB" set LIB=%VCINSTALLDIR%ATLMFC\LIB;%LIB%
@if exist "%VCINSTALLDIR%LIB" set LIB=%VCINSTALLDIR%LIB;%LIB%
@goto setlibpath
:setstorelib
@if exist "%VCINSTALLDIR%LIB\store" set LIB=%VCINSTALLDIR%LIB\store;%LIB%

:setlibpath
@rem LIBPATH
@rem -------
@if "%1" == "store" goto setstorelibpath
@if exist "%VCINSTALLDIR%ATLMFC\LIB" set LIBPATH=%VCINSTALLDIR%ATLMFC\LIB;%LIBPATH%
@if exist "%VCINSTALLDIR%LIB" set LIBPATH=%VCINSTALLDIR%LIB;%LIBPATH%
@goto appendlibpath
:setstorelibpath
@if exist "%VCINSTALLDIR%LIB\store" set LIBPATH=%VCINSTALLDIR%LIB\store;%VCINSTALLDIR%LIB\store\references;%LIBPATH%
:appendlibpath
@if exist "%FrameworkDir%%Framework40Version%" set LIBPATH=%FrameworkDir%%Framework40Version%;%LIBPATH%
@set LIBPATH=%FrameworkDir%%FrameworkVersion%;%LIBPATH%

@goto end

@REM -----------------------------------------------------------------------
:GetVSCommonToolsDir
@set VS140COMNTOOLS=
@call :GetVSCommonToolsDirHelper32 HKLM > nul 2>&1
@if errorlevel 1 call :GetVSCommonToolsDirHelper32 HKCU > nul 2>&1
@if errorlevel 1 call :GetVSCommonToolsDirHelper64  HKLM > nul 2>&1
@if errorlevel 1 call :GetVSCommonToolsDirHelper64  HKCU > nul 2>&1
@exit /B 0

:GetVSCommonToolsDirHelper32
@for /F "tokens=1,2*" %%i in ('reg query "%1\SOFTWARE\Microsoft\VisualStudio\SxS\VS7" /v "14.0"') DO (
	@if "%%i"=="14.0" (
		@SET VS140COMNTOOLS=%%k
	)
)
@if "%VS140COMNTOOLS%"=="" exit /B 1
@SET VS140COMNTOOLS=%VS140COMNTOOLS%Common7\Tools\
@exit /B 0

:GetVSCommonToolsDirHelper64
@for /F "tokens=1,2*" %%i in ('reg query "%1\SOFTWARE\Wow6432Node\Microsoft\VisualStudio\SxS\VS7" /v "14.0"') DO (
	@if "%%i"=="14.0" (
		@SET VS140COMNTOOLS=%%k
	)
)
@if "%VS140COMNTOOLS%"=="" exit /B 1
@SET VS140COMNTOOLS=%VS140COMNTOOLS%Common7\Tools\
@exit /B 0

@REM -----------------------------------------------------------------------
:error_no_VS140COMNTOOLSDIR
@echo ERROR: Cannot determine the location of the VS Common Tools folder.
@goto end

:error_no_VSINSTALLDIR
@echo ERROR: Cannot determine the location of the VS installation.
@goto end

:error_no_VCINSTALLDIR
@echo ERROR: Cannot determine the location of the VC installation.
@goto end

:error_no_FrameworkDIR32
@echo ERROR: Cannot determine the location of the .NET Framework 32bit installation.
@goto end

:error_no_FrameworkVer32
@echo ERROR: Cannot determine the version of the .NET Framework 32bit installation.
@goto end

:error_no_Framework40Version
@echo ERROR: Cannot determine the .NET Framework 4.0 version.
@goto end

:usage
echo Error in script usage. The correct usage is:
echo     %0
echo   or
echo     %0 store
echo   or
echo     %0 10.0.10240.0
echo   or
echo     %0 store 10.0.10240.0

:end




@echo off
::set "lj=%~p0"
::set "lj=%lj:\= %"
::for %%a in (%lj%) do set curdir=%%a

::1-获取属性信息-manifest.ini
CALL readconfig ..\manifest.ini version vnumber
set "vnumber=%vnumber: =%"
CALL readconfig ..\manifest.ini data modelname
set "modelname=%modelname: =%"
set modelname1=%modelname:~0,21%
set modelname2=%modelname:~22,21%
set modelname3=%modelname:~-21%
CALL readconfig ..\manifest.ini id curdir
set "curdir=%curdir: =%"

echo %vnumber%
set str="%vnumber%"
set /a count=0
:GOON
for /f "delims=., tokens=1,*" %%i in (%str%) do (
	if %count%==0 (
       set vnumber_cpu=%%i
    ) 
	if %count%==1 (
       set vnumber_cpu=%vnumber_cpu%.00
    )
	if %count% GTR 1 (
       set vnumber_cpu=%vnumber_cpu%.%%i
    )
    set str="%%j"
	set /a count = %count% + 1
    goto GOON
)
echo %vnumber_cpu%

::2-下载编译依赖库
call get3rdParty.bat
cd ..
cd %curdir%

::3-编译release
rd /q/s build
mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" ..		
msbuild %curdir%.sln /p:Configuration=Release /t:Build
::msbuild %curdir%.sln /p:Configuration=Debug /t:Build
cd ..

rd /q/s build_cpu
mkdir build_cpu
cd build_cpu
cmake -G "Visual Studio 14 2015 Win64"  -DCPU_ONLY=ON ..		
msbuild %curdir%.sln /p:Configuration=Release /t:Build
::msbuild %curdir%.sln /p:Configuration=Debug /t:Build
cd ..

::4-设置版本信息
set versionnum=0.0.0.0
set versioninfo_upd=/s description %vnumber%
set versioninfo_upd=%versioninfo_upd% /s company "TAL Inc."
set versioninfo_upd=%versioninfo_upd% /s copyright "Copyright TAL(C)2018"
set versioninfo_upd=%versioninfo_upd% /s title "AI_Engine"
set versioninfo_upd=%versioninfo_upd% /s product "AI_Engine"
set versioninfo_upd=%versioninfo_upd% /pv "%versionnum%"
script_win\verpatch.exe build\bin\Release\%curdir%.dll  /va %versionnum% %versioninfo_upd%
script_win\verpatch.exe build\bin\Release\performance_testing_GPU.exe  /va %versionnum% %versioninfo_upd%

set versioninfo_upd=/s description %vnumber_cpu%
set versioninfo_upd=%versioninfo_upd% /s company "TAL Inc."
set versioninfo_upd=%versioninfo_upd% /s copyright "Copyright TAL(C)2018"
set versioninfo_upd=%versioninfo_upd% /s title "AI_Engine"
set versioninfo_upd=%versioninfo_upd% /s product "AI_Engine"
set versioninfo_upd=%versioninfo_upd% /pv "%versionnum%"
script_win\verpatch.exe build_cpu\bin\Release\%curdir%.dll  /va %versionnum% %versioninfo_upd%
script_win\verpatch.exe build_cpu\bin\Release\performance_testing_CPU.exe  /va %versionnum% %versioninfo_upd%


::5-拷贝sdk文件
set rfloder=%curdir%_%vnumber%
xcopy api\include\%curdir%.hpp release\%rfloder%\include\ /y

xcopy test\demo.cpp release\%rfloder%\demo\ /y

xcopy data\models\%modelname1% release\%rfloder%\model\ /y
xcopy data\models\%modelname2% release\%rfloder%\model\ /y
xcopy data\models\%modelname3% release\%rfloder%\model\ /y
xcopy data\config.ini release\%rfloder%\model\ /y
xcopy data\images\testing release\%rfloder%\images\testing\ /d /e /y

xcopy build\bin\Release\%curdir%.dll release\%rfloder%\libs\x64\gpu\ /y
xcopy build\bin\Release\%curdir%.lib release\%rfloder%\libs\x64\gpu\ /y

xcopy build_cpu\bin\Release\%curdir%.dll release\%rfloder%\libs\x64\cpu\ /y
xcopy build_cpu\bin\Release\%curdir%.lib release\%rfloder%\libs\x64\cpu\ /y

xcopy build\bin\Release\performance_testing_GPU.exe release\%rfloder%\libs\x64\gpu\ /y
xcopy build_cpu\bin\Release\performance_testing_CPU.exe release\%rfloder%\libs\x64\cpu\ /y

xcopy ..\3rdParty\ft_caffe\cuda9\libs\Win64\facethink_caffe_GPU.dll release\%rfloder%\libs\x64\gpu\ /y
xcopy ..\3rdParty\ft_caffe\cuda9\libs\Win64\facethink_caffe_CPU.dll release\%rfloder%\libs\x64\cpu\ /y

xcopy ..\3rdParty\tbb\lib\x64\v140\Release\tbb.dll release\%rfloder%\libs\x64\gpu\ /y
xcopy ..\3rdParty\tbb\lib\x64\v140\Release\tbb.dll release\%rfloder%\libs\x64\cpu\ /y

xcopy ..\3rdParty\opencv\lib310\opencv_world310.dll release\%rfloder%\libs\x64\gpu\ /y
xcopy ..\3rdParty\opencv\lib310\opencv_world310.dll release\%rfloder%\libs\x64\cpu\ /y

xcopy ..\3rdParty\openblas\0.2.19\Win64\bin\libgcc_s_seh-1.dll release\%rfloder%\libs\x64\cpu\ /y
xcopy ..\3rdParty\openblas\0.2.19\Win64\bin\libgfortran-3.dll release\%rfloder%\libs\x64\cpu\ /y
xcopy ..\3rdParty\openblas\0.2.19\Win64\bin\libopenblas.dll release\%rfloder%\libs\x64\cpu\ /y
xcopy ..\3rdParty\openblas\0.2.19\Win64\bin\libquadmath-0.dll release\%rfloder%\libs\x64\cpu\ /y
xcopy ..\3rdParty\openblas\0.2.19\Win64\bin\libwinpthread-1.dll release\%rfloder%\libs\x64\cpu\ /y

xcopy script_win\performance_run_gpu.bat release\%rfloder%\ /y
xcopy script_win\performance_run_cpu.bat release\%rfloder%\ /y

xcopy manifest.ini release\%rfloder%\ /y

::pause()