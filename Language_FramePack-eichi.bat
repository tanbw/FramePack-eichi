@echo off
:SEL
cls
echo. 1 Japan (ja)------------------------FramePack-Eichi_endframe_ichi
echo. 2 English (en)----------------------FramePack-Eichi_endframe_ichi
echo. 3 Traditional Chinese (Zh-TW)-------FramePack-Eichi_endframe_ichi
echo. 4 Russian (ru)----------------------FramePack-Eichi_endframe_ichi
echo.
echo. 10 Japan (ja)-----------------------FramePack-Eichi_endframe_ichi_f1
echo. 11 English (en)---------------------FramePack-Eichi_endframe_ichi_f1
echo. 12 Traditional Chinese (Zh-TW)------FramePack-Eichi_endframe_ichi_f1
echo. 13 Russian (ru)---------------------FramePack-Eichi_endframe_ichi_f1
echo.
echo. 20 Japan (ja)-----------------------FramePack-Eichi_oneframe_ichi
echo. 21 English (en)---------------------FramePack-Eichi_oneframe_ichi
echo. 22 Traditional Chinese (Zh-TW)------FramePack-Eichi_oneframe_ichi
echo. 23 Russian (ru)---------------------FramePack-Eichi_oneframe_ichi
echo.
echo. 30 FramePack------------------------Original FramePack
echo. 31 FramePack-F1---------------------Original FramePack
echo.
echo. 99 Go to Official FramePack
echo. 00 Go to Official FramePack-Eichi
echo.
set /p Type=Please select language (number):
if "%Type%"=="1" goto JP-1
if "%Type%"=="2" goto EN-2
if "%Type%"=="3" goto TW-3
if "%Type%"=="4" goto RU-4
if "%Type%"=="10" goto JP-10
if "%Type%"=="11" goto EN-11
if "%Type%"=="12" goto TW-12
if "%Type%"=="13" goto RU-13
if "%Type%"=="20" goto JP-20
if "%Type%"=="21" goto EN-21
if "%Type%"=="22" goto TW-22
if "%Type%"=="23" goto RU-23
if "%Type%"=="30" goto FP
if "%Type%"=="31" goto FPF1
if "%Type%"=="99" goto FPO
if "%Type%"=="00" goto FPEO
if "%Type%"=="" goto PP

:JP-1
cls
@echo EndFrame_Ichi_Japan Language...
call environment.bat
cd %~dp0webui
python.exe endframe_ichi.py --server 127.0.0.1 --inbrowser
goto PP

:EN-2
cls
@echo EndFrame_Ichi_EnglishJapan Language...
call environment.bat
cd %~dp0webui
python.exe endframe_ichi.py --server 127.0.0.1 --inbrowser --lang en
goto PP

:TW-3
cls
@echo EndFrame_Ichi_Traditional Chinese Language...
call environment.bat
cd %~dp0webui
python.exe endframe_ichi.py --server 127.0.0.1 --inbrowser --lang zh-tw
goto PP

:RU-4
cls
@echo EndFrame_Ichi_Russian Language...
call environment.bat
cd %~dp0webui
python.exe endframe_ichi.py --server 127.0.0.1 --inbrowser --lang ru
goto PP

:JP-10
cls
@echo EndFrame_Ichi_Japan Language...
call environment.bat
cd %~dp0webui
python.exe endframe_ichi_f1.py --server 127.0.0.1 --inbrowser
goto PP

:EN-11
cls
@echo EndFrame_Ichi_EnglishJapan Language...
call environment.bat
cd %~dp0webui
python.exe endframe_ichi_f1.py --server 127.0.0.1 --inbrowser --lang en
goto PP

:TW-12
cls
@echo EndFrame_Ichi_Traditional Chinese Language...
call environment.bat
cd %~dp0webui
python.exe endframe_ichi_f1.py --server 127.0.0.1 --inbrowser --lang zh-tw
goto PP

:RU-13
cls
@echo EndFrame_Ichi_Russian Language...
call environment.bat
cd %~dp0webui
python.exe endframe_ichi_f1.py --server 127.0.0.1 --inbrowser --lang ru
goto PP

:JP-20
cls
@echo EndFrame_Ichi_Japan Language...
call environment.bat
cd %~dp0webui
python.exe oneframe_ichi.py --server 127.0.0.1 --inbrowser
goto PP

:EN-21
cls
@echo EndFrame_Ichi_EnglishJapan Language...
call environment.bat
cd %~dp0webui
python.exe oneframe_ichi.py --server 127.0.0.1 --inbrowser --lang en
goto PP

:TW-22
cls
@echo EndFrame_Ichi_Traditional Chinese Language...
call environment.bat
cd %~dp0webui
python.exe oneframe_ichi.py --server 127.0.0.1 --inbrowser --lang zh-tw
goto PP

:RU-23
cls
@echo EndFrame_Ichi_Russian Language...
call environment.bat
cd %~dp0webui
python.exe oneframe_ichi.py --server 127.0.0.1 --inbrowser --lang ru
goto PP

:FP
cls
@echo FramePack...
call environment.bat
cd %~dp0webui
python.exe demo_gradio.py --server 127.0.0.1 --inbrowser
goto PP

:FPF1
cls
@echo FramePack F1...
call environment.bat
cd %~dp0webui
python.exe demo_gradio_f1.py --server 127.0.0.1 --inbrowser
goto PP

:FPO
cls
@echo Go to FramePack Official...
Start https://github.com/lllyasviel/FramePack
goto PP

:FPEO
cls
@echo Go to FramePack Official...
Start https://github.com/git-ai-code/FramePack-eichi
goto PP

:PP
pause