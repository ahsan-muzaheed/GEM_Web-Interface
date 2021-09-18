:: Copyright 1998-2018 Epic Games, Inc. All Rights Reserved.
@echo off

pushd %~dp0

::call setup.bat

title GEM website

::Run node server
::If running with frontend web server and accessing outside of localhost pass in --publicIp=<ip_of_machine>
node app %*

popd
pause
