#!/bin/sh


function InstallAndRunTestExecutable()
{
	#adb push ./PopH264 /data/local/tmp && adb push ./libc++_shared.so /data/local/tmp && adb shell "cd /data/local/tmp && chmod +x ./PopH264 && ./PopH264"
	adb push ./PopH264TestApp /data/local/tmp
	RESULT=$?
	if [[ $RESULT -ne 0 ]]; then
		exit $RESULT
	fi
	
	adb push ./libc++_shared.so /data/local/tmp
	RESULT=$?
	if [[ $RESULT -ne 0 ]]; then
		exit $RESULT
	fi
	
	adb push ./libPopH264.so /data/local/tmp
	RESULT=$?
	if [[ $RESULT -ne 0 ]]; then
		exit $RESULT
	fi
	
	adb shell "cd /data/local/tmp && chmod +x ./PopH264TestApp && ./PopH264TestApp"
	
	RESULT=$?
	if [[ $RESULT -ne 0 ]]; then
		exit $RESULT
	fi
}

cd /Volumes/Code/PopH264/PopH264.Android/PopH264TestApp/libs/armeabi-v7a
RESULT=$?
if [[ $RESULT -ne 0 ]]; then
	exit $RESULT
fi
	
InstallAndRunTestExecutable

