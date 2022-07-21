#!/bin/sh

# abort on any command error
set -e

function InstallAndRunTestExecutable()
{
	ABI=$1
	EXE=PopH264TestApp
	
	echo "Push files to device..."
	#adb push ./PopH264 /data/local/tmp && adb push ./libc++_shared.so /data/local/tmp && adb shell "cd /data/local/tmp && chmod +x ./PopH264 && ./PopH264"
	adb push ./PopH264.Android/libPopH264/libs/$ABI/* /data/local/tmp
	#adb push ./libc++_shared.so /data/local/tmp
	#adb push ./libPopH264.so /data/local/tmp
	echo "Run exe..."
	adb shell "cd /data/local/tmp && chmod +x ./$EXE && ./$EXE"
	
	RESULT=$?
	if [[ $RESULT -ne 0 ]]; then
		exit $RESULT
	fi
}

# auto detect ABI
if [ $# -eq 0 ]; then
	echo "Detecting abi..."
	abi=$(adb shell getprop ro.product.cpu.abi)
	echo "Detected ABI = $abi"
	InstallAndRunTestExecutable $abi
else
	InstallAndRunTestExecutable $1
fi

