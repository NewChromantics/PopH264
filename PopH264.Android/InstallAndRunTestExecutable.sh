#!/bin/sh

set -e

CopyAndRunBin () {
    adb push "./PopH264TestApp/libs/$1/PopH264TestApp" /data/local/tmp
    adb push "./PopH264TestApp/libs/$1/libc++_shared.so" /data/local/tmp
    adb push "./PopH264TestApp/libs/$1/PopH264.h" /data/local/tmp
    adb push "./PopH264TestApp/libs/$1/libPopH264.so" /data/local/tmp
    adb shell "cd /data/local/tmp && chmod +x ./PopH264TestApp && ./PopH264TestApp"
}

if [ $# -eq 0 ]; then
    abi=$(adb shell getprop ro.product.cpu.abi)
    echo "Detected ABI = $abi"
    CopyAndRunBin $abi
else
    CopyAndRunBin $1
fi


