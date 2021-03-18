#!/bin/sh

if [ $# -eq 0 ]; then
    echo "Provide an architecture to test"
    echo "Support archs are: arm64-v8a, armeabi-v7a, x86, x86_64"
    exit 1
fi

adb push "./PopH264TestApp/libs/$1/PopH264TestApp" /data/local/tmp
adb push "./PopH264TestApp/libs/$1/libc++_shared.so" /data/local/tmp
adb push "./PopH264TestApp/libs/$1/PopH264.h" /data/local/tmp
adb push "./PopH264TestApp/libs/$1/libPopH264.so" /data/local/tmp
adb shell "cd /data/local/tmp && chmod +x ./PopH264TestApp && ./PopH264TestApp"
