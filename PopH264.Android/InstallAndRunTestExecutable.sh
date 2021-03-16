#!/bin/sh

adb push ./PopH264TestApp /data/local/tmp
adb push ./libc++_shared.so /data/local/tmp
adb push ./PopH264.h /data/local/tmp
adb push ./libPopH264.so /data/local/tmp
adb shell "cd /data/local/tmp && chmod +x ./PopH264TestApp && ./PopH264TestApp"
