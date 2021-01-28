#!/bin/sh

chmod +x $SRC_PATH/TestApp/gradlew
RESULT=$?
if [[ $RESULT -ne 0 ]]; then
	exit $RESULT
fi
		
TestApp/gradlew
RESULT=$?
if [[ $RESULT -ne 0 ]]; then
	exit $RESULT
fi

adb install -r -t ./app/build/outputs/apk/debug/app-debug.apk 
RESULT=$?
if [[ $RESULT -ne 0 ]]; then
	exit $RESULT
fi

adb shell am start -n com.androidcplusplusexample.androidcplusplusexamplejava2/.MainActivity
RESULT=$?
if [[ $RESULT -ne 0 ]]; then
	exit $RESULT
fi

exit 0
