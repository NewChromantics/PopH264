#!/bin/sh
#export ANDROID_NDK_HOME=/usr/local/Cellar/android-ndk/r11c/
#ANDROID_NDK_HOME

#echo "env vars"
#env

# require param
ACTION="$1"

DEFAULT_ACTION="release"

if [ "$ACTION" == "" ]; then
	echo "Defaulting build ACTION to $DEFAULT_ACTION"
	ACTION=$DEFAULT_ACTION
#	echo "Android/build.sh: No action specified"
#	exit 1;
fi


if [ "$BUILD_TARGET_NAME" == "" ]; then
	echo "Android/build.sh: BUILD_TARGET_NAME not specified, expecting PopXyz"
	exit 1;
fi


if [ -z "$ANDROID_API" ]; then
	ANDROID_API="23"
fi


if [ -z "$ANDROID_PLATFORM" ]; then
# tsdk: Minimum platform that supports ifaddrs
	ANDROID_PLATFORM="24"
fi

MAXCONCURRENTBUILDS=1
BUILD_PROJECT_FOLDER=$BUILD_TARGET_NAME.Android

# set android NDK dir
if [ -z "$ANDROID_NDK_HOME" ]; then
	echo "ANDROID_NDK_HOME env var not set"
	exit 1
fi

# to avoid errors like Your APP_BUILD_SCRIPT points to an unknown file using Android ndk-build
# https://stackoverflow.com/questions/6494567/your-app-build-script-points-to-an-unknown-file-using-android-ndk-build
# expect the env var to be set
# this could also be passed into the ndk-build command line
if [ -z "$NDK_PROJECT_PATH" ]; then
	echo "NDK_PROJECT_PATH is not set, defaulting to $BUILD_PROJECT_FOLDER..."
	export NDK_PROJECT_PATH=$BUILD_PROJECT_FOLDER
fi

function BuildAbi()
{
	ANDROID_ABI=$1
	echo "ndk-build $ANDROID_ABI..."
	$ANDROID_NDK_HOME/ndk-build -j$MAXCONCURRENTBUILDS APP_PLATFORM=android-$ANDROID_PLATFORM ANDROID_ABI=$ANDROID_ABI NDK_DEBUG=1 NDK_LOG=1

	RESULT=$?

	if [[ $RESULT -ne 0 ]]; then
		exit $RESULT
	fi

	SRC_PATH="$BUILD_PROJECT_FOLDER/libs/$ANDROID_ABI/lib$BUILD_TARGET_NAME.so"

	# set android NDK dir
	if [ -z "$UNITY_ASSET_PLUGIN_PATH" ]; then
		echo "UNITY_ASSET_PLUGIN_PATH not set, skipping post-build copy of $SRC_PATH"
	else
		DEST_PATH="$UNITY_ASSET_PLUGIN_PATH/$ANDROID_ABI"
		echo "Copying $SRC_PATH to $DEST_PATH"

		mkdir -p $DEST_PATH && cp $SRC_PATH $DEST_PATH

		RESULT=$?
		if [[ $RESULT -ne 0 ]]; then
			exit $RESULT
		fi
	fi
}

#We never pass NDK_DEBUG=1 to vrlib as this generates a duplicate gdbserver
#instead the app using vrlib can set it 
if [ $ACTION == "release" ]; then
	echo "Android/build.sh: $ACTION..."

	BuildAbi armeabi-v7a
	BuildAbi x86
	BuildAbi x86_64
	BuildAbi arm64-v8a
	exit 0
fi

if [ $ACTION == "clean" ]; then
	echo "Android/build.sh: Cleaning..."
	$ANDROID_NDK_HOME/ndk-build clean NDK_DEBUG=0
	#ant clean
	exit $?
fi


# havent exit'd, don't know this command
echo "Android/build.sh: Unknown command $ACTION"
exit 1
