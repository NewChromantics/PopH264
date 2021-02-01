#!/bin/sh
# gr: this should have been set by brew on osx
#export ANDROID_NDK_HOME="/usr/local/share/android-ndk"

#echo "env vars"
#env

# require param
#BUILD_PROJECT_FOLDER=$BUILD_TARGET_NAME.Android
BUILD_PROJECT_FOLDER="$1"

ACTION="$2"

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
	ANDROID_API="28"
fi


if [ -z "$ANDROID_PLATFORM" ]; then
# tsdk: 24 is Minimum platform that supports ifaddrs
	ANDROID_PLATFORM="29"
fi

MAXCONCURRENTBUILDS=1

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

ADDITIONAL_BUILD_FILES=(Source/PopH264.h)


function InstallAndRunTestExecutable()
{
	#adb push ./PopH264 /data/local/tmp && adb push ./libc++_shared.so /data/local/tmp && adb shell "cd /data/local/tmp && chmod +x ./PopH264 && ./PopH264"
	adb push ./PopH264 /data/local/tmp
	adb push ./libc++_shared.so /data/local/tmp
	adb shell "cd /data/local/tmp && chmod +x ./PopH264 && ./PopH264"
	
	RESULT=$?
	if [[ $RESULT -ne 0 ]]; then
		exit $RESULT
	fi
}

function CopyAdditionalBuildFiles()
{
	ANDROID_ABI=$1
	BUILD_PATH="$BUILD_PROJECT_FOLDER/libs/$ANDROID_ABI/"
	echo "CopyAdditionalBuildFiles to $BUILD_PATH"

	for Filename in ${ADDITIONAL_BUILD_FILES[@]}; do
		echo "cp $Filename $BUILD_PATH"
		cp $Filename $BUILD_PATH
		RESULT=$?
		if [[ $RESULT -ne 0 ]]; then
			exit $RESULT
		fi
	done
}

function CopyBuildFilesToUnity()
{
	SRC_PATH="$BUILD_PROJECT_FOLDER/libs/"

	if [ -z "$UNITY_ASSET_PLUGIN_PATH" ]; then
		echo "UNITY_ASSET_PLUGIN_PATH not set, skipping post-build copy of $SRC_PATH"
	else
		DEST_PATH="$UNITY_ASSET_PLUGIN_PATH/"
		echo "Copying $SRC_PATH to $DEST_PATH"

		# -R to copy a directory (recurse)
		mkdir -p $DEST_PATH && cp -R $SRC_PATH $DEST_PATH

		RESULT=$?
		if [[ $RESULT -ne 0 ]]; then
			exit $RESULT
		fi
	fi
}

function BuildAbi()
{
	ANDROID_ABI=$1	
	
	#if [ ! -z "$ANDROID_ABI_ONLY" && "$ANDROID_ABI_ONLY"!="$ANDROID_ABI" ]; then
	#	echo "Skipping ABI [$ANDROID_ABI] (ANDROID_ABI_ONLY=$ANDROID_ABI_ONLY)"
	#	return 0
	#fi
	
	ENABLE_DEBUG_SYMBOLS=$2
	echo "ndk-build $ANDROID_ABI... DEBUG_SYMBOLS=$ENABLE_DEBUG_SYMBOLS"
	$ANDROID_NDK_HOME/ndk-build -j$MAXCONCURRENTBUILDS APP_PLATFORM=android-$ANDROID_PLATFORM ANDROID_ABI=$ANDROID_ABI NDK_DEBUG=$ENABLE_DEBUG_SYMBOLS NDK_LOG=1 V=1

	RESULT=$?

	if [[ $RESULT -ne 0 ]]; then
		exit $RESULT
	fi

	CopyAdditionalBuildFiles $ANDROID_ABI
}

#We never pass NDK_DEBUG=1 to vrlib as this generates a duplicate gdbserver
#instead the app using vrlib can set it 
if [ $ACTION == "release" ]; then
	echo "Android/build.sh: $ACTION..."

	ENABLE_DEBUG_SYMBOLS=0

	BuildAbi armeabi-v7a $ENABLE_DEBUG_SYMBOLS
	BuildAbi x86 $ENABLE_DEBUG_SYMBOLS
	BuildAbi x86_64 $ENABLE_DEBUG_SYMBOLS
	BuildAbi arm64-v8a $ENABLE_DEBUG_SYMBOLS

	CopyBuildFilesToUnity

	exit 0
fi


if [ $ACTION == "clean" ]; then
	echo "Android/build.sh: Cleaning..."
	$ANDROID_NDK_HOME/ndk-build clean NDK_DEBUG=0
	$ANDROID_NDK_HOME/ndk-build clean NDK_DEBUG=1
	#ant clean
	exit $?
fi


# havent exit'd, don't know this command
echo "Android/build.sh: Unknown command $ACTION"
exit 1
