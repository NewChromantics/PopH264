#!/bin/sh
export ANDROID_NDK=/usr/local/Cellar/android-ndk/r11c/

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


if [ -z "$ANDROID_API" ]; then
	ANDROID_API="23"
fi


MAXCONCURRENTBUILDS=8

echo "Android targets..."
#android list targets

echo "Update android project"
#android update project -t android-$ANDROID_API -p . -s

# set android NDK dir
if [ -z "$ANDROID_NDK" ]; then
	echo "ANDROID_NDK env var not set"
	exit 1
fi

# set android NDK dir
if [ -z "$UNITY_ASSET_PLUGIN_PATH" ]; then
	echo "UNITY_ASSET_PLUGIN_PATH env var not set"
	exit 1
fi

function BuildAbi()
{
	ANDROID_ABI=$1
	$ANDROID_NDK/ndk-build -j$MAXCONCURRENTBUILDS NDK_DEBUG=0 NDK_PROJECT_PATH=$SOURCE_ROOT/PopH264.Android/

	RESULT=$?

	if [[ $RESULT -ne 0 ]]; then
		exit $RESULT
	fi

	SRC_PATH="PopH264.Android/libs/$ANDROID_ABI/libPopH264.so"
	DEST_PATH="$UNITY_ASSET_PLUGIN_PATH/$ANDROID_ABI"
	echo "Copying $SRC_PATH to $DEST_PATH"

	mkdir -p $DEST_PATH && cp $SRC_PATH $DEST_PATH

	RESULT=$?
	if [[ $RESULT -ne 0 ]]; then
		exit $RESULT
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
	$ANDROID_NDK/ndk-build clean NDK_DEBUG=0
	#ant clean
	exit $?
fi


# havent exit'd, don't know this command
echo "Android/build.sh: Unknown command $ACTION"
exit 1
