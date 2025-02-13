#!/bin/bash
# gr: this should have been set by brew on osx
#export ANDROID_NDK_HOME="/usr/local/share/android-ndk"

#echo "env vars"
#env

# require param
#BUILD_PROJECT_FOLDER=$BUILD_TARGET_NAME.Android
BUILD_PROJECT_FOLDER="$1"
OUTPUT_DIRECTORY=$BUILD_PROJECT_FOLDER/Build

ACTION="$2"

DEFAULT_ACTION="release"

if [ "$ACTION" == "" ]; then
	echo "Defaulting build ACTION to $DEFAULT_ACTION"
	ACTION=$DEFAULT_ACTION
#	echo "Android/build.sh: No action specified"
#	exit 1;
fi


if [ "$BUILD_TARGET_NAME" == "" ]; then
	echo "Build.sh: BUILD_TARGET_NAME not specified, expecting PopXyz"
	exit 1;
fi


ADDITIONAL_BUILD_FILES=(Source_Wasm/PopH264WebApi.js Source_Wasm/Player.js)



function CopyAdditionalBuildFiles()
{
	echo "CopyAdditionalBuildFiles to $OUTPUT_DIRECTORY"

	for Filename in ${ADDITIONAL_BUILD_FILES[@]}; do
		echo "cp $Filename $OUTPUT_DIRECTORY"
		cp $Filename $OUTPUT_DIRECTORY
		RESULT=$?
		if [[ $RESULT -ne 0 ]]; then
			exit $RESULT
		fi
	done
}

function CopyBuildFilesToUnity()
{
	SRC_PATH="$OUTPUT_DIRECTORY"

	#if [ -z "$UNITY_ASSET_PLUGIN_PATH" ]; then
	#	echo "UNITY_ASSET_PLUGIN_PATH not set, skipping post-build copy of $SRC_PATH"
	#else
	#	DEST_PATH="$UNITY_ASSET_PLUGIN_PATH/"
	#	echo "Copying $SRC_PATH to $DEST_PATH"

		# -R to copy a directory (recurse)
	#	mkdir -p $DEST_PATH && cp -R $SRC_PATH $DEST_PATH

	#	RESULT=$?
	#	if [[ $RESULT -ne 0 ]]; then
	#		exit $RESULT
	#	fi
	#fi
}

function CheckResult()
{
	RESULT=$1
	if [[ $RESULT -ne 0 ]]; then
		exit $RESULT
	fi
}

function Build()
{
	TARGET_NAME=$1
	
	#cd $BUILD_PROJECT_FOLDER
	#make
	# build .js and .wasm output
	
	ExportedFunctions=("_PopH264_GetVersion")
	ExportsJoined=$(printf ",\"%s\"" "${ExportedFunctions[@]}")
	Exports=${ExportsJoined:1}	# substring after first character
	echo Exports: $Exports
	
	
	mkdir -p $OUTPUT_DIRECTORY
	CheckResult $?

	SourcePath=.

	ModuleFilename=$OUTPUT_DIRECTORY/$BUILD_TARGET_NAME.js
	DecoderFilename=$OUTPUT_DIRECTORY/Decoder.js
	WasmFilename=$OUTPUT_DIRECTORY/$BUILD_TARGET_NAME.wasm
	echo ModuleFilename:$ModuleFilename
	echo DecoderFilename:$DecoderFilename
	echo WasmFilename:$WasmFilename

	#	build final decoder js
	cat $ModuleFilename >> $DecoderFilename
	CheckResult $?

	CopyAdditionalBuildFiles $TARGET_NAME
}

#We never pass NDK_DEBUG=1 to vrlib as this generates a duplicate gdbserver
#instead the app using vrlib can set it 
if [ $ACTION == "release" ]; then
	echo "Build.sh: $ACTION..."

	Build $BUILD_TARGET_NAME

	CopyBuildFilesToUnity

	exit 0
fi


if [ $ACTION == "clean" ]; then
	cd $BUILD_PROJECT_FOLDER
	make clean
	#echo "Build.sh: Cleaning..."
	#$ANDROID_NDK_HOME/ndk-build clean NDK_DEBUG=0
	#$ANDROID_NDK_HOME/ndk-build clean NDK_DEBUG=1
	#ant clean
	exit $?
fi


# havent exit'd, don't know this command
echo "Build.sh: Unknown command $ACTION"
exit 1
