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
	echo "Build.sh: BUILD_TARGET_NAME not specified, expecting PopXyz"
	exit 1;
fi


#ADDITIONAL_BUILD_FILES=(Source/PopH264.h)
ADDITIONAL_BUILD_FILES=



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


function Build()
{
	TARGET_NAME=$1
	
	#cd $BUILD_PROJECT_FOLDER
	#make
	# build .js and .wasm output
	
	ExportedFunctions=("_PopH264_GetVersion" "_broadwayGetMajorVersion")
	ExportsJoined=$(printf ",\"%s\"" "${ExportedFunctions[@]}")
	Exports=${ExportsJoined:1}	# substring after first character
	echo Exports: $Exports
	
	OutputDirectory=$BUILD_PROJECT_FOLDER/Build
	mkdir -p $OutputDirectory
	RESULT=$?
	if [[ $RESULT -ne 0 ]]; then
		exit $RESULT
	fi

	OutputFilename=$OutputDirectory/$BUILD_TARGET_NAME.js
	echo OutputFilename:$OutputFilename
	
	SourcePath=.
	Flags="--no-entry -O3"
	Flags="${Flags} -I${SourcePath}/Source/Broadway/Decoder -I${SourcePath}/Source/Broadway/Decoder/inc"
	Flags="${Flags} -s EXPORTED_FUNCTIONS=[$Exports]"
	echo Flags: $Flags
	#emcc Source/BroadwayAll.c Source/PopH264_WasmBroadwayDecoder.c $Flags -o $BUILD_TARGET_NAME.js
	emcc $SourcePath/Source/BroadwayAll.c $SourcePath/Source/Broadway/Decoder/src/Decoder.c $SourcePath/Source/PopH264_WasmBroadwayDecoder.c $Flags -o $OutputFilename

	RESULT=$?
	if [[ $RESULT -ne 0 ]]; then
		exit $RESULT
	fi

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
