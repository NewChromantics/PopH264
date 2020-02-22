#!/bin/sh

if [ -z "$UNITY_ASSET_PLUGIN_PATH" ]; then
	echo "UNITY_ASSET_PLUGIN_PATH env var not set"
	exit 1
fi

# new magic leap lab now sets MLSDK
if [ -z "$MAGIC_LEAP_SDK_PATH" ]; then
	echo "MAGIC_LEAP_SDK_PATH (magic leap sdk, eg /Volumes/Assets/mlsdk/v0.23.0) env var not set"
	exit 1
fi

echo "SOURCE_ROOT = $SOURCE_ROOT"
$MAGIC_LEAP_SDK_PATH/mabu -b $MABU_FILENAME -t $MABU_BUILD_TARGET --out $SOURCE_ROOT/PopH264.Lumin/$MABU_BUILD_TARGET/

RESULT=$?

if [[ $RESULT -ne 0 ]]; then
	exit $RESULT
fi

SRC_PATH="PopH264.Lumin/release_ml1/release_lumin_clang-3.8_aarch64/libPopH264.so"
DEST_PATH="$UNITY_ASSET_PLUGIN_PATH"
echo "Copying $SRC_PATH to $DEST_PATH"

mkdir -p $DEST_PATH && cp $SRC_PATH $DEST_PATH

RESULT=$?
if [[ $RESULT -ne 0 ]]; then
	exit $RESULT
fi

exit 0
