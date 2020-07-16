#!/bin/bash -e

xcodebuild -workspace $BUILDPROJECT/project.xcworkspace -scheme $BUILDSCHEME -showBuildSettings | grep TARGET_BUILD_DIR | sed -e 's/.*TARGET_BUILD_DIR = //')

xcodebuild -workspace $BUILDPROJECT/project.xcworkspace -scheme $BUILDSCHEME -showBuildSettings
xcodebuild -workspace $BUILDPROJECT/project.xcworkspace -list
xcodebuild -workspace $BUILDPROJECT/project.xcworkspace -scheme $BUILDSCHEME

echo "Build Directory ${BUILDPROJECT} contents"

ls $BUILDPROJECT
ls ../$BUILDPROJECT