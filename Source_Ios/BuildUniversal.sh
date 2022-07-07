# exit when any command fails
set -e

PROJECT_NAME=$1
BUILDPATH_IOS="./build/${PROJECT_NAME}_Ios"
BUILDPATH_SIM="./build/${PROJECT_NAME}_IosSimulator"
BUILDPATH_OSX="./build/${PROJECT_NAME}_Osx"
xcodebuild archive -scheme ${PROJECT_NAME}_Ios -archivePath $BUILDPATH_IOS SKIP_INSTALL=NO -sdk iphoneos
xcodebuild archive -scheme ${PROJECT_NAME}_Ios -archivePath $BUILDPATH_SIM SKIP_INSTALL=NO -sdk iphonesimulator
xcodebuild archive -scheme ${PROJECT_NAME}_Osx -archivePath $BUILDPATH_OSX SKIP_INSTALL=NO
xcodebuild -create-xcframework -framework ${BUILDPATH_IOS}.xcarchive/Products/Library/Frameworks/${PROJECT_NAME}_Ios.framework -framework ${BUILDPATH_SIM}.xcarchive/Products/Library/Frameworks/${PROJECT_NAME}_Ios.framework -framework ${BUILDPATH_OSX}.xcarchive/Products/Library/Frameworks/${PROJECT_NAME}_Osx.framework -output ${BUILT_PRODUCTS_DIR}/${FULL_PRODUCT_NAME}
