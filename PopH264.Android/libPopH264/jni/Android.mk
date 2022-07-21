# gr: make path absolute so errors have full path
#		this makes them jump in xcode
LOCAL_PATH := $(abspath $(call my-dir))
APP_MODULE := $(BUILD_TARGET_NAME)


# extra ../ as jni is always prepended
SRC := ../../..
#$(warning $(LOCAL_PATH))	#	debug




include $(CLEAR_VARS)

# full speed arm instead of thumb
LOCAL_ARM_MODE  := arm

#include cflags.mk

#LOCAL_CFLAGS	+= -pg -DNDK_PROFILE # compile with profiling
#LOCAL_CFLAGS	+= -mfpu=neon		# ARM NEON support
LOCAL_CFLAGS	+= -DTARGET_ANDROID

#LOCAL_CFLAGS	+= -Werror			# error on warnings
LOCAL_CFLAGS	+= -Wall
LOCAL_CFLAGS	+= -Wextra
#LOCAL_CFLAGS	+= -Wlogical-op		# not part of -Wall or -Wextra
#LOCAL_CFLAGS	+= -Weffc++			# too many issues to fix for now
LOCAL_CFLAGS	+= -Wno-strict-aliasing		# TODO: need to rewrite some code
LOCAL_CFLAGS	+= -Wno-unused-parameter
LOCAL_CFLAGS	+= -Wno-missing-field-initializers	# warns on this: SwipeAction	ret = {}
LOCAL_CFLAGS	+= -Wno-reorder-ctor
LOCAL_CFLAGS	+= -Wno-multichar	# used in internal Android headers:  DISPLAY_EVENT_VSYNC = 'vsyn',
LOCAL_CFLAGS	+= -Wno-invalid-source-encoding
LOCAL_CFLAGS	+= -Wno-ignored-qualifiers
LOCAL_CFLAGS	+= -Wno-unknown-pragmas	# ignore windows pragmas
LOCAL_CFLAGS	+= -Wno-deprecated-copy-with-user-provided-copy
LOCAL_CFLAGS	+= -Wno-type-limits
LOCAL_CFLAGS	+= -Wno-invalid-offsetof
LOCAL_CFLAGS	+= -Wno-unused-but-set-variable
LOCAL_CFLAGS	+= -Wno-sign-compare
LOCAL_CFLAGS	+= -Wno-unused-variable	# wouldnt normally exclude this, but one specific case which would be messy to ifdef around

#ifeq ($(OVR_DEBUG),1)
#LOCAL_CFLAGS	+= -DOVR_BUILD_DEBUG=1 -O0 -g
#else
LOCAL_CFLAGS	+= -O3
#endif


SOY_PATH = $(SRC)/Source/SoyLib


LOCAL_C_INCLUDES += \
$(LOCAL_PATH)/$(SRC)/Source/Broadway/Decoder	\
$(LOCAL_PATH)/$(SRC)/Source/Broadway/Decoder/inc	\
$(LOCAL_PATH)/$(SOY_PATH)/src	\
$(LOCAL_PATH)/$(SRC)/Source/Json11	\


# use warning as echo
#$(warning $(LOCAL_C_INCLUDES))

#LOCAL_STATIC_LIBRARIES += android-ndk-profiler



# project files
# todo: generate from input from xcode
LOCAL_SRC_FILES  := \
$(SRC)/Source/PopH264.cpp \
$(SRC)/Source/PopH264TestData.cpp \
$(SRC)/Source/TDecoder.cpp \
$(SRC)/Source/TDecoderInstance.cpp \
$(SRC)/Source/TEncoder.cpp \
$(SRC)/Source/TEncoderInstance.cpp \
$(SRC)/Source/BroadwayDecoder.cpp \
$(SRC)/Source/BroadwayAll.c \
$(SRC)/Source/Json11/json11.cpp \
$(SRC)/Source/AndroidDecoder.cpp \

#$(SRC)/Source/AndroidMedia.cpp \


# soy lib files
LOCAL_SRC_FILES  += \
$(SOY_PATH)/src/SoyTypes.cpp \
$(SOY_PATH)/src/SoyAssert.cpp \
$(SOY_PATH)/src/SoyDebug.cpp \
$(SOY_PATH)/src/SoyPixels.cpp \
$(SOY_PATH)/src/memheap.cpp \
$(SOY_PATH)/src/SoyArray.cpp \
$(SOY_PATH)/src/SoyTime.cpp \
$(SOY_PATH)/src/SoyString.cpp \
$(SOY_PATH)/src/SoyH264.cpp \
$(SOY_PATH)/src/SoyPng.cpp \
$(SOY_PATH)/src/SoyImage.cpp \
$(SOY_PATH)/src/SoyStreamBuffer.cpp \
$(SOY_PATH)/src/SoyFourcc.cpp \
$(SOY_PATH)/src/SoyThread.cpp \
$(SOY_PATH)/src/SoyJava.cpp \
$(SOY_PATH)/src/SoyStream.cpp \
$(SOY_PATH)/src/SoyMediaFormat.cpp \
$(SOY_PATH)/src/SoyRuntimeLibrary.cpp \
$(SOY_PATH)/src/SoyPlatform.cpp \

#$(SOY_PATH)/src/SoyOpengl.cpp \
#$(SOY_PATH)/src/SoyOpenglContext.cpp \
#$(SOY_PATH)/src/SoyEvent.cpp \
#$(SOY_PATH)/src/SoyShader.cpp \
#$(SOY_PATH)/src/SoyUnity.cpp \
#$(SOY_PATH)/src/SoyBase64.cpp \
#$(SOY_PATH)/src/SoyGraphics.cpp \


#$(call import-module,android-ndk-profiler)


LOCAL_MODULE := $(APP_MODULE)_static
LOCAL_MODULE_FILENAME := lib$(APP_MODULE)	# outputs libNAME.a
include $(BUILD_STATIC_LIBRARY)




#	build shared library from the static library we just built
include $(CLEAR_VARS)

#LOCAL_LDLIBS	+= -lGLESv3			# OpenGL ES 3.0
#LOCAL_LDLIBS	+= -lEGL			# GL platform interface
LOCAL_LDLIBS  	+= -llog			# logging
#LOCAL_LDLIBS  	+= -landroid		# native windows
#LOCAL_LDLIBS	+= -lz				# For minizip
#LOCAL_LDLIBS	+= -lOpenSLES		# audio
LOCAL_LDLIBS += -lmediandk			# native/ndk mediacodec


#/Volumes/Code/PopH264/clang++:1:1: no such file or directory: '/Volumes/Code/PopH264/PopH264.Android/libPopH264/jni/PopH264'
#LOCAL_LDLIBS += $(LOCAL_PATH)/$(APP_MODULE)

# gr: i think this is the correct way to do it, but it's added to the .so link as -shared poph264.a
#	and producdes NO symbols in the output
#	LOCAL_WHOLE_STATIC_LIBRARIES though, works!
#LOCAL_STATIC_LIBRARIES += $(APP_MODULE)_static
LOCAL_WHOLE_STATIC_LIBRARIES += $(APP_MODULE)_static


LOCAL_STRIP_MODE := none
cmd-strip :=

# gr: when the test app executable tries to run, it can't find the c++shared.so next to it
#	use this to alter the rpath so it finds it
#LOCAL_LDFLAGS	+= -rdynamic
LOCAL_LDFLAGS	+= -Wl,-rpath,.

LOCAL_MODULE := $(APP_MODULE)_shared
LOCAL_MODULE_FILENAME := $(APP_MODULE) # outputs NAME.so

include $(BUILD_SHARED_LIBRARY)



#	build test app
include $(CLEAR_VARS)

LOCAL_C_INCLUDES += \
$(LOCAL_PATH)/$(SOY_PATH)/src	\
$(LOCAL_PATH)/$(SRC)/Source/

# missing
#LOCAL_WHOLE_STATIC_LIBRARIES += libsigchain
#LOCAL_LDFLAGS += \
#	-Wl,--export-dynamic \
#	-Wl,--version-script,art/sigchainlib/version-script.txt
# gr: missing, which means JNI_CreateJavaVM is missing :/
#LOCAL_LDLIBS  	+= -ljvm			# java

# native glue support (hoping this starts JVM)
LOCAL_STATIC_LIBRARIES += android_native_app_glue
#LOCAL_STATIC_LIBRARIES += ndk_helper
#LOCAL_STATIC_LIBRARIES += $(APP_MODULE)_static	# maybe should use shared?
LOCAL_SHARED_LIBRARIES += $(APP_MODULE)_shared	# use shared to determine if any symbols are missing

#LOCAL_STATIC_LIBRARIES += android-ndk-profiler
#LOCAL_SHARED_LIBRARIES := libPopH264

LOCAL_LDLIBS  	+= -llog			# logging

# gr: when the test app executable tries to run, it can't find the c++shared.so next to it
#	use this to alter the rpath so it finds it
#LOCAL_LDFLAGS	+= -rdynamic
LOCAL_LDFLAGS	+= -Wl,-rpath,.

LOCAL_SRC_FILES  := \
$(SRC)/Source_TestApp/PopH264_TestApp.cpp \

# soy lib files
LOCAL_SRC_FILES  += \
$(SOY_PATH)/src/SoyTypes.cpp \
$(SOY_PATH)/src/SoyFilesystem.cpp \

LOCAL_MODULE := $(APP_MODULE)_testapp
LOCAL_MODULE_FILENAME := $(APP_MODULE)TestApp	# may have a better place to get this target name from

include $(BUILD_EXECUTABLE)

#	this declares the module so we can use it above
$(call import-module,android/native_app_glue)


