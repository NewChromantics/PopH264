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


#LOCAL_LDLIBS	+= -lGLESv3			# OpenGL ES 3.0
#LOCAL_LDLIBS	+= -lEGL			# GL platform interface
LOCAL_LDLIBS  	+= -llog			# logging
#LOCAL_LDLIBS  	+= -landroid		# native windows
#LOCAL_LDLIBS	+= -lz				# For minizip
#LOCAL_LDLIBS	+= -lOpenSLES		# audio
LOCAL_LDLIBS += -lmediandk			# native/ndk mediacodec


# gr: when the test app executable tries to run, it can't find the c++shared.so next to it
#	use this to alter the rpath so it finds it
#LOCAL_LDFLAGS	+= -rdynamic
LOCAL_LDFLAGS	+= -Wl,-rpath,.

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


LOCAL_MODULE := $(APP_MODULE)_static
LOCAL_MODULE_FILENAME := $(APP_MODULE)	# outputs libNAME.a
include $(BUILD_STATIC_LIBRARY)




#	build shared library from the static library we just built
include $(CLEAR_VARS)
LOCAL_MODULE := $(APP_MODULE)_shared
LOCAL_MODULE_FILENAME := $(APP_MODULE) # outputs NAME.so
LOCAL_STATIC_LIBRARIES := $(APP_MODULE)_static
include $(BUILD_SHARED_LIBRARY)



#$(call import-module,android-ndk-profiler)
