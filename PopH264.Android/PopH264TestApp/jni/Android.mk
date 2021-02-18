# gr: make path absolute so errors have full path
#		this makes them jump in xcode
LOCAL_PATH := $(abspath $(call my-dir))

$(info $(LOCAL_PATH))	#	debug
$(info $(ANDROID_ABI))	#	debug
$(info $(TARGET_ARCH_ABI))	#	debug


include $(CLEAR_VARS)
LOCAL_MODULE := libPopH264
LOCAL_SRC_FILES := ../../libPopH264/libs/$(ANDROID_ABI)/libPopH264.so
include $(PREBUILT_SHARED_LIBRARY)


# extra ../ as jni is always prepended
SRC := ../../..
#$(warning $(LOCAL_PATH))	#	debug

# gr: get this from env var
APP_MODULE := $(BUILD_TARGET_NAME)

# full speed arm instead of thumb
LOCAL_ARM_MODE  := arm

#include cflags.mk

# This file is included in all .mk files to ensure their compilation flags are in sync
# across debug and release builds.

# NOTE: this is not part of import_vrlib.mk because VRLib itself needs to have these flags
# set, but VRLib's make file cannot include import_vrlib.mk or it would be importing itself.

LOCAL_CFLAGS	:= -DANDROID_NDK
LOCAL_CFLAGS	+= -Werror			# error on warnings
LOCAL_CFLAGS	+= -Wall
LOCAL_CFLAGS	+= -Wextra
#LOCAL_CFLAGS	+= -Wlogical-op		# not part of -Wall or -Wextra
#LOCAL_CFLAGS	+= -Weffc++			# too many issues to fix for now
LOCAL_CFLAGS	+= -Wno-strict-aliasing		# TODO: need to rewrite some code
LOCAL_CFLAGS	+= -Wno-unused-parameter
LOCAL_CFLAGS	+= -Wno-missing-field-initializers	# warns on this: SwipeAction	ret = {}
LOCAL_CFLAGS	+= -Wno-multichar	# used in internal Android headers:  DISPLAY_EVENT_VSYNC = 'vsyn',
LOCAL_CFLAGS	+= -Wno-invalid-source-encoding
#LOCAL_CFLAGS	+= -pg -DNDK_PROFILE # compile with profiling
#LOCAL_CFLAGS	+= -mfpu=neon		# ARM NEON support
LOCAL_CPPFLAGS	:= -Wno-type-limits
LOCAL_CPPFLAGS	+= -Wno-invalid-offsetof

LOCAL_CFLAGS     := -Werror -DANDROID_NDK
LOCAL_CFLAGS	 += -Wno-multichar	# used in internal Android headers:  DISPLAY_EVENT_VSYNC = 'vsyn',

LOCAL_CFLAGS     := -Werror -DTARGET_ANDROID

#ifeq ($(OVR_DEBUG),1)
#LOCAL_CFLAGS	+= -DOVR_BUILD_DEBUG=1 -O0 -g
#else
LOCAL_CFLAGS	+= -O3
#endif


SOY_PATH = $(SRC)/Source/SoyLib

#--------------------------------------------------------
# Unity plugin
#--------------------------------------------------------
include $(CLEAR_VARS)

LOCAL_MODULE := $(APP_MODULE)

LOCAL_C_INCLUDES += \
$(LOCAL_PATH)/$(SOY_PATH)/src	\
$(LOCAL_PATH)/$(SRC)/Source/	


# include explicit java support
# https://stackoverflow.com/a/33945805/355753
#LOCAL_C_INCLUDES += ${JNI_H_INCLUDE}
#LOCAL_SHARED_LIBRARIES += libnativehelper

# missing
#LOCAL_WHOLE_STATIC_LIBRARIES += libsigchain
#LOCAL_LDFLAGS += \
#	-Wl,--export-dynamic \
#	-Wl,--version-script,art/sigchainlib/version-script.txt
# gr: missing, which means JNI_CreateJavaVM is missing :/
#LOCAL_LDLIBS  	+= -ljvm			# java

# native glue support (hoping this starts JVM)
LOCAL_STATIC_LIBRARIES += android_native_app_glue



# use warning as echo
#$(warning $(LOCAL_C_INCLUDES))

#LOCAL_STATIC_LIBRARIES += android-ndk-profiler

LOCAL_SHARED_LIBRARIES := libPopH264

LOCAL_LDLIBS  	+= -llog			# logging

# gr: when the test app executable tries to run, it can't find the c++shared.so next to it
#	use this to alter the rpath so it finds it
#LOCAL_LDFLAGS	+= -rdynamic
LOCAL_LDFLAGS	+= -Wl,-rpath,.

# project files
# todo: generate from input from xcode
LOCAL_SRC_FILES  := \
$(SRC)/Source_TestApp/PopH264_TestApp.cpp \


# soy lib files
LOCAL_SRC_FILES  += \
$(SOY_PATH)/src/SoyTypes.cpp \
$(SOY_PATH)/src/SoyFilesystem.cpp \


#$(warning Build executable)	#	debug
LOCAL_MODULE := "PopH264TestApp"

include $(BUILD_EXECUTABLE)


#$(call import-module,android-ndk-profiler)
$(call import-module,android/native_app_glue)
