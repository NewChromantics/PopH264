LIB_NAME = PopH264
APP_NAME = PopH264TestApp

# X ?= if X env var isn't set, use this default
compiler ?= g++
CC = ${compiler}

# osTarget is the target os eg. Nvidia / Pi4
# tsdk: the naming of the build folders are handled by the github actions
# osTarget is still used for if statements in the make file
osTarget ?= Linux
BUILD_DIR = ../Build

PROJECT_PATH = ..
SRC_PATH = $(PROJECT_PATH)/Source
SOY_PATH = $(PROJECT_PATH)/Source/SoyLib

LIB_INCLUDES = \
-I$(SRC_PATH)	\
-I$(SRC_PATH)/Broadway/Decoder	\
-I$(SRC_PATH)/Broadway/Decoder/inc	\
-I$(SOY_PATH)/src	\
-I$(SRC_PATH)/Json11	\

LIB_SRC_FILES  = \
$(SRC_PATH)/PopH264.cpp \
$(SRC_PATH)/PopH264TestData.cpp \
$(SRC_PATH)/TDecoder.cpp \
$(SRC_PATH)/TDecoderInstance.cpp \
$(SRC_PATH)/TEncoder.cpp \
$(SRC_PATH)/TEncoderInstance.cpp \
$(SRC_PATH)/Json11/json11.cpp \
$(SRC_PATH)/X264Encoder.cpp

# soy lib files
LIB_SRC_FILES  += \
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

APP_SRC_FILES =	\
$(PROJECT_PATH)/Source_TestApp/PopH264_TestApp.cpp	\
$(PROJECT_PATH)/Source_TestApp/PopH264_TestAssetGenerator.cpp	\
$(SOY_PATH)/src/SoyFilesystem.cpp

LIB_DEFINES = \
-DTARGET_LINUX	\

# -g stops symbol stripping
# -f[lag] PIC position independent code (for dyanmically loaded code)
# -rpath $ORIGIN makes the shared object runtimepath be same place as executable
CFLAGS = -c $(LIB_DEFINES) $(LIB_INCLUDES) -std=gnu++17 -fPIC -g
LDFLAGS = -pthread
LDFLAGS += '-Wl,-rpath,$$ORIGIN'
LDFLAGS += '-Wl,-rpath-link=$(BUILD_DIR)'

LIB_LINK_FLAGS = $(LDFLAGS) -shared
APP_LINK_FLAGS = $(LDFLAGS)

# gr: maybe this should include the os lib dir for x264
LIB_LINK_LIBS = \
-lx264

ifeq ($(osTarget),Nvidia)
	# from nvidia samples Rules.mk
	# ARM ABI of the target platform
	TEGRA_ARMABI ?= aarch64-linux-gnu
	TARGET_ROOTFS =
	CUDA_PATH := /usr/local/cuda

	LIB_LINK_FLAGS += -Wl,-rpath-link=$(TARGET_ROOTFS)/lib/$(TEGRA_ARMABI)
	LIB_LINK_FLAGS += -Wl,-rpath-link=$(TARGET_ROOTFS)/usr/lib/$(TEGRA_ARMABI)
	LIB_LINK_FLAGS += -Wl,-rpath-link=$(TARGET_ROOTFS)/usr/lib/$(TEGRA_ARMABI)/tegra
	LIB_LINK_FLAGS += -Wl,-rpath-link=$(TARGET_ROOTFS)/$(CUDA_PATH)/lib64

	LIB_LINK_LIBS += \
		-lpthread -lv4l2 -lEGL -lGLESv2 -lX11 \
		-lnvbuf_utils -lnvjpeg -lnvosd -ldrm \
		-lcuda -lcudart \
		-lnvinfer -lnvparsers \
		-L"$(TARGET_ROOTFS)/$(CUDA_PATH)/lib64" \
		-L"$(TARGET_ROOTFS)/usr/lib/$(TEGRA_ARMABI)" \
		-L"$(TARGET_ROOTFS)/usr/lib/$(TEGRA_ARMABI)/tegra"

	LIB_SRC_FILES += \
		$(SRC_PATH)/NvidiaEncoder.cpp \
		$(SRC_PATH)/nvidia/samples/common/classes/NvV4l2Element.cpp	\
		$(SRC_PATH)/nvidia/samples/common/classes/NvV4l2ElementPlane.cpp	\
		$(SRC_PATH)/nvidia/samples/common/classes/NvVideoEncoder.cpp	\
		$(SRC_PATH)/nvidia/samples/common/classes/NvElement.cpp	\
		$(SRC_PATH)/nvidia/samples/common/classes/NvBuffer.cpp	\
		$(SRC_PATH)/nvidia/samples/common/classes/NvElementProfiler.cpp	\
		$(SRC_PATH)/nvidia/samples/common/classes/NvLogging.cpp	\
		$(SRC_PATH)/nvidia/samples/common/classes/NvUtils.cpp	

	LIB_INCLUDES += \
		-I$(SRC_PATH)/nvidia/include/

	LIB_DEFINES += -DENABLE_NVIDIA
	# gr: we've already defined CFLAGS so need to update it here
	CFLAGS += -DENABLE_NVIDIA
endif


LIB_OBJECTS=$(LIB_SRC_FILES:.cpp=.o)
APP_OBJECTS=$(APP_SRC_FILES:.cpp=.o)

OUT_LIB=$(BUILD_DIR)/lib${LIB_NAME}.so
OUT_APP=$(BUILD_DIR)/${APP_NAME}

# These files (without directory structure) get copied to build dir
OUT_FILE_SOURCES = \
$(SRC_PATH)/PopH264.h	\

# convert to Build/xxx target files
OUT_FILES = $(addprefix ${BUILD_DIR}/,$(notdir ${OUT_FILE_SOURCES}))
#$(info out files...)
#$(info ${OUT_FILES})


GithubWorkflow: $(OUT_APP) $(OUT_LIB)
	$(info ::set-output name=executable:$(OUT_APP))
	$(info ::set-output name=library:$(OUT_LIB))
.PHONY: GithubWorkflow

exec: $(OUT_APP)
.PHONY: exec

lib: $(OUT_LIB) 
.PHONY: lib

# Copy other output files
$(OUT_FILES): $(OUT_FILE_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(info Copying output files $(OUT_FILES))
	cp $(OUT_FILE_SOURCES) $(BUILD_DIR)

$(OUT_APP): $(APP_OBJECTS) $(OUT_LIB)
	$(info Building test app $(OUT_APP))
	mkdir -p $(BUILD_DIR)
	$(CC) $(APP_LINK_FLAGS) $(APP_OBJECTS) -o $@ -L$(BUILD_DIR) -l$(LIB_NAME)

$(OUT_LIB): $(LIB_OBJECTS) $(OUT_FILES)
	$(info Building library $(OUT_LIB))
	$(CC) $(LIB_LINK_FLAGS) -o $@ $(LIB_OBJECTS) $(LIB_LINK_LIBS)
	$(info Listing dependencies of $(OUT_LIB))
	ldd $(OUT_LIB)

.cpp.o:
	$(CC) $(CFLAGS) $(APP_INCLUDES) $(LIB_INCLUDES) $< -o $@

clean:
	# clean build files, anything .o
	$(info Cleaning files)
	rm -f $(APP_OBJECTS)
	rm -f $(LIB_OBJECTS)
	rm -f $(OUT_LIB)
	rm -f $(OUT_APP)
