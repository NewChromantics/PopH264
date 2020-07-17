LIBRARY_NAME = PopH264
TESTAPP_NAME = PopH264TestApp

# X ?= if X env var isn't set, use this default
compiler ?= g++
CC = ${compiler}

# archTarget is the target architecture
# if unspecified, this becomes Linux_Linux/ pick something better :)
archTarget ?= Linux
BUILD_DIR = ../build/Linux_$(archTarget)

SRC = ..
SOY_PATH = $(SRC)/Source/SoyLib

LOCAL_C_INCLUDES = \
-I$(SRC)/Source	\
-I$(SRC)/Source/Broadway/Decoder	\
-I$(SRC)/Source/Broadway/Decoder/inc	\
-I$(SOY_PATH)/src	\
-I$(SRC)/Source/Json11	\

LOCAL_SRC_FILES  = \
$(SRC)/Source/PopH264.cpp \
$(SRC)/Source/PopH264TestData.cpp \
$(SRC)/Source/TDecoder.cpp \
$(SRC)/Source/TDecoderInstance.cpp \
$(SRC)/Source/TEncoder.cpp \
$(SRC)/Source/TEncoderInstance.cpp \
$(SRC)/Source/Json11/json11.cpp \
$(SRC)/Source/X264Encoder.cpp

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

TESTAPP_SRC_FILES =	\
$(SRC)/Source_TestApp/PopH264_TestApp.cpp	\
$(SRC)/Source_TestApp/PopH264_TestAssetGenerator.cpp	\
$(SRC)/Source/SoyLib/src/SoyFilesystem.cpp

# -g stops symbol stripping
# -f[lag] PIC position independent code (for dyanmically loaded code)
# -rpath $ORIGIN makes the shared object runtimepath be same place as executable
CFLAGS = -c -DTARGET_LINUX $(LOCAL_C_INCLUDES) -std=gnu++17 -fPIC -g
LDFLAGS = -pthread
LDFLAGS += '-Wl,-rpath,$$ORIGIN'
LDFLAGS += '-Wl,-rpath-link=$(BUILD_DIR)'

LIB_LINK_FLAGS = $(LDFLAGS) -shared
TEST_APP_LINK_FLAGS = $(LDFLAGS)

LIBRARY_LINK_LIBS = \
-lx264

ifeq ($(archTarget),nvidia)
	# from nvidia samples Rules.mk
	# ARM ABI of the target platform
	TEGRA_ARMABI ?= aarch64-linux-gnu
	TARGET_ROOTFS =
	CUDA_PATH := /usr/local/cuda

	LIB_LINK_FLAGS += -Wl,-rpath-link=$(TARGET_ROOTFS)/lib/$(TEGRA_ARMABI)
	LIB_LINK_FLAGS += -Wl,-rpath-link=$(TARGET_ROOTFS)/usr/lib/$(TEGRA_ARMABI)
	LIB_LINK_FLAGS += -Wl,-rpath-link=$(TARGET_ROOTFS)/usr/lib/$(TEGRA_ARMABI)/tegra
	LIB_LINK_FLAGS += -Wl,-rpath-link=$(TARGET_ROOTFS)/$(CUDA_PATH)/lib64

	LIBRARY_LINK_LIBS += \
		-lpthread -lv4l2 -lEGL -lGLESv2 -lX11 \
		-lnvbuf_utils -lnvjpeg -lnvosd -ldrm \
		-lcuda -lcudart \
		-lnvinfer -lnvparsers \
		-L"$(TARGET_ROOTFS)/$(CUDA_PATH)/lib64" \
		-L"$(TARGET_ROOTFS)/usr/lib/$(TEGRA_ARMABI)" \
		-L"$(TARGET_ROOTFS)/usr/lib/$(TEGRA_ARMABI)/tegra"

	LOCAL_SRC_FILES += \
		$(SRC)/Source/NvidiaEncoder.cpp \
		$(SRC)/Source/nvidia/samples/common/classes/NvV4l2Element.cpp	\
		$(SRC)/Source/nvidia/samples/common/classes/NvV4l2ElementPlane.cpp	\
		$(SRC)/Source/nvidia/samples/common/classes/NvVideoEncoder.cpp	\
		$(SRC)/Source/nvidia/samples/common/classes/NvElement.cpp	\
		$(SRC)/Source/nvidia/samples/common/classes/NvBuffer.cpp	\
		$(SRC)/Source/nvidia/samples/common/classes/NvElementProfiler.cpp	\
		$(SRC)/Source/nvidia/samples/common/classes/NvLogging.cpp	\
		$(SRC)/Source/nvidia/samples/common/classes/NvUtils.cpp	
endif


LOCAL_C_INCLUDES += \
-I$(SRC)/Source/nvidia/include/

LIB_OBJECTS=$(LOCAL_SRC_FILES:.cpp=.o)
TESTAPP_OBJECTS=$(TESTAPP_SRC_FILES:.cpp=.o)

# ARCHITECTURE=${archTarget}
# env var MACHTYPE is this systems arch type, eg; aarch64-unknown-linux-gnu
OUT_LIBRARY=$(BUILD_DIR)/lib${LIBRARY_NAME}.so
OUT_EXECUTABLE=$(BUILD_DIR)/${TESTAPP_NAME}

# gr: automate this somehow so its a list of sources, then OUT_ is just their name with no directory structire
OUT_HEADER=$(BUILD_DIR)/PopH264.h
OUT_HEADER_SRC=$(SRC)/Source/PopH264.h

exec: $(OUT_EXECUTABLE)
.PHONY: exec

lib: $(OUT_LIBRARY) 
.PHONY: lib

# Copy other output files
$(OUT_HEADER): $(OUT_HEADER_SRC)
	mkdir -p $(BUILD_DIR)
	$(info Copying $(OUT_HEADER))
	cp $(OUT_HEADER_SRC) $(OUT_HEADER)

$(OUT_EXECUTABLE): $(TESTAPP_OBJECTS) $(OUT_LIBRARY)
	echo "Building test app $(OUT_EXECUTABLE)"
	$(CC) $(TEST_APP_LINK_FLAGS) $(TESTAPP_OBJECTS) -o $@ -L$(BUILD_DIR) -l$(LIBRARY_NAME)

$(OUT_LIBRARY): $(LIB_OBJECTS) $(OUT_HEADER)
	echo "Building library $(OUT_LIBRARY)"
	$(CC) $(LIB_LINK_FLAGS) -o $@ $(LIB_OBJECTS) $(LIBRARY_LINK_LIBS)
	# ar rcs $@ $(LIB_OBJECTS)
	echo "Listing dependencies"
	ldd $(OUT_LIBRARY)

.cpp.o:
	$(CC) $(CFLAGS) $(LOCAL_C_INCLUDES) $(LIBRARY_C_INCLUDES) $< -o $@

clean:
	# clean build files, anything .o
	echo "Cleaning files"
	rm -f $(TESTAPP_OBJECTS)
	rm -f $(LIB_OBJECTS)	
	#find ../ -type f -name '*.o' -delete
	rm -f $(OUT_LIBRARY)
	rm -f $(OUT_EXECUTABLE)
