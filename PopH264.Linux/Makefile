# X ?= if X env var isn't set, use this default
compiler ?= g++
CC = ${compiler}

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

LIB_LINK_FLAGS = $(LDFLAGS) -shared
TEST_APP_LINK_FLAGS = $(LDFLAGS)

LIBRARY_LINK_LIBS = \
-lx264

LIB_OBJECTS=$(LOCAL_SRC_FILES:.cpp=.o)
TESTAPP_OBJECTS=$(TESTAPP_SRC_FILES:.cpp=.o)

# env var MACHTYPE is this systems arch type, eg; aarch64-unknown-linux-gnu
OUT_LIBRARY=PopH264${architecture}.so
OUT_EXECUTABLE=PopH264TestApp${architecture}

GithubWorkflow: $(OUT_EXECUTABLE) $(OUT_LIBRARY)
	mkdir /build/PopH264${architecture}
	mv $(OUT_EXECUTABLE) $(OUT_LIBRARY) /build/PopH264${architecture}
.PHONY: GithubWorkflow

exec: $(OUT_EXECUTABLE)
.PHONY: exec

lib: $(OUT_LIBRARY) 
.PHONY: lib

$(OUT_EXECUTABLE): $(TESTAPP_OBJECTS) $(OUT_LIBRARY)
	echo "Building test app $(OUT_EXECUTABLE)"
	$(CC) $(TEST_APP_LINK_FLAGS) $(TESTAPP_OBJECTS) -o $@ $(OUT_LIBRARY)

$(OUT_LIBRARY): $(LIB_OBJECTS)
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
