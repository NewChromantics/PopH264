# Raspberry Pi
# CC = arm-linux-gnueabihf-g++

# Standard
CC = g++
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
$(SRC)/Source/Json11/json11.cpp

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

LOCAL_BUILD_FILES =	\
$(SRC)/Source_TestApp/PopH264_TestApp.cpp	\
$(SRC)/Source_TestApp/PopH264_TestAssetGenerator.cpp	\
$(SRC)/Source/SoyLib/src/SoyFilesystem.cpp

CFLAGS = -c -DTARGET_LINUX $(LOCAL_C_INCLUDES) -std=gnu++17
LDFLAGS = -static

OBJECTS=$(LOCAL_SRC_FILES:.cpp=.o) $(LOCAL_BUILD_FILES:.cpp=.o)
LIB_OBJECTS=$(LOCAL_SRC_FILES:.cpp=.o)
BUILD_OBJECTS=$(LOCAL_BUILD_FILES:.cpp=.o)

EXECUTABLE=pop
ARCHITECTURE=${archTarget}

exec: $(EXECUTABLE) clean cleanExec
.PHONY: exec

lib: PopH264.a clean
.PHONY: lib

$(EXECUTABLE): $(BUILD_OBJECTS) PopH264.a
	$(CC) $(LDFLAGS) $^ -o $@

PopH264.a: $(OBJECTS)
	ar rcs $@ $(LIB_OBJECTS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	mkdir -p ./libs/$(ARCHITECTURE)/objects/
	mv $(OBJECTS) ./libs/$(ARCHITECTURE)/objects
	mv PopH264.a ./libs/$(ARCHITECTURE)/
	echo Build complete

cleanExec:
	mv $(EXECUTABLE) ./libs/$(ARCHITECTURE)/