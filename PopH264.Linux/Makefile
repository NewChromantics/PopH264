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

TESTAPP_SRC_FILES =	\
$(SRC)/Source_TestApp/PopH264_TestApp.cpp	\
$(SRC)/Source_TestApp/PopH264_TestAssetGenerator.cpp	\
$(SRC)/Source/SoyLib/src/SoyFilesystem.cpp

CFLAGS = -c -DTARGET_LINUX $(LOCAL_C_INCLUDES) -std=gnu++17 -fPIC
LDFLAGS = '-Wl,-rpath,$$ORIGIN'

LIB_OBJECTS=$(LOCAL_SRC_FILES:.cpp=.o)
TESTAPP_OBJECTS=$(TESTAPP_SRC_FILES:.cpp=.o)

ARCHITECTURE=${archTarget}
OUT_LIBRARY=PopH264.so
OUT_EXECUTABLE=PopH264Test

exec: $(OUT_EXECUTABLE) 
.PHONY: exec

lib: $(OUT_LIBRARY) 
.PHONY: lib

$(OUT_EXECUTABLE): $(TESTAPP_OBJECTS) $(OUT_LIBRARY)
	echo "Building test app $(OUT_EXECUTABLE)"
	$(CC) $(LDFLAGS) $^ -o $@

$(OUT_LIBRARY): $(LIB_OBJECTS)
	echo "Building library $(OUT_LIBRARY)"
	$(CC) -shared -o $@ $(LIB_OBJECTS)
	# ar rcs $@ $(LIB_OBJECTS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	# clean build files, anything .o
	echo "Cleaning files"
	rm -f $(TESTAPP_OBJECTS)
	rm -f $(LIB_OBJECTS)	
	#find ../ -type f -name '*.o' -delete
	rm -f $(OUT_LIBRARY)
	rm -f $(OUT_EXECUTABLE)