CC = arm-linux-gnueabihf-g++
SRC = ..

SOY_PATH = $(SRC)/Source/SoyLib

LOCAL_C_INCLUDES = \
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

CFLAGS = -c -Wall -DTARGET_LINUX $(LOCAL_C_INCLUDES) -std=gnu++17
LDFLAGS = -static

OBJECTS=$(LOCAL_SRC_FILES:.cpp=.o)
EXECUTABLE=pop

all: $(LOCAL_SRC_FILES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJECTS)