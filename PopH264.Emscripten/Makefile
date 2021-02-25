LIB_NAME = PopH264

# X ?= if X env var isn't set, use this default
compiler ?= emcc
CC = ${compiler}

# osTarget is the target os eg. Nvidia / Pi4
# tsdk: the naming of the build folders are handled by the github actions
# osTarget is still used for if statements in the make file
osTarget ?= Wasm

PROJECT_PATH = /Volumes/Code/PopH264
BUILD_DIR = $(PROJECT_PATH)/BuildDirEm
SRC_PATH = $(PROJECT_PATH)/Source
SOY_PATH = $(PROJECT_PATH)/Source/SoyLib

LIB_INCLUDES = \
-I$(SRC_PATH)	\
-I$(SRC_PATH)/Broadway/Decoder	\
-I$(SRC_PATH)/Broadway/Decoder/inc	\
-I$(SOY_PATH)/src	\
-I$(SRC_PATH)/Json11	\

LIB_SRC_FILES  = \
$(SRC_PATH)/PopH264c.c \

LIB_DEFINES = \
-DTARGET_WASM	\
-DTARGET_LINUX	\

EMCC_C_FLAGS = -O3 \
-s NO_EXIT_RUNTIME=1	\
-s 'EXPORTED_FUNCTIONS=["GetVersion","HEAP8", "HEAP16", "HEAP32", "_broadwayGetMajorVersion", "_broadwayGetMinorVersion", "_broadwayInit", "_broadwayExit", "_broadwayCreateStream", "_broadwayPlayStream", "_broadwayOnHeadersDecoded", "_broadwayOnPictureDecoded"]'	\

EMCC_LINK_FLAGS = --no-entry



# -g stops symbol stripping
# -f[lag] PIC position independent code (for dyanmically loaded code)
# -rpath $ORIGIN makes the shared object runtimepath be same place as executable
#CFLAGS = -c $(LIB_DEFINES) $(LIB_INCLUDES) -std=gnu++17 
CFLAGS = -c $(LIB_DEFINES) $(LIB_INCLUDES) 
CFLAGS += $(EMCC_C_FLAGS)
LDFLAGS = $(EMCC_LINK_FLAGS)


LIB_LINK_FLAGS = $(LDFLAGS)

# gr: maybe this should include the os lib dir for x264
LIB_LINK_LIBS = 

LIB_OBJECTS=$(LIB_SRC_FILES:.c=.o)

OUT_LIB=$(BUILD_DIR)/lib${LIB_NAME}.wasm

# These files (without directory structure) get copied to build dir
OUT_FILE_SOURCES = \
$(SRC_PATH)/PopH264.h	\

# convert to Build/xxx target files
OUT_FILES = $(addprefix ${BUILD_DIR}/,$(notdir ${OUT_FILE_SOURCES}))
#$(info out files...)
#$(info ${OUT_FILES})


GithubWorkflow: $(OUT_LIB)
	$(info ::set-output name=library:$(OUT_LIB))
.PHONY: GithubWorkflow

lib: $(OUT_LIB) 
.PHONY: lib

# Copy other output files
$(OUT_FILES): $(OUT_FILE_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(info Copying output files $(OUT_FILES))
	cp $(OUT_FILE_SOURCES) $(BUILD_DIR)

$(OUT_LIB): $(LIB_OBJECTS) $(OUT_FILES)
	$(info Building library $(OUT_LIB))
	$(CC) $(LIB_LINK_FLAGS) -o $@ $(LIB_OBJECTS) $(LIB_LINK_LIBS)
	$(info Listing dependencies of $(OUT_LIB))
	#ldd $(OUT_LIB)

.cpp.o:
	$(info compiling $<)
	$(info flags=$(CFLAGS))
	$(CC) $(CFLAGS) $(LIB_INCLUDES) $< -o $@


.c.o:
	$(info compiling $<)
	$(info flags=$(CFLAGS))
	$(CC) $(CFLAGS) $(LIB_INCLUDES) $< -o $@


clean:
	# clean build files, anything .o
	$(info Cleaning files)
	$(info rm $(BUILD_DIR))
	# gr: one of these deleted .c!
	#rm -f $(LIB_OBJECTS)	
	#rm -f $(OUT_LIB)
	rm -rf $(BUILD_DIR)
