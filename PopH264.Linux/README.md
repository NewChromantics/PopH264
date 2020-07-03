# Cross Compiling OpenGL on Raspberry Pi

The approach taken is based on this blog post: https://desertbot.io/blog/how-to-cross-compile-for-raspberry-pi

The idea is to make a use a docker container containing a MakeFile and have it spit out an executable for the pi.

The gcc compiler for the pi is too old for the version needed so the Dockerfile builds an image pulling in a compiler from [abhiTronix/raspberry-pi-cross-compilers](https://github.com/abhiTronix/raspberry-pi-cross-compilers) using wget and sets it up following the [installation instructions](https://github.com/abhiTronix/raspberry-pi-cross-compilers/wiki/Cross-Compiler:-Installation-Instructions)

PopH264 is shared with the docker container and the make command re: PopH264.Linux is run automatically

`docker run -v <PATHTOFOLDER>/PopH264:/build <Docker Image>`

---

## MakeFile

Based on http://mrbook.org/blog/tutorials/make/ and https://stackoverflow.com/questions/2734719/how-to-compile-a-static-library-in-linux

The makefile takes all the source and build c files and makes objects of them all => uses the source object files to create a lib => creates an executable with this lib and the build objects => moves all objects into the local build folder

#### Note

If \$(SRC)/Source/BroadwayAll.c is in the LOCAL_SRC_FILES list be careful as there already exists a BroadwayAll.o and this command deletes it and leaves you scratchin your head with this error:

```bash
# make
make: *** No rule to make target '../Source/BroadwayAll.c', needed by 'all'.  Stop.
```

Which seems to suggest that this file isnt built with the MakeFile.

## Notes

None of the changes to the any file other than those in this folder have been pushed to the main branch

### A

Creating Object files was breaking the build with

```bash
In file included from ../Source/TEncoderInstance.cpp:21:
../Source/X264Encoder.h:82:16: error: 'x264_picture_t' has not been declared
   82 |  void   Encode(x264_picture_t* InputPicture);
      |                ^~~~~~~~~~~~~~
../Source/X264Encoder.h:92:2: error: 'x264_t' does not name a type
   92 |  x264_t*   mHandle = nullptr;
      |  ^~~~~~
../Source/X264Encoder.h:93:2: error: 'x264_param_t' does not name a type
   93 |  x264_param_t mParam = {0};
      |  ^~~~~~~~~~~~
../Source/X264Encoder.h:94:2: error: 'x264_picture_t' does not name a type
   94 |  x264_picture_t mPicture;
      |  ^~~~~~~~~~~~~~
```

This was stopped by commenting the following lines (7-9) in TEncoderInstance.cpp

```c++
// #if !defined(TARGET_ANDROID)
// #define ENABLE_X264
// #endi
```

### B

Broadway is removed from the build by deleting

```bash
$(SRC)/Source/BroadwayDecoder.cpp \
$(SRC)/Source/BroadwayAll.c \
```

from the Makefile and adding

```c++
#if !defined(TARGET_LINUX)
#define ENABLE_BROADWAY
#endif
```

to TDecoderInstance.cpp

### Addendum

[abhiTronix/raspberry-pi-cross-compilers](https://github.com/abhiTronix/raspberry-pi-cross-compilers) also has compilers for compiling on the pi

---

## Other Approaches

### dockcross

Cross compiling toolchains in Docker images.
https://github.com/dockcross/dockcross

### mmozeiko / rpi

Graphics demos and clang toochain for raspberry pi
https://github.com/mmozeiko/rpi
