# Cross Compiling OpenGL on Raspberry Pi

The approach taken is based on this blog post: https://desertbot.io/blog/how-to-cross-compile-for-raspberry-pi

The idea is to make a use a docker container containing a MakeFile and have it spit out an executable for the pi.

The gcc compiler for the pi is too old for the version needed so the Dockerfile builds an image pulling in a compiler from [abhiTronix/raspberry-pi-cross-compilers](https://github.com/abhiTronix/raspberry-pi-cross-compilers) using wget and sets it up following the [installation instructions](https://github.com/abhiTronix/raspberry-pi-cross-compilers/wiki/Cross-Compiler:-Installation-Instructions)

A volume is shared of using PopH264 and a makefile is run manually
tsdk: this could be set up to run automatically later

The docker image is currently broken on

```bash
tar: /cross-gcc-10.1.0-pi_3+.tar.gz: Cannot open: No such file or directory
tar: Error is not recoverable: exiting now
```

`docker run -it -v ~/PopH264:/build <DockerImage>`

The makefile takes all the source files => object files => static lib

Need 2 makefiles
1 for PopH264
1 for testapp

---

## MakeFile

Based on http://mrbook.org/blog/tutorials/make/

Currently makefile runs building the object files but then breaks on

```bash
In file included from ../Source/BroadwayAll.c:5:
../Source/Broadway/Decoder/src/extraFlags.c:1:10: fatal error: extraFlags.h: No such file or directory
    1 | #include "extraFlags.h"
      |          ^~~~~~~~~~~~~~
compilation terminated.
make: *** [Makefile:48: pop] Error 1
```

Makefile has a clean command-line argument which removes all the object files... at the moment this is manually run using
`make clean`

## Notes

Creating Object files was breaking in the build with

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
