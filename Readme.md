Build Status
==========================

![Build Windows](https://github.com/NewChromantics/PopH264/workflows/Build%20Windows/badge.svg)
![Build Linux](https://github.com/NewChromantics/PopH264/workflows/Build%20Linux/badge.svg)
![Build Apple](https://github.com/NewChromantics/PopH264/workflows/Build%20Apple/badge.svg)
![Build Android](https://github.com/NewChromantics/PopH264/workflows/Build%20Android/badge.svg)
![Create Release](https://github.com/NewChromantics/PopH264/workflows/Create%20Release/badge.svg)

About
============================
PopH264 was created to provide simple & consistent access to low level/native H264 decoders and encoders across various platforms, for use in C-API apps (eg. PopEngine), and Unity. It is certainly not limited to these engines, but that is the bulk of the use.

It is designed to be incredibly simple; does absolutely no syncronisation, no CPU heavy conversions (it specifically tries to use native paths which avoid ANY conversion) with the idea that the caller will implement this on GPU on demand instead, and intends to be purely data-in-data-out.
It internally uses threads where we NEED to, but we try and avoid this, so Push-frame and Pop-frame functions are supposed to be synchronous (although the Pop-frame functions essentially just remove already-buffered data). Again, the intention is for the user to thread these for their own needs.

It is also designed to expose all possible options (encoding and decoding) for obscure edge cases, and expose as much meta information as possible (including timestamps for various parts of encoding/decoding for profiling)

History
--------------------
The project is basically a re-write of PopMovieTexture https://gitlab.com/NewChromantics/PopMovieTexture / http://popmovie.xyz which was a Unity plugin/lib which tried to decode Mp4s, handle audio streams, blit to RGB textures etc. 
Since Unity implemented their own video player the need for a simple video player was wiped out. Whilst not perfect, it does handle 99% of use cases.

PopH264 still has a niche to fill where users need very precise synchronisation, or access to raw YUV planes, custom YUV colour matrix implementation etc.

Video Containers
----------------
PopH264 does not implement any extractors, neither the native ones (which vary in support from platform to platform), nor custom implementations. Because containers are usually so simple, yet vary so wildly with implementations, it became much more apparent a high-level (c#) decode was much more useful for debugging, workarounds with odd formats etc, when the H264 streams stayed the same.

We have an MP4 decoder implementation here, https://github.com/NewChromantics/PopCodecs but you can implement your own, use libav/ffmpeg, use custom streams, etc etc. Many people use PopH264 with their own minimal container for low overheads, or simplicity.

Future
----------------
- Whilst the project is called PopH264, there is actually very little restriction in use of purely H264. The NALU splitting and specific handling of SPS/PPS is really the only restriction.
- There is little holding back PopH264 from handling VP9, HEVC/H265 etc, but there isn't currently the demand, and H264 is still by far the most widely supported format.

- The other benefit of being specific to H264, is that the project does intend to extract other meta (Macroblock information, motion vectors) cross platform.

- PopH264 is unlikely to handle Audio.

- Web build is being redone in a few ways (rewrite of the interface to broadway, Broadway removed from project as of 2025) and planned implementation of [WebCodecs](https://www.w3.org/TR/webcodecs/) and abusing WebRTC for a decoding loopback (no YUV output support though!)

- Web via Unity is very low down on my list personally, if anyone who is experienced in this field want's to help and/or guide me how to bridge js modules, interfacing and unity's integration/build, get in touch. (I have never used unity for a web build)


Sponsorship/Funding
----------------------
Whilst we do happily accept money, we currently haven't setup github sponsoring. If you wish to sponsor via a particular method, send bitcoin, leave an issue or get in touch; [graham@newchromantics.com](mailto:graham@newchromantics.com) / [@soylentgraham](http://www.twitter.com/soylentgraham)

Rather than any trickle payments, we do encourage people to ask for commissioned new features/improvements/platform support. 
Feel free to ask for them in issues, even if you have no budget. (But please submit bugs regardless!)


Financial Contributers (Thank you!)
-----------------------------
These people have already contributed money towards the project. (Get in touch ASAP if I have missed you out!)
- [Condense Reality](https://www.condensereality.com/)

Open Source Attributions
------------------------

See [ATTRIBUTIONS.md](./ATTRIBUTIONS.md)

API Documentation
=========================
The bulk of the API documentation is in PopH264.h, which is kept up-to-date as code changes.

Installation
===========================

Unity
--------------------
Unity deployment is now done via the Unity Git Package repository https://github.com/NewChromantics/PopH264.UnityPackage
- This repository is manually updated! so could be out of date. But the binaries & c# code is copied directly from the releases of this project
- In the unity package manager add https://github.com/NewChromantics/PopH264.UnityPackage
- Done!

Unreal Support
------------------
- Work-in-progress plugin is here https://github.com/NewChromantics/PopH264_UnrealPlugin

Swift/ui support
-----------------
There is a SwiftPackageManager compatible package at https://github.com/NewChromantics/PopH264.swiftpackage
- This repository is manually updated! so could be out of date. But the binaries & c# code is copied directly from the releases of this project


Platform Support
=======================

* As of 2025, PopH264 supports only built-in OS decoders/encoders
  * No longer contains any fallback solutions not provided by the OS/Platform.
  * As before, it provides access to built-in hardware decoders/encoders if available on the platform
  * Simplifies licensing as PopH264 now does not include any patented decoder/encoder algorithms itself.

* Any empty platforms are generally planned, but not yet implemented.

| Platform                                       | Built-in OS Decoding       | Built-in OS Encoding |
|------------------------------------------------|----------------------------|----------------------|
| Windows x86                                    |                            |                      |
| Windows x64                                    | MediaFoundation            | MediaFoundation      |
| Windows UWP Hololens1                          |                            |                      |
| Windows UWP Hololens2                          | MediaFoundation            | MediaFoundation      |
| Linux arm64 for Nvidia Jetson Nano             |                            | V4L2 (Nvidia)        |
| Linux arm for Raspberry PI 1,2,Zero (untested) |                            |                      |
| Linux arm for Raspberry PI 3                   |                            |                      |
| Linux x64 ubuntu                               |                            |                      |
| Osx Intel                                      | AvFoundation               | AvFoundation         |
| Osx Arm64                                      | AvFoundation               | AvFoundation         |
| Ios                                            | AvFoundation               | AvFoundation         |
| Ios Simulator                                  | Untested                   | Untested             |
| Android armeabi-v7a                            | NdkMediaCodec              |                      |
| Android x86                                    | NdkMediaCodec              |                      |
| Android x86_64                                 | NdkMediaCodec              |                      |
| Android arm64-v8a                              | NdkMediaCodec              |                      |
| Magic Leap/Luma (Linux x86)                    | MLMediaCodec Google,Nvidia |                      |
| Web                                            | WebCodecs                  |                      |
| Unity WebGL                                    |                            |                      |

Android
---------------------
- The minimum android API version supported by this binary is 21 (android 8.0), dictated by the `ANDROID_PLATFORM` env var in `/PopH264.Android/Build.sh`.
- But, android 8.0 doesn't support async buffers, and we haven't yet implemented non-async buffer access. See issue #52 https://github.com/NewChromantics/PopH264/issues/52
- We support old & new apis by building for the old platform, then loading `libmediandk.so` at runtime (which is already loaded, not yet lazy-loaded) and finding android 9/10 symbols that we use.


Build Instructions
=======================
Windows
-------------
- Open solutions and build

Osx & Ios
---------------
- Open xcodeprojects and build
- Use `PopH264_Universal` target to build `PopH264.xcframework` which contains OSX, IOS and IOS simulator frameworks.

Linux
----------------
- Compile
  - `make osTarget=arm64` (jetson,pi4 = arm64)

[Start Self Hosted Runner as a Service](https://docs.github.com/en/actions/hosting-your-own-runners/configuring-the-self-hosted-runner-application-as-a-service)

Android
----------------
- Build on OSX by pre-installing the android-ndk with brew
 - `brew install homebrew/cask/android-ndk` note: `android-ndk` seems to be stuck on r13b which doesn't support `-std=c++17`
 - Then build the android scheme in the `xcodeproj`
  
- Build on linux
 - Use this docker container https://hub.docker.com/r/simplatex/android-lightweight
 - based on this article: https://medium.com/@simplatex/how-to-build-a-lightweight-docker-container-for-android-build-c52e4e68997e
 - If needed we can build out own / modify this one but at the moment it works with no issues

- Test run
 - To debug issues there is a script in `/PopH264.Android/` called `InstallAndRunTestExecutable.sh`
 - To test attach your phone and call this script, it will automatically try to find the ABI of your phone `sh ./InstallAndRunTestExecutable.sh`
 - If the ABI isn't automatically detected, you can pass it as the first argument `sh ./InstallAndRunTestExecutable.sh armv7`
 - If you get a `No such file or directory` when trying to use the executable binary then you are probably building for the wrong architecture.
 - If there are API version mismatches with the phone, you should see [runtime] link errors
  - `CANNOT LINK EXECUTABLE "./PopH264TestApp": cannot locate symbol "AMEDIAFORMAT_KEY_CSD_AVC" referenced by "/data/local/tmp/libPopH264.so"...`

Unity Development
==================

Android
-----------------
- Depending on your version of unity you may find the platform architecture meta for each architecture ("ABI" or "Platform") gets lost and you need to set it again.
- Unity seems to currently fail to find the `.so`'s that are put into the managed dll `PopH264Package.dll` via the `asmdef`. Delete the `.asmdef` and build plugins into a non-dll.
- When running you may get `DllNotFound` exceptions;
	- It seems unity can't find the android shared libraries (`libPopH264.so`) in a mangaged dll, so remove the asmdef
	- Sometimes, it just doesn't copy the library into the apk. Double check the platform for `PopH264/armeabi-v7a/libPopH264.so` is `ArmV7` (or the correct one for your platform)
	- Sometimes, this still doesn't help and unity needs to restart.
	- Check that the lib is making it into the `.apk` by checking the libs exist in
		- `YourProject/Temp/StagingArea/libs/armeabi-v7a`
		- Rename your `.apk` to `.zip`, unzip it, it should be present in `/libs/armeabi-v7a`. If not... unity is not putting it in the right place. Check logs, restart unity, clear cache/temp/library

Unity Integration
==============
- Create a `new Pop.H264.Decoder`
- Push h264 data with your own reference frame number/time (normally you would extract this from an mp4/webm/ts/etc container, but it can be just an incrementing number)
- Check your decoder every frame for a new frame

Web Integration
==============
- Include `PopH264.js` as a module. It has been designed to not require any preprocessing.
- Example (published straight from this repository via github pages, demonstrating no need for pre-process) [SrcWeb/](SrcWeb/) (Awkward foldername because of github/jekyll publishing problems)
- Currently webcodecs is under origin trial (or enabled by a flag in chrome). Register here https://developer.chrome.com/origintrials/#/view_trial/-7811493553674125311
- WebCodecs only works under `https` (except localhost which can be http, but still requires an origin trial, with port! the example page has localhost original trial `<meta>`s)

Developer Notes
============================
The following are various notes which PopH264 handles. (Or at least, it should handle and there should be an issue covering it if not).
Or just quirks found along the way that may help other developers.

MediaFoundation
--------------------
- If you try and decode any Nalu before SPS, you get no error, but no output. This include PPS before SPS.
- If you submit an IDR keyframe twice, you should get a frame straight away. (PopH264 now always does this; `todo: option to disable this!`)

MediaFoundation Hololens 2
-----------
Some notes to save investigation (this is gathered from debugging and reading output from visual studio, but PopH264 could do with more APIs to get this information)
- `HEVC/H265` encoder takes only `NV12`
- `H264 QCom hardware encoder` takes only `NV12`
- `H264 MFT encoder` (Microsoft's software encoder) takes `NV12` `IYUV` `YV12` `YUY2`
- `VP9/VP8 hardware encoder` (extension) takes `NV12` `IYUV` `YV12` `YUY2`

Android
---------------
- Android's input & output buffers are NOT threadsafe, in that when you destroy a codec (or flush), the buffer memory is immediately invalidated and reads/writes will seg fault mid-read/write. The PopH264 android decoder now has a lock to wait for threads before destroying codec, and also with read/write threads See #57. Although it is now safe, there may be a chance we've introduced some pauses with locks. (These were all tested by making and destroying 1000's of decoders and frames in the test app)
