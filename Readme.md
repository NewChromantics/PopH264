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
Whilst the project is called PopH264, there is actually very little restriction in use of purely H264. The NALU splitting and specific handling of SPS/PPS is really the only restriction, (as well as the generic Broadway fallback).
There is little holding back PopH264 from handling VP9, HEVC/H265 etc, but there isn't currently the demand, and H264 is still by far the most widely supported format.

The other benefit of being specific to H264, is that the project does intend to extract other meta (Macroblock information, motion vectors) cross platform.

PopH264 is unlikely to handle Audio.

Sponsorship/Funding
----------------------
Whilst we do happily accept money, we currently haven't setup github sponsoring. If you wish to sponsor via a particular method, send bitcoin, leave an issue or get in touch; (graham@newchromantics.com)[mailto:graham@newchromantics.com] / (@soylentgraham)[http://www.twitter.com/soylentgraham]

Rather than any trickle payments, we do encourage people to ask for commissioned new features/improvements/platform support. 
Feel free to ask for them in issues, even if you have no budget. (But please submit bugs regardless!)

Financial Contributers (Thank you!)
-----------------------------
These people have already contributed money towards the project. (Get in touch ASAP if I have missed you out!)
- https://www.condensereality.com/



API Documentation
=========================
The bulk of the API documentation is in PopH264.h, which is kept up-to-date as code changes.

Installation
===========================

Unity
--------------------
Install as a unity package using their scoped registry system;
- In your project's `ProjectName/Packages/manifest.json` add
```
"scopedRegistries": [
    {
      "name": "New Chromantics Packages",
      "url": "https://npm.pkg.github.com/@newchromantics",
      "scopes": [
        "com.newchromantics"
      ]
    }
  ]
```
- Generate a github PAT (a personal access token in your github user-settings)
- In your user directory (`~` on mac, `c:\users\yourname` on windows) add a `.upmconfig.toml` file and add an entry
```
[npmAuth."https://npm.pkg.github.com/@newchromantics"]
token = "your_personal_access_token"
email = "you@youremail.com"
alwaysAuth = true
```
- ~Add `.npmrc` to `ProjectName/Packages/`~ it seems an `.npmrc`(npm authorisation file) file is not required
- Add `"com.newchromantics.poph264": "1.3.3",` to `ProjectName/Packages/manifest.json`
- Thanks to Peter Law http://enigma23.co.uk/blog/how-to-setup-github-packages-and-unity/



Platform Support
=======================
Any empty platforms are generally planned, but not yet implemented.

| Platform       | Software Decoding | Hardware Decoding | Software Encoding | Hardware Encoding | Build Status | 
|----------------|-------------------|-------------------|-------------------|-------------------|--------------|
| Windows x86    |                   |                   |                   |                   |              |
| Windows x64    | Broadway          | MediaFoundation   |                   | MediaFoundation   |              |
| Windows UWP Hololens1 |            |                   |                   |                   |              |
| Windows UWP Hololens2 |            |                   |                   |                   |              |
| Linux arm64 for Nvidia Jetson Nano | Broadway |        | x264              | V4L2 (Nvidia)     |              |
| Linux arm for Raspberry PI 1,2,Zero (untested) | Broadway |    | x264      |                   |              |
| Linux arm for Raspberry PI 3 | Broadway |              | x264              |                   |              |
| Linux x64 ubuntu | Broadway        |                   | x264              |                   |              |
| Osx Intel      | Broadway          | AvFoundation      | x264              | AvFoundation      |              |
| Osx Arm64      | Broadway          | AvFoundation      |                  | AvFoundation      |              |
| Ios            | Broadway          | AvFoundation      |               | AvFoundation      |              |
| Ios Simulator  | Untested          | Untested          | Untested          | Untested          |              |
| Android armeabi-v7a | Broadway     | NdkMediaCodec          |                   |                   |              |
| Android x86    | Broadway          | NdkMediaCodec            |                   |                   |              |
| Android x86_64 | Broadway          | NdkMediaCodec        |                   |                   |              |
| Android arm64-v8a | Broadway       | NdkMediaCodec            |                   |                   |              |
| Magic Leap/Luma (Linux x86) | Broadway  | MLMediaCodec Google,Nvidia|      |                   |              |
| Web            | [Broadway.js*](https://github.com/SoylentGraham/Broadway)   |                   |                   |                   |              |
| Unity WebGL    |                   |                   |                   |                   |              |

* Broadway.js is now with a fork for various bug fixes and optimisations; https://github.com/SoylentGraham/Broadway

Todo:
- List Android min-api/OS level


Unity Decoder Support
-----------------------
- Included is a c# CAPI wrapper with threading support, texture2D output wrapper.
- Turning this into a package is WIP.

Unity Encoder Support
-----------------------
- Still todo, but is a CAPI which can be easily implemented 
- Turning this into a package is WIP.

Unreal Support
------------------
- Not currently planned, but CAPI which hopefully lends itself well to being a class.

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
- Install X264
  - `sudo apt-get install libx264-dev`
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

Unity Development
==================
Macos
------------
When building the plugin, build the OSX framework. The universal framework currently doesn't "install" (copy to /Unity/PopH264/Assets/PopH264)
in `PopH264.cs` disable the define `POPH264_AS_FRAMEWORK` 

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

Misc Notes
============================
The following are various notes which PopH264 handles. (Or at least, it should handle and there should be an issue covering it if not).

MediaFoundation
--------------------
- If you try and decode any Nalu before SPS, you get no error, but no output. This include PPS before SPS.
- If you submit an IDR keyframe twice, you should get a frame straight away. (PopH264 now always does this; `todo: option to disable this!`)

Broadway
-----------------
- If you try and decode an IDR keyframe once then end the stream, you will get no frame out. It requires submitting the frame a second time to get the frame out.
- Similarly, you can submit an IDR/keyframe twice and get that frame immediately output.
- SPS & PPS need to be sent before other packets, or we will get no output. #20
- If you try and decode an intra-frame before keyframe, the decoder will stop with no error and get no frame output. #21
