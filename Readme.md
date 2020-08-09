Build Status
==========================
![Build Windows](https://github.com/NewChromantics/PopH264/workflows/Build%20Windows/badge.svg)
![Build Linux](https://github.com/NewChromantics/PopH264/workflows/Build%20Linux/badge.svg)
![Build Apple](https://github.com/NewChromantics/PopH264/workflows/Build%20Apple/badge.svg)
![Create Release](https://github.com/NewChromantics/PopH264/workflows/Create%20Release/badge.svg)

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
| Osx            | Broadway          | AvFoundation      | x264              | AvFoundation      |              |
| Ios            | Broadway          | AvFoundation      | x264              | AvFoundation      |              |
| Android armeabi-v7a | Broadway     |                   |                   |                   |              |
| Android x86    | Broadway          |                   |                   |                   |              |
| Android x86_64 | Broadway          |                   |                   |                   |              |
| Android arm64-v8a | Broadway       |                   |                   |                   |              |
| Magic Leap/Luma (Linux x86) | Broadway  | MLMediaCodec Google,Nvidia|      |                   |              |
| Web            | Broadway.js       |                   |                   |                   |              |
| Unity WebGL    |                   |                   |                   |                   |              |

Unity Decoder Support
-----------------------
- Included is a c# CAPI wrapper with threading support, texture2D output wrapper.

Unity Encoder Support
-----------------------
- Still todo, but is a CAPI which can be easily implemented 

Unreal Support
------------------
- Not currently planned

Build Instructions
=======================
Windows
-------------
- Open solutions and build

Osx & Ios
---------------
- Open xcodeprojects and build


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
  - `brew install cask android-ndk`
  
- Build on linux
  - Use this docker container https://hub.docker.com/r/simplatex/android-lightweight
  - based on this article: https://medium.com/@simplatex/how-to-build-a-lightweight-docker-container-for-android-build-c52e4e68997e
  - If needed we can build out own / modify this one but at the moment it works with no issues
