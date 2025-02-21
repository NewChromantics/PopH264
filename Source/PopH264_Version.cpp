#include "PopH264.h"
/*
	Version API implementions.

	Split away so we can build ultra-minimal DLLs for testing platforms
*/


namespace PopH264
{
	//	1.2.0	removed access to c++ decoder object
	//	1.2.1	added encoding
	//	1.2.2	Added PopH264_DecoderAddOnNewFrameCallback
	//	1.2.3	PopH264_CreateEncoder now takes JSON instead of encoder name
	//	1.2.4	Added ProfileLevel
	//	1.2.5	Encoder now uses .Keyframe meta setting
	//	1.2.6	X264 now uses ProfileLevel + lots of x264 settings exposed. NOTE: X264 now fully removed from project as of 2025
	//	1.2.7	Added MediaFoundation decoder to windows
	//	1.2.8	Added Test data
	//	1.2.9	Added PopH264_EncoderEndOfStream
	//	1.2.10	Added PopH264_Shutdown
	//	1.2.11	Added nvidia hardware decoder + new settings
	//	1.2.12/13 Temp numbers for continious build fixes
	//	1.2.14	Fixed MediaFoundation encoder not outputing meta
	//	1.2.15	Added KeyFrameFrequency option. AVF now encodes timestamps/framenumbers better producing much smaller packets
	//	1.2.16	Nvidia encoder now outputting input meta
	//	1.3.0	Decoder now created with Json. EnumDecoders added
	//	1.3.1	Mediafoundation decoder working properly 
	//	1.3.x	Meta versions for packaging
	//	1.3.15	Fixed/fixing correct frame number output to match input of decoder
	//	1.3.17	MediaFoundation now doesn't get stuck if we try and decode PPS or frames before SPS
	//	1.3.18	Broadway doesn't get stuck if we dont process in the order SPS, PPS, Keyframe. NOTE: Broadway now fully removed from project as of 2025
	//	1.3.19	Android NDK MediaCodec implementation
	//	1.3.20	Added PopH264_PushEndOfStream API as a clear wrapper for PopH264_PushData(null)
	//	1.3.21	Added AllowBuffering option to decoder so by default LowLatency mode is on for Mediafoundation, which reduces buffering
	//	1.3.22	Version bump for github build
	//	1.3.23	Added extra meta output from decoder (Just MediaFoundation initially)
	//	1.3.24	Version bump for github release
	//	1.3.25	Android wasn't handling COLOR_FormatYUV420SemiPlanar from some devices (samsung s7), is now
	//	1.3.26	Fixed android erroring with mis-aligned/padded buffers. 
	//	1.3.27	Android now outputting ImageRect (cropping rect) meta
	//	1.3.28	Fixed various MediaFoundation memory leaks
	//	1.3.29	Fixed MediaFoundation memory leak. Added Decoder name to meta. Added Unity GetFrameAndMeta interface
	//	1.3.30	Avf (Mac & ios) no longer try and decode SEI nalus (causes -12349 error). Added option to disable this
	//	1.3.31	Added width/height/inputsize hints for android to try and get bigger input buffers; issue #48
	//	1.3.32	Added extra timer debug. Android, if given 0 as size hints, will try and extract hint sizes from SPS
	//	1.3.33	First wasm version of broadway compiled from our repository. NOTE: Broadway now fully removed from project as of 2025
	//	1.3.34	Added some extra macos/ios version number settings for mac-swift app compatibility
	//	1.3.39	Ios now converts greyscale to YUV. Greyscale is blank on ios (fine on osx)

	//	gr: use macros once linux is automated
	//const Soy::TVersion	Version(VERSION_MAJOR,VERSION_MINOR,VERSION_PATCH);
	constexpr int	VersionMajor = 1;
	constexpr int	VersionMinor = 10;
	constexpr int	VersionPatch = 0;
}


__export int32_t PopH264_GetVersion()
{
	auto Version = 0;
	Version += PopH264::VersionMajor;
	Version *= 100;

	Version += PopH264::VersionMinor;
	Version *= 100000;

	Version += PopH264::VersionPatch;

	return Version;
}
