Convert to raw h264 annexb stream;
`ffmpeg -i cat.mov -vcodec copy -an -bsf:v h264_mp4toannexb cat.h264`