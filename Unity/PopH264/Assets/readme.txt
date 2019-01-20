Convert to raw h264 annexb stream;
`ffmpeg -i cat.mov -vcodec copy -an -bsf:v h264_mp4toannexb cat.h264`

Remember... need a baseline profile (need to figure out max supported!)
`ffmpeg -i cat.mov -t 9 -c:v libx264 -profile:v baseline -level 3 -an -bsf:v h264_mp4toannexb cat.h264`


https://docs.microsoft.com/en-us/azure/media-services/previous/media-services-portal-vod-get-started

