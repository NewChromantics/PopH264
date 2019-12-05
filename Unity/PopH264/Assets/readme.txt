Convert to raw h264 annexb stream;
`ffmpeg -i cat.mov -vcodec copy -an -bsf:v h264_mp4toannexb cat.h264`

Remember... need a baseline profile (need to figure out max supported!)
`ffmpeg -i cat.mov -t 9 -c:v libx264 -profile:v baseline -level 3 -an -bsf:v h264_mp4toannexb cat.h264`


https://docs.microsoft.com/en-us/azure/media-services/previous/media-services-portal-vod-get-started


Convert to baseline 
=======
`ffmpeg -i key.mp4 -c:v libx264 -profile:v baseline -level 3 key_baseline.mp4`


Convert to fragmented mp4 for streaming
==================

- Fast start is moov atom at the front
`ffmpeg -i cat_baseline.mp4 -c copy -movflags faststart cat_baseline_fragment.mp4`

- Frag keyframe is fragmented mdats
`ffmpeg -i cat_baseline.mp4 -c copy -movflags frag_keyframe+empty_moov cat_baseline_fragment.mp4`