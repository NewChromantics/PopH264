/*
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

ConvertCaffeToTrtModel is a stand-alone tool used to separate TRT
network conversion from TRT samples, which can be run offline.

The benefits of this approach are:
* Save time for TRT based application running.
* Increases accuracy when running the profiling tool against TRT-based
       application.

To generate the serialized model stream with ConvertCaffeToTrtModel
------------------------------------------------------------------

* Run ConvertCaffeToTrtModel to generate trtModel.cache in your
  current directory:

Command format:
    ConvertCaffeToTrtModel -n <network_name>
                           -l <model_name>
                           -m <mode_name>
                           -o <outBlob_name>[,<outBlob_name>]
                           -f <fp16 or fp32>
                           -b <batch size>
                           -w <workspace_size>
                           -s <store_cache_name>

For example:
    ./ConvertCaffeToTrtModel -n ../../data/Model/GoogleNet_one_class/GoogleNet_modified_oneClass_halfHD.prototxt \
                             -l ../../data/Model/GoogleNet_one_class/GoogleNet_modified_oneClass_halfHD.caffemodel \
                             -m detection -o coverage,bboxes -f fp16 -b 2 \
                             -w 115343360 -s trtModel.cache

To load the serialized model stream in your sample
------------------------------------------------------------------

1. Use TRTModelConv to create a new trtModel.cache file.

2. Use the patch described below to build a new applicaton which loads
   trtModel.cache.
3. Copy trtModel.cache to the folder of the new application.
4. Run the application

Application Patch for Loading trtModel.cache
------------------------------------------------------------------

@@ -216,13 +216,33 @@ int main(int argc, char** argv)
+#define CACHE_MODEL
+#ifdef CACHE_MODEL
+    std::ifstream trtModelFile("trtModel.cache");
+    if (trtModelFile.good())
+    {
+        printf("Using cached TRT model\n");
+        trtModelStream << trtModelFile.rdbuf();
+    }
+    else
+    {
+#endif
         caffeToTRTModel(
             model->getModelDefFileName(),
             model->getModelParaFilemName(),
             outBlobVector,
             bs,
             trtModelStream,
             ! is_fp16);
+#ifdef CACHE_MODEL
+      // cache the trt model
+      printf("Cache TRT model to trtModel.cache\n");
+      std::ofstream trtModelFile("trtModel.cache");
+      trtModelFile << trtModelStream.rdbuf();
+      trtModelFile.close();
+      trtModelStream.seekg(0, trtModelStream.beg);
+    }
+#endif

