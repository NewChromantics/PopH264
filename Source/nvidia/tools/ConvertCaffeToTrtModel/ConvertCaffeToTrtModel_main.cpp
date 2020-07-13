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

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

#ifdef ENABLE_TRT
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "trtModel.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

class Logger : public ILogger {
    void log(Severity severity, const char *msg) override
    {
        if (severity == Logger::Severity::kINTERNAL_ERROR ||
            severity == Logger::Severity::kERROR ||
            severity == Logger::Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class Logger gLogger;

enum net_mode { Classification, Detection } Mode;

static
void caffeToTRTModel (const std::string& deployFile,
                     const std::string& modelFile,
                     // network outputs
                     const std::vector<std::string>& outputs,
                     unsigned int batchSize,
                     unsigned int workSpaceSize,
                     IHostMemory* &trtModelStream,
                     int force_use_fp16)
{
    // create API root class - must span the lifetime of the engine usage
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    // parse the caffe model to populate the network, then set the outputs
    ICaffeParser* parser = createCaffeParser();

    bool useFp16 = builder->platformHasFastFp16();

    if (! force_use_fp16)
        useFp16 = 0;

    std::cout << "useFp16: " << useFp16 << std::endl;
    // create a 16-bit model if it's natively supported
    DataType modelDataType = useFp16 ? DataType::kHALF : DataType::kFLOAT;
    const IBlobNameToTensor *blobNameToTensor =
              parser->parse(deployFile.c_str(),
                                       modelFile.c_str(),
                                       *network,
                                       modelDataType);

    assert(blobNameToTensor != nullptr);
    // the caffe file has no notion of outputs, so we need to manually say
    // which tensors the engine should generate
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine
    builder->setMaxBatchSize(batchSize);
    builder->setMaxWorkspaceSize(workSpaceSize);

    // Eliminate the side-effect from the delay of GPU frequency boost
    builder->setMinFindIterations(3);
    builder->setAverageFindIterations(2);

    // set up the network for paired-fp16 format, only on DriveCX
    if(useFp16)
        builder->setHalf2Mode(true);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();
    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
}

static void usages(char *cmd)
{
    printf("%s\n"
           "    -n <network_name>\n"
           "    -l <model_name>\n"
           "    -m <mode_name>\n"
           "    -o <outBlob_name>[,<outBlob_name>]\n"
           "    -f <fp16 or fp32>\n"
           "    -b <batch_size>\n"
           "    -w <workspace_size>\n"
           "    -s <store_cache_name>\n", cmd);

    printf("For Example: \n");

    printf("    %s -h\n", cmd);

    printf("    %s -n ../../data/Model/GoogleNet_one_class/GoogleNet_modified_oneClass_halfHD.prototxt"
                 " -l ../../data/Model/GoogleNet_one_class/GoogleNet_modified_oneClass_halfHD.caffemodel"
                 " -m detection -o coverage,bboxes -f fp16 -b 2"
                 " -w 115343360"
                 " -s trtModel.cache\n", cmd);
}

static void parse_command(int argc, char** argv, TrtModel *trtModel)
{
    int cc = 0;
    if (argc == 1)
    {
        printf("Please use: %s -h\n", argv[0]);
        exit(0);
    }
    while ((cc = getopt (argc, argv, "n:l:m:o:f:b:w:s:h")) != -1)
    {
        switch (cc)
        {
            case 'n':
                trtModel->input_file_name = optarg;
                if (trtModel->input_file_name == NULL)
                {
                    usages(argv[0]);
                    exit(0);
                }
                break;
            case 'l':
                trtModel->input_model_name = optarg;
                if (trtModel->input_model_name == NULL)
                {
                    usages(argv[0]);
                    exit(0);
                }
                break;
            case 'm':
                trtModel->mode_name = optarg;
                if (trtModel->mode_name == NULL)
                {
                    usages(argv[0]);
                    exit(0);
                }
                if (0 == strcmp(optarg, "classification"))
                {
                    Mode = Classification;
                }
                else if (0 == strcmp(optarg, "detection"))
                {
                    Mode = Detection;
                }
                else
                {
                    usages(argv[0]);
                    exit(0);
                }
                break;
            case 'o':
                trtModel->outBlob_name = optarg;
                if (trtModel->outBlob_name== NULL)
                {
                    usages(argv[0]);
                    exit(0);
                }
                break;
            case 'f':
                trtModel->fp_mode_p = optarg;
                if (trtModel->fp_mode_p == NULL)
                {
                    usages(argv[0]);
                    exit(0);
                }
                if (0 == strcmp(optarg, "fp16"))
                {
                    trtModel->is_fp16 = 1;
                }
                else if (0 == strcmp(optarg, "fp32"))
                {
                    trtModel->is_fp16 = 0;
                }
                else
                {
                    usages(argv[0]);
                    exit(0);
                }
                break;
            case 'b':
                trtModel->batch_size_p = optarg;
                if (trtModel->batch_size_p == NULL)
                {
                    usages(argv[0]);
                    exit(0);
                }
                trtModel->batch_size = atoi(optarg);
                if (trtModel->batch_size == 0)
                {
                    printf ("batch size cannot be 0\n");
                    exit(0);
                }
                break;
            case 'w':
                trtModel->workspace_size_p = optarg;
                if (trtModel->workspace_size_p == NULL)
                {
                    usages(argv[0]);
                    exit(0);
                }
                trtModel->workspace_size = atoi(optarg);
                if (trtModel->workspace_size == 0)
                {
                    printf ("workspace size cannot be 0\n");
                    exit(0);
                }
                break;
            case 's':
                trtModel->store_cached_file_name = optarg;
                if (trtModel->store_cached_file_name == NULL)
                {
                    usages(argv[0]);
                    exit(0);
                }
                break;
            case 'h':
                usages(argv[0]);
                exit(0);
                break;
            case '?':
                if (optopt == 'n' || optopt == 'f' || optopt == 'b')
                    fprintf (stderr, "Option -%c requires an argument.\n",
                               optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr, "Unknown option character `\\x%x'.\n",
                               optopt);
                usages(argv[0]);
                exit(0);
            default:
                usages(argv[0]);
                exit(0);
        }
    }
    std::cout << "input_file_name: " << trtModel->input_file_name << std::endl;
    std::cout << "input_model_name: " << trtModel->input_model_name << std::endl;
    std::cout << "mode_name: " << trtModel->mode_name << std::endl;
    std::cout << "outBlob_name: " << trtModel->outBlob_name << std::endl;
    std::cout << "is_fp16: " << trtModel->is_fp16 << std::endl;
    std::cout << "batch_size: " << trtModel->batch_size << std::endl;
    std::cout << "workspace_size: " << trtModel->workspace_size << std::endl;
    std::cout << "store_cached_file_name: " << trtModel->store_cached_file_name << std::endl;
    return;
}
#endif
int main(int argc, char** argv)
{
#ifdef ENABLE_TRT
    struct timeval tv;
    struct timezone tz;
    struct timeval tv1;
    struct timezone tz1;
    struct TrtModel trtModel;

    parse_command(argc, argv, &trtModel);

    std::cout << "Building a GPU inference engine with batch size = "
              << trtModel.batch_size<< std::endl;

    IHostMemory *trtModelStream{nullptr};

    // collect output blobs information
    std::vector < std::string > outBlobVector;
    if (Mode == Detection)
    {
        trtModel.ptr_p = strchr(trtModel.outBlob_name,',');
        if (trtModel.ptr_p == NULL)
        {
            usages(argv[0]);
            exit(0);
        }

        memset(trtModel.ptr,0,sizeof(trtModel.ptr));
        memcpy(trtModel.ptr, trtModel.outBlob_name,
                (trtModel.ptr_p-trtModel.outBlob_name));
        std::cout << "outBlob_name: "<<trtModel.ptr<< std::endl;
        outBlobVector.push_back(trtModel.ptr);

        memset(trtModel.ptr,0,sizeof(trtModel.ptr));
        memcpy(trtModel.ptr, trtModel.ptr_p+1, strlen(trtModel.ptr_p+1));
        std::cout << "outBlob_name: "<<trtModel.ptr<< std::endl;
        outBlobVector.push_back(trtModel.ptr);
    }
    else
    {
        std::cout << "outBlob_name: "<< trtModel.outBlob_name << std::endl;
        outBlobVector.push_back(trtModel.outBlob_name);
    }

    gettimeofday(&tv, &tz);
    // convert to trt model
    caffeToTRTModel(
        trtModel.input_file_name,
        trtModel.input_model_name,
        outBlobVector,
        trtModel.batch_size,
        trtModel.workspace_size,
        trtModelStream,
        trtModel.is_fp16);
    gettimeofday(&tv1, &tz1);
    std::cout << "model conversion time: "
              << ((tv1.tv_sec-tv.tv_sec)*1000+(tv1.tv_usec-tv.tv_usec)/1000)
              << " ms" << std::endl;

    // cache the trt model
    std::ofstream trtModelFile(trtModel.store_cached_file_name);
    trtModelFile.write((char *)trtModelStream->data(), trtModelStream->size());
    trtModelFile.close();
    trtModelStream->destroy();
    std::cout <<"Cache TRT model to trtModel.cache"<< std::endl;
#endif
    std::cout << "Done." << std::endl;

    return 0;
}
