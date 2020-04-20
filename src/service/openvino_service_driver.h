#ifndef OPENVINO_SERVICE_DRIVER_H
#define OPENVINO_SERVICE_DRIVER_H



#include <string>
#include <iostream>
#include "tensorflow/core/framework/tensor.pb.h"
#include <google/protobuf/util/json_util.h>
#include <tensorflow_serving/apis/prediction_service.grpc.pb.h>
#include <grpc++/server.h>
#include <grpc++/server_builder.h>
#include "server_configure.pb.h"
#include <log.h>

#include <inference_engine.hpp>
#include <details/os/os_filesystem.hpp>
#include "service_common.h"
#include "service_base.h"

#ifdef OPENVINO_SERVICE

using namespace InferenceEngine;
using namespace std;
struct OpenvinoProtoinfoS{
    Precision opevino_dtype;
    int32_t dtypesize;
    int64_t allbytesSize;
    vector<size_t> size_t_dimarr;
    void * pdata;
};


class openvino_service_driver:public tensorflow::serving::PredictionService::Service{
public:
    openvino_service_driver(serving_configure::model_config_list configurelist);
    virtual ~openvino_service_driver(){};
    // Classify.
    virtual ::grpc::Status Classify(::grpc::ServerContext* context, const ::tensorflow::serving::ClassificationRequest* request, ::tensorflow::serving::ClassificationResponse* response);
    // Regress.
    virtual ::grpc::Status Regress(::grpc::ServerContext* context, const ::tensorflow::serving::RegressionRequest* request, ::tensorflow::serving::RegressionResponse* response);
    // Predict -- provides access to loaded TensorFlow model.
    virtual ::grpc::Status Predict(::grpc::ServerContext* context, const ::tensorflow::serving::PredictRequest* request, ::tensorflow::serving::PredictResponse* response);
    // MultiInference API for multi-headed models.
    virtual ::grpc::Status MultiInference(::grpc::ServerContext* context, const ::tensorflow::serving::MultiInferenceRequest* request, ::tensorflow::serving::MultiInferenceResponse* response);
    // GetModelMetadata - provides access to metadata for loaded models.
    virtual ::grpc::Status GetModelMetadata(::grpc::ServerContext* context, const ::tensorflow::serving::GetModelMetadataRequest* request, ::tensorflow::serving::GetModelMetadataResponse* response);
public:
    int32_t loadModel(string modelname,int64_t version,string modeldir);
    InferenceEngine::Blob::Ptr TensorProto_To_OpenvinoInput(const tensorflow::TensorProto & from, InputInfo::Ptr  inputInfoptr);
    int32_t OpenvinoOutput_To_TensorProto(InferRequest &infer_request, DataPtr  outputInfoPtr,tensorflow::TensorProto & outputproto);
private:
    string run_predict_session(const ::tensorflow::serving::PredictRequest* request, ::tensorflow::serving::PredictResponse* response);
    std::map<string,std::tuple<Core,CNNNetwork>>modelsourceMap;

};

#endif




#endif

