#ifndef SERVICE_DRIVER_H
#define SERVICE_DRIVER_H
#include <iostream>
#include "tensorflow/core/framework/tensor.pb.h"
#include <google/protobuf/util/json_util.h>
#include <tensorflow_serving/apis/prediction_service.grpc.pb.h>
#include <grpc++/server.h>
#include <grpc++/server_builder.h>
#include <tensorflow/c/c_api.h>

using namespace std;


struct signatureRelation{
    string modelname;
    string signature;
    vector<string> outputname;
};



class tensorflow_service_driver:public tensorflow::serving::PredictionService::Service{
public:
    tensorflow_service_driver();
    virtual ~tensorflow_service_driver(){};
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
private:
    std::map<string,map<string,signatureRelation>> signatureRelationMap;
    TF_Session * t_sess;
	TF_Graph * t_graph;
};

#endif