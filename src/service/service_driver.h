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

#include <dlfcn.h>


struct signatureRelation{
    string modelname;
    string signature;
    vector<string> outputname;
};


template<typename T>
void mydlsym(void * handle,T & a, string funname) {
     a = (T)(dlsym(handle, funname.c_str()));
}





class tensorflowOp{
public:
    tensorflowOp(string filename){
        auto handle = dlopen(filename.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
        if (! handle) {
            dlerror();
            // LOG_ERROR("dlopen error="<<dlerror());
            return;
        }
        this->TF_DeleteBuffer = decltype(this->TF_DeleteBuffer)(dlsym(handle, "TF_DeleteBuffer"));
        mydlsym(handle,this->TF_DeleteSessionOptions,"TF_DeleteSessionOptions");
        mydlsym(handle,this->TF_LoadSessionFromSavedModel,"TF_LoadSessionFromSavedModel");
        mydlsym(handle,TF_NewBuffer, "TF_NewBuffer");
        mydlsym(handle,TF_NewBufferFromString, "TF_NewBufferFromString");
        mydlsym(handle,TF_NewGraph, "TF_NewGraph");
        mydlsym(handle, TF_NewSessionOptions,"TF_NewSessionOptions");
        mydlsym(handle,TF_NewStatus,"TF_NewSessionOptions");
        mydlsym(handle,TF_Version,"TF_Version");
        
    };
    virtual ~tensorflowOp(){};
    TF_SessionOptions* (*TF_NewSessionOptions)(void);
    TF_Buffer* (*TF_NewBufferFromString)(const void* proto,size_t proto_len);
    TF_Buffer* (*TF_NewBuffer)(void);
    void (*TF_DeleteBuffer)(TF_Buffer*);
    TF_Status* (*TF_NewStatus)(void);
    TF_Graph* (*TF_NewGraph)(void);
    TF_Session* (*TF_LoadSessionFromSavedModel)(const TF_SessionOptions* session_options, const TF_Buffer* run_options,const char* export_dir, const char* const* tags, int tags_len,TF_Graph* graph, TF_Buffer* meta_graph_def, TF_Status* status);
    void (*TF_DeleteSessionOptions)(TF_SessionOptions*);
    char* (*TF_Version)(void);
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