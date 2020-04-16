#ifndef SERVICE_DRIVER_H
#define SERVICE_DRIVER_H
#include <iostream>
#include "tensorflow/core/framework/tensor.pb.h"
#include <google/protobuf/util/json_util.h>
#include <tensorflow_serving/apis/prediction_service.grpc.pb.h>
#include <grpc++/server.h>
#include <grpc++/server_builder.h>
#include <tensorflow/c/c_api.h>
#include "server_configure.pb.h"

#include <inference_engine.hpp>
#include <details/os/os_filesystem.hpp>
// #include <samples/common.hpp>
// #include <samples/ocv_common.hpp>
// #include <samples/classification_results.h>

#include <log.h>

using namespace std;
using namespace InferenceEngine;

#include <dlfcn.h>


struct signatureRelation{
    string modelname;
    string signature;
    vector<string> outputname;
};

struct modelSource{
    TF_Session * t_sess;
	TF_Graph * t_graph;
    // TF_Buffer* metagraph;
    tensorflow::MetaGraphDef metagraph_def;
};


struct protoinfoS{
    TF_DataType dtype;
    Precision opevino_dtype;
    int32_t dtypesize;
    int64_t allbytesSize;
    vector<int64_t> dimarr;
    vector<size_t> size_t_dimarr;
    void * pdata;
};

template<typename T>
void mydlsym(void * handle,T & a, string funname) {
     a = (T)(dlsym(handle, funname.c_str()));
}






class tensorflowOp{
public:
    tensorflowOp(){
        this->handle= nullptr;
    };
    tensorflowOp(string filename){
        LOG_INFO("enter")
        this->handle = nullptr;
        this->handle = dlopen(filename.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
        if (! this->handle) {
            auto perr=  dlerror();
            LOG_ERROR("dlerror="<<perr);
            std::quick_exit(-1);
        }
        LOG_INFO("dl tensorflow ok");
        this->TF_DeleteBuffer = decltype(this->TF_DeleteBuffer)(dlsym(handle, "TF_DeleteBuffer"));
        mydlsym(this->handle,this->TF_DeleteSessionOptions,"TF_DeleteSessionOptions");
        mydlsym(this->handle,this->TF_LoadSessionFromSavedModel,"TF_LoadSessionFromSavedModel");
        mydlsym(this->handle,TF_NewBuffer, "TF_NewBuffer");
        mydlsym(this->handle,TF_NewBufferFromString, "TF_NewBufferFromString");
        mydlsym(this->handle,TF_NewGraph, "TF_NewGraph");
        mydlsym(this->handle,TF_NewSessionOptions,"TF_NewSessionOptions");
        mydlsym(this->handle,TF_NewStatus,"TF_NewStatus");
        mydlsym(this->handle,TF_Version,"TF_Version");
        mydlsym(this->handle,TF_DeleteTensor,"TF_DeleteTensor");
        mydlsym(this->handle,TF_NewTensor,"TF_NewTensor");
        mydlsym(this->handle,TF_GraphOperationByName,"TF_GraphOperationByName");
        mydlsym(this->handle,TF_TensorData,"TF_TensorData");
        mydlsym(this->handle,TF_SessionRun,"TF_SessionRun");
        mydlsym(this->handle,TF_GetCode,"TF_GetCode");
        mydlsym(this->handle,TF_NumDims,"TF_NumDims");
        mydlsym(this->handle,TF_Dim,"TF_Dim");
        mydlsym(this->handle,TF_DeleteStatus,"TF_DeleteStatus");
        mydlsym(this->handle,TF_TensorType,"TF_TensorType");
        
    };
    virtual ~tensorflowOp(){
        if(nullptr != this->handle){
            // dlclose(this->handle);
        }
    };
    TF_SessionOptions* (*TF_NewSessionOptions)(void);
    TF_Buffer* (*TF_NewBufferFromString)(const void* proto,size_t proto_len);
    TF_Buffer* (*TF_NewBuffer)(void);
    void (*TF_DeleteBuffer)(TF_Buffer*);
    TF_Status* (*TF_NewStatus)(void);
    TF_Graph* (*TF_NewGraph)(void);
    TF_Session* (*TF_LoadSessionFromSavedModel)(const TF_SessionOptions* session_options, const TF_Buffer* run_options,const char* export_dir, const char* const* tags, int tags_len,TF_Graph* graph, TF_Buffer* meta_graph_def, TF_Status* status);
    void (*TF_DeleteSessionOptions)(TF_SessionOptions*);
    char* (*TF_Version)(void);
    void (*TF_DeleteTensor)(TF_Tensor*);
    TF_Tensor* (*TF_NewTensor)(TF_DataType, const int64_t* dims, int num_dims, void* data, size_t len,void (*deallocator)(void* data, size_t len, void* arg),void* deallocator_arg);
    TF_Operation* (*TF_GraphOperationByName)(TF_Graph* graph, const char* oper_name);
    void* (*TF_TensorData)(const TF_Tensor*);
    void (*TF_SessionRun)(TF_Session* session,const TF_Buffer* run_options,const TF_Output* inputs, TF_Tensor* const* input_values, int ninputs,const TF_Output* outputs, TF_Tensor** output_values, int noutputs,const TF_Operation* const* target_opers, int ntargets,TF_Buffer* run_metadata,TF_Status*);
    TF_Code (*TF_GetCode)(const TF_Status* s);
    int (*TF_NumDims)(const TF_Tensor*);
    int64_t (*TF_Dim)(const TF_Tensor* tensor, int dim_index);
    void (*TF_DeleteStatus)(TF_Status*);
    TF_DataType (*TF_TensorType)(const TF_Tensor*);
private:
    void * handle;
};



class tensorflow_service_driver:public tensorflow::serving::PredictionService::Service{
public:
    tensorflow_service_driver(serving_configure::model_config_list configurelist);
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
public:
    int32_t loadModel(string modelname,int64_t version,string modeldir);
private:
    // std::map<string,map<string,signatureRelation>> signatureRelationMap;
    // shared_ptr<TF_Tensor> TensorProto_To_TF_Tensor(const tensorflow::TensorProto & from);
    TF_Tensor * TensorProto_To_TF_Tensor(const tensorflow::TensorProto & from);
    int32_t Tensor_To_TensorProto(const TF_Tensor * tensorp,tensorflow::TensorProto & to);
    string run_predict_session(const ::tensorflow::serving::PredictRequest* request, ::tensorflow::serving::PredictResponse* response);
    tensorflowOp TfOp;
    std::map<string,modelSource> modelsourceMap;
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
    int32_t TensorProto_To_OpenvinoInput(const tensorflow::TensorProto & from,InferRequest &infer_request, InputInfo & inputInfo);
    int32_t OpenvinoOutput_To_TensorProto(InferRequest &infer_request, DataPtr  outputInfoPtr,tensorflow::TensorProto & outputproto);
private:
    string run_predict_session(const ::tensorflow::serving::PredictRequest* request, ::tensorflow::serving::PredictResponse* response);
    std::map<string,std::tuple<Core,CNNNetwork>>modelsourceMap;

};


#endif