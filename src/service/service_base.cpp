#include<log.h>
#include "service_base.h"

#include <google/protobuf/util/json_util.h>
#include <tensorflow_serving/apis/prediction_service.grpc.pb.h>
#include <grpc++/server.h>
#include <grpc++/server_builder.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;


#ifdef TENSERFLOW_SERVICE
#include "tensorflow_service_driver.h"
shared_ptr<grpc::Service> ServiceFactory::CreateTensorFlowService(serving_configure::model_config_list configurelist){
    return  make_shared<tensorflow_service_driver>(configurelist);
}
shared_ptr<tensorflow::serving::PredictionService::StubInterface> createNoNetTensorflowClientservice(serving_configure::model_config_list configurelist){
    auto serverinterface = make_shared<tensorflow_service_driver>(configurelist);
    return createNoNetClientservice(serverinterface);
}

shared_ptr<tensorflow::serving::PredictionService::StubInterface> createNoNetClientservice(serving_configure::model_config_list configurelist){
    return createNoNetTensorflowClientservice(configurelist);
}
#endif



#ifdef OPENVINO_SERVICE
#include "openvino_service_driver.h"
std::shared_ptr<grpc::Service> ServiceFactory::CreateOpenVinoService(serving_configure::model_config_list configurelist){
    return make_shared<openvino_service_driver>(configurelist);
}
shared_ptr<tensorflow::serving::PredictionService::StubInterface> createNoNetOpenvinoClientservice(serving_configure::model_config_list configurelist){
    auto serverinterface = make_shared<openvino_service_driver>(configurelist);
    return createNoNetClientservice(serverinterface);
}
shared_ptr<tensorflow::serving::PredictionService::StubInterface> createNoNetClientservice(serving_configure::model_config_list configurelist){
    return createNoNetOpenvinoClientservice(configurelist);
}

#endif











class local_tensor_sering_grpclient:public tensorflow::serving::PredictionService::StubInterface{
public:
    local_tensor_sering_grpclient(shared_ptr<tensorflow::serving::PredictionService::Service> servicePtr){
        this->_servicePtr =servicePtr; 
    };
    virtual ~local_tensor_sering_grpclient(){};
     ::grpc::Status Classify(::grpc::ClientContext* context, const ::tensorflow::serving::ClassificationRequest& request, ::tensorflow::serving::ClassificationResponse* response) override;
    ::grpc::Status Regress(::grpc::ClientContext* context, const ::tensorflow::serving::RegressionRequest& request, ::tensorflow::serving::RegressionResponse* response) override;
    ::grpc::Status Predict(::grpc::ClientContext* context, const ::tensorflow::serving::PredictRequest& request, ::tensorflow::serving::PredictResponse* response) override;
    ::grpc::Status MultiInference(::grpc::ClientContext* context, const ::tensorflow::serving::MultiInferenceRequest& request, ::tensorflow::serving::MultiInferenceResponse* response) override;
    ::grpc::Status GetModelMetadata(::grpc::ClientContext* context, const ::tensorflow::serving::GetModelMetadataRequest& request, ::tensorflow::serving::GetModelMetadataResponse* response) override;
   
private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::ClassificationResponse>* AsyncClassifyRaw(::grpc::ClientContext* context, const ::tensorflow::serving::ClassificationRequest& request, ::grpc::CompletionQueue* cq){return nullptr;};
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::ClassificationResponse>* PrepareAsyncClassifyRaw(::grpc::ClientContext* context, const ::tensorflow::serving::ClassificationRequest& request, ::grpc::CompletionQueue* cq){return nullptr;};
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::RegressionResponse>* AsyncRegressRaw(::grpc::ClientContext* context, const ::tensorflow::serving::RegressionRequest& request, ::grpc::CompletionQueue* cq){return nullptr;};
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::RegressionResponse>* PrepareAsyncRegressRaw(::grpc::ClientContext* context, const ::tensorflow::serving::RegressionRequest& request, ::grpc::CompletionQueue* cq){return nullptr;};
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::PredictResponse>* AsyncPredictRaw(::grpc::ClientContext* context, const ::tensorflow::serving::PredictRequest& request, ::grpc::CompletionQueue* cq){return nullptr;};
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::PredictResponse>* PrepareAsyncPredictRaw(::grpc::ClientContext* context, const ::tensorflow::serving::PredictRequest& request, ::grpc::CompletionQueue* cq){return nullptr;};
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::MultiInferenceResponse>* AsyncMultiInferenceRaw(::grpc::ClientContext* context, const ::tensorflow::serving::MultiInferenceRequest& request, ::grpc::CompletionQueue* cq) {return nullptr;};
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::MultiInferenceResponse>* PrepareAsyncMultiInferenceRaw(::grpc::ClientContext* context, const ::tensorflow::serving::MultiInferenceRequest& request, ::grpc::CompletionQueue* cq){return nullptr;};
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::GetModelMetadataResponse>* AsyncGetModelMetadataRaw(::grpc::ClientContext* context, const ::tensorflow::serving::GetModelMetadataRequest& request, ::grpc::CompletionQueue* cq){return nullptr;};
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::GetModelMetadataResponse>* PrepareAsyncGetModelMetadataRaw(::grpc::ClientContext* context, const ::tensorflow::serving::GetModelMetadataRequest& request, ::grpc::CompletionQueue* cq){return nullptr;};
    shared_ptr<tensorflow::serving::PredictionService::Service> _servicePtr;
};

::grpc::Status local_tensor_sering_grpclient::Classify(::grpc::ClientContext* context, const ::tensorflow::serving::ClassificationRequest& request, ::tensorflow::serving::ClassificationResponse* response){
    return this->_servicePtr->Classify(nullptr,&request,response);
    // return Status::OK;
}
::grpc::Status local_tensor_sering_grpclient::Regress(::grpc::ClientContext* context, const ::tensorflow::serving::RegressionRequest& request, ::tensorflow::serving::RegressionResponse* response) {
    return this->_servicePtr->Regress(nullptr,&request,response);
}
::grpc::Status local_tensor_sering_grpclient::Predict(::grpc::ClientContext* context, const ::tensorflow::serving::PredictRequest& request, ::tensorflow::serving::PredictResponse* response) {
     return this->_servicePtr->Predict(nullptr,&request,response);
}
::grpc::Status local_tensor_sering_grpclient::MultiInference(::grpc::ClientContext* context, const ::tensorflow::serving::MultiInferenceRequest& request, ::tensorflow::serving::MultiInferenceResponse* response){
    return Status::OK;
}
::grpc::Status local_tensor_sering_grpclient::GetModelMetadata(::grpc::ClientContext* context, const ::tensorflow::serving::GetModelMetadataRequest& request, ::tensorflow::serving::GetModelMetadataResponse* response) {
    return Status::OK;
}



shared_ptr<tensorflow::serving::PredictionService::StubInterface> createNoNetClientservice(shared_ptr<tensorflow::serving::PredictionService::Service> serviceptr){
    return  make_shared<local_tensor_sering_grpclient>(serviceptr);
}


int tensor_serving_local_server(shared_ptr<grpc::Service> service_ptr,string addr,serving_configure::model_config_list congifureList) {
	ServerBuilder builder;
	builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
	builder.RegisterService(service_ptr.get());
    std::unique_ptr<Server> LocalServer;
	LocalServer.reset(builder.BuildAndStart().release());
	std::cout << "grpc server listening on: " << addr.c_str() << std::endl;
	LocalServer->Wait();
	return 0;
}

void loadconfigure(string configurefile,serving_configure::model_config_list & congifureList){
	ifstream jsonfile(configurefile);
    if(!jsonfile.is_open()){
        LOG_ERROR("can't find configurefile="<<configurefile);
		std::quick_exit(-1);
    }

    string line;
    string jsonstr;
    while (getline(jsonfile, line))
    {
        jsonstr += line;
    }
    jsonfile.close();

	// LOG_TRACE("jsonstr="<<jsonstr.c_str());
    google::protobuf::util::JsonParseOptions opt;
	opt.ignore_unknown_fields = true;
    auto ret =google::protobuf::util::JsonStringToMessage(jsonstr, &congifureList,opt);
	if (google::protobuf::util::Status::OK != ret) {
		LOG_ERROR("jsontomessage,use default value, err:"<<ret.ToString().c_str());
        std::quick_exit(-1);
	}
    LOG_INFO("configure json="<<congifureList.DebugString());
}