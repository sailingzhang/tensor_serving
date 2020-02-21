#ifndef SERVICE_BASE_H
#define SERVICE_BASE_H

#include <string>
#include <iostream>
#include <tensorflow_serving/apis/prediction_service.grpc.pb.h>
#include "server_configure.pb.h"

using namespace std;

void loadconfigure(string configurefile,serving_configure::model_config_list & congifureList);
int tensor_serving_local_server(serving_configure::model_config_list congifureList);
shared_ptr<tensorflow::serving::PredictionService::StubInterface> createNoNetTensorflowClientservice(serving_configure::model_config_list configurelist);
shared_ptr<tensorflow::serving::PredictionService::StubInterface> createNoNetClientservice(shared_ptr<tensorflow::serving::PredictionService::Service> serviceptr);
class ServiceFactory{
public:
   static std::shared_ptr<grpc::Service>  CreateTensorFlowService(serving_configure::model_config_list configurelist);
};

#endif