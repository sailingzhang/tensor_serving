#ifndef SERVICE_BASE_H
#define SERVICE_BASE_H

#include <string>
#include <iostream>
#include <tensorflow_serving/apis/prediction_service.grpc.pb.h>
#include "server_configure.pb.h"

using namespace std;


class ServiceFactory{
public:
   static std::shared_ptr<grpc::Service>  CreateTensorFlowService(serving_configure::model_config_list configurelist);
};

#endif