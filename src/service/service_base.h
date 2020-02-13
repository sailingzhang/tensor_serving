#ifndef SERVICE_BASE_H
#define SERVICE_BASE_H

#include <string>
#include <iostream>
#include <tensorflow_serving/apis/prediction_service.grpc.pb.h>

using namespace std;


class ServiceFactory{
public:
   static std::shared_ptr<grpc::Service>  CreateTensorFlowService(string path);
};

#endif