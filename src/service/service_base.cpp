#include "service_base.h"
#include "service_driver.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

shared_ptr<grpc::Service> ServiceFactory::CreateTensorFlowService(string path){
    return  make_shared<tensorflow_service_driver>();
}