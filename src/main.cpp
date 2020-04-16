#include <iostream>
#include <fstream>
#include <thread>
#include <log.h>
#include <grpc++/grpc++.h>
#include "service/service_base.h"
// #include <grpc++/grpc++.h>
#include <google/protobuf/util/json_util.h>

using namespace std;

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;



int main(int argc,char *argv[]){
	log_init("s_log.conf");
    LOG_INFO("hello world");
	serving_configure::model_config_list congifureList;
	loadconfigure(argv[1],congifureList);
    auto tensorflow_serviceptr = ServiceFactory::CreateTensorFlowService(congifureList);
    auto openvino_serviceptr = ServiceFactory::CreateOpenVinoService(congifureList);
    std::thread  tf_thread(tensor_serving_local_server,tensorflow_serviceptr,"0.0.0.0:9001",congifureList);
    std::thread  openvino_thread(tensor_serving_local_server,openvino_serviceptr,"0.0.0.0:9002",congifureList);
    tf_thread.join();
    openvino_thread.join();
    
    return 0;
}