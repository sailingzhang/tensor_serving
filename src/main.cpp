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

    #ifdef TENSERFLOW_SERVICE
    auto serviceptr = ServiceFactory::CreateTensorFlowService(congifureList);
    LOG_INFO("tensorflow type")
    #endif

    #ifdef OPENVINO_SERVICE
    auto serviceptr = ServiceFactory::CreateOpenVinoService(congifureList);
    LOG_INFO("openvino type");
    #endif

    std::thread  s_thread(tensor_serving_local_server,serviceptr,"0.0.0.0:9001",congifureList);
    s_thread.join();
    
    return 0;
}