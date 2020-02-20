#include <iostream>
#include <fstream>
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
    tensor_serving_local_server(congifureList);
    return 0;
}