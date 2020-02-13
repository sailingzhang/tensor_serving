#include <iostream>
#include <log.h>
#include <grpc++/grpc++.h>
#include "service/service_base.h"

using namespace std;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

int local_server(int argc, char* argv[]) {
	std::string server_address("0.0.0.0:9001");
    // string configurefile = argv[1];
    auto tensorflowServicePtr = ServiceFactory::CreateTensorFlowService("");

	ServerBuilder builder;
	// Listen on the given address without any authentication mechanism.
	builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
	// Register "service" as the instance through which we'll communicate with
	// clients. In this case it corresponds to an *synchronous* service.
	builder.RegisterService(tensorflowServicePtr.get());
	// Finally assemble the server.
	//std::unique_ptr<Server> server(builder.BuildAndStart());
    std::unique_ptr<Server> LocalServer;
	LocalServer.reset(builder.BuildAndStart().release());
	std::cout << "grpc server listening on: " << server_address.c_str() << std::endl;
	LocalServer->Wait();
	return 0;
}



int main(int argc,char *argv[]){
    LOG_INFO("hello world");
    local_server(argc,argv);
    return 0;
}