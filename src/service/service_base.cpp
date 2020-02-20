#include<log.h>
#include "service_base.h"
#include "service_driver.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

shared_ptr<grpc::Service> ServiceFactory::CreateTensorFlowService(serving_configure::model_config_list configurelist){
    return  make_shared<tensorflow_service_driver>(configurelist);
}

int tensor_serving_local_server(serving_configure::model_config_list congifureList) {
	std::string server_address("0.0.0.0:9001");
    auto tensorflowServicePtr = ServiceFactory::CreateTensorFlowService(congifureList);
	
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