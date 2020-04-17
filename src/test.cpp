#include <iostream>
#include <iostream>
#include <fstream>
#include <thread>
#include <log.h>
#include <grpc++/grpc++.h>
#include "service/service_base.h"
// #include <grpc++/grpc++.h>
#include <google/protobuf/util/json_util.h>

using namespace std;

int main_test(int argc,char *argv[]){
    cout<<"hello world"<<endl;
    serving_configure::model_config_list congifureList;
	loadconfigure(argv[1],congifureList);
    // auto client = createNoNetOpenvinoClientservice(congifureList);
    return 0;
}