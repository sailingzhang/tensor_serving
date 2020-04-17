#include "service_common.h"

string composeModelNameKey(const string & modelname,const int64_t & version){
    return modelname +"@" + to_string(version);
}