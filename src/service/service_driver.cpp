#include "service_driver.h"
#include <fstream>
#include<typeinfo>
#include <log.h>
#include <sstream>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;





/* To make tensor release happy...*/
static void dummy_deallocator(void* data, size_t len, void* arg)
{
}
static void free_deallocator(void* data, size_t len, void* arg)
{
    if(nullptr != data){
        free(data);
    }
}

string composeModelNameKey(const string & modelname,const int64_t & version){
    return modelname +"@" + to_string(version);
}


tensorflow_service_driver::tensorflow_service_driver(serving_configure::model_config_list configurelist){
        LOG_DEBUG("enter");
        tensorflowOp  tmop("./libtensorflow.so.1");
        this->TfOp = tmop;
        LOG_INFO("tensorflwo version="<<TfOp.TF_Version());
        auto configs = configurelist.config();
        for(auto it = configs.begin();it != configs.end();it++){
            if(it->model_platform() != "tensorflow"){
                continue;
            }
            LOG_INFO("tensorflow service load, modelname="<<it->name()<<" version="<<it->version()<<" path="<<it->base_path());
            this->loadModel(it->name(),it->version(),it->base_path());
        }
}

int32_t tensorflow_service_driver::loadModel(string modelname,int64_t version,string modeldir){
        LOG_DEBUG("load,enter");
        auto &&modelnamekey = composeModelNameKey(modelname,version);
        TF_SessionOptions* opt = TfOp.TF_NewSessionOptions();
        TF_Buffer* run_options = TfOp.TF_NewBufferFromString("", 0);
        TF_Buffer* metagraph = TfOp.TF_NewBuffer();
        TF_Status* s = TfOp.TF_NewStatus();
        const char* tags[] = {"serve"};
        TF_Graph* graph = TfOp.TF_NewGraph();
        TF_Session* session = TfOp.TF_LoadSessionFromSavedModel(opt, run_options, modeldir.c_str(), tags, 1, graph, metagraph, s);
        if(this->TfOp.TF_GetCode(s) != TF_OK){
            LOG_ERROR("status err,code="<<this->TfOp.TF_GetCode(s));
            std::quick_exit(-1);
        }

        tensorflow::MetaGraphDef metagraph_def;
        metagraph_def.ParseFromArray(metagraph->data, metagraph->length);
        modelSource onemodel;
        onemodel.t_sess = session;
        onemodel.t_graph = graph;
        onemodel.metagraph_def = metagraph_def;
        LOG_INFO("register model="<<modelnamekey);
        this->modelsourceMap[modelnamekey] = onemodel;

        TfOp.TF_DeleteBuffer(run_options);
        TfOp.TF_DeleteSessionOptions(opt);
        TfOp.TF_DeleteBuffer(metagraph);
        TfOp.TF_DeleteStatus(s);
        LOG_INFO("load,exit");
        return 0;
}

TF_Tensor * tensorflow_service_driver::TensorProto_To_TF_Tensor(const tensorflow::TensorProto & from){
    LOG_DEBUG("enter");
    protoinfoS ret;
    ret.allbytesSize = 1;
    switch (from.dtype())
    {
        case tensorflow::DataType::DT_FLOAT:
            ret.dtype = TF_FLOAT;
            ret.dtypesize = sizeof(float);
            ret.pdata=(void *)from.float_val().begin();
            break;
        case tensorflow::DataType::DT_DOUBLE:
            ret.dtype = TF_DOUBLE;
            ret.dtypesize = sizeof(double);
            ret.pdata =(void *) from.double_val().begin();
            break;
        case tensorflow::DataType::DT_BOOL:
            ret.dtype = TF_BOOL;
            ret.dtypesize = sizeof(bool);
            ret.pdata = (void *)from.bool_val().begin();
            break;
        case tensorflow::DataType::DT_INT32:
            ret.dtype = TF_INT32;
            ret.dtypesize = sizeof(int32_t);
            ret.pdata = (void *)from.int_val().begin();
            break;
        case tensorflow::DataType::DT_INT64:
            ret.dtype = TF_INT64;
            ret.dtypesize = sizeof(int64_t);
            ret.pdata =(void *) from.int64_val().begin();
            break;
        default:
            LOG_ERROR("not support such type="<<from.dtype());
            return nullptr;
            break;
    }
    
    auto allsize = 1;
    auto &dim = from.tensor_shape().dim();
    ret.dimarr.resize(dim.size(),0);
    for(auto i =0;i < dim.size();i++){
        ret.dimarr[i] = dim[i].size();
        allsize= allsize* dim[i].size();
    }
    ret.allbytesSize = ret.dtypesize * allsize;
    auto gettensorptr = this->TfOp.TF_NewTensor(ret.dtype,ret.dimarr.data(),ret.dimarr.size(),ret.pdata,ret.allbytesSize,dummy_deallocator,nullptr);
    return gettensorptr;
}

int32_t tensorflow_service_driver::Tensor_To_TensorProto(const TF_Tensor * tensorp,tensorflow::TensorProto & outputproto){
    auto type =this->TfOp.TF_TensorType(tensorp);
    auto allsize = 1;
    auto  numdims = this->TfOp.TF_NumDims(tensorp);
    for(auto j = 0; j < numdims; j++){
        auto dimsize = this->TfOp.TF_Dim(tensorp,j);
        outputproto.mutable_tensor_shape()->add_dim()->set_size(dimsize);
        allsize *= dimsize;
    }

    switch (type)
    {
        case TF_FLOAT:
            {
                outputproto.set_dtype(tensorflow::DataType::DT_FLOAT);
                auto datap = (const float *)this->TfOp.TF_TensorData(tensorp);
                for(auto dataindex = 0; dataindex < allsize;dataindex++){
                    outputproto.add_float_val(datap[dataindex]);
                }  
            }
            break;
        case TF_DOUBLE:
            {
                outputproto.set_dtype(tensorflow::DataType::DT_DOUBLE);
                auto datap = (const double *)this->TfOp.TF_TensorData(tensorp);
                for(auto dataindex = 0; dataindex < allsize;dataindex++){
                    outputproto.add_double_val(datap[dataindex]);
                }
            }            
            break;
        case TF_BOOL:
            {
                outputproto.set_dtype(tensorflow::DataType::DT_BOOL);
                auto datap = (const bool *)this->TfOp.TF_TensorData(tensorp);
                for(auto dataindex = 0; dataindex < allsize;dataindex++){
                    outputproto.add_bool_val(datap[dataindex]);
                } 
            }

            break;
        case TF_INT32:
            {
                outputproto.set_dtype(tensorflow::DataType::DT_INT32);
                auto datap = (const int32_t *)this->TfOp.TF_TensorData(tensorp);
                for(auto dataindex = 0; dataindex < allsize;dataindex++){
                    outputproto.add_int_val(datap[dataindex]);
                }
            }    
            break;
        case TF_INT64:
            {
                outputproto.set_dtype(tensorflow::DataType::DT_INT64);
                auto datap = (const int64_t *)this->TfOp.TF_TensorData(tensorp);
                for(auto dataindex = 0; dataindex < allsize;dataindex++){
                    outputproto.add_int64_val(datap[dataindex]);
                } 
            }
            break;
        default:
            LOG_ERROR("not support such type="<<type);
            return -1;
            break;
    }
    return 0;

}

std::tuple<string,int32_t> split_tensorname(string namestr){
        tuple<string,int32_t> ret = make_tuple<string,int32_t>("",0);
        namestr.replace(namestr.rfind(":"),1," ");
        std::stringstream s(namestr);
        s>>get<0>(ret)>>get<1>(ret);
        LOG_DEBUG("tensorname="<<get<0>(ret)<<" index="<<get<1>(ret));
        return ret;
}

string tensorflow_service_driver::run_predict_session( const ::tensorflow::serving::PredictRequest* request, ::tensorflow::serving::PredictResponse* response)
{
    LOG_DEBUG("enter");
    string ret ;
	TF_Status * s=this->TfOp.TF_NewStatus();
    std::vector<TF_Tensor*> input_values;
    std::vector<TF_Tensor*> output_values;
    auto relasefun = [&](){
        this->TfOp.TF_DeleteStatus(s);
        for(auto it =input_values.begin();it != input_values.end();it++){
            this->TfOp.TF_DeleteTensor(*it);
        }
        for(auto it =output_values.begin();it != output_values.end();it++){
            this->TfOp.TF_DeleteTensor(*it);
        }
    };


    auto &base_modelname = request->model_spec().name();
    auto &&modelversion = request->model_spec().version().value();
    auto modelname = composeModelNameKey(base_modelname,modelversion);
    auto &signaturename = request->model_spec().signature_name();
    LOG_DEBUG("modelname="<<modelname<<" signaturename="<<signaturename);
    auto it = this->modelsourceMap.find(modelname);
    if(it == this->modelsourceMap.end()){
        LOG_ERROR("no find such model,name="<<modelname);
        ret ="no find such model,name=" + modelname;
        relasefun();
        return ret;
    }
    TF_Session * sess = it->second.t_sess;
    TF_Graph * graph = it->second.t_graph;
    auto & metagraph_def = it->second.metagraph_def;
    const auto &signature_def_map = metagraph_def.signature_def();
    auto signaturemapIt = signature_def_map.find(signaturename);
    if(signaturemapIt == signature_def_map.end()){
        LOG_ERROR("no find such signaturename="<<signaturename);
        ret = "no find such signaturename="+signaturename;
        relasefun();
        return ret;
    }
    const auto &signature_def = signaturemapIt->second;
    const auto & outputsinfo = signature_def.outputs();
    const auto & inputsinfo = signature_def.inputs();


    std::vector<TF_Output> input_names;
	
    for(auto it = request->inputs().begin();it != request->inputs().end();it++){
        // auto tensorp = TensorProto_To_TF_Tensor(it->second);
        //auto proinfo = protoinfo(it->second);
        //auto tensorp = this->TfOp.TF_NewTensor(proinfo.dtype,proinfo.dimarr.data(),proinfo.dimarr.size(),proinfo.pdata,proinfo.allbytesSize,dummy_deallocator,nullptr);
        auto tensorp = this->TensorProto_To_TF_Tensor(it->second);
        if(nullptr == tensorp){
            LOG_ERROR("tensorp is null");
            ret = "tensorp is null";
            relasefun();
            return  ret;
        }
        auto infoIt =inputsinfo.find(it->first);
        if(infoIt == inputsinfo.end()){
            LOG_ERROR("no find such input name="<<it->first);
            ret = "no find such input name="+it->first;
            relasefun();
            return ret;
        }
        
        auto inputsplit = split_tensorname(infoIt->second.name());
        LOG_DEBUG("get inputname,first="<<infoIt->first<<" second="<<infoIt->second.name()<<" split0="<<get<0>(inputsplit)<<" split1="<<get<1>(inputsplit));
        TF_Operation* input_name=this->TfOp.TF_GraphOperationByName(graph,get<0>(inputsplit).c_str());
        if(nullptr == input_name){
            LOG_ERROR("input_name is null");
            ret = "input_name is null";
            relasefun();
            return ret;
        }
        input_names.push_back({input_name, get<1>(inputsplit)});
        input_values.push_back(tensorp);
    }

    
    vector<string> outputSignatureNameVec;
    std::vector<TF_Output> output_names;
    for(auto outputIt = outputsinfo.begin(); outputIt != outputsinfo.end();outputIt++){
        auto outputsplit = split_tensorname(outputIt->second.name());
        LOG_DEBUG("get outputname,first="<<outputIt->first<<" second="<<outputIt->second.name()<<" split0="<<get<0>(outputsplit)<<" split1="<<get<1>(outputsplit));
        TF_Operation* output_name =this->TfOp.TF_GraphOperationByName(graph,get<0>(outputsplit).c_str());
        if(nullptr == output_name){
            LOG_ERROR("output_name is null");
            ret = "output_name is null";
            relasefun();
            return ret;
        }
        outputSignatureNameVec.push_back(outputIt->first);
        output_names.push_back({output_name,get<1>(outputsplit)});
    }
    // std::vector<TF_Tensor*> output_values(output_names.size(), nullptr);
    output_values.resize(output_names.size(),nullptr);
    this->TfOp.TF_SessionRun(sess,nullptr,input_names.data(),input_values.data(),input_names.size(),output_names.data(),output_values.data(),output_names.size(),nullptr,0,nullptr,s);
    if(this->TfOp.TF_GetCode(s) != TF_OK){
        LOG_ERROR("status err,code="<<this->TfOp.TF_GetCode(s));
        ret = "status err,code="+ to_string(this->TfOp.TF_GetCode(s));
        relasefun();
        return ret;
    }
    auto &outputmap =  *response->mutable_outputs();
    for(auto i = 0;i < output_values.size();i++){
        tensorflow::TensorProto outputproto;
        Tensor_To_TensorProto(output_values[i],outputproto);    
        outputmap[outputSignatureNameVec[i]] = std::move(outputproto);
        LOG_DEBUG("one output,sname="<<outputSignatureNameVec[i]);
    }
    relasefun();
    return ret;

}




    // Classify.
::grpc::Status tensorflow_service_driver::Classify(::grpc::ServerContext* context, const ::tensorflow::serving::ClassificationRequest* request, ::tensorflow::serving::ClassificationResponse* response){
    return Status::OK;
}
    // Regress.
::grpc::Status tensorflow_service_driver::Regress(::grpc::ServerContext* context, const ::tensorflow::serving::RegressionRequest* request, ::tensorflow::serving::RegressionResponse* response){
    return Status::OK;
}
    // Predict -- provides access to loaded TensorFlow model.
::grpc::Status tensorflow_service_driver::Predict(::grpc::ServerContext* context, const ::tensorflow::serving::PredictRequest* request, ::tensorflow::serving::PredictResponse* response){
    auto &&ret =  this->run_predict_session(request,response);
    if(ret.empty()){
        return Status::OK;
    }else{
        LOG_ERROR("Predict error="<<ret);
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,ret);
    }
    
}
    // MultiInference API for multi-headed models.
::grpc::Status tensorflow_service_driver::MultiInference(::grpc::ServerContext* context, const ::tensorflow::serving::MultiInferenceRequest* request, ::tensorflow::serving::MultiInferenceResponse* response){
return Status::OK;
}
    // GetModelMetadata - provides access to metadata for loaded models.
::grpc::Status tensorflow_service_driver::GetModelMetadata(::grpc::ServerContext* context, const ::tensorflow::serving::GetModelMetadataRequest* request, ::tensorflow::serving::GetModelMetadataResponse* response){
return Status::OK;
}