#include "service_driver.h"
#include <fstream>
#include<typeinfo>
#include <log.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;





/* To make tensor release happy...*/
static void dummy_deallocator(void* data, size_t len, void* arg)
{
}

tensorflow_service_driver::tensorflow_service_driver(){
        LOG_DEBUG("enter");
    // string model_fname= "test.pb";
    // this->t_sess=load_graph(model_fname.c_str(),&this->t_graph);
        tensorflowOp  tmop("libtensorflow.so.1");
        this->TfOp = tmop;
        LOG_INFO("tensorflwo version="<<TfOp.TF_Version());
        string saved_model_dir ="mtcnnmodel";
        this->loadModel("mtcnnmodel",saved_model_dir);

}

int32_t tensorflow_service_driver::loadModel(string modelname,string modeldir){
        LOG_DEBUG("load,enter");
        TF_SessionOptions* opt = TfOp.TF_NewSessionOptions();
        TF_Buffer* run_options = TfOp.TF_NewBufferFromString("", 0);
        TF_Buffer* metagraph = TfOp.TF_NewBuffer();
        TF_Status* s = TfOp.TF_NewStatus();
        const char* tags[] = {"serve"};
        TF_Graph* graph = TfOp.TF_NewGraph();
        TF_Session* session = TfOp.TF_LoadSessionFromSavedModel(opt, run_options, modeldir.c_str(), tags, 1, graph, metagraph, s);

        tensorflow::MetaGraphDef metagraph_def;
        metagraph_def.ParseFromArray(metagraph->data, metagraph->length);
        modelSource onemodel;
        onemodel.t_sess = session;
        onemodel.t_graph = graph;
        onemodel.metagraph_def = metagraph_def;
        this->modelsourceMap[modelname] = onemodel;

        TfOp.TF_DeleteBuffer(run_options);
        TfOp.TF_DeleteSessionOptions(opt);
        TfOp.TF_DeleteBuffer(metagraph);
        TfOp.TF_DeleteStatus(s);
        LOG_INFO("load,exit");
        return 0;
}

TF_Tensor * tensorflow_service_driver::TensorProto_To_TF_Tensor(const tensorflow::TensorProto & from){
    LOG_DEBUG("enter");
    auto &dim = from.tensor_shape().dim();
    int64_t dimarr[100];
    for(auto i =0;i < dim.size();i++){
        dimarr[i] = dim[i].size();
        LOG_DEBUG("one dim="<<dimarr[i]);
    }
    auto gettensorptr = this->TfOp.TF_NewTensor(TF_FLOAT,dimarr,dim.size(),(void *)from.float_val().begin(),sizeof(float)*from.float_val().size(),dummy_deallocator,nullptr);
    return gettensorptr;
    // auto deleteop = this->TfOp.TF_DeleteTensor;
    // shared_ptr<TF_Tensor> sp(gettensorptr,[=](TF_Tensor *p){ (*deleteop)(p);});
    // return sp;
}


int32_t tensorflow_service_driver::run_predict_session( const ::tensorflow::serving::PredictRequest* request, ::tensorflow::serving::PredictResponse* response)
{
    LOG_DEBUG("enter");

    auto &modelname = request->model_spec().name();
    auto &signaturename = request->model_spec().signature_name();
    LOG_DEBUG("modelname="<<modelname<<" signaturename="<<signaturename);
    auto it = this->modelsourceMap.find(modelname);
    if(it == this->modelsourceMap.end()){
        LOG_ERROR("no find such model,name="<<modelname);
        return -1;
    }
    TF_Session * sess = it->second.t_sess;
    TF_Graph * graph = it->second.t_graph;
    auto & metagraph_def = it->second.metagraph_def;
    const auto &signature_def_map = metagraph_def.signature_def();
    const auto &signature_def = signature_def_map.at(signaturename);
    const auto & outputsinfo = signature_def.outputs();
    const auto & inputsinfo = signature_def.inputs();

	TF_Status * s=this->TfOp.TF_NewStatus();

    int32_t request_index =0;
    std::vector<TF_Output> input_names;
	std::vector<TF_Tensor*> input_values;
    for(auto it = request->inputs().begin();it != request->inputs().end();it++){
        auto protop = TensorProto_To_TF_Tensor(it->second);
        auto infoIt =inputsinfo.find(it->first);
        if(infoIt == inputsinfo.end()){
            LOG_ERROR("no find such input name="<<it->first);
            return -1;
        }
        TF_Operation* input_name=this->TfOp.TF_GraphOperationByName(graph,infoIt->second.name().c_str());
        input_names.push_back({input_name, request_index});
        input_values.push_back(protop);
    }

    


    vector<string> outputnameVec;
    std::vector<TF_Output> output_names;
    for(auto outputIt = outputsinfo.begin(); outputIt != outputsinfo.end();outputIt++){
        TF_Operation* output_name =this->TfOp.TF_GraphOperationByName(graph,outputIt->second.name().c_str());
        outputnameVec.push_back(outputIt->second.name());
        output_names.push_back({output_name,0});
    }
    std::vector<TF_Tensor*> output_values(output_names.size(), nullptr);
    this->TfOp.TF_SessionRun(sess,nullptr,input_names.data(),input_values.data(),input_names.size(),output_names.data(),output_values.data(),output_names.size(),nullptr,0,nullptr,s);
    assert(this->TfOp.TF_GetCode(s) == TF_OK);
    auto &outputmap =  *response->mutable_outputs();
    for(auto i = 0;i < output_values.size();i++){
        tensorflow::TensorProto outputproto;
        outputproto.set_dtype(tensorflow::DataType::DT_FLOAT);
        auto allsize = 1;
        auto  numdims = this->TfOp.TF_NumDims(output_values[i]);
        for(auto j = 0; j < numdims; j++){
            auto dimsize = this->TfOp.TF_Dim(output_values[i],j);
            outputproto.mutable_tensor_shape()->add_dim()->set_size(dimsize);
            allsize *= dimsize;
        }
        auto datap = (const float *)this->TfOp.TF_TensorData(output_values[i]);
        for(auto dataindex = 0; dataindex < allsize;dataindex++){
            outputproto.add_float_val(datap[dataindex]);
        }    
        outputmap[outputnameVec[i]] = std::move(outputproto);
    }

	this->TfOp.TF_DeleteStatus(s);
    for(auto it =input_values.begin();it != input_values.end();it++){
        this->TfOp.TF_DeleteTensor(*it);
    }
    for(auto it =output_values.begin();it != input_values.end();it++){
        this->TfOp.TF_DeleteTensor(*it);
    }
	// TF_DeleteTensor(output_values[0]);
	// TF_DeleteTensor(output_values[1]);
	// TF_DeleteTensor(input_tensor);
    return 0;

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
    this->run_predict_session(request,response);
    return Status::OK;
}
    // MultiInference API for multi-headed models.
::grpc::Status tensorflow_service_driver::MultiInference(::grpc::ServerContext* context, const ::tensorflow::serving::MultiInferenceRequest* request, ::tensorflow::serving::MultiInferenceResponse* response){
return Status::OK;
}
    // GetModelMetadata - provides access to metadata for loaded models.
::grpc::Status tensorflow_service_driver::GetModelMetadata(::grpc::ServerContext* context, const ::tensorflow::serving::GetModelMetadataRequest* request, ::tensorflow::serving::GetModelMetadataResponse* response){
return Status::OK;
}