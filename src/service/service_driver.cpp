#include "service_driver.h"
#include <fstream>
#include<typeinfo>
#include <log.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;



/*
static int load_file(const std::string & fname, std::vector<char>& buf)
{
	std::ifstream fs(fname, std::ios::binary | std::ios::in);

	if(!fs.good())
	{
		std::cerr<<fname<<" does not exist"<<std::endl;
		return -1;
	}


	fs.seekg(0, std::ios::end);
	int fsize=fs.tellg();

	fs.seekg(0, std::ios::beg);
	buf.resize(fsize);
	fs.read(buf.data(),fsize);

	fs.close();

	return 0;

}

TF_Session * load_graph(const char * frozen_fname, TF_Graph ** p_graph)
{
	TF_Status* s = TF_NewStatus();

	TF_Graph* graph = TF_NewGraph();

	std::vector<char> model_buf;

	load_file(frozen_fname,model_buf);

	TF_Buffer graph_def = {model_buf.data(), model_buf.size(), nullptr};

	TF_ImportGraphDefOptions* import_opts = TF_NewImportGraphDefOptions();
	TF_ImportGraphDefOptionsSetPrefix(import_opts, "");
	TF_GraphImportGraphDef(graph, &graph_def, import_opts, s);

	if(TF_GetCode(s) != TF_OK)
	{
		printf("load graph failed!\n Error: %s\n",TF_Message(s));

		return nullptr;
	}

	TF_SessionOptions* sess_opts = TF_NewSessionOptions();
	TF_Session* session = TF_NewSession(graph, sess_opts, s);
	assert(TF_GetCode(s) == TF_OK);


	TF_DeleteStatus(s);


	*p_graph=graph;

	return session;
}
*/


/* To make tensor release happy...*/
// static void dummy_deallocator(void* data, size_t len, void* arg)
// {
// }

// shared_ptr<TF_Tensor> TensorProto_To_TF_Tensor(const tensorflow::TensorProto & from){
//     auto &dim = from.tensor_shape().dim();
//     int64_t dimarr[100];
//     for(auto i =0;i < dim.size();i++){
//         dimarr[i] = dim[i].size();
//     }
//     auto gettensorptr = TF_NewTensor(TF_FLOAT,dimarr,dim.size(),(void *)from.float_val().begin(),sizeof(float)*from.float_val().size(),dummy_deallocator,nullptr);
//     shared_ptr<TF_Tensor> sp(gettensorptr,[](TF_Tensor *p){TF_DeleteTensor(p);});
//     return sp;
// }

// void run_predict_session(TF_Session * sess, TF_Graph * graph, const ::tensorflow::serving::PredictRequest* request, ::tensorflow::serving::PredictResponse* response)
// {

// 	/* tensorflow related*/
// 	TF_Status * s= TF_NewStatus();

//     int32_t request_index =0;
//     std::vector<TF_Output> input_names;
// 	std::vector<TF_Tensor*> input_values;
//     for(auto it = request->inputs().begin();it != request->inputs().end();it++){
//         auto protoptr = TensorProto_To_TF_Tensor(it->second);
//         TF_Operation* input_name=TF_GraphOperationByName(graph,it->first.c_str());
//         input_names.push_back({input_name, request_index});
//         input_values.push_back(protoptr.get());
//     }

//     vector<string> outputnameVec;
//     std::vector<TF_Output> output_names;
//     for(auto i = 0; i < outputnameVec.size();i++){
//         TF_Operation* output_name = TF_GraphOperationByName(graph,outputnameVec[i].c_str());
//         output_names.push_back({output_name,0});
//     }
//     std::vector<TF_Tensor*> output_values(output_names.size(), nullptr);
//     TF_SessionRun(sess,nullptr,input_names.data(),input_values.data(),input_names.size(),output_names.data(),output_values.data(),output_names.size(),nullptr,0,nullptr,s);
//     assert(TF_GetCode(s) == TF_OK);
//     auto &outputmap =  *response->mutable_outputs();
//     for(auto i = 0;i < output_values.size();i++){
//         tensorflow::TensorProto outputproto;
//         outputproto.set_dtype(tensorflow::DataType::DT_FLOAT);
//         auto allsize = 1;
//         auto  numdims = TF_NumDims(output_values[i]);
//         for(auto j = 0; j < numdims; j++){
//             auto dimsize = TF_Dim(output_values[i],j);
//             outputproto.mutable_tensor_shape()->add_dim()->set_size(dimsize);
//             allsize *= dimsize;
//         }
//         auto datap = (const float *)TF_TensorData(output_values[i]);
//         for(auto dataindex = 0; dataindex < allsize;dataindex++){
//             outputproto.add_float_val(datap[dataindex]);
//         }
//         outputmap[outputnameVec[i]] = std::move(outputproto);
//     }

// 	TF_DeleteStatus(s);
// 	// TF_DeleteTensor(output_values[0]);
// 	// TF_DeleteTensor(output_values[1]);
// 	// TF_DeleteTensor(input_tensor);

// }




// tensorflow_service_driver::tensorflow_service_driver(){
//     // string model_fname= "test.pb";
//     // this->t_sess=load_graph(model_fname.c_str(),&this->t_graph);
//         LOG_INFO("enter");
//         string saved_model_dir ="mtcnnmodel";
//         TF_SessionOptions* opt = TF_NewSessionOptions();
//         TF_Buffer* run_options = TF_NewBufferFromString("", 0);
//         TF_Buffer* metagraph = TF_NewBuffer();
//         TF_Status* s = TF_NewStatus();
//         const char* tags[] = {"serve"};
//         TF_Graph* graph = TF_NewGraph();
//         TF_Session* session = TF_LoadSessionFromSavedModel(opt, run_options, saved_model_dir.c_str(), tags, 1, graph, metagraph, s);
//         TF_DeleteBuffer(run_options);
//         TF_DeleteSessionOptions(opt);
//         tensorflow::MetaGraphDef metagraph_def;
//         metagraph_def.ParseFromArray(metagraph->data, metagraph->length);
//         TF_DeleteBuffer(metagraph);

//         // Retrieve the regression signature from meta graph def.
//         const auto signature_def_map = metagraph_def.signature_def();
//         const auto signature_def = signature_def_map.at("regress_x_to_y");
//         auto &s_inputs = signature_def.inputs(); 
//         for(auto it = s_inputs.begin();it != s_inputs.end();it++){
//             LOG_INFO("first="<<it->first<<" name="<<it->second.name());
//         }

//         // const string input_name =
//         //     signature_def.inputs().at(tensorflow::kRegressInputs).name();
//         // const string output_name =
//         //     signature_def.outputs().at(tensorflow::kRegressOutputs).name();


//         this->t_sess = session;
//         this->t_graph = graph;
// }


tensorflow_service_driver::tensorflow_service_driver(){
        LOG_DEBUG("enter");
    // string model_fname= "test.pb";
    // this->t_sess=load_graph(model_fname.c_str(),&this->t_graph);
        tensorflowOp  TfOp("libtensorflow.so.1");
        LOG_DEBUG("get tfop ok");
        LOG_INFO("tensorflwo version="<<TfOp.TF_Version());
        string saved_model_dir ="mtcnnmodel";
        TF_SessionOptions* opt = TfOp.TF_NewSessionOptions();
        TF_Buffer* run_options = TfOp.TF_NewBufferFromString("", 0);
        TF_Buffer* metagraph = TfOp.TF_NewBuffer();
        TF_Status* s = TfOp.TF_NewStatus();
        const char* tags[] = {"serve"};
        TF_Graph* graph = TfOp.TF_NewGraph();
        TF_Session* session = TfOp.TF_LoadSessionFromSavedModel(opt, run_options, saved_model_dir.c_str(), tags, 1, graph, metagraph, s);
        TfOp.TF_DeleteBuffer(run_options);
        TfOp.TF_DeleteSessionOptions(opt);
        tensorflow::MetaGraphDef metagraph_def;
        metagraph_def.ParseFromArray(metagraph->data, metagraph->length);
        TfOp.TF_DeleteBuffer(metagraph);

        // Retrieve the regression signature from meta graph def.
        const auto signature_def_map = metagraph_def.signature_def();
        
        // for(auto &signature_def:signature_def_map){
        //     auto &s_inputs = signature_def.inputs(); 
        //     for(auto it = s_inputs.begin();it != s_inputs.end();it++){
        //         LOG_INFO("first="<<it->first<<" name="<<it->second.name());
        //     }
        // }
        const auto signature_def = signature_def_map.at("pnetSignature");
        auto &s_inputs = signature_def.inputs(); 
        for(auto it = s_inputs.begin();it != s_inputs.end();it++){
            LOG_INFO("first="<<it->first<<" name="<<it->second.name());
        }
        this->t_sess = session;
        this->t_graph = graph;
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
    // run_predict_session(this->t_sess,this->t_graph,request,response);
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