#include "openvino_service_driver.h"
#ifdef OPENVINO_SERVICE

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

openvino_service_driver::openvino_service_driver(serving_configure::model_config_list configurelist){
    auto configs = configurelist.config();
    for(auto it = configs.begin();it != configs.end();it++){
        if(it->model_platform() != "openvino"){
            continue;
        }
        LOG_INFO("openvino service load, modelname="<<it->name()<<" version="<<it->version()<<" path="<<it->base_path());
        this->loadModel(it->name(),it->version(),it->base_path());
    }
    return ;
}

int32_t openvino_service_driver::loadModel(string modelname,int64_t version,string modeldir){
    LOG_INFO("begin openvino load ,modelname="<<modelname<<" version="<<version<<" dir="<<modeldir);
    auto &&modelnamekey = composeModelNameKey(modelname,version);
    Core ie;
    CNNNetwork network = ie.ReadNetwork(modeldir+"/model.xml", modeldir+"/model.bin");
    LOG_INFO("network  batchsize"<<network.getBatchSize());
    // for(auto it = network.getInputShapes().begin();it != network.getInputShapes().end();it++){
    //     for(auto shapeIt= it->second.begin();shapeIt != it->second.end();shapeIt++){
    //         LOG_INFO(" shape,first="<<shapeIt<<" second="<<it->second);
    //     }
        
    // }
    
    auto inputinfo = network.getInputsInfo();
    for(auto it = inputinfo.begin(); it != inputinfo.end();it++){
        string dimstr ="";
        auto inputptr= it->second;
        if(nullptr == inputptr){
            LOG_ERROR("inputptr is null,name="<<it->first);
            continue;
        }
        auto tensorDesc = inputptr->getTensorDesc();
        for(auto dimIt = tensorDesc.getDims().begin();dimIt != tensorDesc.getDims().end();dimIt++){
            dimstr += (to_string(*dimIt)+",");
        }
        LOG_INFO(" inputname="<<inputptr->name()<<" dims="<<dimstr);
    }
    this->modelsourceMap[modelnamekey]= {ie,network};
    LOG_INFO("openvino load over,modelname="<<modelname);
    // network.setBatchSize(1);
    return 0;
}

int32_t openvino_service_driver::TensorProto_To_OpenvinoInput(const tensorflow::TensorProto & from,InferRequest &infer_request, InputInfo & inputInfo){
    LOG_DEBUG("enter,inputname="<<inputInfo.name()<<" datatype="<<from.dtype());
    OpenvinoProtoinfoS ret;
    ret.allbytesSize = 1;
    switch (from.dtype())
    {
        case tensorflow::DataType::DT_FLOAT:
            ret.opevino_dtype = Precision::FP32;
            ret.dtypesize = sizeof(float);
            ret.pdata=(void *)from.float_val().begin();
            break;
        case tensorflow::DataType::DT_BOOL:
            ret.opevino_dtype = Precision::BOOL;
            ret.dtypesize = sizeof(bool);
            ret.pdata = (void *)from.bool_val().begin();
            break;
        case tensorflow::DataType::DT_INT32:
            ret.opevino_dtype = Precision::I32;
            ret.dtypesize = sizeof(int32_t);
            ret.pdata = (void *)from.int_val().begin();
            break;
        case tensorflow::DataType::DT_INT64:
            ret.opevino_dtype = Precision::I64;
            ret.dtypesize = sizeof(int64_t);
            ret.pdata =(void *) from.int64_val().begin();
            break;
        default:
            LOG_ERROR("not support such type="<<from.dtype());
            return -1;
            break;
    }
    
    auto allsize = 1;
    auto &dim = from.tensor_shape().dim();
    ret.size_t_dimarr.resize(dim.size(),0);
    for(auto i =0;i < dim.size();i++){
        ret.size_t_dimarr[i] = dim[i].size();
        allsize= allsize* dim[i].size();
    }
    ret.allbytesSize = ret.dtypesize * allsize;

    
    LOG_DEBUG("inputinfo precision="<<inputInfo.getPrecision()<<" layout="<<inputInfo.getLayout());


    // InputInfo::Ptr input_info = it->second;
    // std::string input_name = it->first;
    inputInfo.getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    // inputInfo.setLayout(Layout::NHWC);
    // inputInfo.setPrecision(ret.opevino_dtype);

    
    

    
    auto getdims = inputInfo.getTensorDesc().getDims();
    for(auto & v:getdims){
        LOG_DEBUG("get dim="<<v);
    }

    auto layout =  inputInfo.getTensorDesc().getLayoutByDims(getdims);
    LOG_DEBUG("layout ="<<layout);
    InferenceEngine::TensorDesc tDesc(inputInfo.getPrecision(),inputInfo.getTensorDesc().getDims(),InferenceEngine::Layout::NHWC);
    auto blobptr = InferenceEngine::make_shared_blob<float>(tDesc,static_cast<float *>(ret.pdata));
    // auto blobptr = InferenceEngine::make_shared_blob<float>(tDesc);
    for(auto it = blobptr->begin();it != blobptr->end();it++){
        LOG_DEBUG("blot data="<<*it);
    }
    infer_request.SetBlob(inputInfo.name(), blobptr);
    LOG_DEBUG("exit");
    return 0;

}

int32_t openvino_service_driver::OpenvinoOutput_To_TensorProto(InferRequest &infer_request, DataPtr outputInfoPtr,tensorflow::TensorProto & outputproto){
    auto type = outputInfoPtr->getPrecision();
    auto allsize = 1;
    auto  numdims = outputInfoPtr->getDims().size();
    for(size_t j = 0; j < numdims; j++){
        auto dimsize = outputInfoPtr->getDims()[j];
        outputproto.mutable_tensor_shape()->add_dim()->set_size(dimsize);
        allsize *= dimsize;
    }
    Blob::Ptr outputBlob = infer_request.GetBlob(outputInfoPtr->getName());
    
    switch (type)
    {
        case Precision::FP32:
            {
                outputproto.set_dtype(tensorflow::DataType::DT_FLOAT);
                // auto datap = (const float *)this->TfOp.TF_TensorData(tensorp);
                auto datap = outputBlob->buffer().as<float*>();
                for(auto dataindex = 0; dataindex < allsize;dataindex++){
                    outputproto.add_float_val(datap[dataindex]);
                }  
            }
            break;
        case Precision::BOOL:
            {
                outputproto.set_dtype(tensorflow::DataType::DT_BOOL);
                auto datap = outputBlob->buffer().as<bool*>();
                for(auto dataindex = 0; dataindex < allsize;dataindex++){
                    outputproto.add_bool_val(datap[dataindex]);
                } 
            }

            break;
        case Precision::I32:
            {
                outputproto.set_dtype(tensorflow::DataType::DT_INT32);
                auto datap = outputBlob->buffer().as<int32_t*>();
                for(auto dataindex = 0; dataindex < allsize;dataindex++){
                    outputproto.add_int_val(datap[dataindex]);
                }
            }    
            break;
        case Precision::I64:
            {
                outputproto.set_dtype(tensorflow::DataType::DT_INT64);
                auto datap = outputBlob->buffer().as<int64_t*>();
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

string openvino_service_driver::run_predict_session(const ::tensorflow::serving::PredictRequest* request, ::tensorflow::serving::PredictResponse* response){
    string ret;
    auto &base_modelname = request->model_spec().name();
    auto &&modelversion = request->model_spec().version().value();
    auto modelname = composeModelNameKey(base_modelname,modelversion);
    auto &signaturename = request->model_spec().signature_name();
    LOG_DEBUG("modelname="<<modelname<<" signaturename="<<signaturename);
    auto modelTt = this->modelsourceMap.find(modelname);
    if(modelTt == this->modelsourceMap.end()){
        LOG_ERROR("no find such model,name="<<modelname);
        ret ="no find such model,name=" + modelname;
        return ret;
    }
    auto & ie = std::get<Core>(modelTt->second);
    auto & network =std::get<CNNNetwork>(modelTt->second);
    network.setBatchSize(1);
    ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU");
    InferRequest infer_request = executable_network.CreateInferRequest();

    auto getInputInfo =  network.getInputsInfo();
    auto req_inputs = request->inputs();
    for(auto infoIt = getInputInfo.begin();infoIt != getInputInfo.end();infoIt++){
        auto it = req_inputs.find(infoIt->first);
        if(it == req_inputs.end()){
            ret = "no find such input,name="+infoIt->first;
            LOG_ERROR(ret);
            return ret;
        }
        LOG_DEBUG("find input name="<<it->first);
        auto & tensorproto = it->second;
        auto & inputinfo  = infoIt->second;
        this->TensorProto_To_OpenvinoInput(tensorproto,infer_request,*inputinfo);
    }


    // for(auto it = request->inputs().begin();it != request->inputs().end();it++){
    //     auto infoIt = getInputInfo.find(it->first);
    //     if(infoIt ==  getInputInfo.end()){
    //         ret = "no find such input,name="+it->first;
    //         LOG_ERROR(ret);
    //         for(auto debugIt = getInputInfo.begin();debugIt != getInputInfo.end();debugIt++){
    //             LOG_INFO("real input name="<<debugIt->first);
    //         }
    //         return ret;
    //     }
    //     LOG_DEBUG("find input name="<<it->first);
    //     auto & tensorproto = it->second;
    //     auto & inputinfo  = infoIt->second;
    //     this->TensorProto_To_OpenvinoInput(tensorproto,infer_request,*inputinfo);

    // }


    infer_request.Infer();

    auto &outputmap =  *response->mutable_outputs();
    // for(auto i = 0;i < output_values.size();i++){
    //     tensorflow::TensorProto outputproto;
    //     Tensor_To_TensorProto(output_values[i],outputproto);    
    //     outputmap[outputSignatureNameVec[i]] = std::move(outputproto);
    //     LOG_DEBUG("one output,sname="<<outputSignatureNameVec[i]);
    // }
    auto  OutputsInfo = network.getOutputsInfo();
    for(auto it = OutputsInfo.begin(); it != OutputsInfo.end();it++){
        DataPtr output_info = it->second;
        std::string output_name = it->first;
        // output_info->setPrecision(Precision::FP32);
        // Blob::Ptr output = infer_request.GetBlob(output_name);
        tensorflow::TensorProto tensorproto;
        this->OpenvinoOutput_To_TensorProto(infer_request,output_info,tensorproto);
        LOG_DEBUG("output name="<<output_name<<" data="<<tensorproto.DebugString());
        outputmap[output_name] = std::move(tensorproto);
    }

    return ret;
}


    // Classify.
::grpc::Status openvino_service_driver::Classify(::grpc::ServerContext* context, const ::tensorflow::serving::ClassificationRequest* request, ::tensorflow::serving::ClassificationResponse* response){
    return Status::OK;
}
    // Regress.
::grpc::Status openvino_service_driver::Regress(::grpc::ServerContext* context, const ::tensorflow::serving::RegressionRequest* request, ::tensorflow::serving::RegressionResponse* response){
    return Status::OK;
}
    // Predict -- provides access to loaded TensorFlow model.
::grpc::Status openvino_service_driver::Predict(::grpc::ServerContext* context, const ::tensorflow::serving::PredictRequest* request, ::tensorflow::serving::PredictResponse* response){
    auto ret =this->run_predict_session(request,response);
    if(ret.empty()){
        return Status::OK;
    }else{
        LOG_ERROR("Predict error="<<ret);
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,ret);
    }
   
}
    // MultiInference API for multi-headed models.
::grpc::Status openvino_service_driver::MultiInference(::grpc::ServerContext* context, const ::tensorflow::serving::MultiInferenceRequest* request, ::tensorflow::serving::MultiInferenceResponse* response){
return Status::OK;
}
    // GetModelMetadata - provides access to metadata for loaded models.
::grpc::Status openvino_service_driver::GetModelMetadata(::grpc::ServerContext* context, const ::tensorflow::serving::GetModelMetadataRequest* request, ::tensorflow::serving::GetModelMetadataResponse* response){
return Status::OK;
}



#endif