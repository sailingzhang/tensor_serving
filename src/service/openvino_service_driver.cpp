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
        if(!it->isload()){
            LOG_INFO("modelname="<<it->name()<<" isable load");
            continue;
        }
        LOG_INFO("...................................begin...................................................................");
        LOG_INFO("openvino service load, modelname="<<it->name()<<" version="<<it->version()<<" path="<<it->base_path());
        this->loadModel(it->name(),it->version(),it->base_path(),*it);
        LOG_INFO("....................................end..................................................................");
    }
    return ;
}

int32_t openvino_service_driver::loadModel(string modelname,int64_t version,string modeldir,serving_configure::model_config & configure){
    LOG_INFO("begin openvino load ,modelname="<<modelname<<" version="<<version<<" dir="<<modeldir);
    auto &&modelnamekey = composeModelNameKey(modelname,version);
    Core ie;
    CNNNetwork network = ie.ReadNetwork(modeldir+"/model.xml", modeldir+"/model.bin");
    LOG_INFO("network  batchsize"<<network.getBatchSize());
    network.setBatchSize(1);

    auto getInputInfo =  network.getInputsInfo();
    auto getOutputInfo = network.getOutputsInfo();
    for(auto infoIt = getInputInfo.begin();infoIt != getInputInfo.end();infoIt++){
        string shape="";
        auto  inputInfoptr  = infoIt->second;
        //set input info here.
        // inputInfoptr->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
        LOG_INFO("set layout="<<configure.layout());
        switch (configure.layout())
        {
        case serving_configure::NHWC:
            inputInfoptr->setLayout(Layout::NHWC);
            break;
        case serving_configure::NCHW:
            inputInfoptr->setLayout(Layout::NCHW);
            break;
        default:
            break;
        }
        
        auto dims = inputInfoptr->getTensorDesc().getDims();
        for(auto& dim:dims){
            shape = shape +","+to_string(dim);
        }
        inputInfoptr->setPrecision(InferenceEngine::Precision::U8);
        LOG_INFO("openvino input name="<<infoIt->first<<" shape="<<shape<<" precision="<<inputInfoptr->getPrecision());
    }
    for(auto outinfoIt = getOutputInfo.begin();outinfoIt != getOutputInfo.end();outinfoIt++){
        string shape="";
        auto outputinfoptr = outinfoIt->second;
        auto dims = outputinfoptr->getTensorDesc().getDims();
        for(auto& dim:dims){
            shape = shape +","+to_string(dim);
        }
        LOG_INFO("openvino output name="<<outinfoIt->first<<" shape="<<shape<<" precision="<<outputinfoptr->getPrecision());
    }
     
    string device_type="CPU";
    switch (configure.device())
    {
    case serving_configure::CPU:
        device_type ="CPU";
        break;
    case serving_configure::GPU:
        device_type ="GPU";
        break;
    default:
        break;
    }
    LOG_INFO("device type="<<device_type);
    // std::cout << ie.GetVersions(device_type) << std::endl;
    
    ExecutableNetwork executable_network =  ie.LoadNetwork(network, device_type);
    LOG_INFO("load ok")
    InferRequest infer_request = executable_network.CreateInferRequest();

    this->modelsourceMap[modelnamekey]={getInputInfo,infer_request,getOutputInfo};
    LOG_INFO("openvino load over,modelname="<<modelname);
    return 0;
}

InferenceEngine::Blob::Ptr  openvino_service_driver::TensorProto_To_OpenvinoInput(const tensorflow::TensorProto & from, InputInfo::Ptr  inputInfoptr,shared_ptr<OpenvinoProtoinfoS> openvinoprotoinfoPtr){
    LOG_DEBUG("enter,inputname="<<inputInfoptr->name()<<" datatype="<<from.dtype());
    InferenceEngine::Blob::Ptr blobptr_ret;
    OpenvinoProtoinfoS ret;
    ret.allbytesSize = 1;
    auto getdims = inputInfoptr->getTensorDesc().getDims();
    switch (from.dtype())
    {
        case tensorflow::DataType::DT_UINT8:
        {
            ret.opevino_dtype = Precision::U8;
            ret.dtypesize = sizeof(PrecisionTrait<Precision::U8>::value_type);
            ret.uint8Ptr = shared_ptr<uint8_t>(new uint8_t[from.int_val().size()],[](uint8_t *p){delete [] p;});
            for(auto i = 0;i < from.int_val().size();i++){
                ret.uint8Ptr.get()[i] = static_cast<uint8_t>(from.int_val()[i]);
            }
            ret.pdata=ret.uint8Ptr.get();
            InferenceEngine::TensorDesc tDesc(ret.opevino_dtype,getdims,InferenceEngine::Layout::NHWC);
            auto blobptr = InferenceEngine::make_shared_blob<uint8_t>(tDesc,static_cast<uint8_t *>(ret.pdata));
            blobptr_ret = blobptr;
            for(auto it = blobptr->begin();it != blobptr->end();it++){
                LOG_DEBUG("blot data="<<*it);
            }
        }
        break;
        case tensorflow::DataType::DT_INT8:
        {
            ret.opevino_dtype = Precision::I8;
            ret.dtypesize = sizeof(PrecisionTrait<Precision::I8>::value_type);
            ret.int8Ptr = shared_ptr<int8_t>(new int8_t[from.int_val().size()],[](int8_t *p){delete [] p;});
            for(auto i = 0;i < from.int_val().size();i++){
                ret.int8Ptr.get()[i] = static_cast<int8_t>(from.int_val()[i]);
            }
            ret.pdata=ret.int8Ptr.get();
            InferenceEngine::TensorDesc tDesc(ret.opevino_dtype,getdims,InferenceEngine::Layout::NHWC);
            auto blobptr = InferenceEngine::make_shared_blob<int8_t>(tDesc,static_cast<int8_t *>(ret.pdata));
            blobptr_ret = blobptr;
            for(auto it = blobptr->begin();it != blobptr->end();it++){
                LOG_DEBUG("blot data="<<*it);
            }
        }
        break;
        case tensorflow::DataType::DT_BFLOAT16:
        {
            ret.opevino_dtype = Precision::FP16;
            ret.dtypesize = sizeof(PrecisionTrait<Precision::FP16>::value_type);
            ret.f16Ptr = shared_ptr<PrecisionTrait<Precision::FP16>::value_type>(new PrecisionTrait<Precision::FP16>::value_type[from.int_val().size()],[](PrecisionTrait<Precision::FP16>::value_type *p){delete [] p;});
            for(auto i = 0;i < from.half_val().size();i++){
                ret.f16Ptr.get()[i] = static_cast<PrecisionTrait<Precision::FP16>::value_type>(from.half_val()[i]);
            }
            ret.pdata=ret.f16Ptr.get();
            InferenceEngine::TensorDesc tDesc(ret.opevino_dtype,getdims,InferenceEngine::Layout::NHWC);
            auto blobptr = InferenceEngine::make_shared_blob<PrecisionTrait<Precision::FP16>::value_type>(tDesc,static_cast<PrecisionTrait<Precision::FP16>::value_type *>(ret.pdata));
            blobptr_ret = blobptr;
            for(auto it = blobptr->begin();it != blobptr->end();it++){
                LOG_DEBUG("blot data="<<*it);
            }
        }
            break;
        case tensorflow::DataType::DT_FLOAT:
        {
            ret.opevino_dtype = Precision::FP32;
            ret.dtypesize = sizeof(float);
            ret.pdata=(void *)from.float_val().begin();
            InferenceEngine::TensorDesc tDesc(ret.opevino_dtype,getdims,InferenceEngine::Layout::NHWC);
            auto blobptr = InferenceEngine::make_shared_blob<PrecisionTrait<Precision::FP32>::value_type>(tDesc,static_cast<PrecisionTrait<Precision::FP32>::value_type *>(ret.pdata));
            blobptr_ret = blobptr;
            for(auto it = blobptr->begin();it != blobptr->end();it++){
                LOG_DEBUG("blot data="<<*it);
            }
        }
        break;
        case tensorflow::DataType::DT_BOOL:
        {
            ret.opevino_dtype = Precision::BOOL;
            ret.dtypesize = sizeof(bool);
            ret.pdata = (void *)from.bool_val().begin();
            InferenceEngine::TensorDesc tDesc(ret.opevino_dtype,getdims,InferenceEngine::Layout::NHWC);
            auto blobptr = InferenceEngine::make_shared_blob<PrecisionTrait<Precision::BOOL>::value_type>(tDesc,static_cast<PrecisionTrait<Precision::BOOL>::value_type *>(ret.pdata));
            blobptr_ret = blobptr;
            for(auto it = blobptr->begin();it != blobptr->end();it++){
                LOG_DEBUG("blot data="<<*it);
            }
        }
        break;
        case tensorflow::DataType::DT_INT32:
        {
            ret.opevino_dtype = Precision::I32;
            ret.dtypesize = sizeof(int32_t);
            ret.pdata = (void *)from.int_val().begin();
            InferenceEngine::TensorDesc tDesc(ret.opevino_dtype,getdims,InferenceEngine::Layout::NHWC);
            auto blobptr = InferenceEngine::make_shared_blob<PrecisionTrait<Precision::I32>::value_type>(tDesc,static_cast<PrecisionTrait<Precision::I32>::value_type *>(ret.pdata));
            blobptr_ret = blobptr;
            for(auto it = blobptr->begin();it != blobptr->end();it++){
                LOG_DEBUG("blot data="<<*it);
            }
        }
        break;
        case tensorflow::DataType::DT_INT64:
        {
            ret.opevino_dtype = Precision::I64;
            ret.dtypesize = sizeof(int64_t);
            ret.pdata =(void *) from.int64_val().begin();
            InferenceEngine::TensorDesc tDesc(ret.opevino_dtype,getdims,InferenceEngine::Layout::NHWC);
            auto blobptr = InferenceEngine::make_shared_blob<PrecisionTrait<Precision::I64>::value_type>(tDesc,static_cast<PrecisionTrait<Precision::I64>::value_type *>(ret.pdata));
            blobptr_ret = blobptr;
            for(auto it = blobptr->begin();it != blobptr->end();it++){
                LOG_DEBUG("blot data="<<*it);
            }
        }
        break;
        default:
            LOG_ERROR("not support such type="<<from.dtype());
            return nullptr;
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

    
    LOG_DEBUG("inputinfo precision="<<inputInfoptr->getPrecision()<<" layout="<<inputInfoptr->getLayout());


    // InputInfo::Ptr input_info = it->second;
    // std::string input_name = it->first;
    // inputInfoptr->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    // inputInfoptr->setLayout(Layout::NHWC);
    // inputInfoptr->setPrecision(ret.opevino_dtype);

    

    
    
    // for(auto & v:getdims){
    //     LOG_DEBUG("real input dim="<<v);
    // }
    // for( auto & v:ret.size_t_dimarr){
    //     LOG_DEBUG("client dim="<<v);
    // }

    auto layout =  inputInfoptr->getTensorDesc().getLayoutByDims(ret.size_t_dimarr);
    LOG_DEBUG("layout ="<<layout);
    // InferenceEngine::TensorDesc tDesc(inputInfo.getPrecision(),inputInfo.getTensorDesc().getDims(),InferenceEngine::Layout::NHWC);
    
   
    // auto blobptr = InferenceEngine::make_shared_blob<float>(tDesc);

    
    // infer_request.SetBlob(inputInfo.name(), blobptr);
    LOG_DEBUG("exit");
    return  blobptr_ret;


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
        case Precision::FP16:
            {
                outputproto.set_dtype(tensorflow::DataType::DT_FLOAT);
                // auto datap = (const float *)this->TfOp.TF_TensorData(tensorp);
                
                auto datap = outputBlob->buffer().as<PrecisionTrait<Precision::FP16>::value_type *>();
                for(auto dataindex = 0; dataindex < allsize;dataindex++){
                    outputproto.add_float_val(datap[dataindex]);
                }  
            }
            break;
        case Precision::FP32:
            {
                outputproto.set_dtype(tensorflow::DataType::DT_FLOAT);
                // auto datap = (const float *)this->TfOp.TF_TensorData(tensorp);
                auto datap = outputBlob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
                // auto datap = outputBlob->buffer().as<PrecisionTrait<Precision::FP16>::value_type *>();
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
    std::map<string,InferenceEngine::Blob::Ptr> Datamap;
    std::vector<shared_ptr<OpenvinoProtoinfoS>> protoinfoptrVec;

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
    auto & getInputInfo = std::get<InferenceEngine::InputsDataMap>(modelTt->second);
    auto & infer_request =std::get<InferenceEngine::InferRequest>(modelTt->second);
    auto & OutputsInfo  = std::get<InferenceEngine::OutputsDataMap>(modelTt->second);

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
        auto  inputinfoptr  = infoIt->second;
        
        auto infoptr = make_shared<OpenvinoProtoinfoS>() ;
        protoinfoptrVec.push_back(infoptr);
        Datamap[it->first]= this->TensorProto_To_OpenvinoInput(tensorproto,inputinfoptr,infoptr);
    }

    for(auto it= Datamap.begin();it != Datamap.end();it++){
        infer_request.SetBlob(it->first,it->second);
    }

    infer_request.Infer();

    auto &outputmap =  *response->mutable_outputs();

    for(auto it = OutputsInfo.begin(); it != OutputsInfo.end();it++){
        DataPtr output_info = it->second;
        std::string output_name = it->first;
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