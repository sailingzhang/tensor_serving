# tensor_serving

It is a common Tenser Serving for   Neural network inference engine. You can access the Neural network by Tenser Serving no matter what type of engine backend. It support Tensorflow1.x ,Openvino.   Supporting TensorRt is in plan.   


Build   
need gcc/g++8.x or later  
need cmake3  
(1)git clone https://github.com/sailingzhang/tensor_serving.git  
(2)git clone https://github.com/sailingzhang/thirdpart.git  
(3)mkdir build ;cd build   
(4)cmake ../tensor_serving/src  
(5)make -j8  