#!/bin/bash
set -e


basepath=`pwd`/..
mygit_path=${basepath}/..
build_dir=/tmp/tensor_serving_build
releasedir=/app/tensor_serving
# isubuntu=`cat /etc/os-release|grep ubuntu` 
isubuntu="yes"

mkdir -p ${releasedir}
cp start.sh ${releasedir}
cp s_log.conf  ${releasedir}
cp docker_serving_model.json ${releasedir}

cd ${build_dir}
cmake ${basepath}/src
make -j4

if [ -z $isubuntu ]; then
    echo "not ubuntu"
    cp ${mygit_path}/thirdpart/openvino_2020.1.023/inference_engine/centos_lib/* ${releasedir}
else
    echo "ubuntu"
    cp ${mygit_path}/thirdpart/openvino_2020.1.023/inference_engine/ubuntu_lib/* ${releasedir}
fi
cp tensor_serving ${releasedir}



cp ${basepath}/resource/Dockerfile  .
sudo docker build -t openvino_tensor_serving:latest .





