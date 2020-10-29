#!/bin/bash
set -e


basepath=`pwd`/..
mygit_path=${basepath}/..
build_dir=/tmp/tensor_serving_build
releasedir=${build_dir}/tensor_serving_release
# isubuntu=`cat /etc/os-release|grep ubuntu` 
isubuntu=""

mkdir -p ${releasedir}
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
sudo docker build -t tensor_serving_test:0.1 .





