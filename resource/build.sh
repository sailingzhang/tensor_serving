#!/bin/bash
set -e


basepath=`pwd`/..
mygit_path=${basepath}/..
build_dir=/tmp/openvino_tensor_serving
docker_builddir=${build_dir}/release
tmp_releasedir=${docker_builddir}/openvino_tensor_serving
releasedir=/app/openvino_tensor_serving
# isubuntu=`cat /etc/os-release|grep ubuntu` 
isubuntu="yes"

rm -rf ${releasedir}
rm -rf  ${tmp_releasedir}
mkdir -p ${build_dir}
mkdir -p ${releasedir}
mkdir -p ${tmp_releasedir}

cp start.sh ${tmp_releasedir}
cp s_log.conf  ${tmp_releasedir}
cp docker_serving_model.json ${tmp_releasedir}

cd ${build_dir}
cmake ${basepath}/src
make -j4

if [ -z $isubuntu ]; then
    echo "not ubuntu"
    cp ${mygit_path}/thirdpart/openvino_2020.1.023/inference_engine/centos_lib/* ${tmp_releasedir}
else
    echo "ubuntu"
    cp ${mygit_path}/thirdpart/openvino_2020.1.023/inference_engine/ubuntu_lib/* ${tmp_releasedir}
fi
cp tensor_serving ${tmp_releasedir}


cd ${docker_builddir}
cp ${basepath}/resource/Dockerfile  .
sudo docker image rm -f openvino_tensor_serving:latest
sudo docker build -t openvino_tensor_serving:latest .






