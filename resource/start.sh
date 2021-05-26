#!/bin/bash
BASEDIR=$(dirname $(realpath "$0"))
echo basedir=${BASEDIR}
chmod +x ${BASEDIR}/tensor_serving
# cp ${BASEDIR}/*.proto  ${BASEDIR}/share
# cp -rf ${BASEDIR}/test  ${BASEDIR}/share

configdir=/app_config/openvino_tensorserving

mkdir -p ${configdir}

if [ ! -f "${configdir}/s_log.conf" ]; then
	cp  -f ${BASEDIR}/s_log.conf  ${configdir}
fi
if [ ! -f "${configdir}/docker_serving_model.json" ]; then
	cp  -f ${BASEDIR}/docker_serving_model.json ${configdir}
fi


# cp -rf ${BASEDIR}/webapp/cactus /var/www

# pkill -9 grpcwebproxy
# pkill -9 cactusServer
# pkill -9 grpcwebproxy
# pkill -9 nginx
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${BASEDIR}

${BASEDIR}/tensor_serving  ${configdir}/docker_serving_model.json
# pkill -9 grpcwebproxy
# pkill -9 nginx





# docker container   run --rm  -v /app_config/servingmodel:/app_config/servingmodel -p 9001:9001 -it openvino_tensor_serving:latest   /app/openvino_tensor_serving/tensor_serving /app_config/servingmodel/docker_serving_model.json