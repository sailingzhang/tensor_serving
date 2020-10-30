#!/bin/bash
BASEDIR=$(dirname $(realpath "$0"))
echo basedir=${BASEDIR}
chmod +x ${BASEDIR}/tensor_serving
# cp ${BASEDIR}/*.proto  ${BASEDIR}/share
# cp -rf ${BASEDIR}/test  ${BASEDIR}/share


# if [ ! -f "${BASEDIR}/share/s_log.conf" ]; then
# 	cp  -f ${BASEDIR}/s_log.conf  ${BASEDIR}/share
# fi
# if [ ! -f "${BASEDIR}/share/cactus_configure.json" ]; then
# 	cp  -f ${BASEDIR}/openvino_cactus_configure.json  ${BASEDIR}/share/cactus_configure.json
# fi


# cp -rf ${BASEDIR}/webapp/cactus /var/www

# pkill -9 grpcwebproxy
# pkill -9 cactusServer
# pkill -9 grpcwebproxy
# pkill -9 nginx
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${BASEDIR}

${BASEDIR}/tensor_serving  ${BASEDIR}/servingmodel/docker_serving_model.json
# pkill -9 grpcwebproxy
# pkill -9 nginx
