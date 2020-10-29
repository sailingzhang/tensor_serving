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

cp -rf ${BASEDIR}/webapp/cactus  /var/www
nginx -c  ${BASEDIR}/webapp/cactus/nginx_static.conf  &
${BASEDIR}/grpcwebproxy --backend_addr=localhost:8000  --run_tls_server=false --use_websockets --backend_max_call_recv_msg_size=9242880  --allow_all_origins  >/dev/null 2>&1 &
${BASEDIR}/cactusServer  ${BASEDIR}/share/cactus_configure.json

${BASEDIR}/tensor_serving 
# pkill -9 grpcwebproxy
# pkill -9 nginx
