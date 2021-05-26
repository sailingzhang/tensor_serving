

ARG VERSION=latest
ARG BUILD_IMAGE=cactusRelease:${TF_VERSION}


FROM ubuntu:20.04
# MAINTAINER sailingzhang



######################################arg#########################################################################
ARG APPDIR=/app/openvino_tensor_serving 
ENV LD_LIBRARY_PATH="${APPDIR}:${LD_LIBRARY_PATH}"


###############################tensorflow serving#############################################################


################################################################################################################
#ENV PATH ${APPDIR}:$PATH
WORKDIR  ${APPDIR}
COPY openvino_tensor_serving  ${APPDIR}
RUN chmod +x start.sh &&\
    apt-get -y update &&\
    apt-get install  -y net-tools &&\
    apt-get install -y iputils-ping &&\
#     apt-get install -y nginx &&\
#     apt-get install -y  openssl &&\
#     rm -rf /tmp/* && \
    apt-get clean && rm -rf /var/lib/apt/lists/* 
# RUN echo "root:xtygajp2243374"|chpasswd 
###########################################################################################################
#COPY cactus ${APPDIR}







##############################################################################################################

# EXPOSE 8080
# EXPOSE 8000
# EXPOSE 81


EXPOSE 9001

# EXPOSE 8002
# COPY app ${APPDIR}
# RUN chmod +x ${APPDIR}/start
# ENTRYPOINT ["./start.sh"]




