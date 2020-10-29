

ARG VERSION=latest
ARG BUILD_IMAGE=cactusRelease:${TF_VERSION}


FROM ubuntu:20.04
# MAINTAINER sailingzhang

######################################arg#########################################################################
ARG APPDIR=/app 



###############################tensorflow serving#############################################################


################################################################################################################
#ENV PATH ${APPDIR}:$PATH
WORKDIR  ${APPDIR}
COPY tensor_serving_release  ${APPDIR}
# RUN chmod +x start.sh &&\
#     apt-get update &&\
#     apt-get install -y nginx &&\
#     apt-get install -y  openssl &&\
#     rm -rf /tmp/* && \
#     apt-get clean && rm -rf /var/lib/apt/lists/* 
# RUN echo "root:xtygajp2243374"|chpasswd 
###########################################################################################################
#COPY cactus ${APPDIR}






##############################################################################################################

EXPOSE 8080
EXPOSE 8000
EXPOSE 81

# EXPOSE 8002
# COPY app ${APPDIR}
# RUN chmod +x ${APPDIR}/start
ENTRYPOINT ["./start.sh"]




