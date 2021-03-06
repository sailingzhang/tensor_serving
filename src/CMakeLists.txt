cmake_minimum_required (VERSION 2.6)
project (vision) 

cmake_policy(PUSH)






SET(SOURCE_ROOT ${PROJECT_SOURCE_DIR}/..)
SET(THIRD_PART ${SOURCE_ROOT}/../thirdpart)

# set(InferenceEngine_DIR /opt/intel/openvino/inference_engine/share/)
# find_package(InferenceEngine 2.1 REQUIRED)

SET(GRPC_DIR ${THIRD_PART}/grpc1.28.1)
SET(OPEVINO_DIR ${THIRD_PART}/openvino_2020.1.023)


file(READ "/etc/issue" ETC_ISSUE)
string(REGEX MATCH "Debian|Ubuntu" DIST ${ETC_ISSUE})
if(DIST STREQUAL "Debian")
    message(STATUS ">>>> Found Debian <<<<")
elseif(DIST STREQUAL "Ubuntu")
	SET(GRPC_LIB_DIR ${GRPC_DIR}/ubuntu_lib)
	SET(OPENVINO_LIB_DIR ${OPEVINO_DIR}/inference_engine/ubuntu_lib)
	SET(OPENVINO_LIB_NAME 
		libinference_engine.so
		libngraph.so
		# libtbb.so.2
	)
	message(STATUS ">>>> Found Ubuntu,grpclib_dir=${GRPC_LIB_DIR}, openvino_lib_dir=${OPENVINO_LIB_DIR}<<<<")
else()
	SET(GRPC_LIB_DIR ${GRPC_DIR}/centos_lib)
	SET(OPENVINO_LIB_DIR ${OPEVINO_DIR}/inference_engine/centos_lib)
	SET(OPENVINO_LIB_NAME 
		libinference_engine_c_api.so
		libinference_engine_legacy.so
		libinference_engine_nn_builder.so
		libinference_engine.so
		libinference_engine_transformations.so
		libMKLDNNPlugin.so
		libngraph.so
		libtbbmalloc.so.2
		libtbb.so.2
	)
	message(STATUS ">>>> Found Other system, grpclib_dir=${GRPC_LIB_DIR}, openvino_lib_dir=${OPENVINO_LIB_DIR}<<<<")
endif()


SET(INCLUDE_LIST 
	${PROJECT_SOURCE_DIR}/include/  	
	${GRPC_DIR}/include/
    ${THIRD_PART}/log/
    ${THIRD_PART}/commoncpp/
	${THIRD_PART}/libtensorflow1.15/include
	# ${THIRD_PART}/opencv4.2.0/include/opencv4/
	${PROJECT_SOURCE_DIR}/tensor_proto
	${PROJECT_SOURCE_DIR}/service
	# ${PROJECT_SOURCE_DIR}/dataset
	# ${PROJECT_SOURCE_DIR}/detect_face
	# ${PROJECT_SOURCE_DIR}/identify_face
#	${SOURCE_ROOT}/3rdparty/breakpad/include
	# ${SOURCE_ROOT}/modules/commonlib/include
	# ${InferenceEngine_INCLUDE}
	${OPEVINO_DIR}/inference_engine
    ${OPEVINO_DIR}/inference_engine/include
    ${OPEVINO_DIR}/inference_engine/samples/cpp/common
)
INCLUDE_DIRECTORIES(${INCLUDE_LIST})

SET(LIB_DIRECTORY_LIST
	${PROJECT_SOURCE_DIR}
	# ${SOURCE_ROOT}/modules/authenticationcheck
	${GRPC_LIB_DIR}
	${OPENVINO_LIB_DIR}
	${THIRD_PART}/libtensorflow1.15/lib/linux_cpu
	# ${THIRD_PART}/opencv4.2.0/lib
	# ${THIRD_PART}/opencv4.2.0/lib/opencv4/3rdparty
#	${SOURCE_ROOT}/3rdparty/breakpad/lib
	# ${SOURCE_ROOT}/3rdparty/grpc/lib/linux64_4.8.5gcc

	/opt/intel/openvino/inference_engine/lib/intel64/
	/opt/intel/openvino/inference_engine/external/gna/lib
	/opt/intel/openvino/inference_engine/external/mkltiny_lnx/lib
	/opt/intel/openvino/inference_engine/external/tbb/lib
)
# set(CMAKE_C_VISIBILITY_PRESET hidden)
# set(CMAKE_CXX_VISIBILITY_PRESET hidden)
MESSAGE(STATUS "lib direcotry=: " ${LIB_DIRECTORY_LIST})
# MESSAGE(STATUS "openvino lib =: " ${InferenceEngine_LIBRARIES})
SET(LIB_LIST

	libgrpc++.a	
	libprotoc.a
	libgrpc++_error_details.a
	libgrpc++_reflection.a
	libgrpc_unsecure.a
	libaddress_sorting.a
	libcares.a
	libgrpc++_alts.a
	libgrpc.a
	libgrpc_cronet.a
	libgrpc_plugin_support.a
	libgrpcpp_channelz.a
	libprotobuf-lite.a
	libupb.a
	libgpr.a
	libprotobuf.a
	libz.a

	libabsl_bad_optional_access.a
	libabsl_base.a
	libabsl_dynamic_annotations.a
	libabsl_log_severity.a
	libabsl_spinlock_wait.a
	libabsl_str_format_internal.a
	libabsl_strings.a
	libabsl_strings_internal.a
	libabsl_throw_delegate.a
	libabsl_int128.a
	libabsl_raw_logging_internal.a


	${OPENVINO_LIB_NAME}

	# libinference_engine.so
	# libngraph.so
	# libtbb.so.2


	# libabsl_str_format_internal.a
	# libabsl_strings_internal.a
	# libabsl_base.a

	# ${InferenceEngine_LIBRARIES}
	# ${IE_SAMPLE_DEPENDENCIES}

	m
	dl
	# libstdc++.a
	libstdc++fs.a
	# libc.a
	# libm.a
	# libdl.a
	pthread

	
)


link_directories(${LIB_DIRECTORY_LIST})
AUX_SOURCE_DIRECTORY(${THIRD_PART}/commoncpp SRC_LIST)
AUX_SOURCE_DIRECTORY(${THIRD_PART}/log SRC_LIST)
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/service SRC_LIST)
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/tensor_proto/google/protobuf SRC_LIST)
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/tensor_proto/tensorflow/core/framework SRC_LIST)
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/tensor_proto/tensorflow/core/example SRC_LIST) 
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/tensor_proto/tensorflow/core/protobuf SRC_LIST)
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/tensor_proto/tensorflow_serving/apis SRC_LIST)
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR} SRC_LIST)

# set(TEST_SOURCE ${SRC_LIST})
set(MAIN_SOURCE ${SRC_LIST})
# list(REMOVE_ITEM TEST_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
# list(REMOVE_ITEM MAIN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)

MESSAGE(STATUS "include dir is: " ${INCLUDE_LIST})
# add_library(_vision SHARED  ${SRC_LIST})
# add_executable(tensorgrpcC  ${TEST_SOURCE})
add_executable(tensor_serving  ${MAIN_SOURCE})



# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)


IF(WIN32)
	add_definitions(-DWIN32 -D_CONSOLE -D_WIN32_WINNT=0x600  -D_UNICODE)
	target_link_libraries(_vision crypto decrepit gpr grpc grpc++ libCoreTransporter libprotobuf libprotoc ssl zlibstatic log4cplus${LIB_NAME})
ELSE()
	add_definitions(-std=c++17)
	add_definitions(-static-libgcc)
	add_definitions(-static-libstdc++)
	add_definitions(-DELPP_THREAD_SAFE)
	target_link_libraries(tensor_serving ${LIB_LIST})
ENDIF()

# execute_process(
#     COMMAND bash -c "ldd tensorgrpcC | awk '{print $3}' | xargs -i cp -L {}  ~/tmp"
# )
# ldd tensorgrpcC | awk '{print $3}' | xargs -i cp -L {}  ~/tmp

cmake_policy(POP)
