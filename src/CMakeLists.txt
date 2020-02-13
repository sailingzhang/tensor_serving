cmake_minimum_required (VERSION 2.6)
project (vision) 

cmake_policy(PUSH)



SET(SOURCE_ROOT ${PROJECT_SOURCE_DIR}/..)
SET(THIRD_PART ${SOURCE_ROOT}/../thirdpart)




SET(INCLUDE_LIST 
	${PROJECT_SOURCE_DIR}/include/  	
	${THIRD_PART}/grpc1.27.1/include/
	${THIRD_PART}/log/
	${THIRD_PART}/libtensorflow1.15/include
	# ${THIRD_PART}/opencv4.2.0/include/opencv4/
	${PROJECT_SOURCE_DIR}/tensor_proto
	${PROJECT_SOURCE_DIR}/service
	# ${PROJECT_SOURCE_DIR}/dataset
	# ${PROJECT_SOURCE_DIR}/detect_face
	# ${PROJECT_SOURCE_DIR}/identify_face
#	${SOURCE_ROOT}/3rdparty/breakpad/include
	# ${SOURCE_ROOT}/modules/commonlib/include
)
INCLUDE_DIRECTORIES(${INCLUDE_LIST})

SET(LIB_DIRECTORY_LIST
	${PROJECT_SOURCE_DIR}
	# ${SOURCE_ROOT}/modules/authenticationcheck
	${THIRD_PART}/grpc1.27.1/linuxlib
	${THIRD_PART}/libtensorflow1.15/lib/linux_cpu
	# ${THIRD_PART}/opencv4.2.0/lib
	# ${THIRD_PART}/opencv4.2.0/lib/opencv4/3rdparty
#	${SOURCE_ROOT}/3rdparty/breakpad/lib
	# ${SOURCE_ROOT}/3rdparty/grpc/lib/linux64_4.8.5gcc
)
# set(CMAKE_C_VISIBILITY_PRESET hidden)
# set(CMAKE_CXX_VISIBILITY_PRESET hidden)
MESSAGE(STATUS "lib direcotry=: " ${LIB_DIRECTORY_LIST})
SET(LIB_LIST


	# libgpr.a 
	# libgrpc++.a 
	# libprotobuf.a
	# libprotoc.a	
	
	# # libboringssl.a 
	# # libgrpc++_core_stats.a
	# # libgrpc_cronet.a 
	# libgrpc++_error_details.a 
	# libgrpc++_reflection.a 
	# libcares.a 
	# libgrpc_unsecure.a 
	# libgrpc++_unsecure.a 
	# libz.a


	
	libgrpc++.a
	libprotobuf.a
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
	# libprotobuf-lite.a

	libupb.a
	libz.a

	libgpr.a


	libtensorflow.so
	libtensorflow_framework.so

	m
	dl
	libstdc++.a
	libstdc++fs.a
	libc.a
	# libm.a
	# libdl.a
	pthread
)

link_directories(${LIB_DIRECTORY_LIST})

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
