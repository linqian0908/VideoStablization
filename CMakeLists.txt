add_executable(VideoStablization VideoStablization.cpp)


# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)

include_directories(../../lib/local/LandmarkDetector/include)
include_directories(../../lib/local/FaceAnalyser/include)		
include_directories(../../exe/VideoStablization/)		

target_link_libraries(VideoStablization LandmarkDetector)
target_link_libraries(VideoStablization FaceAnalyser)
target_link_libraries(VideoStablization dlib)

target_link_libraries(VideoStablization ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES})

install (TARGETS VideoStablization DESTINATION ${CMAKE_BINARY_DIR}/bin)
