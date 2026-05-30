cmake_minimum_required(VERSION 3.20)
project(grand-pattern-opencl LANGUAGES C)

set(CMAKE_C_STANDARD 11)

find_package(OpenCL REQUIRED)

# Copy kernel files to build directory
file(GLOB CL_KERNELS "${CMAKE_SOURCE_DIR}/src/kernels/*.cl")
file(COPY ${CL_KERNELS} DESTINATION ${CMAKE_BINARY_DIR}/kernels)

add_executable(test_opencl_kernels
    src/host/main.c
    src/host/opencl_context.c
    src/host/test_cosine.c
    src/host/test_batch_predict.c
    src/host/test_balance.c
    src/host/test_decay.c
    src/host/test_vibe.c
    src/host/test_correlation.c
    src/host/test_merge.c
    src/host/test_e2e.c
)

target_include_directories(test_opencl_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/src/host
    ${OpenCL_INCLUDE_DIRS}
)

target_link_libraries(test_opencl_kernels ${OpenCL_LIBRARY} m)

target_compile_definitions(test_opencl_kernels PRIVATE
    CL_KERNEL_DIR="${CMAKE_BINARY_DIR}/kernels"
)
