set(CMAKE_CUDA_ARCHITECTURES 90a)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(vllm-flash-attention LANGUAGES CXX CUDA)


# Set the C++ standard to use
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler options
add_compile_options(-Wno-deprecated-declarations)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Debug build settings
# set(CMAKE_BUILD_TYPE Debug)

# CUDA options
# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math -t 8 \
#                       -gencode arch=compute_90a,code=sm_90a")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G --expt-relaxed-constexpr --use_fast_math -t 8")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g -G")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -lineinfo --expt-relaxed-constexpr --use_fast_math -gencode arch=compute_90a,code=sm_90a")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g  -lineinfo --expt-relaxed-constexpr --use_fast_math -gencode arch=compute_90a,code=sm_90a")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g -lineinfo --expt-relaxed-constexpr --use_fast_math -gencode arch=compute_90a,code=sm_90a")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g -lineinfo --expt-relaxed-constexpr --use_fast_math -gencode arch=compute_90a,code=sm_90a")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=--verbose,--warn-on-local-memory-usage -Xnvlink=--verbose -Xptxas=--verbose")

# Python settings
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "PYTHON_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}, PYTHON_LIBRARIES: ${Python3_LIBRARIES}")

# PyTorch settings
set(TORCH_PATH "/usr/local/lib/python3.10/dist-packages/torch")
set(CMAKE_PREFIX_PATH ${TORCH_PATH}/share/cmake/Torch)
# CUDA architectures setting
# set(TORCH_CUDA_ARCH_LIST "Ampere" CACHE STRING "List of CUDA architectures")
set(TORCH_CUDA_ARCH_LIST "Hopper" CACHE STRING "List of CUDA architectures")
find_package(Torch REQUIRED)
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")

# Include directories
# set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/cutlass)
set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cutlass)
include_directories(
    ${CUTLASS_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# switch
add_definitions(-DFLASHATTENTION_DISABLE_ALIBI)
add_definitions(-DFLASHATTENTION_DISABLE_DROPOUT)
add_definitions(-DFLASHATTENTION_DISABLE_UNEVEN_K)
add_definitions(-DFLASHATTENTION_DISABLE_LOCAL)
add_definitions(-DC_DEBUG)
# add_definitions(-DMLA_DEBUG)
# add_definitions(-DLP_DEBUG)


# force use sm90a for cutlass
string(REGEX REPLACE "sm_90" "sm_90a" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
string(REGEX REPLACE "compute_90" "compute_90a" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})

# # Create a shared library
# set(FLASH_ATTN_CU
#     # "flash_fwd_hdim128_fp16_sm90.cu"
#     # "flash_fwd_hdim128_bf16_sm90.cu"
#     # "flash_fwd_hdim128_e4m3_sm90.cu"
#     # "flash_fwd_hdim256_fp16_sm90.cu"
#     # "flash_fwd_hdim256_bf16_sm90.cu"
#     # "flash_fwd_hdim256_e4m3_sm90.cu"
#     # "flash_fwd_hdim512_e4m3_sm90.cu"
#     "flash_fwd_hdim576_e4m3_sm90.cu"
#     )

# add_library(vllm_flash_attn_2_cuda SHARED
#     "flash_api.cpp"
#     ${FLASH_ATTN_CU})

# target_link_libraries(vllm_flash_attn_2_cuda PUBLIC ${TORCH_LIBRARIES} ${Python3_LIBRARIES})
# set_target_properties(vllm_flash_attn_2_cuda PROPERTIES CUDA_ARCHITECTURES "90a")

# # Executable
# add_executable(main main.cpp)
# target_link_libraries(main PUBLIC vllm_flash_attn_2_cuda)

add_executable(check_tensor check_tensor.cpp)
target_link_libraries(check_tensor torch)
set_target_properties(check_tensor PROPERTIES CUDA_ARCHITECTURES "90a")