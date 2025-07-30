# GPU utility components
# This CMakeLists.txt is included only when SPADES_GPU_SUPPORT is enabled

if(SPADES_GPU_SUPPORT AND SPADES_CUDA_SUPPORT)
    project(gpu_utils CXX CUDA)
    
    # Collect GPU utility sources
    set(GPU_UTILS_SOURCES
        gpu_device.cpp
        gpu_memory.cpp
    )
    
    set(GPU_UTILS_HEADERS
        gpu_device.hpp
        gpu_memory.hpp
    )
    
    # Create the GPU utilities library
    add_library(gpu_utils STATIC ${GPU_UTILS_SOURCES} ${GPU_UTILS_HEADERS})
    
    # Set CUDA properties
    set_target_properties(gpu_utils PROPERTIES
        CXX_STANDARD 17
    )
    
    # Link CUDA libraries
    target_link_libraries(gpu_utils 
        ${GPU_LIBRARIES}
    )
    
    # Include directories
    target_include_directories(gpu_utils PRIVATE
        ${CMAKE_SOURCE_DIR}/src/common
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    
    # Export for other modules to use
    set(GPU_UTILS_LIBRARY gpu_utils PARENT_SCOPE)
    
else()
    # Create empty library when GPU support is disabled
    add_library(gpu_utils INTERFACE)
    set(GPU_UTILS_LIBRARY gpu_utils PARENT_SCOPE)
endif() 