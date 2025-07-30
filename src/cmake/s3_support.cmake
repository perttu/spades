# S3 Support Configuration for SPAdES
# This file handles AWS SDK dependencies and S3 support options

option(SPADES_S3_SUPPORT "Enable S3 support for SPAdES" OFF)
option(SPADES_S3_AWS_SDK "Use AWS SDK for S3 operations" ON)
option(SPADES_S3_CURL "Use libcurl for S3 operations (alternative to AWS SDK)" OFF)

if(SPADES_S3_SUPPORT)
    message(STATUS "S3 support enabled")
    
    if(SPADES_S3_AWS_SDK)
        # Find AWS SDK
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(AWS_SDK aws-cpp-sdk-s3)
        endif()
        
        if(AWS_SDK_FOUND)
            message(STATUS "Found AWS SDK: ${AWS_SDK_VERSION}")
            add_definitions(-DSPADES_S3_SUPPORT)
            add_definitions(-DSPADES_S3_AWS_SDK)
            
            # Add AWS SDK libraries
            set(SPADES_S3_LIBRARIES ${AWS_SDK_LIBRARIES})
            set(SPADES_S3_INCLUDE_DIRS ${AWS_SDK_INCLUDE_DIRS})
            set(SPADES_S3_LDFLAGS ${AWS_SDK_LDFLAGS_OTHER})
            
        else()
            message(WARNING "AWS SDK not found. S3 support will be limited.")
            # Fall back to basic S3 support without AWS SDK
            add_definitions(-DSPADES_S3_SUPPORT)
            add_definitions(-DSPADES_S3_BASIC)
        endif()
        
    elseif(SPADES_S3_CURL)
        # Use libcurl for S3 operations
        find_package(CURL REQUIRED)
        if(CURL_FOUND)
            message(STATUS "Found libcurl: ${CURL_VERSION}")
            add_definitions(-DSPADES_S3_SUPPORT)
            add_definitions(-DSPADES_S3_CURL)
            
            set(SPADES_S3_LIBRARIES ${CURL_LIBRARIES})
            set(SPADES_S3_INCLUDE_DIRS ${CURL_INCLUDE_DIRS})
            
        else()
            message(FATAL_ERROR "libcurl not found. Cannot enable S3 support with CURL backend.")
        endif()
        
    else()
        # Basic S3 support without external dependencies
        message(STATUS "Using basic S3 support (no external dependencies)")
        add_definitions(-DSPADES_S3_SUPPORT)
        add_definitions(-DSPADES_S3_BASIC)
    endif()
    
    # Add S3 filesystem source files
    set(SPADES_S3_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/common/utils/filesystem/s3_file_system.cpp
    )
    
    # Add S3 filesystem headers
    set(SPADES_S3_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/common/utils/filesystem/s3_file_system.hpp
    )
    
else()
    message(STATUS "S3 support disabled")
endif()

# Function to link S3 libraries to a target
function(target_link_s3 TARGET_NAME)
    if(SPADES_S3_SUPPORT)
        target_link_libraries(${TARGET_NAME} ${SPADES_S3_LIBRARIES})
        target_include_directories(${TARGET_NAME} PRIVATE ${SPADES_S3_INCLUDE_DIRS})
        if(SPADES_S3_LDFLAGS)
            target_link_options(${TARGET_NAME} PRIVATE ${SPADES_S3_LDFLAGS})
        endif()
    endif()
endfunction() 