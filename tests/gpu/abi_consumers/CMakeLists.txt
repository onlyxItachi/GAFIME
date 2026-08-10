cmake_minimum_required(VERSION 3.26)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(gafime_abi_consumers LANGUAGES C)
    include(CTest)
    set(GAFIME_ABI_CONSUMER_PAYLOAD_PATH "" CACHE FILEPATH
        "Payload shared library used by standalone ABI consumer tests")
    set(GAFIME_ABI_CONSUMER_BACKEND_KIND "2" CACHE STRING
        "GafimeBackendKind value for the payload under test")
    set(GAFIME_ABI_CONSUMER_EXPECTED_ROUTE_COUNT "3" CACHE STRING
        "Expected supported numeric-route count")
    if(NOT DEFINED GAFIME_ABI_CONSUMER_TEST_SUFFIX)
        set(GAFIME_ABI_CONSUMER_TEST_SUFFIX "standalone")
    endif()
else()
    enable_language(C)
endif()

if(NOT DEFINED GAFIME_ABI_CONSUMER_TEST_SUFFIX)
    set(GAFIME_ABI_CONSUMER_TEST_SUFFIX "payload")
endif()

if(DEFINED GAFIME_ABI_CONSUMER_PAYLOAD_TARGET AND
   TARGET ${GAFIME_ABI_CONSUMER_PAYLOAD_TARGET})
    set(_gafime_abi_payload "$<TARGET_FILE:${GAFIME_ABI_CONSUMER_PAYLOAD_TARGET}>")
    set(_gafime_abi_payload_target "${GAFIME_ABI_CONSUMER_PAYLOAD_TARGET}")
elseif(GAFIME_ABI_CONSUMER_PAYLOAD_PATH)
    set(_gafime_abi_payload "${GAFIME_ABI_CONSUMER_PAYLOAD_PATH}")
    set(_gafime_abi_payload_target "")
else()
    message(FATAL_ERROR
        "Set GAFIME_ABI_CONSUMER_PAYLOAD_TARGET or GAFIME_ABI_CONSUMER_PAYLOAD_PATH")
endif()

set(_gafime_abi_suffix "${GAFIME_ABI_CONSUMER_TEST_SUFFIX}")

function(gafime_configure_c_consumer target)
    set_target_properties(${target} PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF
    )
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
    target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Werror)
    elseif(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX)
    endif()
    if(_gafime_abi_payload_target)
        add_dependencies(${target} ${_gafime_abi_payload_target})
    endif()
endfunction()

set(_gafime_abi10_target "gafime_abi_1_0_c_consumer_${_gafime_abi_suffix}")
add_executable(${_gafime_abi10_target} abi_1_0_c_consumer.c)
gafime_configure_c_consumer(${_gafime_abi10_target})
add_test(
    NAME ${_gafime_abi10_target}
    COMMAND ${_gafime_abi10_target}
        "${_gafime_abi_payload}"
        "${GAFIME_ABI_CONSUMER_BACKEND_KIND}"
)
set_tests_properties(${_gafime_abi10_target} PROPERTIES SKIP_RETURN_CODE 77)

if(GAFIME_ABI_CONSUMER_ENABLE_DYNAMIC_ARITY)
    set(_gafime_abi10_dynamic_target
        "gafime_abi_1_0_dynamic_arity6_${_gafime_abi_suffix}")
    add_executable(
        ${_gafime_abi10_dynamic_target}
        abi_1_0_dynamic_arity6_consumer.c
    )
    gafime_configure_c_consumer(${_gafime_abi10_dynamic_target})
    add_test(
        NAME ${_gafime_abi10_dynamic_target}
        COMMAND ${_gafime_abi10_dynamic_target}
            "${_gafime_abi_payload}"
            "${GAFIME_ABI_CONSUMER_BACKEND_KIND}"
    )
    set_tests_properties(${_gafime_abi10_dynamic_target} PROPERTIES SKIP_RETURN_CODE 77)
endif()

set(_gafime_abi11_target "gafime_abi_1_1_c_consumer_${_gafime_abi_suffix}")
add_executable(${_gafime_abi11_target} abi_1_1_c_consumer.c)
gafime_configure_c_consumer(${_gafime_abi11_target})
add_test(
    NAME ${_gafime_abi11_target}
    COMMAND ${_gafime_abi11_target}
        "${_gafime_abi_payload}"
        "${GAFIME_ABI_CONSUMER_BACKEND_KIND}"
        "${GAFIME_ABI_CONSUMER_EXPECTED_ROUTE_COUNT}"
)
set_tests_properties(${_gafime_abi11_target} PROPERTIES SKIP_RETURN_CODE 77)

option(GAFIME_ABI_CONSUMER_ENABLE_TYPED_BASELINE
    "Build and run the historical pre-freeze typed ABI 1.1 baseline consumer" OFF)
if(GAFIME_ABI_CONSUMER_ENABLE_TYPED_BASELINE)
    set(_gafime_abi11_typed_target
        "gafime_abi_1_1_typed_c_consumer_${_gafime_abi_suffix}")
    add_executable(
        ${_gafime_abi11_typed_target}
        abi_1_1_typed_c_consumer.c
    )
    gafime_configure_c_consumer(${_gafime_abi11_typed_target})
    add_test(
        NAME ${_gafime_abi11_typed_target}
        COMMAND ${_gafime_abi11_typed_target}
            "${_gafime_abi_payload}"
            "${GAFIME_ABI_CONSUMER_BACKEND_KIND}"
            "${GAFIME_ABI_CONSUMER_EXPECTED_ROUTE_COUNT}"
    )
    set_tests_properties(
        ${_gafime_abi11_typed_target}
        PROPERTIES SKIP_RETURN_CODE 77
    )
endif()

set(_gafime_cross_generation_target
    "gafime_abi_cross_generation_consumer_${_gafime_abi_suffix}")
add_executable(
    ${_gafime_cross_generation_target}
    abi_cross_generation_consumer.c
)
gafime_configure_c_consumer(${_gafime_cross_generation_target})
add_test(
    NAME ${_gafime_cross_generation_target}
    COMMAND ${_gafime_cross_generation_target}
        "${_gafime_abi_payload}"
        "${GAFIME_ABI_CONSUMER_BACKEND_KIND}"
)
set_tests_properties(${_gafime_cross_generation_target} PROPERTIES SKIP_RETURN_CODE 77)

set(_gafime_fixture_target "gafime_abi_1_2_route_fixture_${_gafime_abi_suffix}")
add_library(${_gafime_fixture_target} SHARED abi_1_2_route_fixture.c)
set_target_properties(${_gafime_fixture_target} PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
)
target_compile_definitions(${_gafime_fixture_target} PRIVATE GAFIME_GPU_BUILDING_DLL)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${_gafime_fixture_target} PRIVATE -Wall -Wextra -Werror)
elseif(MSVC)
    target_compile_options(${_gafime_fixture_target} PRIVATE /W4 /WX)
endif()

set(_gafime_route_test_target "gafime_abi_route_compatibility_${_gafime_abi_suffix}")
add_executable(${_gafime_route_test_target} abi_route_compatibility_test.c)
gafime_configure_c_consumer(${_gafime_route_test_target})
add_dependencies(${_gafime_route_test_target} ${_gafime_fixture_target})
add_test(
    NAME ${_gafime_route_test_target}
    COMMAND ${_gafime_route_test_target} "$<TARGET_FILE:${_gafime_fixture_target}>"
)

find_program(GAFIME_RUSTC_EXECUTABLE NAMES rustc)
if(GAFIME_RUSTC_EXECUTABLE)
    set(_gafime_rust_output
        "${CMAKE_CURRENT_BINARY_DIR}/gafime_abi_1_1_rust_consumer_${_gafime_abi_suffix}${CMAKE_EXECUTABLE_SUFFIX}")
    set(_gafime_rust_target "gafime_abi_1_1_rust_consumer_${_gafime_abi_suffix}_build")
    add_custom_command(
        OUTPUT "${_gafime_rust_output}"
        COMMAND "${GAFIME_RUSTC_EXECUTABLE}"
            --edition=2021
            -Dwarnings
            "${CMAKE_CURRENT_LIST_DIR}/abi_1_1_rust_consumer.rs"
            -o "${_gafime_rust_output}"
        DEPENDS "${CMAKE_CURRENT_LIST_DIR}/abi_1_1_rust_consumer.rs"
        VERBATIM
    )
    add_custom_target(${_gafime_rust_target} ALL DEPENDS "${_gafime_rust_output}")
    if(_gafime_abi_payload_target)
        add_dependencies(${_gafime_rust_target} ${_gafime_abi_payload_target})
    endif()
    add_test(
        NAME "gafime_abi_1_1_rust_consumer_${_gafime_abi_suffix}"
        COMMAND "${_gafime_rust_output}"
            "${_gafime_abi_payload}"
            "${GAFIME_ABI_CONSUMER_BACKEND_KIND}"
            "${GAFIME_ABI_CONSUMER_EXPECTED_ROUTE_COUNT}"
    )
    set_tests_properties(
        "gafime_abi_1_1_rust_consumer_${_gafime_abi_suffix}"
        PROPERTIES SKIP_RETURN_CODE 77
    )
else()
    message(STATUS "rustc not found; standalone Rust ABI consumer will not be built")
endif()

unset(_gafime_abi_payload)
unset(_gafime_abi_payload_target)
unset(_gafime_abi_suffix)
