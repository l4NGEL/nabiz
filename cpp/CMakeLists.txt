cmake_minimum_required(VERSION 3.20)
project(nabiz_cikarim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime C++ SDK — third_party/onnxruntime/ altına indirilir (git dışı,
# bkz. ../README.md "Faz 4" bölümü). CI'da/başka makinede farklıysa -DORT_DIR=... geç.
set(ORT_DIR "${CMAKE_SOURCE_DIR}/../third_party/onnxruntime" CACHE PATH "ONNX Runtime SDK dizini")

if(NOT EXISTS "${ORT_DIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime SDK bulunamadı: ${ORT_DIR}\n"
        "İndir: https://github.com/microsoft/onnxruntime/releases (onnxruntime-win-x64-<sürüm>.zip)\n"
        "third_party/onnxruntime/ altına çıkar, ya da -DORT_DIR=<yol> geç.")
endif()

add_executable(nabiz_cikarim src/nabiz_cikarim.cpp)
target_include_directories(nabiz_cikarim PRIVATE "${ORT_DIR}/include")
target_link_libraries(nabiz_cikarim PRIVATE "${ORT_DIR}/lib/onnxruntime.lib")

if(MSVC)
    target_compile_options(nabiz_cikarim PRIVATE /W4 /EHsc)
endif()

# onnxruntime.dll çalışma zamanında gerekli — derlenen ikilinin yanına kopyala
add_custom_command(TARGET nabiz_cikarim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_DIR}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:nabiz_cikarim>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_DIR}/lib/onnxruntime_providers_shared.dll"
        "$<TARGET_FILE_DIR:nabiz_cikarim>"
)
