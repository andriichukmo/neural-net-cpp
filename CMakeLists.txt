cmake_minimum_required(VERSION 3.22)
project(neural-net-cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(external/eigen)

add_executable(neural-net-cpp
    main.cpp
    nnet.cpp
    linear_layer.cpp
    activation_function.cpp
    loss_function.cpp
    except.cpp
    tests.cpp
)
