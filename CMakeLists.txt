cmake_minimum_required(VERSION 3.22)
project(neural-net-cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(external/eigen)

add_executable(neural-net-cpp
    main.cpp
    softmax.cpp
    mnist_loader.cpp
    dataloader.cpp
    nnet.cpp
    layer.cpp
    activation_function.cpp
    loss_function.cpp
    except.cpp
    tests.cpp
)
