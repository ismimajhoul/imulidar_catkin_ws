# Install script for directory: /home/bahi/imulidar_catkin_ws/src/robot_controller

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/bahi/imulidar_catkin_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/bahi/imulidar_catkin_ws/build/robot_controller/catkin_generated/installspace/robot_controller.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/robot_controller/cmake" TYPE FILE FILES
    "/home/bahi/imulidar_catkin_ws/build/robot_controller/catkin_generated/installspace/robot_controllerConfig.cmake"
    "/home/bahi/imulidar_catkin_ws/build/robot_controller/catkin_generated/installspace/robot_controllerConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/robot_controller" TYPE FILE FILES "/home/bahi/imulidar_catkin_ws/src/robot_controller/package.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/robot_controller" TYPE PROGRAM FILES
    "/home/bahi/imulidar_catkin_ws/src/robot_controller/scripts/robot_controller.py"
    "/home/bahi/imulidar_catkin_ws/src/robot_controller/scripts/imu_position_estimation.py"
    "/home/bahi/imulidar_catkin_ws/src/robot_controller/scripts/ins_position_estimator.py"
    "/home/bahi/imulidar_catkin_ws/src/robot_controller/scripts/torque_pos_imu_ins.py"
    "/home/bahi/imulidar_catkin_ws/src/robot_controller/scripts/mov_diag.py"
    "/home/bahi/imulidar_catkin_ws/src/robot_controller/scripts/calib.py"
    "/home/bahi/imulidar_catkin_ws/src/robot_controller/scripts/equadiff.py"
    "/home/bahi/imulidar_catkin_ws/src/robot_controller/scripts/kalman.py"
    )
endif()

