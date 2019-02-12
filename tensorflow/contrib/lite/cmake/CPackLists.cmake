
set(CPACK_SET_DESTDIR ON)

set(CPACK_COMPONENTS_ALL libraries headers)
set(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "Tensorflow Lite library")
set(CPACK_COMPONENT_HEADERS_DISPLAY_NAME "Tensorflow Lite headers")

set(CPACK_GENERATOR "RPM")
set(CPACK_PACKAGE_NAME "tensorflowmobile")
set(CPACK_PACKAGE_RELEASE 1)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
"TensorFlow's lightweight solution for mobile and embedded devices")

set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "1")
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_DESCRIPTION
"TensorFlow Lite is TensorFlow's lightweight solution for mobile and embedded
devices. It enables low-latency inference of on-device machine learning models
with a small binary size and fast performance supporting hardware acceleration.

See the documentation: https://www.tensorflow.org/mobile/tflite/")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")

set(CPACK_RESOURCE_FILE_LICENSE "${TENSORFLOW_ROOT_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${TENSORFLOW_LITE_ROOT_DIR}/README.md")

set(CPACK_STRIP_FILES tensorflow-lite)

# rpm options
set(CPACK_RPM_COMPONENT_INSTALL TRUE)
set(CPACK_RPM_PACKAGE_SUMMARY ${CPACK_PACKAGE_DESCRIPTION_SUMMARY})
set(CPACK_RPM_PACKAGE_DESCRIPTION ${CPACK_PACKAGE_DESCRIPTION})
set(CPACK_RPM_PACKAGE_URL "https://www.tensorflow.org/mobile/tflite/")

# message("To install on Red Hat (after cmake with options && make)")
# message("make package")
# message("sudo rpm -ivh tensorflowlite-***Linux.rpm\n\n")

include(CPack)
