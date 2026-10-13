# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/markodev/esp/esp-idf/components/bootloader/subproject"
  "/home/markodev/esp32cam_dsp_benchmark/build/bootloader"
  "/home/markodev/esp32cam_dsp_benchmark/build/bootloader-prefix"
  "/home/markodev/esp32cam_dsp_benchmark/build/bootloader-prefix/tmp"
  "/home/markodev/esp32cam_dsp_benchmark/build/bootloader-prefix/src/bootloader-stamp"
  "/home/markodev/esp32cam_dsp_benchmark/build/bootloader-prefix/src"
  "/home/markodev/esp32cam_dsp_benchmark/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/markodev/esp32cam_dsp_benchmark/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/markodev/esp32cam_dsp_benchmark/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()
