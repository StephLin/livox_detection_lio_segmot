# Livox Detection V1.1 - A Fork for LIO-SEGMOT

The detector can run at least 20 FPS on 2080TI for 200m*100m range detection. The provided model was trained on LivoxDataset_v1.0 within 1w pointcloud sequence.

## Introduction

Livox Detection is a robust,real time detection package for [*Livox LiDARs*](https://www.livoxtech.com/). The detector is designed for L3 and L4 autonomous driving. It can effectively detect within 200*100m range under different vehicle speed conditions(`0~120km/h`). In addition, the detector can perform effective detection in different scenarios, such as high-speed scenes, structured urban road scenes, complex intersection scenes and some unstructured road scenes, etc. In addition, the detector is currently able to effectively detect 3D bounding boxes of five types of objects: `cars`, `trucks`, `bus`, `bimo` and `pedestrians`.

## Dependencies

- `python3.6+`
- `tensorflow1.13+` (tested on 1.13.0)
- `pybind11`
- `ros`
- `LIO-SEGMOT`

## Installation

1. Clone this repository.
2. Clone `pybind11` from [pybind11](https://github.com/pybind/pybind11).
```bash
$ cd utils/lib_cpp
$ git clone https://github.com/pybind/pybind11.git
```
3. Compile C++ module in utils/lib_cpp by running the following command.
```bash
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
```
4. copy the `lib_cpp.so` to root directory:
```bash
$ cp lib_cpp.so ../../../
```

5. Download the [pre_trained model](https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/github/Livox_detection1.1_model.zip) and unzip it to the root directory.

## Run

```bash
python3 ros_main.py
```
