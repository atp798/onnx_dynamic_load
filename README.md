# Onnx dynamic load

## Introduction
This is a wrapper used to load onnxruntime dynamically to inference model.

## Feature
- No c interface exposed.
- Library loader is thread-safe.
- Dynamic library load.
- Library closed once unneeded, no more occupation. 

## Usage
1. Download: `git clone --recursive https://github.com/atp798/onnx_dynamic_load.git`

2. Build: `mkdir build && cd build && cmake ../ && make`

3. Run sample: `./onnxruntime_dynamic_loader.out`

License
-------
MIT license (© 2020 ATP)
