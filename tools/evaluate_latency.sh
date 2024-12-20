#!/usr/bin/env bash
# https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html#trtexec

trtexec --onnx=path_to_onnx_file \
  --fp16 \
  --iterations=2000 \
  --verbose \
  --device=0