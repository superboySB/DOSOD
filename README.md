# 复现笔记

## 笔记本侧复现
```sh
docker build -f docker/laptop.dockerfile -t dosod_image:laptop --network=host --progress=plain .

docker run -itd --privileged --name=dosod-laptop \
--volume /tmp/.X11-unix:/tmp/.X11-unix \
--env DISPLAY=$DISPLAY \
--env QT_X11_NO_MITSHM=1 \
--gpus all \
--network=host \
dosod_image:laptop /bin/bash

docker exec -it dosod-laptop /bin/bash

cd /workspace && git clone https://github.com/superboySB/DOSOD.git

cd DOSOD
```
当我们敲定分类相关的labels for testing以后，运行就分为四步。

Step 1: generate texts embeddings
```sh
python tools/generate_text_prompts_dosod.py \
configs/dosod/dosod_mlp3x_l_100e_1x8gpus_obj365v1_goldg_train_lvis_minival.py \
/workspace/dosod_weights/dosod_mlp3x_l.pth \
--text data/texts/coco_class_texts.json \
--out-dir demo/
```
Step 2: reparameterize model weights
```sh
python tools/reparameterize_dosod.py \
--model /workspace/dosod_weights/dosod_mlp3x_l.pth  \
--out-dir demo/ \
--text-embed demo/xxxxx.npy
```
Step 3: export onnx using rep-style config
```sh
python deploy/export_onnx.py \
configs/dosod/rep_dosod_mlp3x_s_100e_1x8gpus_obj365v1_goldg_train_lvis_minival.py \ 
demo/xxx.pth \
--without-nms \
--work-dir demo/
```
Step 4: run onnx demo
```sh
python deploy/onnx_demo.py \
demo/xxxx.onnx \
demo/sample_images/bus.jpg \
data/texts/coco_class_texts.json \
--output-dir demo/ --onnx-nms
```

## RDK侧复现
Comming soon!