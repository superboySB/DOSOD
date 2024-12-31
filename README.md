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
运行
```sh
python tools/generate_text_prompts_dosod.py path_to_config_file path_to_model_file --text path_to_texts_json_file --out-dir dir_to_save_embedding_npy_file
```

## RDK侧复现