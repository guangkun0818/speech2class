# Speech to class
This is `Pytorch Lightning` and `torchaudio` based repo of Speaker Recognition & Voice Activity Detection training scripts. Still development and maintainance are ongoing. Please read the following and source codes for details.

***Feel free to reach me 609946862@qq.com or issue if any problem encountered!***

## Environment set up
The envs required for vpr and vad task training of this repo contains all components for `speech2text` repo. So basically you only need one built image with `Dockerfile.build` of this repo to support 3 tasks.

```bash
docker build -t asr_vpr_vad_training . -f Dockerfile.build
nvidia-docker run --ipc=host --name=vpr-training-runtime -itd -v /home:/home -v /data:/data asr_vpr_vad_training:latest /bin/bash
```

## Training
Both VPR and VAD tasks are supported. You can build your own system by just specifying the task type in yaml depending on what task supposed to address. Please refer to demos from `config/*.yaml` for the details of configuration of each components. Nevertheless, you might need to calibrate your config related with training pipeline based on different datasets and configured models you applied.

```yaml
task:
  type: &task_type "VAD"
  name: "crdnn-vad"
  export_path: "tasks/debug"
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python build_task.py \
    --config_path="config/resnet18.yaml" || exit 1
```

## Formatted
```bash
find ./ -path "./runtime" -prune -o iname "*.py" -print | xargs yapf -i --style google
```