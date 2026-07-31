# RealSense 3대 기반 데이터 수집 및 ACT 모델 학습 가이드 (Type E: 증강형)

리얼센스(RealSense) 3대를 사용하여 데이터를 수집하고 학습(ACT 모델)을 진행하는 전체 과정을 정리한 가이드입니다. 

4팀이 사용하실 **타입 E (증강형)** 학습 소스에 맞춰 작성되었습니다.

---

## 1단계: 장비 연결 및 식별자 확인
명령어를 실행하기 전에 먼저 연결된 카메라의 시리얼 번호와 로봇 팔의 포트를 확인해야 합니다.

1. **카메라 시리얼 확인**: 3대의 리얼센스 각각의 시리얼 번호를 메모합니다.
   ```bash
   lerobot-find-cameras realsense
   ```
2. **로봇 포트 확인**: 리더 암과 팔로워 암의 USB 포트를 확인합니다.
   ```bash
   lerobot-find-port
   ```

---

## 2단계: 데이터 수집 (Record)

리얼센스 3대의 부하를 고려하여, 녹화 중에는 인코딩하지 않고 에피소드 종료 후 인코딩하는 방식(`streaming_encoding=false`)을 권장합니다. 또한 초기 프레임 안정화를 위해 `warmup_s: 15` 옵션을 각 카메라에 추가합니다.

터미널에서 아래 명령어를 본인의 환경에 맞게(시리얼 번호 및 포트) 수정하여 실행하세요:

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_follower \
  --robot.cameras="{ \
       top: {type: intelrealsense, serial_number_or_name: '시리얼번호1', width: 640, height: 480, fps: 30, warmup_s: 15}, \
       side: {type: intelrealsense, serial_number_or_name: '시리얼번호2', width: 640, height: 480, fps: 30, warmup_s: 15}, \
       wrist: {type: intelrealsense, serial_number_or_name: '시리얼번호3', width: 640, height: 480, fps: 30, warmup_s: 15} \
  }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_leader \
  --display_data=true \
  --dataset.repo_id=data/my_task_3rs \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=30 \
  --dataset.single_task="여기에 작업 설명 작성 (예: pink block into the box)" \
  --dataset.vcodec=h264 \
  --dataset.streaming_encoding=false \
  --dataset.encoder_threads=4
```

> [!TIP] 
> **수집 중 키보드 조작법**
> - `→ (오른쪽 화살표)` : 현재 에피소드를 성공으로 저장하고 다음 녹화로 이동합니다.
> - `← (왼쪽 화살표)` : 현재 에피소드를 삭제하고 다시 녹화합니다. (실패한 데이터는 학습에 방해가 되므로 절대 저장하지 마세요.)
> - `ESC` : 수집을 모두 마치고 녹화를 종료합니다. 이후 전체 데이터가 h264로 인코딩됩니다.

---

## 3단계: 모델 학습 (Train) - 타입 E (증강)

데이터 수집이 완료되면, ACT 모델을 이용해 로봇을 학습시킵니다. 4팀이 선택하신 **타입 E (증강형)** 명령어입니다. 데이터의 다양성을 높여주는 `image_transforms` 옵션이 켜져 있는 것이 특징입니다.

먼저, 환경 변수를 설정하거나 쉘 스크립트 파일 최상단에 추가하여 실행하세요. `REPO_ID`는 수집 단계에서 입력한 값과 일치해야 합니다.

```bash
# 환경 변수 설정
export REPO_ID="data/my_task_3rs"
export DATASET_ROOT="여기에_데이터셋_저장경로_입력"
export OUTPUT_DIR="outputs/act_typeE"

# 학습 명령어 실행 (Type E는 고정 Batch 48을 권장합니다)
PYTORCH_ALLOC_CONF=expandable_segments:True lerobot-train \
  --policy.type=act \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=act_typeE_augment \
  --policy.device=cuda \
  --batch_size=48 \
  --num_workers=4 \
  --steps=80000 \
  --save_freq=10000 \
  --log_freq=200 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.n_obs_steps=1 \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.dim_model=512 \
  --policy.dim_feedforward=3200 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=1 \
  --policy.n_heads=8 \
  --policy.use_vae=true \
  --policy.kl_weight=10.0 \
  --policy.optimizer_lr=1.2e-5 \
  --policy.optimizer_lr_backbone=1.2e-5 \
  --policy.optimizer_weight_decay=1e-4 \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --policy.push_to_hub=false \
  --wandb.enable=false
```

---

### 타입 E (증강) 변경점 요약
1. **학습 배치 크기**: `batch_size=48` (VRAM 용량이 부족할 경우 상황에 맞게 32 등으로 조절하세요)
2. **학습률 조정**: `optimizer_lr=1.2e-5`, `optimizer_lr_backbone=1.2e-5`로 변경
3. **이미지 증강(Augmentation) 켜기**: `dataset.image_transforms.enable=true`, `dataset.image_transforms.max_num_transforms=3` 옵션 추가
