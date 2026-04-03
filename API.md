# `api/` 设计文档

本文档定义一个运行在 `Qwen3-TTS` 项目根目录下的 `FastAPI` 服务。

说明：

- 具体职责代码拆分放在根目录 `api/` 下
- 启动脚本直接运行 `api/main.py`
- 不修改 `qwen_tts/` 下官方源码

目标：

- API 侧只新增 `FastAPI` / `uvicorn`
- 不修改官方训练脚本
- 保留官方 3 类推理能力
- 增加“单 speaker 训练草稿 -> 试听 -> 保存到音色库”的产品流程
- 默认所有数据都放在项目根目录下的 `data/` 中
- 允许调用方通过 `training-audio-dir` 为单次训练指定外部音频目录
- 默认自动选择 1 张可用 GPU，并在整个服务生命周期内固定使用
- 所有 GPU 任务串行执行，避免用户本机显存被并发请求打爆
- 不自动降级到 CPU；若只能使用 CPU，必须在终端显式确认后才允许启动

---

## 1. 总体能力

`api/` 提供以下能力：

- `POST /voiceDesign`
  - 调用 `Qwen3TTSModel.generate_voice_design(...)`
- `POST /clone`
  - 调用 `Qwen3TTSModel.generate_voice_clone(...)`
- `POST /customVoice`
  - 调用 `Qwen3TTSModel.generate_custom_voice(...)`
- `POST /trainVoice`
  - 基于用户上传的数据集执行单 speaker 微调
  - 训练结果先保存到草稿区
  - 训练完成后自动生成试听音频
- `GET /trainVoice/{taskId}`
  - 查询训练任务状态
  - 返回试听音频地址、草稿模型路径等
- `POST /saveVoice`
  - 将草稿音色保存到正式音色库
- `GET /voices`
  - 列出正式音色库中的音色
- `GET /healthz`
  - 健康检查
  - 返回当前绑定设备和队列状态
- `GET /files/...`
  - 读取 `data/` 下的预览音频等静态文件

---

## 2. 目录约定

默认目录如下：

```text
api/
  app.py
  common.py
  config.py
  device.py
  main.py
  runtime.py
  schemas.py
  service.py
start_api_mac.sh
start_api_windows.bat
data/
  voiceLibrary/
    drafts/
      {taskId}/
        training/
          train_raw.jsonl
          train_with_codes.jsonl
          train.log
          checkpoint-epoch-0/
          checkpoint-epoch-1/
        preview/
          preview.wav
        meta.json
    voices/
      {voiceId}/
        model/
          config.json
          model.safetensors
          ...
        preview/
          preview.wav
        meta.json
```

说明：

- `drafts/` 是草稿训练区
- `voices/` 是正式音色库
- 默认情况下，用户上传的数据集会落到草稿目录的 `dataset/` 下
- 如果请求里传了 `training-audio-dir`，则训练音频和 `manifest.json` 改为落到该外部目录
- 使用外部目录时，正式音色库只记录路径，不再重复拷贝整份训练音频
- 官方训练产物目录名本来就叫 `checkpoint-epoch-*`

---

## 3. 请求字段约定

### 3.1 `modelId`

内外统一使用 `modelId`。

例子：

- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `data/voiceLibrary/voices/voice_xxx/model`

### 3.2 `voice`

对外统一叫 `voice`，不暴露底层 `speaker` 命名。

`/customVoice` 内部会映射为：

- `voice` -> `speaker`

### 3.3 音频字段

训练和推理都支持以下音频输入形式：

- 本地路径
- HTTP/HTTPS URL
- `data:audio/...;base64,...` 数据 URI
- 原始 base64 字符串

训练接口默认会把这些输入落到 `data/voiceLibrary/drafts/{taskId}/dataset/` 下。

如果请求里显式传入 `training-audio-dir`，则训练音频改为落到该目录下。

这个字段是**请求参数**，由外部调用方自行管理目录生命周期。

### 3.4 运行设备

服务不在每个请求里接收 `device`。

设备选择规则：

- 服务启动时按 `CUDA > MPS > CPU确认` 的顺序选择设备
- 默认自动选择 1 张 GPU
- 整个服务生命周期内固定使用这 1 张 GPU
- 所有训练、试听、配音都共用这张 GPU
- 不做运行中切卡

如果用户显式通过启动参数指定 `--device cuda:1`，则优先使用该设备。

如果用户显式通过启动参数指定 `--device mps`，则优先使用 Apple Metal / MPS。

---

## 3.5 队列与资源策略

为避免用户本机显存被并发请求打爆，API 服务采用以下策略：

- 服务内部只有 **1 个 GPU worker**
- 所有 GPU 任务都必须串行执行
- GPU 任务包括：
  - `/voiceDesign`
  - `/clone`
  - `/customVoice`
  - `/trainVoice`
  - 训练完成后的自动试听
- 业务上可以区分“训练请求”和“配音请求”
- 但物理执行层永远只有一个 GPU 执行口

推荐行为：

- 训练任务进入训练队列
- 配音任务进入配音队列
- 实际执行时统一交给单一 GPU worker 串行处理
- 队列长度应有限制，超过限制直接返回忙碌错误

这样可以保证：

- 不会多个模型同时占用显存
- 不会训练和配音同时冲击显存
- 不会因为请求洪峰把用户电脑卡死

---

## 4. 接口设计

## 4.1 `POST /voiceDesign`

用于声音设计推理。

请求：

```json
{
  "modelId": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
  "text": "欢迎来到我们的节目。",
  "language": "Chinese",
  "instruct": "成熟稳重的男声，语速适中",
  "responseFormat": "base64"
}
```

返回：

```json
{
  "ok": true,
  "modelId": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
  "sampleRate": 24000,
  "audioBase64": "..."
}
```

如果 `responseFormat = "wav"`，直接返回 `audio/wav` 二进制。

---

## 4.2 `POST /clone`

用于基于参考音频的克隆推理。

请求：

```json
{
  "modelId": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "text": "欢迎来到我们的节目。",
  "language": "Chinese",
  "refAudio": "data:audio/wav;base64,...",
  "refText": "欢迎来到我们的节目。",
  "xVectorOnlyMode": false,
  "responseFormat": "base64"
}
```

---

## 4.3 `POST /customVoice`

用于正式音色配音。

请求：

```json
{
  "modelId": "data/voiceLibrary/voices/voice_xxx/model",
  "voice": "user_001_voice",
  "text": "欢迎来到我们的节目。",
  "language": "Chinese",
  "instruct": "更温柔一点",
  "responseFormat": "base64"
}
```

说明：

- 如果模型只支持一个 `voice`，允许不传，服务端会自动补上
- 如果模型支持多个 `voice`，则必须显式传入

---

## 4.4 `POST /trainVoice`

唯一训练接口。

该接口做的事情：

1. 创建草稿目录
2. 保存用户上传的数据集
3. 生成 `train_raw.jsonl`
4. 调用官方 `prepare_data.py`
5. 调用官方 `sft_12hz.py`
6. 找到最新 `checkpoint`
7. 自动生成试听音频

请求：

```json
{
  "modelId": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "tokenizerModelId": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
  "speakerName": "user_001_voice",
  "training-audio-dir": "/Users/demo/qwen-datasets/task-001",
  "samples": [
    {
      "audio": "data:audio/wav;base64,...",
      "text": "你好，欢迎使用。"
    },
    {
      "audio": "data:audio/wav;base64,...",
      "text": "今天下午三点开会。"
    }
  ],
  "refAudio": "data:audio/wav;base64,...",
  "previewText": "欢迎来到我们的节目。",
  "previewInstruct": "温柔、自然、偏轻松",
  "language": "Chinese",
  "batchSize": 8,
  "lr": 2e-6,
  "numEpochs": 3
}
```

响应：

```json
{
  "ok": true,
  "taskId": "train_20260402_xxxxxxxx",
  "status": "queued",
  "queuePosition": 1,
  "trainingAudioDir": "/Users/demo/qwen-datasets/task-001",
  "trainingAudioManagedExternally": true
}
```

说明：

- 这里的训练是单 speaker 微调
- `previewInstruct` 不参与训练，只用于训练完成后的试听
- 当前训练数据仍然必须有文本
- 训练使用服务启动时已经选定的设备，不在请求中单独传 `device`
- 如果传了 `training-audio-dir`，服务会把 `refAudio`、`samples[*].audio` 和 `manifest.json` 落到该目录
- 如果没传 `training-audio-dir`，则继续使用默认草稿目录下的 `dataset/`

---

## 4.5 `GET /trainVoice/{taskId}`

查询训练状态。

返回：

```json
{
  "ok": true,
  "taskId": "train_20260402_xxxxxxxx",
  "status": "preview_ready",
  "speakerName": "user_001_voice",
  "jobId": "trainVoice_20260402_xxxxxxxx",
  "queuePosition": 1,
  "draftModelId": "data/voiceLibrary/drafts/train_20260402_xxxxxxxx/training/checkpoint-epoch-2",
  "trainingAudioDir": "/Users/demo/qwen-datasets/task-001",
  "trainingAudioManagedExternally": true,
  "manifestPath": "/Users/demo/qwen-datasets/task-001/manifest.json",
  "previewAudioUrl": "/files/voiceLibrary/drafts/train_20260402_xxxxxxxx/preview/preview.wav",
  "logUrl": "/files/voiceLibrary/drafts/train_20260402_xxxxxxxx/training/train.log"
}
```

状态枚举：

- `queued`
- `running`
- `preview_ready`
- `failed`
- `rejected`
- `saved`

---

## 4.6 `POST /saveVoice`

把草稿模型转成正式音色。

请求：

```json
{
  "taskId": "train_20260402_xxxxxxxx",
  "voiceName": "客服女声A"
}
```

保存逻辑：

- 拷贝草稿目录中的 `preview/`
- 将最新 `checkpoint-epoch-*` 拷贝到正式目录的 `model/`
- 生成新的 `voiceId`
- 正式模型路径写入 `modelId`
- 如果训练使用默认目录，则把草稿里的 `dataset/` 一起拷贝到正式目录
- 如果训练使用 `training-audio-dir` 外部目录，则只在正式音色 `meta.json` 里记录该路径，不重复拷贝数据集

返回：

```json
{
  "ok": true,
  "voiceId": "voice_20260402_xxxxxxxx",
  "voiceName": "客服女声A",
  "voice": "user_001_voice",
  "modelId": "data/voiceLibrary/voices/voice_20260402_xxxxxxxx/model",
  "previewAudioUrl": "/files/voiceLibrary/voices/voice_20260402_xxxxxxxx/preview/preview.wav",
  "trainingAudioDir": "/Users/demo/qwen-datasets/task-001",
  "trainingAudioManagedExternally": true,
  "manifestPath": "/Users/demo/qwen-datasets/task-001/manifest.json"
}
```

---

## 4.7 `GET /voices`

列出正式音色库。

返回：

```json
{
  "ok": true,
  "voices": [
    {
      "voiceId": "voice_20260402_xxxxxxxx",
      "voiceName": "客服女声A",
      "voice": "user_001_voice",
      "modelId": "data/voiceLibrary/voices/voice_20260402_xxxxxxxx/model",
      "previewAudioUrl": "/files/voiceLibrary/voices/voice_20260402_xxxxxxxx/preview/preview.wav",
      "trainingAudioDir": "/Users/demo/qwen-datasets/task-001",
      "trainingAudioManagedExternally": true,
      "manifestPath": "/Users/demo/qwen-datasets/task-001/manifest.json"
    }
  ]
}
```

---

## 5. 实现要点

## 5.1 实现结构

实现基于：

- `FastAPI`
- `uvicorn`
- `threading`
- `subprocess`
- `urllib`
- `soundfile`

其中：

- `api/main.py` 负责 CLI、设备选择、启动打印、`uvicorn.run(...)`
- `api/app.py` 负责 `FastAPI` 路由和异常处理
- `api/runtime.py` 负责模型缓存、队列、训练任务编排
- `api/service.py` 负责接口业务逻辑
- `api/schemas.py` 负责请求模型

## 5.2 训练不重写，只编排官方脚本

训练流程必须直接调用官方脚本：

- `finetuning/prepare_data.py`
- `finetuning/sft_12hz.py`

这样做的好处：

- 不偏离官方训练逻辑
- 后续官方修训练脚本时更容易跟进

## 5.3 推理做模型缓存

服务可以缓存已经加载的 `modelId`，但必须受显存约束控制。

推荐实现：

- 默认只保留 1 个活动模型在显存中
- 新模型加载前主动释放旧模型
- 不允许多个模型长期同时常驻显存

## 5.4 单 GPU worker 串行执行

推荐实现：

- 训练和推理共用 1 个 GPU worker
- 所有 GPU 任务进入统一执行口
- worker 一次只执行 1 个任务
- 队列达到上限时，新请求直接返回忙碌错误

这样比“多个接口各自开线程直接跑 GPU”稳定得多。

## 5.5 草稿与正式库分离

- 草稿只放在 `data/voiceLibrary/drafts/`
- 用户点击保存后，才进入 `data/voiceLibrary/voices/`

## 5.6 禁止自动降级到 CPU

默认策略：

- 如果发现可用 CUDA 设备，则正常启动
- 如果没有 CUDA 但有 MPS，则使用 MPS 正常启动
- 如果既没有 CUDA 也没有 MPS，则 **不自动切到 CPU**

必须在启动终端中给出明确提示：

```text
No CUDA or MPS device detected.
Running Qwen3-TTS on CPU will be very slow.
Do you want to continue on CPU? [y/N]
```

只有用户输入以下任一值才允许继续启动：

- `y`
- `yes`

其他输入或直接回车：

- 终止启动
- 返回非 0 退出码

这样可以避免用户在不知情的情况下进入极慢的 CPU 模式。

---

## 6. 启动方式

```bash
python api/main.py --host 0.0.0.0 --port 8000
```

或者：

```bash
./start_api_mac.sh --host 0.0.0.0 --port 8000
```

Windows：

```bat
start_api_windows.bat --host 0.0.0.0 --port 8000
```

可选参数：

- `--host`
- `--port`
- `--device`
- `--dtype`
- `--flash-attn`
- `--data-dir`
- `--max-gpu-queue-size`

安装方式建议：

```bash
pip install -e ".[api]"
```

### 6.1 启动时设备选择

推荐行为：

- 如果用户未传 `--device`
  - 先自动探测 CUDA 设备
  - 没有 CUDA 再尝试 MPS
  - 都没有才弹终端确认 CPU
- `--device` 默认值为 `auto`
- 如果用户传了 `--device cuda:0`
  - 使用该设备
- 如果用户传了 `--device mps`
  - 使用 MPS
- 如果用户传了 `--device cpu`
  - 仍然需要终端二次确认

### 6.2 启动时终端输出

服务启动时必须在终端直接打印当前运行设备信息，不能让用户猜。

示例：

```text
Qwen3-TTS API starting...
Device mode: CUDA
Selected device: cuda:0
Device name: NVIDIA GeForce RTX 4090
Data dir: /path/to/Qwen3-TTS/data
API listening on http://0.0.0.0:8000
```

如果当前走 MPS：

```text
Qwen3-TTS API starting...
Device mode: MPS
Selected device: mps
Device name: Apple M2
Data dir: /path/to/Qwen3-TTS/data
API listening on http://0.0.0.0:8000
```

如果既没有 CUDA 也没有 MPS：

```text
No CUDA or MPS device detected.
Running Qwen3-TTS on CPU will be very slow.
Do you want to continue on CPU? [y/N]
```

只有用户输入 `y` 或 `yes` 才继续启动。

### 6.3 服务状态接口建议

`GET /healthz` 建议至少返回：

```json
{
  "ok": true,
  "status": "healthy",
  "selectedDevice": "cuda:0",
  "deviceName": "NVIDIA GeForce RTX 4090",
  "queueStatus": {
    "running": 1,
    "queued": 0
  },
  "dataDir": "/path/to/Qwen3-TTS/data"
}
```

---

## 7. 前端推荐流程

1. 用户上传多段音频并填写文本
2. 调 `POST /trainVoice`
3. 轮询 `GET /trainVoice/{taskId}`
4. 状态变成 `preview_ready` 后播放 `previewAudioUrl`
5. 用户满意则调 `POST /saveVoice`
6. 后续用返回的 `modelId + voice` 调 `POST /customVoice`
