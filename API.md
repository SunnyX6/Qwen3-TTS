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
- 默认自动选择 1 张可用 GPU，并在整个服务生命周期内固定使用
- 所有 GPU 任务串行执行，避免用户本机显存被并发请求打爆
- 不自动降级到 CPU；若只能使用 CPU，必须在终端显式确认后才允许启动

---

## 1. 总体能力

`api/` 提供以下能力：

- `POST /api/voiceDesign`
  - 调用 `Qwen3TTSModel.generate_voice_design(...)`
- `POST /api/clone`
  - 调用 `Qwen3TTSModel.generate_voice_clone(...)`
- `POST /api/customVoice`
  - 调用 `Qwen3TTSModel.generate_custom_voice(...)`
- `POST /api/trainVoice`
  - 基于用户上传的数据集执行单 speaker 微调
  - 训练结果先保存到草稿区
  - 训练完成后自动生成试听音频
- `GET /api/trainVoice/{taskId}`
  - 查询训练任务状态
  - 返回试听音频地址、草稿模型路径等
- `POST /api/deployVoice`
  - 将草稿音色保存到正式音色库
- `GET /api/voices`
  - 列出正式音色库中的音色
- `GET /api/healthz`
  - 健康检查
  - 返回当前绑定设备和队列状态
- `GET /api/files/...`
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
          train.log
          checkpoint-epoch-{latest}/
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
- 训练完成后，草稿目录只保留 `train.log`、最新 `checkpoint-epoch-*` 和 `preview.wav`
- `train_raw.jsonl`、`train_with_codes.jsonl` 等中间产物会在任务结束后清理

---

## 3. 请求字段约定

### 3.1 `modelId`

内外统一使用 `modelId`。

例子：

- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `Qwen3-TTS-12Hz-1.7B-Base`
- `data/voiceLibrary/voices/voice_xxx/model`

> API and loaders now prefer the project-local `./models` directory. For example, `Qwen/Qwen3-TTS-12Hz-1.7B-Base` will first resolve to `./models/Qwen3-TTS-12Hz-1.7B-Base` if it exists, and simple names like `Qwen3-TTS-12Hz-1.7B-Base` are treated as `./models/Qwen3-TTS-12Hz-1.7B-Base` by default.

### 3.2 `voice`

对外统一叫 `voice`，不暴露底层 `speaker` 命名。

`/api/customVoice` 内部会映射为：

- `voice` -> `speaker`

### 3.3 音频字段

`/api/clone` 和 `/api/trainVoice` 的音频输入统一使用 `multipart/form-data` 上传文件。

服务端不再接收音频字段的本地路径、URL、`data:audio` 或原始 base64 字符串。

字段约定：

- `/api/clone`：`refAudio`（单文件）
- `/api/trainVoice`：`refAudio`（单文件）、`sampleAudios`（多文件）、`sampleTexts`（与 `sampleAudios` 一一对应）
- `/api/clone` 继续保留 `refText`

训练接口会在服务端临时目录中使用这些上传音频，仅用于当前训练任务的数据准备阶段，不会持久化保存到 `data/` 目录。

### 3.4 公共生成参数

以下参数为公共生成参数：

- `seed`
- `maxNewTokens`
- `temperature`
- `topP`
- `repetitionPenalty`

适用范围：

- `/api/voiceDesign`
- `/api/clone`
- `/api/customVoice`
- `/api/trainVoice`

默认值固定为：

- `seed = 0`
- `maxNewTokens = 2048`
- `temperature = 0.9`
- `topP = 1.0`
- `repetitionPenalty = 1.05`

其中 `/api/trainVoice` 里的这 5 个参数只用于训练完成后的自动试听生成，不参与训练本身。

### 3.5 运行设备

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

## 3.6 队列与资源策略

为避免用户本机显存被并发请求打爆，API 服务采用以下策略：

- 服务内部只有 **1 个 GPU worker**
- 所有 GPU 任务都必须串行执行
- GPU 任务包括：
  - `/api/voiceDesign`
  - `/api/clone`
  - `/api/customVoice`
  - `/api/trainVoice`
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

## 4.1 `POST /api/voiceDesign`

用于声音设计推理。

请求：

```json
{
  "modelId": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
  "text": "欢迎来到我们的节目。",
  "language": "Chinese",
  "instruct": "成熟稳重的男声，语速适中",
  "responseFormat": "base64",
  "seed": 0,
  "maxNewTokens": 2048,
  "temperature": 0.9,
  "topP": 1.0,
  "repetitionPenalty": 1.05
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

## 4.2 `POST /api/clone`

用于基于参考音频的克隆推理。

请求类型：`multipart/form-data`

```bash
curl -X POST "http://127.0.0.1:8000/api/clone" \
  -F "modelId=Qwen/Qwen3-TTS-12Hz-1.7B-Base" \
  -F "text=欢迎来到我们的节目。" \
  -F "language=Chinese" \
  -F "refAudio=@/path/to/ref.wav" \
  -F "refText=欢迎来到我们的节目。" \
  -F "xVectorOnlyMode=false" \
  -F "responseFormat=base64" \
  -F "seed=0" \
  -F "maxNewTokens=2048" \
  -F "temperature=0.9" \
  -F "topP=1.0" \
  -F "repetitionPenalty=1.05"
```

说明：

- `xVectorOnlyMode = false` 时，必须传 `refText`
- `refAudio` 必须是上传文件（`multipart` file part）
- `refAudio` 会按 `demo.py` 的方式先做归一化，再送进模型
- `seed`、`maxNewTokens`、`temperature`、`topP`、`repetitionPenalty` 都可以直接通过 API 透传给底层生成逻辑

---

## 4.3 `POST /api/customVoice`

用于正式音色配音。

请求：

```json
{
  "modelId": "data/voiceLibrary/voices/voice_xxx/model",
  "voice": "user_001_voice",
  "text": "欢迎来到我们的节目。",
  "language": "Chinese",
  "instruct": "更温柔一点",
  "responseFormat": "base64",
  "seed": 0,
  "maxNewTokens": 2048,
  "temperature": 0.9,
  "topP": 1.0,
  "repetitionPenalty": 1.05
}
```

说明：

- 如果模型只支持一个 `voice`，允许不传，服务端会自动补上
- 如果模型支持多个 `voice`，则必须显式传入

---

## 4.4 `POST /api/trainVoice`

唯一训练接口。

该接口做的事情：

1. 创建草稿目录
2. 在系统临时目录中准备用户上传音频
3. 生成 `train_raw.jsonl`
4. 调用官方 `prepare_data.py`
5. 调用官方 `sft_12hz.py`
6. 找到最新 `checkpoint`
7. 自动生成试听音频
8. 清理中间产物，只保留 `train.log`、最新 `checkpoint` 和 `preview.wav`

请求：

请求类型：`multipart/form-data`

```bash
curl -X POST "http://127.0.0.1:8000/api/trainVoice" \
  -F "modelId=Qwen/Qwen3-TTS-12Hz-1.7B-Base" \
  -F "speakerName=user_001_voice" \
  -F "previewText=欢迎来到我们的节目。" \
  -F "previewInstruct=温柔、自然、偏轻松" \
  -F "language=Chinese" \
  -F "batchSize=8" \
  -F "lr=2e-6" \
  -F "numEpochs=3" \
  -F "seed=0" \
  -F "maxNewTokens=2048" \
  -F "temperature=0.9" \
  -F "topP=1.0" \
  -F "repetitionPenalty=1.05" \
  -F "refAudio=@/path/to/ref.wav" \
  -F "sampleAudios=@/path/to/sample1.wav" \
  -F "sampleTexts=你好，欢迎使用。" \
  -F "sampleAudios=@/path/to/sample2.wav" \
  -F "sampleTexts=今天下午三点开会。"
```

响应：

```json
{
  "ok": true,
  "taskId": "train_20260402_xxxxxxxx",
  "status": "queued",
  "queuePosition": 1
}
```

说明：

- 这里的训练是单 speaker 微调
- 服务端固定使用 `Qwen/Qwen3-TTS-Tokenizer-12Hz`，调用方不需要也不能再传 `tokenizerModelId`
- `previewInstruct` 不参与训练，只用于训练完成后的试听
- `seed`、`maxNewTokens`、`temperature`、`topP`、`repetitionPenalty` 在这个接口里只作用于训练完成后的自动试听生成
- 当前训练数据仍然必须有文本
- `sampleAudios` 和 `sampleTexts` 必须按顺序一一对应，数量必须一致
- 训练使用服务启动时已经选定的设备，不在请求中单独传 `device`
- 上传音频只在训练任务的数据准备阶段暂存于系统临时目录，不会持久化保存到 `data/`

---

## 4.5 `GET /api/trainVoice/{taskId}`

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
  "previewAudioUrl": "/api/files/voiceLibrary/drafts/train_20260402_xxxxxxxx/preview/preview.wav",
  "logUrl": "/api/files/voiceLibrary/drafts/train_20260402_xxxxxxxx/training/train.log"
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

## 4.6 `POST /api/deployVoice`

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

返回：

```json
{
  "ok": true,
  "voiceId": "voice_20260402_xxxxxxxx",
  "voiceName": "客服女声A",
  "voice": "user_001_voice",
  "modelId": "data/voiceLibrary/voices/voice_20260402_xxxxxxxx/model",
  "previewAudioUrl": "/api/files/voiceLibrary/voices/voice_20260402_xxxxxxxx/preview/preview.wav"
}
```

---

## 4.7 `GET /api/voices`

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
      "previewAudioUrl": "/api/files/voiceLibrary/voices/voice_20260402_xxxxxxxx/preview/preview.wav"
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
- `--models-dir`
- `--max-gpu-queue-size`

安装方式建议：

```bash
pip install -e ".[runtime,api]"
```

> Before installing API dependencies, install a PyTorch build that matches the target machine from https://pytorch.org/get-started/locally/ . The project does not auto-pick a GPU-specific PyTorch build for users.

> `--flash-attn` is enabled by default. If `flash_attn` is missing or broken, API startup now exits immediately with an installation hint instead of waiting until the first inference request fails.

> The API runs as a single process. `--max-gpu-queue-size` defaults to `2`, which means at most 2 waiting GPU jobs can queue behind the currently running job.

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
API listening on http://0.0.0.0:8000/api
```

如果当前走 MPS：

```text
Qwen3-TTS API starting...
Device mode: MPS
Selected device: mps
Device name: Apple M2
Data dir: /path/to/Qwen3-TTS/data
API listening on http://0.0.0.0:8000/api
```

如果既没有 CUDA 也没有 MPS：

```text
No CUDA or MPS device detected.
Running Qwen3-TTS on CPU will be very slow.
Do you want to continue on CPU? [y/N]
```

只有用户输入 `y` 或 `yes` 才继续启动。

### 6.3 服务状态接口建议

`GET /api/healthz` 建议至少返回：

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
2. 调 `POST /api/trainVoice`
3. 轮询 `GET /api/trainVoice/{taskId}`
4. 状态变成 `preview_ready` 后播放 `previewAudioUrl`
5. 用户满意则调 `POST /api/deployVoice`
6. 后续用返回的 `modelId + voice` 调 `POST /api/customVoice`
