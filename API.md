# `api/` 设计文档

本文档定义一个运行在 `Qwen3-TTS` 项目根目录下的 `FastAPI` 服务。

说明：

- 本文档描述当前 `api/` 实现
- 若文档与代码存在冲突，以代码和 `CUSTOM_SPEAKER_BANK_ARCHITECTURE.md` 为准

说明：

- 具体职责代码拆分放在根目录 `api/` 下
- 启动脚本直接运行 `api/main.py`
- 不修改 `qwen_tts/` 下官方源码

目标：

- API 侧只新增 `FastAPI` / `uvicorn`
- 训练与推理都由仓内模块统一编排
- 保留官方 3 类推理能力
- 增加“单 speaker 训练 -> 自动注册到音色库 -> 直接用于 customVoice”的产品流程
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
  - 同一条 HTTP 连接以 `SSE` 持续返回训练状态
  - 训练成功后自动注册到当前 `CustomVoice` backbone 的音色库
- `POST /api/transcribe`
  - 独立的便利转录接口
  - 不参与训练流程，只负责语音转文本
- `GET /api/voices`
  - 列出当前 backbone 下可用的内置 speaker 和自定义 speaker
- `DELETE /api/voices/{voiceId}`
  - 删除一个已注册的自定义音色
- `GET /api/healthz`
  - 健康检查
  - 返回当前绑定设备和队列状态

---

## 2. 目录约定

默认目录如下：

```text
api/
  app.py
  asr.py
  common.py
  config.py
  device.py
  main.py
  runtime.py
  schemas.py
  service.py
start_api_mac.sh
start_api_windows.bat
models/
  asr/
    faster-whisper/
      large-v3/
      large-v3-turbo/
      ...
    faster-whisper-cache/
    funasr/
      speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
      speech_fsmn_vad_zh-cn-16k-common-pytorch/
      punc_ct-transformer_zh-cn-common-vocab272727-pytorch/
      speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online/
data/
  voices/
    index.json
    {voiceId}/
      model/
        speaker.safetensors
        speaker_config.json
      training_summary.json
      meta.json
```

说明：

- `voices/` 是已注册的自定义音色库，`index.json` 是注册索引
- 训练日志只输出到 API 服务终端，不落盘保存为 `train.log`
- 训练过程使用项目内 `data/train/tmp/` 工作目录，训练完成后会自动清理子目录
- `models/asr/` 是转录接口的推荐本地模型目录
- 若启动时传了 `--models-dir /your/path/models`，则 ASR 目录会变成 `/your/path/models/asr`

---

## 3. 请求字段约定

### 3.1 `modelId`

内外统一使用 `modelId`。

例子：

- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `Qwen3-TTS-12Hz-1.7B-Base`
- `data/voices/voice_xxx/model`

> API and loaders now prefer the project-local `./models` directory. For example, `Qwen/Qwen3-TTS-12Hz-1.7B-Base` will first resolve to `./models/Qwen3-TTS-12Hz-1.7B-Base` if it exists, and simple names like `Qwen3-TTS-12Hz-1.7B-Base` are treated as `./models/Qwen3-TTS-12Hz-1.7B-Base` by default.

### 3.2 `voice`

对外统一叫 `voice`，不暴露底层 `speaker` 命名。

`/api/customVoice` 内部会映射为：

- `voice` -> `speaker`

### 3.3 音频字段

`/api/clone`、`/api/trainVoice` 和 `/api/transcribe` 的音频输入统一使用 `multipart/form-data` 上传文件。

服务端不再接收音频字段的本地路径、URL、`data:audio` 或原始 base64 字符串。

字段约定：

- `/api/clone`：`refAudio`（单文件）
- `/api/trainVoice`：`refAudio`（单文件）、`sampleAudios`（多文件）、`sampleTexts`（与 `sampleAudios` 一一对应）
- `/api/transcribe`：`audios`（多文件）
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

默认值固定为：

- `seed = 0`
- `maxNewTokens = 2048`
- `temperature = 0.9`
- `topP = 1.0`
- `repetitionPenalty = 1.05`

### 3.5 运行设备

服务不在每个请求里接收 `device`。

设备选择规则：

- 服务启动时按 `CUDA > MPS > CPU确认` 的顺序选择设备
- 默认自动选择 1 张 GPU
- 整个服务生命周期内固定使用这 1 张 GPU
- 所有训练和推理都共用这张 GPU
- 不做运行中切卡

如果用户显式通过启动参数指定 `--device cuda:1`，则优先使用该设备。

如果用户显式通过启动参数指定 `--device mps`，则优先使用 Apple Metal / MPS。

---

## 3.6 队列与资源策略

为避免用户本机显存被并发请求打爆，API 服务采用以下策略：

- 服务内部只有 **1 个 GPU worker**
- 所有 GPU 任务都必须串行执行
- GPU 任务包括：
  - `/api/transcribe`
  - `/api/voiceDesign`
  - `/api/clone`
  - `/api/customVoice`
  - `/api/trainVoice`
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

## 4.1 `POST /api/transcribe`

独立的便利转录接口。

请求类型：`multipart/form-data`

```bash
curl -X POST "http://127.0.0.1:8000/api/transcribe" \
  -F "language=Auto" \
  -F "provider=auto" \
  -F "modelSize=large-v3" \
  -F "audios=@/path/to/a.wav" \
  -F "audios=@/path/to/b.wav"
```

请求字段：

- `language`
  - 可选值：`Auto`、`Chinese`、`Cantonese`、`English`、`Japanese`、`Korean`
- `provider`
  - 可选值：`auto`、`faster-whisper`、`funasr`
  - 其中 `funasr` 只接受 `Chinese` 或 `Cantonese`
- `modelSize`
  - 当前用于 `faster-whisper`
  - 可选值：`medium`、`medium.en`、`large-v2`、`large-v3`、`large-v3-turbo`
- `audios`
  - 一个或多个上传音频文件

返回：

```json
{
  "ok": true,
  "providerRequested": "auto",
  "languageRequested": "Auto",
  "modelSize": "large-v3",
  "total": 2,
  "successCount": 2,
  "failedCount": 0,
  "results": [
    {
      "index": 0,
      "fileName": "a.wav",
      "ok": true,
      "text": "额度耗尽，强制睡觉",
      "languageDetected": "Chinese",
      "languageCode": "zh",
      "providerUsed": "funasr",
      "error": null
    },
    {
      "index": 1,
      "fileName": "b.wav",
      "ok": true,
      "text": "今天下午三点开会。",
      "languageDetected": "Chinese",
      "languageCode": "zh",
      "providerUsed": "funasr",
      "error": null
    }
  ]
}
```

说明：

- 该接口和训练接口没有耦合，只是独立的便利转录能力
- 服务端在内存中解码音频，并统一转换到 `16kHz`
- `provider=auto` 时，会优先做自动判断；识别到 `zh` / `yue` 时转到 `FunASR`
- 单个文件失败不会拖垮整批请求，错误会落在对应 `results[i].error`
- 当前实现会优先读取项目内本地目录；若本地目录不存在，则按上游库默认方式自动下载

本地模型目录约定：

- `faster-whisper`
  - `<models-dir>/asr/faster-whisper/<modelSize>`
  - 例：`./models/asr/faster-whisper/large-v3`
- `FunASR`
  - `<models-dir>/asr/funasr/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
  - `<models-dir>/asr/funasr/speech_fsmn_vad_zh-cn-16k-common-pytorch`
  - `<models-dir>/asr/funasr/punc_ct-transformer_zh-cn-common-vocab272727-pytorch`
  - `<models-dir>/asr/funasr/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online`

推荐手动下载命令：

```bash
# faster-whisper large-v3
pip install -U "huggingface_hub[cli]"
huggingface-cli download Systran/faster-whisper-large-v3 \
  --local-dir ./models/asr/faster-whisper/large-v3

# faster-whisper large-v3-turbo
huggingface-cli download Systran/faster-whisper-large-v3-turbo \
  --local-dir ./models/asr/faster-whisper/large-v3-turbo

# FunASR Chinese ASR + VAD + PUNC
pip install -U modelscope
modelscope download --model iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
  --local_dir ./models/asr/funasr/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
modelscope download --model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch \
  --local_dir ./models/asr/funasr/speech_fsmn_vad_zh-cn-16k-common-pytorch
modelscope download --model iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch \
  --local_dir ./models/asr/funasr/punc_ct-transformer_zh-cn-common-vocab272727-pytorch

# FunASR Cantonese
modelscope download --model iic/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online \
  --local_dir ./models/asr/funasr/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online
```

---

## 4.2 `POST /api/voiceDesign`

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

## 4.3 `POST /api/clone`

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

## 4.4 `POST /api/customVoice`

用于正式音色配音。

请求：

```json
{
  "voice": "user_001_voice",
  "text": "欢迎来到我们的节目。",
  "language": "Chinese",
  "dialect": "beijing_dialect",
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
- `customVoice` 额外支持可选 `dialect`
- 当前 `dialect` 只支持 `beijing_dialect` 和 `sichuan_dialect`
- 传 `dialect` 时，`language` 仍然是独立字段；建议使用 `Chinese` 或 `Auto`
- `instruct` 是风格/表现控制，和 `dialect` 无关

---

## 4.5 `POST /api/trainVoice`

唯一训练接口。

该接口做的事情：

1. 创建项目内 `data/train/tmp/` 工作目录
2. 读取上传音频并在内存中统一转换到 `24kHz`
3. 调用仓内训练模块编码训练样本
4. 执行 speaker package 训练
5. 将训练产出的 voice package 直接注册到 `data/voices/`
6. 训练结束后自动清理对应工作子目录

请求：

请求类型：`multipart/form-data`

```bash
curl -X POST "http://127.0.0.1:8000/api/trainVoice" \
  -F "modelId=Qwen/Qwen3-TTS-12Hz-1.7B-Base" \
  -F "speakerName=user_001_voice" \
  -F "language=Chinese" \
  -F "batchSize=8" \
  -F "lr=2e-6" \
  -F "numEpochs=3" \
  -F "refAudio=@/path/to/ref.wav" \
  -F "sampleAudios=@/path/to/sample1.wav" \
  -F "sampleTexts=你好，欢迎使用。" \
  -F "sampleAudios=@/path/to/sample2.wav" \
  -F "sampleTexts=今天下午三点开会。"
```

响应：

响应类型：`text/event-stream`

```text
id: 2
event: status
data: {"ok":true,"taskId":"train_20260412_113000_ab12cd34","status":"queued","speakerName":"user_001_voice","voiceId":null,"voice":null,"baseModelId":"models/Qwen3-TTS-12Hz-1.7B-Base","jobId":"trainVoice_20260412_113000_ef56ab78","queuePosition":1,"error":null}

id: 3
event: status
data: {"ok":true,"taskId":"train_20260412_113000_ab12cd34","status":"running","speakerName":"user_001_voice","voiceId":null,"voice":null,"baseModelId":"models/Qwen3-TTS-12Hz-1.7B-Base","jobId":"trainVoice_20260412_113000_ef56ab78","queuePosition":1,"error":null}

id: 4
event: status
data: {"ok":true,"taskId":"train_20260412_113000_ab12cd34","status":"completed","speakerName":"user_001_voice","voiceId":"voice_20260412_113045_123456","voice":"user_001_voice","baseModelId":"models/Qwen3-TTS-12Hz-1.7B-Base","jobId":"trainVoice_20260412_113000_ef56ab78","queuePosition":1,"error":null}
```

说明：

- 这里的训练是单 speaker 微调
- 服务端固定使用 `Qwen/Qwen3-TTS-Tokenizer-12Hz`，调用方不需要也不能再传 `tokenizerModelId`
- 当前训练数据仍然必须有文本
- `sampleAudios` 和 `sampleTexts` 必须按顺序一一对应，数量必须一致
- 训练使用服务启动时已经选定的设备，不在请求中单独传 `device`
- 上传音频不会持久化保存到 `data/`，只在内存中解码和重采样
- 训练进度日志只输出到 API 服务终端
- 调用方需要按 `SSE` 流持续读取响应体，而不是等一个普通 JSON
- 状态会按 `queued`、`running`、`completed` / `failed` / `rejected` 推进
- 连接空闲时服务端会发送 `: keep-alive`

---

## 4.6 `GET /api/voices`

列出当前服务 `custom_voice_model_id` 对应 backbone 下可用的音色：

- 模型内置 speaker
- 已注册且与当前 backbone 兼容的 custom speaker

返回：

```json
{
  "ok": true,
  "voices": [
    {
      "voiceId": "voice_20260402_xxxxxxxx",
      "speaker": "user_001_voice",
      "voice": "user_001_voice",
      "speakerName": "user_001_voice",
      "baseModelId": "models/Qwen3-TTS-12Hz-1.7B-Base",
      "source": "custom",
      "deletable": true
    },
    {
      "voiceId": null,
      "speaker": "Serena",
      "voice": "Serena",
      "speakerName": "Serena",
      "baseModelId": "models/Qwen3-TTS-12Hz-1.7B-Base",
      "enabled": true,
      "source": "builtin",
      "deletable": false
    }
  ]
}
```

---

## 4.7 `DELETE /api/voices/{voiceId}`

删除一个已注册的自定义音色。

返回：

```json
{
  "ok": true,
  "voiceId": "voice_20260412_102050_944573",
  "speaker": "user_001_voice",
  "baseModelId": "models/Qwen3-TTS-12Hz-1.7B-Base"
}
```

---

## 5. 实现要点

## 5.1 实现结构

实现基于：

- `FastAPI`
- `uvicorn`
- `threading`
- `queue`
- `soundfile`
- `librosa`

其中：

- `api/main.py` 负责 CLI、设备选择、启动打印、`uvicorn.run(...)`
- `api/app.py` 负责 `FastAPI` 路由和异常处理
- `api/asr.py` 负责转录模型调度和 provider 选择
- `api/runtime.py` 负责模型缓存、队列、训练任务编排
- `api/service.py` 负责接口业务逻辑
- `api/schemas.py` 负责请求模型

## 5.2 训练由仓内模块编排

训练流程当前直接调用仓内训练模块：

- `qwen_tts.training.encode_training_records(...)`
- `qwen_tts.training.train_speaker_package(...)`

当前实现特点：

- 上传音频在内存中完成解码和重采样
- 不生成 `train.log`
- 成功后直接注册 voice package 到 `VoiceRegistry`

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

- `data/voices/` 保存已注册的自定义音色 package
- 训练过程只使用项目内 `data/train/tmp/` 工作目录，任务结束后自动清理
- 训练成功后服务端直接完成注册，不存在单独的 `deployVoice` 保存步骤

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

如果需要 `POST /api/transcribe`：

```bash
pip install -e ".[runtime,api,asr]"
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
  "dataDir": "/path/to/Qwen3-TTS/data",
  "modelsDir": "/path/to/Qwen3-TTS/models",
  "asrModelsDir": "/path/to/Qwen3-TTS/models/asr"
}
```

---

## 7. 前端推荐流程

1. 用户上传多段音频并填写文本
2. 调 `POST /api/trainVoice`
3. 持续读取同一个 `POST /api/trainVoice` 响应流中的 `SSE`
4. 收到 `completed` 事件后，取返回里的 `voice` 或 `speakerName`
5. 后续直接用该 `voice` 调 `POST /api/customVoice`
