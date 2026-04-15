# `api/` 设计文档

本文档定义一个运行在 `Qwen3-TTS` 项目根目录下的 `FastAPI` 服务。

说明：

- 本文档描述当前 `api/` 实现
- 若文档与代码存在冲突，以代码和 `CUSTOM_SPEAKER_BANK_ARCHITECTURE.md` 为准

说明：

- HTTP 路由、请求模型和异常处理放在根目录 `api/` 下
- 队列、运行时状态、任务执行和训练编排放在根目录 `runtime/` 下
- 启动脚本直接运行根目录 `main.py`
- 不修改 `qwen_tts/` 下官方源码

目标：

- API 侧只新增 `FastAPI` / `uvicorn`
- 训练与推理都由仓内模块统一编排
- 保留官方 3 类推理能力
- 增加“单 speaker 训练 -> 自动注册到音色库 -> 直接用于 customVoice”的产品流程
- 默认所有数据都放在项目根目录下的 `data/` 中
- 主进程只负责 API、队列、请求分发、统一 cancel
- 训练/推理运行时、模型加载、设备解析都在 request-scoped 子进程内完成
- 所有 GPU 任务串行执行，避免用户本机显存被并发请求打爆
- `--device auto` 不会隐式降级到 CPU；若部署机器只能用 CPU，必须显式用 `--device cpu` 启动

---

## 1. 总体能力

`api/` 提供以下能力：

- `POST /api/voiceDesign`
  - 调用 `Qwen3TTSModel.generate_voice_design(...)`
  - 调用方必须通过 query 参数传 `requestId`
- `POST /api/clone`
  - 调用 `Qwen3TTSModel.generate_voice_clone(...)`
  - 调用方必须通过 query 参数传 `requestId`
- `POST /api/customVoice`
  - 调用 `Qwen3TTSModel.generate_custom_voice(...)`
  - 调用方必须通过 query 参数传 `requestId`
- `POST /api/trainVoice`
  - 基于用户上传的数据集执行单 speaker 微调
  - 调用方必须通过 query 参数传 `requestId`
  - 同一条 HTTP 连接会一直阻塞到训练终态，再返回单个 JSON
  - 训练成功后自动注册到当前 `CustomVoice` backbone 的音色库
- `POST /api/translate`
  - 独立的语音转文本（ASR）接口
  - 不参与训练流程，只负责语音转文本
  - 调用方必须通过 query 参数传 `requestId`
- `POST /api/cancel`
  - 通用取消接口
  - 请求体必须带 `kind` 和 `requestId`
  - 若目标任务正在运行，请求会一直阻塞到对应 request-scoped worker 进程真正退出后再返回
- `GET /api/voices`
  - 列出当前 backbone 下可用的内置 speaker 和自定义 speaker
- `DELETE /api/voices/{voiceId}`
  - 删除一个已注册的自定义音色
- `GET /api/healthz`
  - 健康检查
  - 返回主进程调度状态、队列状态和当前启动配置

---

## 2. 目录约定

默认目录如下：

```text
main.py
api/
  exceptions.py
  schemas.py
  server.py
runtime/
  catalog.py
  executor.py
  state.py
  task.py
start_api_linux.sh
start_api_windows.bat
models/
  asr/
    faster-whisper/
      large-v3/
      large-v3-turbo/
      ...
    faster-whisper-cache/
data/
  voices/
    index.json
    {voiceId}/
      model/
        speaker.safetensors
      meta.json
```

说明：

- `voices/` 是已注册的自定义音色库，`index.json` 是注册索引
- 训练日志只输出到 API 服务终端，不落盘保存为 `train.log`
- 训练过程使用项目内 `data/train/tmp/` 工作目录，训练完成后会自动清理子目录
- `models/asr/` 是语音转文本（ASR）接口的推荐本地模型目录
- 若启动时传了 `--models-dir /your/path/models`，则 ASR 目录会变成 `/your/path/models/asr`

---

## 3. 请求字段约定

### 3.1 `modelId`

API 对 `modelId` 使用严格白名单，不接受任意本地路径或简写名。

当前支持的官方完整 id 如下：

- `POST /api/voiceDesign`
  - `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `POST /api/clone`
  - `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
  - `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- `POST /api/customVoice`
  - `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
  - `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `POST /api/trainVoice`
  - `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
  - `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
  - `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
  - `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`

解析规则：

- API 会把这些官方完整 id 解析到本地 `./models/<model-leaf>` 目录
- 对应目录必须已经存在且包含完整模型文件
- Python loader 支持的简写名或任意本地路径并不是 API 的对外契约

### 3.2 `speaker`

`/api/customVoice` 对外直接使用 `speaker`。

### 3.3 音频字段

`/api/clone`、`/api/trainVoice` 和 `/api/translate` 的音频输入统一使用 `multipart/form-data` 上传文件。

服务端不再接收音频字段的本地路径、URL、`data:audio` 或原始 base64 字符串。

字段约定：

- `/api/clone`：`refAudio`（单文件）
- `/api/trainVoice`：`refAudio`（单文件）、`sampleAudios`（多文件）、`sampleTexts`（与 `sampleAudios` 一一对应）
- `/api/translate`：`audios`（多文件）
- `/api/clone` 继续保留 `refText`

训练接口会在服务端临时目录中使用这些上传音频，仅用于当前训练任务的数据准备阶段，不会持久化保存到 `data/` 目录。

所有 GPU 重任务接口统一使用 `requestId` 作为调用方自定义任务标识：

- `POST /api/translate?requestId=...`
- `POST /api/voiceDesign?requestId=...`
- `POST /api/clone?requestId=...`
- `POST /api/customVoice?requestId=...`
- `POST /api/trainVoice?requestId=...`

统一取消接口：

- `POST /api/cancel`
  - body: `{"kind":"translate|voiceDesign|clone|customVoice|trainVoice","requestId":"..."}`

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

- 服务启动时只接收 `--device` 配置，不在主进程里加载模型或初始化运行时
- 每个 request-scoped 子进程启动时才真正解析设备并加载运行时
- `--device auto` 会在子进程里按 `CUDA > MPS` 选择可用加速设备
- 若没有可用 CUDA/MPS，则不会隐式进入 CPU 模式
- CPU-only 部署必须显式用 `--device cpu` 启动，显式参数本身就是服务端确认
- 不做运行中切卡

如果用户显式通过启动参数指定 `--device cuda:1`，则优先使用该设备。

如果用户显式通过启动参数指定 `--device mps`，则优先使用 Apple Metal / MPS。

---

## 3.6 队列与资源策略

为避免用户本机显存被并发请求打爆，API 服务采用以下策略：

- 服务内部只有 **1 个 GPU worker**
- 所有 GPU 任务都必须串行执行
- GPU 任务包括：
  - `/api/translate`
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

## 4.1 `POST /api/translate`

独立的语音转文本（ASR）接口。

说明：`translate` 是接口名，不代表“额外多一步流程”；音频转文本本身就是 ASR。

请求类型：`multipart/form-data`

```bash
curl -X POST "http://127.0.0.1:8000/api/translate?requestId=translate_20260415_001" \
  -F "language=Auto" \
  -F "modelSize=large-v3" \
  -F "audios=@/path/to/a.wav" \
  -F "audios=@/path/to/b.wav"
```

请求字段：

- `language`
  - 可选值（与 `qwen3-tts` 语言口径对齐）：`Auto`、`Chinese`、`English`、`Japanese`、`Korean`、`German`、`French`、`Russian`、`Portuguese`、`Spanish`、`Italian`、`Beijing_Dialect`、`Sichuan_Dialect`
  - 服务端会自动映射到 ASR 语言码；`Beijing_Dialect` / `Sichuan_Dialect` 会映射为中文识别
- `modelSize`
  - 当前用于 `faster-whisper`
  - 可选值：`medium`、`large-v2`、`large-v3`、`large-v3-turbo`
  - 内部映射：当 `language=English` 且 `modelSize=medium` 时，服务端自动使用 `medium.en`
- `audios`
  - 一个或多个上传音频文件

返回：

```json
{
  "ok": true,
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
      "providerUsed": "faster-whisper",
      "error": null
    },
    {
      "index": 1,
      "fileName": "b.wav",
      "ok": true,
      "text": "今天下午三点开会。",
      "languageDetected": "Chinese",
      "languageCode": "zh",
      "providerUsed": "faster-whisper",
      "error": null
    }
  ]
}
```

HTTP 状态码：

- `200`：全部文件识别成功
- `207`：部分成功、部分失败（失败详情在 `results[i].error`）
- `422`：全部文件识别失败（请求字段合法，但批处理内全部失败）
- `400`：请求参数校验失败（如 `language` / `modelSize` 不支持）
- `404`：所需本地 ASR 模型目录不存在

说明：

- 该接口和训练接口没有耦合，只是独立的语音转文本能力
- 服务端在内存中解码音频，并统一转换到 `16kHz`
- 服务端统一使用 `faster-whisper`
- 单个文件失败不会拖垮整批请求，错误会落在对应 `results[i].error`
- 仅从本地模型目录加载，不做兜底自动下载

本地模型目录约定：

- `faster-whisper`
  - `<models-dir>/asr/faster-whisper/<modelSize>`
  - 例：`./models/asr/faster-whisper/large-v3`

推荐手动下载命令：

```bash
# faster-whisper medium (ModelScope)
pip install -U modelscope
modelscope download --model Systran/faster-whisper-medium \
  --local_dir ./models/asr/faster-whisper/medium

# faster-whisper medium (Hugging Face)
pip install -U "huggingface_hub[cli]"
huggingface-cli download Systran/faster-whisper-medium \
  --local-dir ./models/asr/faster-whisper/medium

# faster-whisper large-v2 (ModelScope)
modelscope download --model Systran/faster-whisper-large-v2 \
  --local_dir ./models/asr/faster-whisper/large-v2

# faster-whisper large-v2 (Hugging Face)
huggingface-cli download Systran/faster-whisper-large-v2 \
  --local-dir ./models/asr/faster-whisper/large-v2

# faster-whisper large-v3 (ModelScope)
modelscope download --model Systran/faster-whisper-large-v3 \
  --local_dir ./models/asr/faster-whisper/large-v3

# faster-whisper large-v3 (Hugging Face)
huggingface-cli download Systran/faster-whisper-large-v3 \
  --local-dir ./models/asr/faster-whisper/large-v3

# faster-whisper large-v3-turbo (Hugging Face)
huggingface-cli download Systran/faster-whisper-large-v3-turbo \
  --local-dir ./models/asr/faster-whisper/large-v3-turbo
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

请求 URL：

```text
POST /api/voiceDesign?requestId=voice_design_20260415_001
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
curl -X POST "http://127.0.0.1:8000/api/clone?requestId=clone_20260415_001" \
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
  "speaker": "user_001_voice",
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

请求 URL：

```text
POST /api/customVoice?requestId=custom_voice_20260415_001
```

说明：

- `speaker` 是请求必填字段
- 训练完成后，直接把返回里的 `speaker` 传给 `customVoice`
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
curl -X POST "http://127.0.0.1:8000/api/trainVoice?requestId=user_001_train_20260414" \
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

响应类型：`application/json`

```json
{
  "ok": true,
  "requestId": "user_001_train_20260414",
  "taskId": "user_001_train_20260414",
  "status": "completed",
  "speaker": "user_001_voice",
  "voiceId": "voice_20260412_113045_123456",
  "baseModelId": "/abs/path/to/Qwen3-TTS/models/Qwen3-TTS-12Hz-1.7B-Base",
  "error": null
}
```

若训练过程中被另一条请求取消，则当前 `POST /api/trainVoice` 会返回：

```json
{
  "ok": false,
  "requestId": "user_001_train_20260414",
  "status": "canceled",
  "kind": "trainVoice",
  "error": "Canceled by request"
}
```

说明：

- 这里的训练是单 speaker 微调
- `requestId` 为必填 query 参数，由调用方预先生成
- 服务端固定使用 `Qwen/Qwen3-TTS-Tokenizer-12Hz`，调用方不需要也不能再传 `tokenizerModelId`
- 当前训练数据仍然必须有文本
- `sampleAudios` 和 `sampleTexts` 必须按顺序一一对应，数量必须一致
- 训练使用服务启动时传入的 `--device` 策略，并在子进程里解析真实设备，不在请求中单独传 `device`
- 上传音频不会持久化保存到 `data/`，只在内存中解码和重采样
- 训练进度日志只输出到 API 服务终端
- 该接口不会返回 `queued` / `running` 等中间态给调用方，只返回终态 JSON
- 可能的终态包括：`completed`、`failed`、`canceled`
- 成功响应稳定返回的训练结果字段为 `requestId`、`taskId`、`status`、`speaker`、`voiceId`、`baseModelId`、`error`
- 若 `requestId` 已存在，服务会返回 `409 Conflict`
- 若前端需要支持用户中途取消，必须在发起训练前先生成 `requestId`

---

## 4.6 `POST /api/cancel`

统一取消接口。

请求：

```bash
curl -X POST "http://127.0.0.1:8000/api/cancel" \
  -H "Content-Type: application/json" \
  -d '{
    "kind": "trainVoice",
    "requestId": "user_001_train_20260414"
  }'
```

响应：

```json
{
  "ok": true,
  "requestId": "user_001_train_20260414",
  "kind": "trainVoice"
}
```

说明：

- `kind` 和 `requestId` 必须同时提供
- 若任务仍在队列中，接口会直接移出队列后返回成功
- 若任务已经开始运行，接口会等待对应 request-scoped worker 进程真正退出后再返回成功
- 若任务不存在，返回 `404`
- 若任务已经 `completed` 或 `failed`，返回 `409`
- 若重复取消同一个运行中的任务，后续 `POST /api/cancel` 也会等到同一个终态

---

## 4.7 `GET /api/voices`

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
      "baseModelId": "/abs/path/to/Qwen3-TTS/models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
      "enabled": true,
      "createdAt": "2026-04-02T10:20:30.000000",
      "supportedDialects": [
        "beijing_dialect",
        "sichuan_dialect"
      ],
      "nativeDialect": null,
      "source": "custom",
      "deletable": true
    },
    {
      "voiceId": null,
      "speaker": "Serena",
      "baseModelId": "/abs/path/to/Qwen3-TTS/models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
      "enabled": true,
      "supportedDialects": [
        "beijing_dialect",
        "sichuan_dialect"
      ],
      "nativeDialect": null,
      "source": "builtin",
      "deletable": false
    },
    {
      "voiceId": null,
      "speaker": "Dylan",
      "baseModelId": "/abs/path/to/Qwen3-TTS/models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
      "enabled": true,
      "supportedDialects": [
        "beijing_dialect",
        "sichuan_dialect"
      ],
      "nativeDialect": "beijing_dialect",
      "source": "builtin",
      "deletable": false
    }
  ]
}
```

说明：

- `supportedDialects` 表示当前 CustomVoice backbone 可接受的方言控制参数
- `nativeDialect` 只表示该 speaker 的原生方言属性；训练出来的自定义 speaker 固定为 `null`
- 返回里的 `baseModelId` 是历史字段名；当前实现里它实际表示该条音色绑定的 CustomVoice backbone 本地模型目录

---

## 4.8 `DELETE /api/voices/{voiceId}`

删除一个已注册的自定义音色。

返回：

```json
{
  "ok": true,
  "voiceId": "voice_20260412_102050_944573",
  "speaker": "user_001_voice",
  "baseModelId": "/abs/path/to/Qwen3-TTS/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"
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
- `multiprocessing`
- `soundfile`
- `librosa`

其中：

- 根目录 `main.py` 负责 CLI、启动 `uvicorn`
- `api/exceptions.py` 负责异常到 HTTP 响应的映射
- `api/server.py` 负责 `FastAPI` 路由和异常处理
- `api/schemas.py` 负责请求模型
- `runtime/catalog.py` 负责模型白名单和本地模型目录校验
- `runtime/executor.py` 同时承担：
  - 主进程控制面：队列、请求分发、统一 cancel
  - 子进程任务入口：设备选择、GPU 初始化、模型加载、实际任务执行
- `runtime/state.py` 负责 request-scoped 运行时状态
- `runtime/task.py` 负责 ASR、推理、训练编排和 voice package 注册

## 5.2 训练由仓内模块编排

训练流程当前直接调用仓内训练模块：

- `qwen_tts.training.encode_training_records(...)`
- `qwen_tts.training.train_speaker_package(...)`

当前实现特点：

- 上传音频在内存中完成解码和重采样
- 不生成 `train.log`
- 成功后直接注册 voice package 到 `VoiceRegistry`

## 5.3 请求级运行时进程

服务默认在单次 GPU 请求完成后退出对应的运行时 worker，避免长期占用公共 GPU 资源。

当前实现要求：

- 推理和训练都使用 request-scoped worker
- worker 只处理一个任务，任务完成立即退出
- 模型、显存、解码缓冲和中间张量跟随 worker 一起释放
- 不提供 keep-warm 模式，也不允许运行时资源常驻

## 5.4 主进程控制面 + 单 GPU worker 串行执行

推荐实现：

- 控制面留在主进程
- 主进程只负责 API、内存队列、请求分发、统一 cancel
- 每个 GPU 请求启动独立 request-scoped worker
- 设备选择、`torch/cuda` 初始化、模型加载、训练和推理都只发生在子进程
- 队列达到上限时，新请求直接返回忙碌错误

这样既保证了请求结束后 worker 退出、运行时资源被系统回收，也保证了主进程退出时不会再遗留额外中间进程。

## 5.5 草稿与正式库分离

- `data/voices/` 保存已注册的自定义音色 package
- 训练过程只使用项目内 `data/train/tmp/` 工作目录，任务结束后自动清理
- 训练成功后服务端直接完成注册，不存在单独的 `deployVoice` 保存步骤

## 5.6 禁止隐式降级到 CPU

默认策略：

- 如果启动参数是 `--device auto`
  - 子进程请求运行时优先尝试 CUDA
  - 没有 CUDA 再尝试 MPS
  - 如果既没有 CUDA 也没有 MPS，则 **不会隐式切到 CPU**
- 如果部署方明确接受 CPU 模式，必须显式使用 `--device cpu`

原因：

- 当前架构里主进程只是控制面，不负责运行时初始化
- 模型加载、设备解析、训练/推理都在 request-scoped 子进程里完成
- 所以 CPU 模式必须体现在明确的启动参数里，而不是靠子进程里的交互式确认

这样可以避免服务在 `--device auto` 下悄悄掉到极慢的 CPU 模式。

---

## 6. 启动方式

```bash
python main.py --host 0.0.0.0 --port 8000
```

如果机器只能跑 CPU：

```bash
python main.py --host 0.0.0.0 --port 8000 --device cpu --no-flash-attn
```

或者：

```bash
./start_api_linux.sh --host 0.0.0.0 --port 8000
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

如果需要 `POST /api/translate`：

```bash
pip install -e ".[runtime,api,asr]"
```

> Before installing API dependencies, install a PyTorch build that matches the target machine from https://pytorch.org/get-started/locally/ . The project does not auto-pick a GPU-specific PyTorch build for users.

> `--flash-attn` is enabled by default. If `flash_attn` is missing or broken, API startup now exits immediately with an installation hint instead of waiting until the first inference request fails.

> The API keeps the control plane in the main process and serializes GPU work through a main-process queue. Runtime jobs run in fresh request-scoped child processes and release runtime resources when each request finishes.

> `--max-gpu-queue-size` defaults to `2`, which means at most 2 waiting GPU jobs can queue behind the currently running job.

### 6.1 启动时设备选择

推荐行为：

- 如果用户未传 `--device`
  - 主进程只记录 `--device=auto`
  - 子进程在真正执行请求时先探测 CUDA
  - 没有 CUDA 再尝试 MPS（仅 macOS）
  - 都没有则请求直接失败，不会隐式切到 CPU
- `--device` 默认值为 `auto`
- 如果用户传了 `--device cuda:0`
  - 子进程固定使用该设备
- 如果用户传了 `--device mps`
  - 子进程固定使用 MPS
- 如果用户传了 `--device cpu`
  - 视为显式接受 CPU 模式
  - 子进程直接使用 CPU，不再做交互式确认

### 6.2 启动时终端输出

主进程只负责控制面，所以启动日志不保证提前打印“最终运行设备”。

示例：

```text
Qwen3-TTS API starting...
Data dir: /path/to/Qwen3-TTS/data
API listening on http://0.0.0.0:8000/api
```

如果需要强制 CPU 模式，应该在启动命令里显式传 `--device cpu --no-flash-attn`，而不是等待服务在请求期内弹交互提示。

### 6.3 服务状态接口建议

`GET /api/healthz` 建议至少返回：

```json
{
  "ok": true,
  "status": "healthy",
  "selectedDevice": "auto",
  "deviceMode": "",
  "deviceName": "",
  "queueStatus": {
    "activeJob": null,
    "queuedCount": 0,
    "queuedJobs": []
  },
  "runtimePolicy": "executor-queue+single-request-child-process",
  "dataDir": "/path/to/Qwen3-TTS/data",
  "modelsDir": "/path/to/Qwen3-TTS/models",
  "asrModelsDir": "/path/to/Qwen3-TTS/models/asr"
}
```

说明：

- `selectedDevice` 是服务启动时传入的 `--device` 配置
- 主进程不解析运行时设备，所以 `deviceMode` / `deviceName` 在 `--device auto` 下可能为空
- 真实执行设备由每个请求对应的子进程在运行时解析

---

## 7. 前端推荐流程

1. 用户上传多段音频并填写文本
2. 前端先生成一个唯一的 `requestId`
3. 调 `POST /api/trainVoice?requestId=...`
4. 如用户中途取消，另开一条请求调 `POST /api/cancel`
5. `POST /api/cancel` body 传：
   `{"kind":"trainVoice","requestId":"..."}`
6. `POST /api/trainVoice` 返回 `completed` 后，取返回里的 `speaker`
7. 后续直接用该 `speaker` 调 `POST /api/customVoice?requestId=...`
