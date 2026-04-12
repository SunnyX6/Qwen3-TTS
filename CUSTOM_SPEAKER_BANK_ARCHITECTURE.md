# Qwen3-TTS 自定义 Speaker Bank 架构方案

本文档定义 `Qwen3-TTS` 的唯一目标架构：

- 不再把自定义 speaker 当成一堆完整 checkpoint 的目录集合
- 不再要求调用方传模型路径
- 不再调用 `finetuning/prepare_data.py` 和 `finetuning/sft_12hz.py`
- 直接按 `Qwen3-TTS` 官方 `CustomVoice` 模型的真实架构二开
- 让自定义训练出来的 speaker 最终也能像官方内置 speaker 一样，通过
  `generate_custom_voice(..., speaker="xxx")` 直接调用

---

## 1. 结论

本项目的最终形态固定为：

- 共享一个 `CustomVoice` 主模型作为运行时 backbone
- 所有自定义 speaker 训练结果都发布为 `speaker package`
- 所有正式 speaker package 统一存放在 `data/voices/{voiceId}/`
- 服务维护一份正式的 `speaker bank`
- 训练成功后立即注册 speaker，不等额外发布动作
- 调用方始终只传 `speaker`
- 服务内部负责把 `speaker` 解析到：
  - 官方内置 speaker
  - 或本地发布的自定义 speaker package

最终调用方式保持为：

```python
model.generate_custom_voice(
    text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    language="Chinese",
    speaker="sunny_female_01",
    instruct="用特别愤怒的语气说",
)
```

调用方不传模型目录，不传 checkpoint 路径，不关心内部注册逻辑。

---

## 2. 官方架构事实

本方案建立在以下事实上：

1. 官方内置 speaker 不是“一堆独立模型”
- `Qwen3-TTS-12Hz-1.7B-CustomVoice` 只有一套主模型权重
- 多个内置 speaker 共享同一套模型参数
- speaker 只是当前模型配置里的 `talker_config.spk_id` 映射和对应 embedding 槽位

2. 官方 `generate_custom_voice()` 调 speaker 的前提是：
- 当前加载模型已经知道这些 speaker
- `speaker` 会被解析到 `spk_id`
- 然后进入当前模型的生成路径

3. 官方 `finetuning/sft_12hz.py` 只是一个“能训练出完整 checkpoint”的脚本
- 它不是生产级的 speaker bank 架构
- 它的输出是完整模型 checkpoint
- 它不负责统一管理多个自定义 speaker

因此，本项目不再把官方 `finetuning` 目录当作正式架构，只把它当作可复用逻辑来源。

---

## 3. 最终目录

唯一有效目录结构如下：

```text
data/
  train/
    {taskId}/
      dataset/
      training/
      export/
      meta.json
  voices/
    index.json
    {voiceId}/
      meta.json
      model/
        speaker.safetensors
        speaker_config.json
```

说明：

- `data/train` 只放训练中的草稿和中间产物
- `data/voices` 只放正式发布后的 speaker package
- `data/voices/index.json` 是正式 speaker bank 的权威注册表
- `voiceLibrary` 目录彻底废弃
- 正式发布目录里不再存 `preview.wav`

---

## 4. 正式发布物定义

每个自定义 speaker 发布后固定为一个 `speaker package`。

目录：

```text
data/voices/{voiceId}/
  meta.json
  model/
    speaker.safetensors
    speaker_config.json
```

### 4.1 `meta.json`

建议结构：

```json
{
  "voiceId": "voice_20260411_xxxxxxxx",
  "speaker": "sunny_female_01",
  "displayName": "Sunny Female 01",
  "sourceTaskId": "train_20260411_xxxxxxxx",
  "baseModelId": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "tokenizerType": "12hz",
  "ttsModelType": "custom_voice",
  "enabled": true,
  "createdAt": "2026-04-11T18:30:00"
}
```

### 4.2 `speaker_config.json`

建议结构：

```json
{
  "schemaVersion": 1,
  "speaker": "sunny_female_01",
  "slotId": 3000,
  "baseModelId": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "tokenizerType": "12hz",
  "adapterType": "lora",
  "loraRank": 16
}
```

### 4.3 `speaker.safetensors`

固定保存该 speaker 的可挂载参数，不保存完整主模型。内容固定为：

- `speaker_embedding`
- `codec_embedding_slot_3000`
- `lora` 适配器参数

该文件不包含整套 backbone 权重。

---

## 5. 训练产物定义

训练目录仍然使用：

```text
data/train/{taskId}/
```

但训练的最终目标不再是产出完整 checkpoint 作为正式发布物，而是产出：

- 草稿训练状态
- 中间导出目录 `export/`
- 可发布的 `speaker package`

训练完成后，`export/` 中产出的内容会被立即注册并搬入：

```text
data/voices/{voiceId}/
```

---

## 6. 训练方案

### 6.1 训练入口

训练阶段不再调用：

- `finetuning/prepare_data.py`
- `finetuning/sft_12hz.py`

这两个脚本只作为逻辑参考，不作为正式执行入口。

正式训练入口改为项目内的新训练模块，例如：

```text
qwen_tts/training/
  data_pipeline.py
  speaker_package_train.py
  export_package.py
```

补充约束：

- 运行时共享 backbone 固定是 `CustomVoice`
- 但训练源模型固定使用同家族的 `Base`
- 原因不是“兼容补丁”，而是官方只有 `Base` 挂了 `speaker_encoder`
- 因此系统必须先把请求解析成一对模型：
  - 训练源模型：`Qwen3-TTS-...-Base`
  - 运行时 backbone：同尺寸同 tokenizer 的 `Qwen3-TTS-...-CustomVoice`
- 训练完成后导出的 package 只绑定 `CustomVoice` backbone，不绑定 `Base`

### 6.2 可复用逻辑来源

训练实现直接复用官方源码里的底层逻辑：

- tokenizer 编码逻辑
- `audio_codes` 数据准备方式
- `ref_audio` / `ref_mels` 处理方式
- `Qwen3TTSForConditionalGeneration` 的前向路径
- 官方训练使用的 loss 结构

来源文件：

- `finetuning/prepare_data.py`
- `finetuning/sft_12hz.py`
- `qwen_tts/core/models/modeling_qwen3_tts.py`

### 6.3 训练目标

训练时固定采用：

- 冻结共享 backbone 主参数
- 只训练 speaker-specific 参数

唯一训练对象固定为：

- 自定义 speaker 的 `codec_embedding` 槽位 `3000`
- `talker` 主体上的 LoRA 参数
- `code_predictor` 上的 LoRA 参数

不训练：

- tokenizer
- `speaker_encoder`
- 完整 backbone 全量参数

### 6.4 训练输出

训练完成后直接导出 speaker package 所需的参数：

- `speaker_embedding`
- `codec_embedding_slot_3000`
- `talker` LoRA 权重
- `code_predictor` LoRA 权重

不导出完整 checkpoint 作为正式发布物。

---

## 7. 自动注册方案

### 7.1 语义

训练成功后的唯一收口动作是：

- 把训练任务导出的 `speaker package` 立刻注册进正式 `speaker bank`

系统不再保留“先训练，再单独发布才能可用”的双路径。

因此：

- 试听和正式调用走同一条 `speaker -> generate_custom_voice()` 路径
- 训练成功后，speaker 立即可用于 `customVoice`
- 如果用户不满意，直接删除该 speaker

系统不再负责：

- 搬运 `preview.wav`
- 暴露训练目录路径
- 把完整 checkpoint 当正式模型发布
- 用 `publishVoice/deployVoice` 决定 speaker 是否可用

### 7.2 发布流程

1. `POST /api/trainVoice` 完成上传并创建任务，立即返回 `taskId`
2. 前端或调用方基于 `taskId` 订阅训练事件流
3. 训练任务完成，导出 `speaker package`
4. 校验 `speaker package` 完整性
5. 校验 `speaker` 名称是否冲突
6. 分配新的 `voiceId`
7. 写入 `data/voices/{voiceId}/`
8. 原子更新 `data/voices/index.json`
9. 运行时 `voice_registry` 立即刷新当前进程内映射
10. 更新 `data/train/{taskId}/meta.json`，写入 `voiceId/speaker`

训练接口的终态不是“preview_ready”，而是“registered”。

### 7.3 冲突规则

注册时固定执行以下校验：

- 不能与官方内置 speaker 重名
- 不能与已发布自定义 speaker 重名
- 不能发布到不兼容的 `baseModelId`

重名直接拒绝，不做覆盖，不做模糊匹配。

---

## 8. Speaker Bank

### 8.1 注册表

`data/voices/index.json` 是唯一权威注册表。

建议结构：

```json
{
  "schemaVersion": 1,
  "voices": [
    {
      "voiceId": "voice_20260411_xxxxxxxx",
      "speaker": "sunny_female_01",
      "baseModelId": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
      "path": "data/voices/voice_20260411_xxxxxxxx",
      "enabled": true,
      "createdAt": "2026-04-11T18:30:00"
    }
  ]
}
```

### 8.2 启动加载

服务启动时只加载：

- `index.json`
- 必要的 voice metadata

服务不会把所有 speaker package 权重一次性全读进内存。

运行时内存中只保留：

- speaker 名到 package metadata 的映射
- 已经被最近使用过的 package 权重缓存

这和官方模型内置的 `spk_id` 本质一样，都是小规模 metadata 映射，不是“全量模型索引进内存”。

### 8.3 热更新

`speaker bank` 不是“启动快照”，而是运行时注册表。

固定行为如下：

- 训练成功后注册 speaker，当前 API 进程内立即可见
- 删除 speaker 后，当前 API 进程内立即不可见
- 后续请求调用 `get_supported_speakers()` / `generate_custom_voice()` 时读取最新注册表快照
- 不允许依赖重启 API 服务让 speaker 生效

当前 `uvicorn` 单进程部署下，由 `voice_registry` 负责热刷新。

---

## 9. 运行时推理

### 9.1 `get_supported_speakers()`

`Qwen3TTSModel.get_supported_speakers()` 改为返回并集：

- 当前 backbone 模型内置 speaker
- `speaker bank` 里与当前 backbone 兼容的自定义 speaker

因此，`Qwen3-TTS` 自己就能知道自己有哪些可用 speaker。

这里不能只在模型初始化时算一次。

`get_supported_speakers()` 必须在调用时读取当前 `voice_registry` 快照，否则新 speaker 仍会退化成重启后才能看到。

### 9.2 `generate_custom_voice()`

`generate_custom_voice()` 的调用方式不变，但内部行为改为：

1. 先检查 `speaker` 是否属于当前 backbone 内置 speaker
2. 如果是，直接走官方原生逻辑
3. 如果不是，去 `speaker bank` 查找
4. 找到后读取对应 `speaker package`
5. 将 package 注入当前共享 `CustomVoice` 主模型
6. 临时把该 speaker 绑定到 `slotId = 3000`
7. 调用现有生成路径
8. 生成结束后恢复模型现场

### 9.3 注入方式

自定义 speaker 的激活固定为：

- 临时覆盖 `codec_embedding.weight[3000]`
- 临时激活当前 speaker 的 LoRA adapter
- 临时在运行时 speaker map 中挂上：

```python
{speaker.lower(): 3000}
```

这样就能继续复用官方 `generate_custom_voice()` 的 speaker 路径，而不需要改动调用方接口。

训练完成后的第一次试听，也必须走这一条正式路径，不再允许单独做“草稿试听专用逻辑”。

### 9.4 批量调用

批量调用时，如果一批请求里混有多个不同 custom speaker：

- 先按 speaker 分组
- 每组依次激活对应 package
- 分组生成
- 最后按输入顺序还原结果

不允许在同一次前向里同时激活多个 custom speaker package。

---

## 10. 共享主模型

本方案固定使用共享 `CustomVoice` 主模型作为 backbone。

要求：

- 自定义 speaker package 必须绑定某个具体 `CustomVoice` backbone
- 例如：
  - `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
  - `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`

自定义 speaker package 不能脱离 backbone 单独存在。

也就是说：

- 自定义 speaker 不是一套完整模型
- 它只是一个可挂载到特定 backbone 上的 speaker 增量包

---

## 11. API 与 SDK 行为

### 11.1 SDK

SDK 目标行为固定为：

```python
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = model.generate_custom_voice(
    text="你好。",
    language="Chinese",
    speaker="sunny_female_01",
    instruct="平静自然",
)
```

调用方不传 `voiceId`、不传 `model path`、不传训练目录。

### 11.2 服务接口

服务接口目标语义固定为：

- `POST /api/trainVoice`
  - 创建训练任务
  - 训练成功后立即注册 speaker
- `GET /api/trainVoice/{taskId}/events`
  - 训练状态 SSE 主通道
- `GET /api/voices`
  - 返回正式 speaker bank 条目
- `DELETE /api/voices/{voiceId}`
  - 删除 speaker 并从运行时注册表反注册
- `POST /api/customVoice`
  - 继续只收 `speaker`

约束：

- 训练进度只走 SSE
- 不把文件上传的 `POST /api/trainVoice` 本身做成长连接事件流
- SSE 首帧必须返回当前任务快照，因此不再保留单独的训练状态查询接口
- 任务创建和任务订阅必须分离，否则断线恢复、应用重启恢复、任务列表回填都会变脏

### 11.3 Preview

训练后的试听统一改为：

- 训练完成后，speaker 已经注册成功
- 前端直接调用一次 `customVoice(speaker=...)`
- 后端走正式 `generate_custom_voice()` 路径
- 音频以 bytes/base64 直接返回给调用方

不再依赖：

- `preview.wav`
- `/api/files/...`
- 训练目录里的静态音频文件
- 草稿试听专用模型路径

---

## 12. 代码落位

目标代码结构固定为：

```text
qwen_tts/
  inference/
    qwen3_tts_model.py
    voice_registry.py
    voice_package.py
    voice_router.py
  training/
    data_pipeline.py
    speaker_package_train.py
    export_package.py
```

说明：

- `voice_registry.py`
  - 负责 `index.json`
  - 负责 speaker 名冲突校验
  - 负责 metadata 查询
  - 负责运行时热刷新与反注册
- `voice_package.py`
  - 负责 speaker package 的读写格式
- `voice_router.py`
  - 负责把 `speaker` 路由到 built-in 或 custom package
- `speaker_package_train.py`
  - 负责训练过程
- `export_package.py`
  - 负责把训练产物导出为正式 package

---

## 13. 明确删除项

以下内容在正式实现时全部直接删除，不保留兼容：

- `voiceLibrary` 目录
- `/api/files/...` 文件路由
- 将完整 checkpoint 当正式发布物
- 训练完成后落地 `preview.wav` 再供前端下载
- 调用方传模型路径来选择自定义 speaker
- 通过多个完整 checkpoint 的路径路由 speaker
- 官方 `finetuning` 脚本作为正式训练入口
- `publishVoice/deployVoice` 作为 speaker 可见性的开关
- 草稿试听和正式调用分成两套链路

---

## 14. 实施顺序

实施顺序固定为：

1. 清理旧目录与旧协议
   - 移除 `voiceLibrary`
   - 正式目录统一为 `data/voices`
2. 重写训练模块
   - 不再调用官方脚本
   - 只复用底层逻辑
3. 定义 speaker package 格式
4. 实现 `speaker bank`
5. 改造 `Qwen3TTSModel.get_supported_speakers()`
6. 改造 `Qwen3TTSModel.generate_custom_voice()`
7. 改造训练完成后的自动注册与删除接口
8. 去掉训练预览的文件落地逻辑

---

## 15. 最终判定标准

当且仅当满足以下条件时，改造才算完成：

1. 自定义训练 speaker 训练成功后无需重启即可出现在 `get_supported_speakers()` 中
2. 调用方只传 `speaker` 即可生成
3. 训练完成后的试听与正式调用走同一条 `customVoice -> generate_custom_voice()` 路径
4. 删除 speaker 后无需重启即可从 `get_supported_speakers()` 中消失
5. 不需要传模型路径
6. 服务端没有 `voiceLibrary`
7. 正式发布目录固定为 `data/voices/{voiceId}`
8. 正式发布物中没有 `preview.wav`
9. 训练入口不再调用官方 `finetuning` 脚本

这就是 `Qwen3-TTS` 自定义 speaker 统一管理的最终架构。
