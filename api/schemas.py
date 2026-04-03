from __future__ import annotations

from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class BaseRequestModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class AudioRequestModel(BaseRequestModel):
    modelId: str
    text: str
    language: str = "Auto"
    responseFormat: Literal["base64", "wav"] = "base64"


class VoiceDesignRequest(AudioRequestModel):
    instruct: str


class CloneRequest(AudioRequestModel):
    refAudio: str
    refText: Optional[str] = None
    xVectorOnlyMode: bool = False


class CustomVoiceRequest(AudioRequestModel):
    voice: Optional[str] = None
    instruct: Optional[str] = None


class TrainSample(BaseModel):
    audio: str
    text: str


class TrainVoiceRequest(BaseRequestModel):
    modelId: str
    tokenizerModelId: str
    speakerName: str
    samples: list[TrainSample]
    refAudio: str
    previewText: str
    previewInstruct: Optional[str] = None
    language: str = "Auto"
    batchSize: int = 8
    lr: float = 2e-6
    numEpochs: int = 3
    training_audio_dir: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("training-audio-dir", "trainingAudioDir", "training_audio_dir"),
        serialization_alias="trainingAudioDir",
    )


class SaveVoiceRequest(BaseRequestModel):
    taskId: str
    voiceName: Optional[str] = None
