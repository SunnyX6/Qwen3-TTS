from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from fastapi import File, Form, UploadFile
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

DEFAULT_ASR_MODEL_SIZE = "large-v3"
DEFAULT_GENERATION_SEED = 0
DEFAULT_GENERATION_MAX_NEW_TOKENS = 2048
DEFAULT_GENERATION_TEMPERATURE = 0.9
DEFAULT_GENERATION_TOP_P = 1.0
DEFAULT_GENERATION_REPETITION_PENALTY = 1.05
DEFAULT_TRAIN_TOKENIZER_MODEL_ID = "Qwen/Qwen3-TTS-Tokenizer-12Hz"


class BaseRequestModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class GenerationOptionsModel(BaseRequestModel):
    seed: int = Field(default=DEFAULT_GENERATION_SEED)
    max_new_tokens: Optional[int] = Field(
        default=DEFAULT_GENERATION_MAX_NEW_TOKENS,
        validation_alias=AliasChoices("maxNewTokens", "max_new_tokens"),
        serialization_alias="maxNewTokens",
    )
    top_p: Optional[float] = Field(
        default=DEFAULT_GENERATION_TOP_P,
        validation_alias=AliasChoices("topP", "top_p"),
        serialization_alias="topP",
    )
    temperature: Optional[float] = DEFAULT_GENERATION_TEMPERATURE
    repetition_penalty: Optional[float] = Field(
        default=DEFAULT_GENERATION_REPETITION_PENALTY,
        validation_alias=AliasChoices("repetitionPenalty", "repetition_penalty"),
        serialization_alias="repetitionPenalty",
    )


class AudioRequestModel(GenerationOptionsModel):
    text: str
    language: str = "Auto"
    responseFormat: Literal["base64", "wav"] = "base64"


class ModelBoundAudioRequestModel(AudioRequestModel):
    modelId: str


class VoiceDesignRequest(ModelBoundAudioRequestModel):
    instruct: str


class CustomVoiceRequest(AudioRequestModel):
    modelId: Optional[str] = None
    speaker: str
    instruct: Optional[str] = None
    dialect: Optional[str] = None


class CloneRequestForm(ModelBoundAudioRequestModel):
    refText: Optional[str] = None
    xVectorOnlyMode: bool = False


@dataclass
class CloneRequest:
    form: CloneRequestForm
    refAudio: UploadFile

    @classmethod
    def as_form(
        cls,
        modelId: str = Form(...),
        text: str = Form(...),
        language: str = Form("Auto"),
        refText: Optional[str] = Form(None),
        xVectorOnlyMode: bool = Form(False),
        responseFormat: Literal["base64", "wav"] = Form("base64"),
        seed: int = Form(DEFAULT_GENERATION_SEED),
        maxNewTokens: int = Form(DEFAULT_GENERATION_MAX_NEW_TOKENS),
        topP: float = Form(DEFAULT_GENERATION_TOP_P),
        temperature: float = Form(DEFAULT_GENERATION_TEMPERATURE),
        repetitionPenalty: float = Form(DEFAULT_GENERATION_REPETITION_PENALTY),
        refAudio: UploadFile = File(...),
    ) -> "CloneRequest":
        return cls(
            form=CloneRequestForm(
                modelId=modelId,
                text=text,
                language=language,
                refText=refText,
                xVectorOnlyMode=xVectorOnlyMode,
                responseFormat=responseFormat,
                seed=seed,
                max_new_tokens=maxNewTokens,
                top_p=topP,
                temperature=temperature,
                repetition_penalty=repetitionPenalty,
            ),
            refAudio=refAudio,
        )

    async def to_payload(self) -> dict:
        payload = self.form.model_dump(exclude_none=True)
        payload["refAudioBytes"] = await self.refAudio.read()
        payload["refAudioFilename"] = self.refAudio.filename
        return payload


class TrainVoiceRequestForm(BaseRequestModel):
    modelId: str
    speakerName: str
    sample_texts: list[str] = Field(
        validation_alias=AliasChoices("sampleTexts", "sample_texts"),
        serialization_alias="sampleTexts",
    )
    language: str = "Auto"
    batchSize: int = 8
    lr: float = 2e-6
    numEpochs: int = 3


@dataclass
class TrainVoiceRequest:
    form: TrainVoiceRequestForm
    refAudio: UploadFile
    sampleAudios: list[UploadFile]

    @classmethod
    def as_form(
        cls,
        modelId: str = Form(...),
        speakerName: str = Form(...),
        sampleTexts: list[str] = Form(...),
        language: str = Form("Auto"),
        batchSize: int = Form(8),
        lr: float = Form(2e-6),
        numEpochs: int = Form(3),
        refAudio: UploadFile = File(...),
        sampleAudios: list[UploadFile] = File(...),
    ) -> "TrainVoiceRequest":
        return cls(
            form=TrainVoiceRequestForm(
                modelId=modelId,
                speakerName=speakerName,
                sample_texts=sampleTexts,
                language=language,
                batchSize=batchSize,
                lr=lr,
                numEpochs=numEpochs,
            ),
            refAudio=refAudio,
            sampleAudios=sampleAudios,
        )

    async def to_payload(self) -> dict:
        sample_texts = self.form.sample_texts
        if len(self.sampleAudios) != len(sample_texts):
            raise ValueError("`sampleAudios` and `sampleTexts` must have the same length")
        if not self.sampleAudios:
            raise ValueError("`sampleAudios` must be a non-empty list")

        samples = []
        for upload_file, sample_text in zip(self.sampleAudios, sample_texts):
            samples.append(
                {
                    "audioBytes": await upload_file.read(),
                    "audioFilename": upload_file.filename,
                    "text": sample_text,
                }
            )

        payload = self.form.model_dump(exclude_none=True)
        payload.pop("sample_texts", None)
        payload["tokenizerModelId"] = DEFAULT_TRAIN_TOKENIZER_MODEL_ID
        payload["samples"] = samples
        payload["refAudioBytes"] = await self.refAudio.read()
        payload["refAudioFilename"] = self.refAudio.filename
        return payload


class TranscribeRequestForm(BaseRequestModel):
    language: str = "Auto"
    modelSize: str = Field(default=DEFAULT_ASR_MODEL_SIZE)


@dataclass
class TranscribeRequest:
    form: TranscribeRequestForm
    audios: list[UploadFile]

    @classmethod
    def as_form(
        cls,
        language: str = Form("Auto"),
        modelSize: str = Form(DEFAULT_ASR_MODEL_SIZE),
        audios: list[UploadFile] = File(...),
    ) -> "TranscribeRequest":
        return cls(
            form=TranscribeRequestForm(
                language=language,
                modelSize=modelSize,
            ),
            audios=audios,
        )

    async def to_payload(self) -> dict:
        if not self.audios:
            raise ValueError("`audios` must be a non-empty list")

        payload = self.form.model_dump(exclude_none=True)
        payload["audios"] = []
        for upload_file in self.audios:
            audio_bytes = await upload_file.read()
            if not audio_bytes:
                raise ValueError(f"`audios` contains an empty file: {upload_file.filename or 'unknown'}")
            payload["audios"].append(
                {
                    "audioBytes": audio_bytes,
                    "audioFilename": upload_file.filename,
                }
            )
        return payload
