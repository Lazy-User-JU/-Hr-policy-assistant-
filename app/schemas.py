from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(max_length=1000)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty.")
        return value


class Citation(BaseModel):
    document: str
    version: int
    section: str | None
    quote: str


class AskResponse(BaseModel):
    status: Literal["answered", "refused"]
    answer: str
    citations: list[Citation]
    refusal_reason: str | None = None
    answered_by: Literal["llm", "extractive"] | None = None


class UploadResponse(BaseModel):
    status: Literal["indexed", "unchanged"]
    document: str
    version: int
    chunks: int


class Evidence(BaseModel):
    excerpt: int
    quote: str


class ModelReply(BaseModel):
    answerable: bool
    answer: str = ""
    evidence: list[Evidence] = []
