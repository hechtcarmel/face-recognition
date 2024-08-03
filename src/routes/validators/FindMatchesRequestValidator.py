from pydantic import BaseModel, field_validator
from typing import List, Optional
import os

from src.services.FaceRecognitionService import ModelType


class FindMatchesRequestValidator(BaseModel):
    selfie_paths: List[str]
    folder_path: str
    output_path: Optional[str] = None
    model_name: ModelType = 'Facenet'

    # @field_validator('selfie_paths')
    # @classmethod
    # def check_selfie_paths(cls, v):
    #     if not os.path.exists(v):
    #         raise ValueError(f"Selfie path {v} does not exist")
    #     return v

    # @field_validator('folder_path')
    # @classmethod
    # def check_folder_path(cls, v):
    #     if not os.path.exists(v):
    #         raise ValueError("Folder path does not exist")
    #     return v
