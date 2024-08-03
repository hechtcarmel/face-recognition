import json
import logging
import os
from pathlib import Path
from typing import List, Set, Iterable, Literal, Optional

from deepface import DeepFace

ModelType = Literal[
    "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet"]


class FaceRecognitionService:

    @staticmethod
    def process_and_save_results(selfie_paths: List[str], folder_path: str, output_path: str,
                                 model_name: ModelType = 'VGG-Face', threshold: Optional[float] = None) -> list[str]:
        """
        Process the target images and save the results to a JSON file.

        :param selfie_paths: List of paths to target images.
        :param folder_path: Path to the folder containing images to compare.
        :param output_path: Path to save the results JSON file.
        :param model_name: Name of the model to use for face recognition.
        """
        logging.info(f"Processing target images: {selfie_paths}. Comparing with images in {folder_path}")
        matches = FaceRecognitionService._find_matches(selfie_paths, folder_path, model_name, threshold=threshold)
        FaceRecognitionService._write_results_to_json(matches, output_path)
        logging.info(f"Mathces={output_path}")

        logging.info(f"Results written to {output_path}")
        return list(matches)


    @staticmethod
    def _find_matches(selfie_paths: List[str], folder_path: str, model_name: str, threshold: Optional[float] = None) -> Set[str]:
        """
        Find matches for the target images in the specified folder.

        :param selfie_paths: List of paths to target images.
        :param folder_path: Path to the folder containing images to compare.
        :param model_name: Name of the model to use for face recognition.
        :return: Set of file paths that match the target images.
        """
        matches = set()
        for target_path in selfie_paths:
            try:
                result = DeepFace.find(img_path=target_path, db_path=folder_path, model_name=model_name, threshold=threshold)
                for match in result[0]['identity']:
                    matches.add(match)
            except Exception as e:
                logging.error(f"Error processing {target_path}: {e}")
        return matches

    @staticmethod
    def _write_results_to_json(results: Iterable[str], output_path: str) -> None:
        """
        Write the results to a JSON file.

        :param results: Iterable of file paths that match the target images.
        :param output_path: Path to save the results JSON file.
        """
        output = Path(output_path)
        with output.open('w') as f:
            json.dump(list(results), f, indent=4)
