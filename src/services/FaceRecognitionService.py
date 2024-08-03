import json
import logging
from typing import List, Set, Iterable
from deepface import DeepFace
from pathlib import Path


class FaceRecognitionService:
    @staticmethod
    def process_and_save_results(selfie_paths: List[str], folder_path: str, output_path: str, model_name: str = 'VGG-Face') -> None:
        """
        Process the target images and save the results to a JSON file.

        :param selfie_paths: List of paths to target images.
        :param folder_path: Path to the folder containing images to compare.
        :param output_path: Path to save the results JSON file.
        :param model_name: Name of the model to use for face recognition.
        """
        logging.info(f"Processing target images: {selfie_paths}. Comparing with images in {folder_path}")
        matches = FaceRecognitionService._find_matches(selfie_paths, folder_path, model_name)
        FaceRecognitionService._write_results_to_json(matches, output_path)
        logging.info(f"Results written to {output_path}")

    @staticmethod
    def _find_matches(selfie_paths: List[str], folder_path: str, model_name: str) -> Set[str]:
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
                result = DeepFace.find(img_path=target_path, db_path=folder_path, model_name=model_name)
                for match in result[0]['identity']:
                    matches.add(match)
            except Exception as e:
                logging.error(f"Error processing {target_path}: {e}")
        return matches

    @staticmethod
    def _verify_match(target_path: Path, file_path: Path, model_name: str) -> bool:
        """
        Verify if the target image matches the file image using the specified model.

        :param target_path: Path to the target image.
        :param file_path: Path to the file image.
        :param model_name: Name of the model to use for face recognition.
        :return: True if the images match, False otherwise.
        """
        result = DeepFace.verify(img1_path=str(target_path), img2_path=str(file_path), model_name=model_name)
        logging.debug(f"Verified for target: {target_path} and file: {file_path}. Result: {result['verified']}")
        return result["verified"]

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