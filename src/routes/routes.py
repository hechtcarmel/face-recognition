from http import HTTPStatus

from flask import Blueprint, request, jsonify
# import torch as tc
import tempfile
import tensorflow as tf
from pydantic import ValidationError

from src.routes.validators.FindMatchesRequestValidator import FindMatchesRequestValidator
from src.services.FaceRecognitionService import FaceRecognitionService

matcher = Blueprint('match', __name__)

@matcher.route('/matches', methods=['POST'])
def find_matches():
    try:
        data = FindMatchesRequestValidator(**request.json)
    except ValidationError as e:
        return jsonify({"error": str(e.errors())}), HTTPStatus.BAD_REQUEST

    output_path = data.output_path or tempfile.NamedTemporaryFile(suffix='.json', dir='tmp').name

    FaceRecognitionService.process_and_save_results(
        selfie_paths=data.selfie_paths, folder_path=data.folder_path, output_path=output_path
    )


    return jsonify({"message": "Processing completed", "output_path": output_path}), HTTPStatus.OK

@matcher.route('/gpu/avalilable', methods=['GET'])
def gpu_availability():
    # Use pytorch to check if GPU is available
    gpu_available = tf.test.is_gpu_available(cude_only=True)#tc.cuda.is_available()
    return jsonify({"gpu_available": gpu_available}), HTTPStatus.OK