import logging
import tempfile
from http import HTTPStatus
import tensorflow as tf
from flask import Flask, jsonify, request
from pydantic import ValidationError

from src.routes.validators.FindMatchesRequestValidator import FindMatchesRequestValidator
from src.services.FaceRecognitionService import FaceRecognitionService

logging.basicConfig(level=logging.DEBUG)




app = Flask(__name__)


# app.register_blueprint(matcher)

@app.route('/', methods=['POST'])
def find_matches():
    try:
        data = FindMatchesRequestValidator(**request.json)
    except ValidationError as e:
        return jsonify({"error": list(e.errors())}), HTTPStatus.BAD_REQUEST

    output_path = data.output_path or tempfile.NamedTemporaryFile(suffix='.json', dir='tmp').name

    matches = FaceRecognitionService.process_and_save_results(selfie_paths=data.selfie_paths,
        folder_path=data.folder_path, output_path=output_path, model_name=data.model_name)

    return jsonify({"output_path": output_path, "matches": matches}), HTTPStatus.OK


@app.route('/gpu/avalilable', methods=['GET'])
def gpu_availability():
    gpu_available = tf.config.list_physical_devices('GPU')
    return jsonify({"gpu_available": gpu_available}), HTTPStatus.OK


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
