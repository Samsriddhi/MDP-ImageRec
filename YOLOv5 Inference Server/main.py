# import time
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from model import *

# app = Flask(__name__)
# CORS(app)
# model = load_model()
# from model import load_opencv_templates
# load_opencv_templates()

# # model = None
# @app.route('/status', methods=['GET'])
# def status():
#     """
#     This is a health check endpoint to check if the server is running
#     :return: a json object with a key "result" and value "ok"
#     """
#     return jsonify({"result": "ok"})

# @app.route('/image', methods=['POST'])
# def image_predict():
#     """
#     This is the main endpoint for the image prediction algorithm
#     :return: a json object with a key "result" and value a dictionary with keys "obstacle_id" and "image_id"
#     """
#     file = request.files['file']
#     filename = file.filename
#     file.save(os.path.join('uploads', filename))
#     # filename format: "<timestamp>_<obstacle_id>_<signal>.jpeg"
#     constituents = file.filename.split("_")
#     obstacle_id = constituents[1]

#     ## Week 8 ## 
#     #signal = constituents[2].strip(".jpg")
#     #image_id = predict_image(filename, model, signal)

#     ## Week 9 ## 
#     # We don't need to pass in the signal anymore
#     image_id = predict_image_week_9(filename,model)

#     # Return the obstacle_id and image_id
#     result = {
#         "obstacle_id": obstacle_id,
#         "image_id": image_id
#     }
#     return jsonify(result)

# @app.route('/stitch', methods=['GET'])
# def stitch():
#     """
#     This is the main endpoint for the stitching command. Stitches the images using two different functions, in effect creating two stitches, just for redundancy purposes
#     """
#     img = stitch_image()
#     img.show()
#     img2 = stitch_image_own()
#     img2.show()
#     return jsonify({"result": "ok"})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)



import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# import only what we use from model1.py
from model3 import (
    load_model,
    load_opencv_templates,
    predict_image_yolo,
    predict_image_opencv,
    stitch_image,
    stitch_image_own,
)

app = Flask(__name__)
CORS(app)

# ensure uploads dir exists
os.makedirs("uploads", exist_ok=True)

# init once at startup
model = load_model()
load_opencv_templates()


def _extract_obstacle_id(filename: str) -> str:
    """
    expected filename format: "<timestamp>_<obstacle_id>_<...>.jpg"
    if format is unexpected, return "NA" instead of crashing.
    """
    parts = filename.split("_")
    return parts[1] if len(parts) >= 2 else "NA"


@app.route('/status', methods=['GET'])
def status():
    return jsonify({"result": "okk"})


@app.route('/imageyolo', methods=['POST'])
def image_yolo():
    """
    YOLO-only prediction. Fast path; no OpenCV fallback.
    Returns: {"obstacle_id": "...", "image_id": "..."}
    """
    if 'file' not in request.files:
        return jsonify({"error": "missing file"}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "empty filename"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join('uploads', filename)
    file.save(save_path)

    obstacle_id = _extract_obstacle_id(filename)
    image_id = predict_image_yolo(filename, model)

    return jsonify({"obstacle_id": obstacle_id, "image_id": image_id})


@app.route('/imageopencv', methods=['POST'])
def image_opencv():
    """
    OpenCV-only prediction. Thorough template matching.
    Returns: {"obstacle_id": "...", "image_id": "..."}
    """
    if 'file' not in request.files:
        return jsonify({"error": "missing file"}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "empty filename"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join('uploads', filename)
    file.save(save_path)

    obstacle_id = _extract_obstacle_id(filename)
    image_id = predict_image_opencv(filename)

    return jsonify({"obstacle_id": obstacle_id, "image_id": image_id})


@app.route('/stitch', methods=['GET'])
def stitch():
    """
    Stitches images from YOLO and own_results into two separate strips.
    """
    img1 = stitch_image()
    img2 = stitch_image_own()
    # no GUI .show() in server context
    return jsonify({"result": "ok"})


# (optional) legacy notice if someone still calls /image
@app.route('/image', methods=['POST'])
def legacy_image():
    return jsonify({"error": "endpoint moved. use /imageyolo or /imageopencv"}), 410


if __name__ == '__main__':
    # debug=True is handy; flip to False in production
    app.run(host='0.0.0.0', port=5000, debug=False)
