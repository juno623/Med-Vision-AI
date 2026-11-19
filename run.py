import os

os.environ.setdefault("USE_TF", "0")                 # we only use PyTorch; avoids the Keras 3 error
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from app import create_app

app = create_app()

if __name__ == "__main__":
    # debug=False on purpose: the debug reloader would load the OCR model twice
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)),
            debug=False, threaded=True)
