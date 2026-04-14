import traceback
from PIL import Image
import gradio as gr
from fastapi import FastAPI
from service.predict import workflow

import os
from flask import Flask, render_template, request
from PIL import Image

# apna real prediction function import karo
# example:
# from service.predict import workflow
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TREATMENTS = {
    "Apple___Apple_scab": "Use fungicide and remove infected leaves.",
    "Apple___Black_rot": "Prune infected branches and apply proper fungicide.",
    "Tomato___Late_blight": "Use copper-based fungicide and avoid overhead watering.",
    "Healthy": "No treatment needed. Plant is healthy."
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    treatment = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(save_path)

            img = Image.open(save_path).convert("RGB")
            prediction, treatment = workflow(img)
    return render_template(
        "index.html",
        prediction=prediction,
        treatment=treatment
    )
    return render_template(
        "index.html",
        prediction=prediction,
        treatment=treatment,
        image_path=image_path,
        error=error
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)





workflow = None
WORKFLOW_READY = False

# FastAPI instance
app = FastAPI()

def get_workflow():
    global workflow, WORKFLOW_READY
    if workflow is None:
        try:
            from service.predict import workflow as wf
            workflow = wf
            WORKFLOW_READY = True
            print("[INFO] Workflow loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load workflow: {e}")
            traceback.print_exc()
            WORKFLOW_READY = False
    return workflow


def process_image(image):
    wf = get_workflow()
    if not WORKFLOW_READY or wf is None:
        return "Error", "The prediction service failed to load. Please check the server logs."
    try:
        disease_name, remedy = wf(image)
        return disease_name, remedy
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        traceback.print_exc()
        return "Error", f"Prediction failed: {str(e)}"

# Create the Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Remedy"),
    ],
    title="Classify Plant Diseases and Get Remedies",
    allow_flagging="never",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    server_host = "0.0.0.0"
    print(f"[INFO] Starting Gradio app on {server_host}:{port}")
    print(f"[INFO] Workflow ready: {WORKFLOW_READY}")
    iface.launch(
        server_name=server_host,
        server_port=port,
        share=False,
        inbrowser=False,
        prevent_thread_lock=True,
        show_error=True,
        show_api=False,
    )