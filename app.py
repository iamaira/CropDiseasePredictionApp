import os
import sys
import traceback
from fastapi import FastAPI
import gradio as gr
import uvicorn

# Create app first to respond to health checks
app = FastAPI()

# Health check endpoint for deployment
@app.get("/health")
def health_check():
    return {"status": "ok"}

try:
    from service.predict import workflow
    WORKFLOW_READY = True
except Exception as e:
    print(f"[ERROR] Failed to load workflow: {e}")
    traceback.print_exc()
    WORKFLOW_READY = False

def process_image(image):
    if not WORKFLOW_READY:
        return "Error", "The prediction service failed to load. Please check the server logs."
    try:
        disease_name, remedy = workflow(image)
        return disease_name, remedy
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        traceback.print_exc()
        return "Error", f"Prediction failed: {str(e)}"

# Create the Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(
        image_mode="RGB",
        sources="upload",
        label="Upload Plant Disease Image",
        show_download_button=True,
        type="pil",
    ),
    outputs=[
        gr.Textbox(label="Prediction", placeholder="Disease Prediction"),
        gr.Markdown(label="Remedy"),
    ],
    title="Classify Plant Diseases and Get Remedies",
    allow_flagging="never",
)

# Mount Gradio app with FastAPI
app = gr.mount_gradio_app(app, iface, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    server_host = "0.0.0.0"
    print(f"[INFO] Starting app on {server_host}:{port}")
    print(f"[INFO] Workflow ready: {WORKFLOW_READY}")
    uvicorn.run(app, host=server_host, port=port, timeout_keep_alive=60)
