import os
import traceback
import gradio as gr

workflow = None
WORKFLOW_READY = False


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
    inputs=gr.Image(
        image_mode="RGB",
        type="pil",
    ),
    outputs=[
        gr.Textbox(label="Prediction", placeholder="Disease Prediction"),
        gr.Markdown(label="Remedy"),
    ],
    title="Classify Plant Diseases and Get Remedies",
    flagging_mode="never",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    server_host = "0.0.0.0"
    print(f"[INFO] Starting Gradio app on {server_host}:{port}")
    print(f"[INFO] Workflow ready: {WORKFLOW_READY}")
    iface.launch(
        server_name=server_host,
        server_port=port,
        share=True,
        inbrowser=False,
        prevent_thread_lock=True,
        show_error=True,
        show_api=False,
        api_name=False,
        root_path="",
        auth=None,
        enable_queue=False,
    )
