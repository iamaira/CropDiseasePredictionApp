import os

import gradio as gr

from service.predict import workflow

def process_image(image):
    disease_name, remedy = workflow(image)
    return disease_name, remedy


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
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    server_host = "0.0.0.0"
    print(f"[INFO] Starting Gradio app on {server_host}:{port}")
    iface.launch(server_name=server_host, server_port=port, share=False)
