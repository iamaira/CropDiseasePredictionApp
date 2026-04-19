import os
import traceback
from PIL import Image
from flask import Flask, render_template, request
from service.predict import workflow

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    treatment = None
    error = None
    uploaded_image = None

    if request.method == "POST":
        file = request.files.get("file")
        print(f"[INFO] POST received, file={getattr(file, 'filename', None)}", flush=True)

        if file and file.filename != "":
            save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(save_path)
            uploaded_image = "/" + save_path.replace("\\", "/")

            try:
                img = Image.open(save_path).convert("RGB")

                result = workflow(img, file.filename)

                if result is None:
                    prediction = "Error"
                    treatment = "Workflow returned nothing."
                elif isinstance(result, tuple) and len(result) == 2:
                    prediction, treatment = result
                else:
                    prediction = "Error"
                    treatment = f"Unexpected workflow result: {result}"

                print(
                    f"[INFO] workflow result: prediction={prediction!r}, treatment={treatment!r}",
                    flush=True,
                )

            except Exception as e:
                error = f"Prediction failed: {e}"
                print("[ERROR] Prediction failed:", e, flush=True)
                traceback.print_exc()
        else:
            error = "Please upload a valid image file."

    return render_template(
        "index.html",
        prediction=prediction,
        treatment=treatment,
        error=error,
        uploaded_image=uploaded_image,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)