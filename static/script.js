const fileInput = document.getElementById("fileInput");
const previewBox = document.getElementById("previewBox");
const previewImage = document.getElementById("previewImage");
const form = document.getElementById("predictForm");
const loader = document.getElementById("loaderOverlay");
const welcomeScreen = document.getElementById("welcome-screen");

if (fileInput) {
    fileInput.addEventListener("change", function () {
        const file = this.files[0];

        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                previewImage.src = e.target.result;
                previewBox.classList.remove("hidden");
            };
            reader.readAsDataURL(file);
        }
    });
}

if (form) {
    form.addEventListener("submit", function () {
        if (loader) {
            loader.classList.remove("hidden");
        }
    });
}

if (welcomeScreen) {
    welcomeScreen.addEventListener("click", function () {
        welcomeScreen.classList.add("hide");
    });
}