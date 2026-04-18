const fileInput = document.getElementById("fileInput");
const previewBox = document.getElementById("previewBox");
const previewImage = document.getElementById("previewImage");
const form = document.getElementById("predictForm");
const loader = document.getElementById("loaderOverlay");
const welcomeScreen = document.getElementById("welcome-screen");

// Image preview
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

// Show loader on submit
if (form) {
    form.addEventListener("submit", function () {
        if (loader) {
            loader.classList.remove("hidden");
        }
    });
}

// Welcome screen logic
if (welcomeScreen) {
    const alreadyStarted = sessionStorage.getItem("welcomeShown");

    // Agar pehle click ho chuka hai, to welcome screen hide rakho
    if (alreadyStarted === "true") {
        welcomeScreen.classList.add("hide");
    } else {
        // First time click pe hide karo aur session me save karo
        welcomeScreen.addEventListener("click", function () {
            welcomeScreen.classList.add("hide");
            sessionStorage.setItem("welcomeShown", "true");
        });
    }
}