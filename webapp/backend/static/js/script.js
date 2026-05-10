document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.querySelector("[data-upload-zone]");
  const fileInput = document.querySelector("#image");

  if (dropzone && fileInput) {
    const updateState = () => {
      const hasFile = fileInput.files && fileInput.files.length > 0;
      dropzone.classList.toggle("has-file", hasFile);
      dropzone.dataset.filename = hasFile ? fileInput.files[0].name : "";
    };

    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("is-dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, () => {
        dropzone.classList.remove("is-dragover");
      });
    });

    fileInput.addEventListener("change", updateState);
    updateState();
  }

  const summary = document.querySelector(".result-highlight");
  if (summary) {
    summary.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
});
