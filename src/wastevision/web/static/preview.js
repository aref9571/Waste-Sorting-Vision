const input = document.getElementById("image");
const preview = document.getElementById("preview");

input.addEventListener("change", () => {
  if (preview.src) {
    URL.revokeObjectURL(preview.src);
  }
  const file = input.files[0];
  if (file) {
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
  } else {
    preview.removeAttribute("src");
    preview.hidden = true;
  }
});
