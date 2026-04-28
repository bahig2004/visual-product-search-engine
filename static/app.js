const form = document.getElementById("searchForm");
const resultsGrid = document.getElementById("resultsGrid");
const statusText = document.getElementById("statusText");
const dropZone = document.getElementById("dropZone");
const imageInput = document.getElementById("imageInput");
const searchBtn = document.getElementById("searchBtn");
const resultsMeta = document.getElementById("resultsMeta");
const queryPreviewBox = document.getElementById("queryPreviewBox");
const queryPreviewImage = document.getElementById("queryPreviewImage");
const queryFileName = document.getElementById("queryFileName");

function setStatus(text) {
  statusText.textContent = text;
}

function setLoadingState(isLoading) {
  searchBtn.disabled = isLoading;
  searchBtn.textContent = isLoading ? "Searching..." : "Search Similar Products";
}

function updatePreview(file) {
  if (!file) {
    queryPreviewBox.classList.add("hidden");
    queryPreviewImage.removeAttribute("src");
    queryFileName.textContent = "";
    return;
  }
  const objectUrl = URL.createObjectURL(file);
  queryPreviewImage.src = objectUrl;
  queryFileName.textContent = file.name;
  queryPreviewBox.classList.remove("hidden");
}

function setFileOnInput(file) {
  const dataTransfer = new DataTransfer();
  dataTransfer.items.add(file);
  imageInput.files = dataTransfer.files;
  updatePreview(file);
}

function renderResults(payload) {
  resultsGrid.innerHTML = "";
  resultsMeta.textContent = "";
  const results = payload.results || [];
  if (!results.length) {
    setStatus("No results found.");
    return;
  }

  setStatus(`Retrieved ${results.length} result(s).`);
  resultsMeta.textContent = `Top K: ${payload.top_k} | Count: ${payload.count}`;
  for (const item of results) {
    const card = document.createElement("article");
    card.className = "card";
    const image = item.image_url
      ? `<img src="${item.image_url}" alt="Result ${item.rank}" />`
      : `<img alt="No preview available" />`;
    card.innerHTML = `
      ${image}
      <div class="card-body">
        <p class="card-title">#${item.rank} | score ${Number(item.score).toFixed(4)}</p>
        <div class="muted">category: ${item.category || "N/A"}</div>
        <div>${item.filename || item.image_id || "unknown"}</div>
      </div>
    `;
    resultsGrid.appendChild(card);
  }
}

function isValidImage(file) {
  return Boolean(file && file.type && file.type.startsWith("image/"));
}

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove("dragover");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (!isValidImage(file)) {
    setStatus("Please drop a valid image file.");
    return;
  }
  setFileOnInput(file);
  setStatus("Image ready. Click search.");
});

dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    imageInput.click();
  }
});

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (!file) {
    updatePreview(null);
    return;
  }
  if (!isValidImage(file)) {
    setStatus("Please choose a valid image file.");
    imageInput.value = "";
    updatePreview(null);
    return;
  }
  updatePreview(file);
  setStatus("Image ready. Click search.");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = imageInput.files?.[0];
  if (!file) {
    setStatus("Please choose or drop an image first.");
    return;
  }

  const formData = new FormData(form);
  setLoadingState(true);
  setStatus("Searching for visually similar products...");
  resultsGrid.innerHTML = "";
  resultsMeta.textContent = "";

  try {
    const response = await fetch("/search", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Search failed.");
    }
    renderResults(payload);
  } catch (error) {
    setStatus(error.message);
  } finally {
    setLoadingState(false);
  }
});
