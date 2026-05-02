
const form = document.getElementById("searchForm");
const textSearchForm = document.getElementById("textSearchForm");
const resultsGrid = document.getElementById("resultsGrid");
const emptyState = document.getElementById("emptyState");
const statusText = document.getElementById("statusText");
const dropZone = document.getElementById("dropZone");
const imageInput = document.getElementById("imageInput");
const searchBtn = document.getElementById("searchBtn");
const textSearchBtn = document.getElementById("textSearchBtn");
const keywordInput = document.getElementById("keywordInput");
const topKText = document.getElementById("topKText");
const resultsMeta = document.getElementById("resultsMeta");
const queryPreviewBox = document.getElementById("queryPreviewBox");
const queryPreviewImage = document.getElementById("queryPreviewImage");
const queryFileName = document.getElementById("queryFileName");
const visualPanel = document.getElementById("visualPanel");
const modeTabs = document.querySelectorAll(".mode-tab");

function setStatus(text) {
  statusText.textContent = text;
}

function setLoadingVisual(isLoading) {
  searchBtn.disabled = isLoading;
  searchBtn.textContent = isLoading ? "Searching…" : "Find similar";
}

function setLoadingText(isLoading) {
  textSearchBtn.disabled = isLoading;
  textSearchBtn.textContent = isLoading ? "Searching…" : "Search";
}

function showResultsGrid(show) {
  if (show) {
    resultsGrid.classList.remove("hidden");
    emptyState.classList.add("hidden");
  } else {
    resultsGrid.classList.add("hidden");
    emptyState.classList.remove("hidden");
  }
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
  const mode = payload.mode || "visual";
  const metricHint =
    mode === "keyword"
      ? "keyword match"
      : payload.results?.[0]?.metric === "cosine"
        ? "cosine"
        : "similarity";

  if (!results.length) {
    setStatus("No results found. Try another keyword or image.");
    showResultsGrid(false);
    return;
  }

  setStatus(`Showing ${results.length} result(s).`);
  const parts = [
    mode === "keyword" ? `Keyword: “${payload.query || ""}”` : "Visual match",
    `max ${payload.top_k}`,
    `${payload.count} returned`,
    metricHint,
  ];
  resultsMeta.textContent = parts.filter(Boolean).join(" · ");

  for (const item of results) {
    const card = document.createElement("article");
    card.className = "card";
    const scoreLabel =
      mode === "keyword"
        ? `match ${Number(item.score).toFixed(2)}`
        : `score ${Number(item.score).toFixed(4)}`;
    const image = item.image_url
      ? `<img src="${item.image_url}" alt="" loading="lazy" decoding="async" />`
      : `<div class="card-placeholder">No preview</div>`;
    card.innerHTML = `
      ${image}
      <div class="card-body">
        <p class="card-title">#${item.rank} · ${scoreLabel}</p>
        <div class="muted">${item.category || "—"}</div>
        <div class="card-filename">${item.filename || item.image_id || ""}</div>
      </div>
    `;
    resultsGrid.appendChild(card);
  }
  showResultsGrid(true);
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
  setStatus("Image ready. Run visual search.");
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
  setStatus("Image ready. Run visual search.");
});

textSearchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const q = (keywordInput.value || "").trim();
  if (!q) {
    setStatus("Enter a search term (e.g. blouse).");
    keywordInput.focus();
    return;
  }
  let topK = parseInt(topKText.value, 10);
  if (Number.isNaN(topK) || topK < 1) topK = 24;
  setLoadingText(true);
  setStatus("Searching catalog…");
  resultsGrid.innerHTML = "";
  resultsMeta.textContent = "";

  try {
    const response = await fetch("/search_text", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ q, top_k: topK }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Search failed.");
    }
    renderResults(payload);
  } catch (error) {
    setStatus(error.message || "Search failed.");
    showResultsGrid(false);
  } finally {
    setLoadingText(false);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = imageInput.files?.[0];
  if (!file) {
    setStatus("Choose or drop an image first.");
    return;
  }

  const formData = new FormData(form);
  setLoadingVisual(true);
  setStatus("Finding visually similar products…");
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
    setStatus(error.message || "Search failed.");
    showResultsGrid(false);
  } finally {
    setLoadingVisual(false);
  }
});

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const mode = tab.dataset.mode;
    modeTabs.forEach((t) => {
      const active = t.dataset.mode === mode;
      t.classList.toggle("is-active", active);
      t.setAttribute("aria-selected", active ? "true" : "false");
    });
    if (mode === "keyword") {
      visualPanel.classList.add("hidden");
      keywordInput.focus();
    } else {
      visualPanel.classList.remove("hidden");
    }
  });
});

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

