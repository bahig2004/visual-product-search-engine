const textSearchForm = document.getElementById("textSearchForm");
const resultsGrid = document.getElementById("resultsGrid");
const emptyState = document.getElementById("emptyState");
const statusText = document.getElementById("statusText");
const textSearchBtn = document.getElementById("textSearchBtn");
const keywordInput = document.getElementById("keywordInput");
const topKText = document.getElementById("topKText");
const resultsMeta = document.getElementById("resultsMeta");
const generatedAnswer = document.getElementById("generatedAnswer");
const exampleChips = document.querySelectorAll(".example-chip");
const dropHint = document.getElementById("dropHint");

let catalogFeatureCache = null;

function setStatus(text) {
  statusText.textContent = text;
}

function setLoadingText(isLoading) {
  textSearchBtn.disabled = isLoading;
  textSearchBtn.textContent = isLoading ? "Searching..." : "Search";
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

function resetResultsState() {
  resultsGrid.innerHTML = "";
  resultsMeta.textContent = "";
  generatedAnswer.textContent = "";
  generatedAnswer.classList.add("hidden");
}

function toPositiveTopK() {
  let topK = parseInt(topKText.value, 10);
  if (Number.isNaN(topK) || topK < 1) topK = 10;
  return topK;
}

function renderResults(payload) {
  resultsGrid.innerHTML = "";
  resultsMeta.textContent = `mode: ${payload.mode || "rag"} | top_k: ${payload.top_k || 0} | count: ${payload.count || 0}`;

  generatedAnswer.textContent = payload.answer || "";
  generatedAnswer.classList.toggle("hidden", !payload.answer);

  const results = payload.results || [];
  if (!results.length) {
    setStatus("No matching context found.");
    showResultsGrid(false);
    return;
  }

  setStatus(`Showing ${results.length} result(s).`);
  for (const item of results) {
    const card = document.createElement("article");
    card.className = "card";
    const image = item.image_url
      ? `<img src="${item.image_url}" alt="" loading="lazy" decoding="async" />`
      : `<div class="card-placeholder">No preview</div>`;
    card.innerHTML = `
      ${image}
      <div class="card-body">
        <p class="card-title">#${item.rank} | score ${Number(item.score).toFixed(4)}</p>
        <div class="muted">${item.category || "unknown category"}</div>
        <div class="muted">${item.filename || ""}</div>
      </div>
    `;
    resultsGrid.appendChild(card);
  }
  showResultsGrid(true);
}

function cosineSimilarity(a, b) {
  if (a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (!normA || !normB) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function buildImageFeatureVector(image) {
  const size = 24;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    throw new Error("Canvas context is unavailable.");
  }
  ctx.drawImage(image, 0, 0, size, size);
  const data = ctx.getImageData(0, 0, size, size).data;
  let r = 0;
  let g = 0;
  let b = 0;
  let r2 = 0;
  let g2 = 0;
  let b2 = 0;
  const totalPixels = size * size;
  for (let idx = 0; idx < data.length; idx += 4) {
    const rv = data[idx];
    const gv = data[idx + 1];
    const bv = data[idx + 2];
    r += rv;
    g += gv;
    b += bv;
    r2 += rv * rv;
    g2 += gv * gv;
    b2 += bv * bv;
  }
  const meanR = r / totalPixels;
  const meanG = g / totalPixels;
  const meanB = b / totalPixels;
  const stdR = Math.sqrt(Math.max(0, r2 / totalPixels - meanR * meanR));
  const stdG = Math.sqrt(Math.max(0, g2 / totalPixels - meanG * meanG));
  const stdB = Math.sqrt(Math.max(0, b2 / totalPixels - meanB * meanB));
  return [meanR, meanG, meanB, stdR, stdG, stdB];
}

async function loadImageFromSource(source) {
  return await new Promise((resolve, reject) => {
    const image = new Image();
    if (typeof source === "string") {
      image.crossOrigin = "anonymous";
      image.src = source;
    } else {
      image.src = URL.createObjectURL(source);
    }
    image.onload = () => {
      resolve(image);
      if (typeof source !== "string") {
        URL.revokeObjectURL(image.src);
      }
    };
    image.onerror = () => reject(new Error("Unable to read image data."));
  });
}

async function ensureCatalogFeatures() {
  if (catalogFeatureCache) return catalogFeatureCache;

  const response = await fetch("/catalog-images", { headers: { Accept: "application/json" } });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Unable to load catalog images.");
  }

  const features = [];
  for (const item of payload.items || []) {
    try {
      const image = await loadImageFromSource(item.image_url);
      const vector = buildImageFeatureVector(image);
      features.push({ ...item, vector });
    } catch (_error) {
      // Skip images that cannot be read in browser.
    }
  }
  catalogFeatureCache = features;
  return features;
}

async function runImageSearch(file) {
  const topK = toPositiveTopK();
  setLoadingText(true);
  setStatus("Searching by dropped image...");
  resetResultsState();

  try {
    if (!file || !file.type.startsWith("image/")) {
      throw new Error("Drop a valid image file.");
    }

    const uploadedImage = await loadImageFromSource(file);
    const uploadedVector = buildImageFeatureVector(uploadedImage);
    const catalog = await ensureCatalogFeatures();

    const ranked = catalog
      .map((item) => ({ ...item, score: cosineSimilarity(uploadedVector, item.vector) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);

    renderResults({
      mode: "image",
      top_k: topK,
      count: ranked.length,
      answer: ranked.length
        ? "Results ranked by visual similarity from dropped image."
        : "No matching products found for this image.",
      results: ranked.map((item, index) => ({
        rank: index + 1,
        score: item.score,
        category: item.category,
        filename: item.filename,
        image_path: item.image_path,
        image_url: item.image_url,
      })),
    });
  } catch (error) {
    setStatus(error.message || "Image search failed.");
    showResultsGrid(false);
  } finally {
    setLoadingText(false);
  }
}

textSearchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const q = (keywordInput.value || "").trim();
  if (!q) {
    setStatus("Enter a search query.");
    keywordInput.focus();
    return;
  }
  const topK = toPositiveTopK();

  setLoadingText(true);
  setStatus("Searching catalog...");
  resetResultsState();

  try {
    const response = await fetch("/search", {
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

["dragenter", "dragover"].forEach((eventName) => {
  textSearchForm.addEventListener(eventName, (event) => {
    event.preventDefault();
    textSearchForm.classList.add("drop-active");
    if (dropHint) {
      dropHint.textContent = "Release image to search by picture.";
    }
  });
});

["dragleave", "dragend"].forEach((eventName) => {
  textSearchForm.addEventListener(eventName, () => {
    textSearchForm.classList.remove("drop-active");
    if (dropHint) {
      dropHint.textContent = "Tip: Drag and drop an image on the search bar to find visually similar products.";
    }
  });
});

textSearchForm.addEventListener("drop", async (event) => {
  event.preventDefault();
  textSearchForm.classList.remove("drop-active");
  if (dropHint) {
    dropHint.textContent = "Tip: Drag and drop an image on the search bar to find visually similar products.";
  }
  const file = event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files[0] : null;
  await runImageSearch(file);
});

exampleChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    keywordInput.value = chip.dataset.query || "";
    keywordInput.focus();
  });
});
