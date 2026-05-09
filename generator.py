from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


COLOR_WORDS = {
    "white",
    "black",
    "blue",
    "red",
    "green",
    "yellow",
    "pink",
    "purple",
    "orange",
    "brown",
    "gray",
    "grey",
    "beige",
    "navy",
    "cream",
    "gold",
    "silver",
}

COLOR_RGB_CENTERS = {
    "red": (220, 50, 47),
    "orange": (230, 126, 34),
    "yellow": (241, 196, 15),
    "green": (46, 204, 113),
    "blue": (52, 152, 219),
    "purple": (155, 89, 182),
    "pink": (231, 76, 160),
    "brown": (139, 90, 43),
    "beige": (220, 198, 160),
    "gray": (130, 130, 130),
}


@dataclass
class RagDocument:
    doc_id: str
    text: str
    metadata: dict[str, Any]


class RagGenerator:
    """Build retrieval documents from local product images."""

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    @staticmethod
    def _nearest_color_name(r: int, g: int, b: int) -> str:
        best_name = "gray"
        best_dist = float("inf")
        for name, (cr, cg, cb) in COLOR_RGB_CENTERS.items():
            dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
            if dist < best_dist:
                best_dist = dist
                best_name = name
        return best_name

    def _extract_colors_from_image(self, image_path: Path) -> list[str]:
        # Use center crop to reduce background influence and classify dominant tones.
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
            left = int(width * 0.2)
            top = int(height * 0.2)
            right = max(left + 1, int(width * 0.8))
            bottom = max(top + 1, int(height * 0.8))
            cropped = rgb.crop((left, top, right, bottom)).resize((64, 64))
            pixels = list(cropped.getdata())

        total = max(1, len(pixels))
        black_count = 0
        white_count = 0
        color_bins: dict[str, int] = {}

        for r, g, b in pixels:
            max_c = max(r, g, b)
            min_c = min(r, g, b)
            delta = max_c - min_c
            v = max_c / 255.0
            s = 0.0 if max_c == 0 else delta / max_c

            if v < 0.22:
                black_count += 1
                continue
            if v > 0.82 and s < 0.2:
                white_count += 1
                continue
            if s < 0.12:
                color_bins["gray"] = color_bins.get("gray", 0) + 1
                continue
            name = self._nearest_color_name(r, g, b)
            color_bins[name] = color_bins.get(name, 0) + 1

        detected: list[str] = []
        if (black_count / total) >= 0.12:
            detected.append("black")
        if (white_count / total) >= 0.2:
            detected.append("white")

        sorted_bins = sorted(color_bins.items(), key=lambda item: item[1], reverse=True)
        for name, count in sorted_bins[:2]:
            if count / total >= 0.12:
                detected.append(name)

        # De-duplicate while preserving order.
        unique: list[str] = []
        for name in detected:
            if name not in unique:
                unique.append(name)
        return unique

    def build_from_images_folder(self, images_dir: Path) -> list[RagDocument]:
        if not images_dir.exists():
            raise FileNotFoundError(f"Missing images folder: {images_dir}")

        docs: list[RagDocument] = []
        for image_path in sorted(images_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in self.IMAGE_EXTS:
                continue

            category = image_path.parent.name
            name_without_ext = image_path.stem.replace("_", " ").replace("-", " ")
            name_tokens = {tok.lower() for tok in name_without_ext.split()}
            text_colors = sorted(name_tokens.intersection(COLOR_WORDS))
            image_colors = self._extract_colors_from_image(image_path)
            colors = sorted({*text_colors, *image_colors})
            rel = image_path.relative_to(images_dir.parent).as_posix()
            doc_text = (
                f"Product category {category}. File name {image_path.name}. "
                f"Search terms {name_without_ext}."
            )
            docs.append(
                RagDocument(
                    doc_id=f"doc_{len(docs) + 1}",
                    text=doc_text,
                    metadata={
                        "category": category,
                        "filename": image_path.name,
                        "image_path": rel,
                        "colors": colors,
                        "image_colors": image_colors,
                    },
                )
            )

        if not docs:
            raise RuntimeError("No images found to build RAG documents.")
        return docs
