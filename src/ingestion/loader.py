"""
Ingestion pipeline for German labor law documents.

Uses the official XML API from gesetze-im-internet.de — much more reliable
than HTML parsing. Each law has a direct XML download URL.

Source: https://www.gesetze-im-internet.de/hinweise.html
XML structure defined in: https://www.gesetze-im-internet.de/dtd/1.01/gii-norm.dtd
"""

import logging
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Direct XML download URLs — official, stable, public domain
GESETZE_XML_SOURCES = {
    "KSchG":  "https://www.gesetze-im-internet.de/kschg/xml.zip",
    "AGG":    "https://www.gesetze-im-internet.de/agg/xml.zip",
    "ArbZG":  "https://www.gesetze-im-internet.de/arbzg/xml.zip",
    "MuSchG": "https://www.gesetze-im-internet.de/muschg_2018/xml.zip",
    "EntgFG": "https://www.gesetze-im-internet.de/entgfg/xml.zip",
    "BGB":    "https://www.gesetze-im-internet.de/bgb/xml.zip",
}

# For BGB we only want employment-related sections (§§ 611-630)
BGB_RELEVANT_PARAGRAPHS = {f"§ {i}" for i in range(611, 631)}


@dataclass
class LegalDocument:
    source: str
    paragraph: str
    title: str
    text: str
    url: str
    doc_type: str = "Gesetz"


def clean_text(text: str) -> str:
    """Normalize whitespace in German legal text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_xml_zip(name: str, zip_bytes: bytes, url: str) -> list[LegalDocument]:
    """Parse XML zip archive from gesetze-im-internet.de."""
    import zipfile
    import io

    documents = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        xml_files = [f for f in zf.namelist() if f.endswith(".xml")]

        for xml_file in xml_files:
            with zf.open(xml_file) as f:
                content = f.read()

            soup = BeautifulSoup(content, "xml")

            # Each <norm> tag is one paragraph
            for norm in soup.find_all("norm"):
                # Paragraph number e.g. "§ 1"
                enbez = norm.find("enbez")
                # Paragraph title e.g. "Sozial ungerechtfertigte Kündigungen"
                titel = norm.find("titel")
                # Content
                text_node = norm.find("Content") or norm.find("content")

                if not text_node:
                    continue

                paragraph = enbez.get_text(strip=True) if enbez else ""
                title = titel.get_text(strip=True) if titel else ""
                text = clean_text(text_node.get_text())

                if len(text) < 30:
                    continue

                # For BGB: only keep employment law sections
                if name == "BGB" and paragraph not in BGB_RELEVANT_PARAGRAPHS:
                    continue

                documents.append(LegalDocument(
                    source=name,
                    paragraph=paragraph,
                    title=title,
                    text=text,
                    url=url.replace("xml.zip", ""),
                ))

    logger.info(f"✓ {name}: {len(documents)} paragraphs loaded")
    return documents


def fetch_law_xml(name: str, url: str) -> list[LegalDocument]:
    """Download and parse a single law via XML zip."""
    try:
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()
        return parse_xml_zip(name, response.content, url)
    except Exception as e:
        logger.error(f"✗ Failed to load {name} from {url}: {e}")
        return []


def load_all_laws() -> list[LegalDocument]:
    """Load all configured German labor law sources via XML API."""
    all_docs = []
    for name, url in GESETZE_XML_SOURCES.items():
        docs = fetch_law_xml(name, url)
        all_docs.extend(docs)
    logger.info(f"Total: {len(all_docs)} paragraphs across {len(GESETZE_XML_SOURCES)} laws")
    return all_docs


def save_raw(documents: list[LegalDocument], output_dir: str = "data/raw") -> None:
    """Save parsed documents as JSON for reproducibility."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for doc in documents:
        safe_para = doc.paragraph.replace(" ", "_").replace("§", "para").replace("/", "-")
        file_name = f"{doc.source}_{safe_para}.json"
        with open(output_path / file_name, "w", encoding="utf-8") as f:
            json.dump(asdict(doc), f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(documents)} documents to {output_dir}/")


if __name__ == "__main__":
    docs = load_all_laws()
    save_raw(docs)
