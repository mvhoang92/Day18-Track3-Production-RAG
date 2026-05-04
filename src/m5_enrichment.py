"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json, re
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _get_openai_client():
    """Get OpenAI client if API key is available."""
    if OPENAI_API_KEY:
        from openai import OpenAI

        return OpenAI(api_key=OPENAI_API_KEY)
    return None


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    if not text or not text.strip():
        return ""

    # Try OpenAI first
    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. Chỉ trả về tóm tắt, không giải thích.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    # Fallback: Extractive summarization (take first 2 sentences)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 2:
        return text
    return ". ".join(sentences[:2]) + "."


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    if not text or not text.strip():
        return []

    # Try OpenAI first
    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng, không đánh số.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            content = resp.choices[0].message.content.strip()
            questions = content.split("\n")
            # Clean up: remove numbering, bullets, etc.
            cleaned = []
            for q in questions:
                q = q.strip()
                q = re.sub(r"^[\d.\-\)\*]+[\s]*", "", q)
                if q and len(q) > 5:
                    cleaned.append(q)
            return cleaned[:n_questions]
        except Exception:
            pass

    # Fallback: Generate simple questions based on keywords
    fallback_questions = _generate_fallback_questions(text, n_questions)
    return fallback_questions


def _generate_fallback_questions(text: str, n_questions: int) -> list[str]:
    """Generate simple questions using extractive method."""
    questions = []

    # Extract key phrases (nouns and verbs)
    words = text.split()
    key_phrases = [w for w in words if len(w) > 4][:10]

    if key_phrases:
        questions.append(f"Thông tin về {key_phrases[0]} là gì?")

    # Check for specific patterns
    if any(word in text.lower() for word in ["ngày", "tháng", "năm", "lần"]):
        questions.append("Có bao nhiêu?")
    if any(word in text.lower() for word in ["được", "phép", "quyền"]):
        questions.append("Nhân viên được làm gì?")
    if any(word in text.lower() for word in ["không", "được", "cấm"]):
        questions.append("Điều gì không được phép?")

    return questions[:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    if not text or not text.strip():
        return text

    # Try OpenAI first
    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu duy nhất, không giải thích thêm.",
                    },
                    {
                        "role": "user",
                        "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
                    },
                ],
                max_tokens=80,
            )
            context = resp.choices[0].message.content.strip()
            return f"{context}\n\n{text}"
        except Exception:
            pass

    # Fallback: Simple contextual prefix
    if document_title:
        return f"[Trích từ: {document_title}] {text}"
    return text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    if not text or not text.strip():
        return {}

    # Try OpenAI first
    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON với các trường: topic (chủ đề chính), entities (danh sách tên riêng/tổ chức), category (policy|hr|it|finance|security|other), language (vi|en). Không giải thích, chỉ trả về JSON.',
                    },
                    {"role": "user", "content": text[:500]},  # Limit text length
                ],
                max_tokens=150,
            )
            result = json.loads(resp.choices[0].message.content)
            # Ensure required fields exist
            return {
                "topic": result.get("topic", "unknown"),
                "entities": result.get("entities", []),
                "category": result.get("category", "other"),
                "language": result.get("language", "vi"),
            }
        except (json.JSONDecodeError, Exception):
            pass

    # Fallback: Rule-based metadata extraction
    return _extract_fallback_metadata(text)


def _extract_fallback_metadata(text: str) -> dict:
    """Extract metadata using simple rules."""
    category = "other"
    topic = "unknown"

    # Simple keyword-based category detection
    category_keywords = {
        "hr": ["nhân viên", "phép", "lương", "thử việc", "thâm niên", "chế độ"],
        "policy": ["chính sách", "quy định", "điều lệ", "quyền lợi", "nghĩa vụ"],
        "it": ["mật khẩu", "VPN", "email", "máy tính", "phần mềm", "bảo mật"],
        "finance": ["chi phí", "ngân sách", "thanh toán", "hóa đơn", "tài chính"],
        "security": ["bảo mật", "an toàn", "phòng cháy", "chữa cháy"],
    }

    text_lower = text.lower()
    for cat, keywords in category_keywords.items():
        if any(kw in text_lower for kw in keywords):
            category = cat
            break

    # Extract potential entities (capitalized words)
    entities = re.findall(r"[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯ][a-zàáâãèéêìíòóôõùúăđĩũơư]+(?:\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯ][a-zàáâãèéêìíòóôõùúăđĩũơư]+)*", text)
    entities = list(set(entities))[:5]  # Limit to 5 entities

    # Determine topic from first few words
    words = text.split()
    topic = " ".join(words[:3]) if len(words) >= 3 else text[:50]

    return {
        "topic": topic,
        "entities": entities,
        "category": category,
        "language": "vi",
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    if not chunks:
        return []

    enriched_chunks = []

    for chunk in chunks:
        text = chunk.get("text", "")
        chunk_metadata = chunk.get("metadata", {})

        if not text or not text.strip():
            continue

        # Apply enrichment techniques based on methods
        summary = ""
        hypothesis_questions = []
        enriched_text = text
        auto_metadata = {}

        if "summary" in methods or "full" in methods:
            summary = summarize_chunk(text)

        if "hyqa" in methods or "full" in methods:
            hypothesis_questions = generate_hypothesis_questions(text)

        if "contextual" in methods or "full" in methods:
            source = chunk_metadata.get("source", "")
            enriched_text = contextual_prepend(text, source)

        if "metadata" in methods or "full" in methods:
            auto_metadata = extract_metadata(text)

        # Create EnrichedChunk
        enriched_chunks.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=hypothesis_questions,
                auto_metadata={**chunk_metadata, **auto_metadata},
                method="+".join(methods),
            )
        )

    return enriched_chunks


# ─── Main ────────────────────────────────────────────────


if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")

    # Test enrich_chunks
    print("\n=== Enrich Chunks Demo ===")
    test_chunks = [
        {"text": sample, "metadata": {"source": "Sổ tay VinUni 2024"}},
        {
            "text": "Mật khẩu WiFi công ty phải thay đổi mỗi 90 ngày. Sử dụng WPA3 hoặc WPA2-AES.",
            "metadata": {"source": "Chính sách IT"},
        },
    ]
    enriched = enrich_chunks(test_chunks)
    for e in enriched:
        print(f"\nOriginal: {e.original_text[:60]}...")
        print(f"Enriched: {e.enriched_text[:60]}...")
        print(f"Metadata: {e.auto_metadata}")