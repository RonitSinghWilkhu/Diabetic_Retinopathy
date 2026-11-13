
import logging
import re
from pathlib import Path
from typing import Any

from app.rag.generation import RAGAnswerGenerator
from app.services.pdf_generator import generate_patient_report


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# EVIDENCE NORMALIZATION
# ---------------------------------------------------------

def _normalise_evidence(evidence: list[dict]) -> list[dict]:
    """
    Preserve actual retrieved evidence metadata.
    Remove duplicate passages and retain source page numbers.
    """

    normalised = []
    seen = set()

    for item in evidence:
        if not isinstance(item, dict):
            continue

        text = item.get("text", "")
        source = item.get("source", "Unknown source")
        pages = item.get("pages", [])

        if not isinstance(text, str) or not text.strip():
            continue

        if not isinstance(pages, (list, tuple)):
            pages = [pages] if pages else []

        key = (
            str(source),
            tuple(str(page) for page in pages),
            text.strip(),
        )

        if key in seen:
            continue

        seen.add(key)

        normalised.append({
            "evidence_id": item.get("evidence_id"),
            "source": str(source),
            "pages": list(pages),
            "text": text.strip(),
        })

    return normalised


# ---------------------------------------------------------
# RAG SECTION GENERATION
# ---------------------------------------------------------

def _generate_section(
    generator: RAGAnswerGenerator,
    question: str,
    image_context: dict[str, Any] | None = None,
) -> dict:
    """
    Generate one evidence-grounded report section.

    If generation fails, other report sections can continue.
    """

    try:
        result = generator.answer(
            question,
            image_context=image_context,
        )

        return {
            "answer": result.answer or "",
            "evidence": result.evidence or [],
            "sources": result.sources or [],
            "error": None,
        }

    except Exception:
        logger.exception(
            "RAG report section generation failed: %s",
            question,
        )

        return {
            "answer": "",
            "evidence": [],
            "sources": [],
            "error": (
                "Information could not be retrieved "
                "from the clinical knowledge base."
            ),
        }


def _remove_leading_heading(
    content: str,
    heading_names: tuple[str, ...],
) -> str:
    """
    Prevent duplicate headings when the PDF generator
    already inserts the section heading.
    """

    if not content:
        return ""

    lines = content.strip().splitlines()

    while lines and not lines[0].strip():
        lines.pop(0)

    if lines:
        first_line = re.sub(
            r"^#{1,6}\s*",
            "",
            lines[0].strip(),
        ).strip()

        first_line = first_line.strip("*").strip()

        for heading in heading_names:
            if first_line.casefold() == heading.casefold():
                lines.pop(0)
                break

    return "\n".join(lines).strip()


def _section_answer(
    result: dict,
    fallback: str,
    heading_names: tuple[str, ...] = (),
) -> str:
    """
    Return generated content without duplicate section headings.
    """

    answer = result.get("answer", "")

    if answer:
        return _remove_leading_heading(
            answer,
            heading_names,
        )

    return result.get("error") or fallback


# ---------------------------------------------------------
# REPORT DATA ASSEMBLY
# ---------------------------------------------------------

def build_report_data(
    analysis: Any,
    original_image_path: str | Path,
    gradcam_paths: dict[str, Path] | None = None,
    patient_name: str | None = None,
    generator: RAGAnswerGenerator | None = None,
) -> dict:
    """
    Assemble prediction, Grad-CAM and RAG information
    into the structured data expected by the PDF generator.

    This function does not query the database.
    """

    predicted_class = analysis.predicted_class

    if not predicted_class:
        raise ValueError(
            "Analysis has no predicted DR class."
        )

    gradcam_paths = gradcam_paths or {}

    class_label = predicted_class.replace("_", " ")

    image_context = {
        "predicted_class": predicted_class,
        "predicted_class_index": analysis.predicted_class_index,
        "confidence": analysis.confidence,
        "probabilities": analysis.probabilities,
        "model_name": analysis.model_name,
        "explanation": {
            "available": bool(gradcam_paths),
            "explained_class": predicted_class,
        },
    }

    if generator is None:
        generator = RAGAnswerGenerator()

    # =====================================================
    # 1. PATIENT-FRIENDLY EXPLANATION
    # =====================================================

    explanation_result = _generate_section(
        generator=generator,
        question=(
            f"Explain the AI screening result for diabetic "
            f"retinopathy class '{class_label}' in simple "
            f"language suitable for a patient. Explain what "
            f"the screening result means, what it does not "
            f"mean, and why an ophthalmologist must confirm "
            f"the findings through clinical examination. "
            f"Avoid technical jargon and raw academic citations. "
            f"Do not claim this is a confirmed diagnosis."
        ),
        image_context=image_context,
    )

    # =====================================================
    # 2. WHAT THE PREDICTED DR CLASS MEANS
    # =====================================================

    patient_meaning_result = _generate_section(
        generator=generator,
        question=(
            f"Explain the general meaning of diabetic "
            f"retinopathy severity class '{class_label}' "
            f"for a patient. Describe the relevant retinal "
            f"changes, what they generally indicate, and "
            f"whether this severity can progress. Use simple "
            f"patient-friendly language. Do not assume that "
            f"the patient has clinically confirmed disease. "
            f"Use only information supported by the retrieved "
            f"clinical evidence."
        ),
    )

    # =====================================================
    # 3. CLINICAL INFORMATION
    # =====================================================

    clinical_result = _generate_section(
        generator=generator,
        question=(
            f"Provide a clinically detailed explanation of "
            f"diabetic retinopathy severity class '{class_label}'. "
            f"Describe its clinical characteristics, relevant "
            f"retinal findings, disease progression and "
            f"evidence-supported management considerations. "
            f"Use professional terminology appropriate for "
            f"the clinician appendix. Use only retrieved "
            f"clinical evidence. Do not invent guidelines."
        ),
    )

    # =====================================================
    # 4. RECOMMENDED NEXT STEPS
    # =====================================================

    next_steps_result = _generate_section(
        generator=generator,
        question=(
            f"For diabetic retinopathy severity class "
            f"'{class_label}', provide evidence-supported "
            f"clinical follow-up recommendations and next "
            f"steps. Explain appropriate ophthalmologist "
            f"assessment, referral urgency and possible "
            f"treatment options where supported by the "
            f"knowledge base. Distinguish general guidance "
            f"from individual treatment decisions. "
            f"Do not prescribe treatment."
        ),
    )

    # =====================================================
    # 5. PREVENTION GUIDANCE
    # =====================================================

    prevention_result = _generate_section(
        generator=generator,
        question=(
            f"Provide patient-friendly prevention and eye-care "
            f"guidance relevant to diabetic retinopathy class "
            f"'{class_label}'. Cover diabetes management, "
            f"blood glucose and blood pressure management, "
            f"regular retinal screening, follow-up appointments "
            f"and other modifiable risk factors where supported "
            f"by retrieved evidence. Use clear actionable "
            f"language. Do not promise a cure or reversal."
        ),
    )

    # =====================================================
    # 6. PRACTICAL SUMMARY FOR CLINICIANS
    # =====================================================

    practical_summary_result = _generate_section(
        generator=generator,
        question=(
            f"Prepare a concise practical clinical summary "
            f"for healthcare professionals dealing with "
            f"diabetic retinopathy class '{class_label}'. "
            f"Organize it using clear headings and numbered "
            f"steps. Include assessment considerations, "
            f"follow-up, referral and management considerations "
            f"only where supported by the retrieved clinical "
            f"evidence. Do not invent recommendations or "
            f"patient-specific treatment decisions."
        ),
    )

    # =====================================================
    # 7. ICO GUIDELINES
    # =====================================================

    ico_guidelines_result = _generate_section(
        generator=generator,
        question=(
            f"Retrieve and summarize relevant International "
            f"Council of Ophthalmology (ICO) diabetic "
            f"retinopathy guidelines applicable to class "
            f"'{class_label}'. Include relevant screening, "
            f"assessment, referral and management guidance "
            f"as detailed excerpts or structured summaries. "
            f"Preserve the guideline's actual recommendations "
            f"and terminology. Only include information "
            f"present in the retrieved knowledge base. "
            f"If ICO guideline evidence is unavailable, "
            f"explicitly state that it was not retrieved. "
            f"Do not attribute recommendations to ICO "
            f"without supporting source evidence."
        ),
    )

    # =====================================================
    # 8. COLLECT EVIDENCE FROM ALL SECTIONS
    # =====================================================

    all_results = [
        explanation_result,
        patient_meaning_result,
        clinical_result,
        next_steps_result,
        prevention_result,
        practical_summary_result,
        ico_guidelines_result,
    ]

    all_evidence = []

    for result in all_results:
        all_evidence.extend(
            result.get("evidence", [])
        )

    evidence = _normalise_evidence(all_evidence)

    # =====================================================
    # 9. ASSEMBLE PDF DATA
    # =====================================================

    return {
        # Prediction information.
        "analysis_id": analysis.id,
        "patient_name": patient_name,
        "predicted_class": predicted_class,
        "confidence": analysis.confidence,
        "probabilities": analysis.probabilities,
        "model_name": analysis.model_name,

        # Original and Grad-CAM images.
        "original_image_path": str(original_image_path),

        "gradcam_heatmap_path": (
            str(gradcam_paths["heatmap"])
            if gradcam_paths.get("heatmap")
            else None
        ),

        "gradcam_overlay_path": (
            str(gradcam_paths["overlay"])
            if gradcam_paths.get("overlay")
            else None
        ),

        # Patient summary.
        "explanation": _section_answer(
            explanation_result,
            "Patient-friendly explanation could not be retrieved.",
            ("Understanding Your Result",),
        ),

        "patient_meaning": _section_answer(
            patient_meaning_result,
            "Information about this DR class could not be retrieved.",
            ("What Mild DR Means", "What Your DR Class Means"),
        ),

        "prevention": [
            _section_answer(
                prevention_result,
                "Prevention guidance could not be retrieved.",
                ("Prevention Guidance",),
            )
        ],

        # Clinician appendix.
        "clinical_information": _section_answer(
            clinical_result,
            "Clinical information could not be retrieved.",
            ("Clinical Information",),
        ),

        "next_steps": [
            _section_answer(
                next_steps_result,
                "Recommended next steps could not be retrieved.",
                ("Recommended Next Steps",),
            )
        ],

        "practical_summary": _section_answer(
            practical_summary_result,
            "Practical clinical summary could not be retrieved.",
            ("Practical Summary for Clinicians",),
        ),

        "ico_guidelines": _section_answer(
            ico_guidelines_result,
            "Relevant ICO guideline information was not retrieved.",
            ("ICO Guidelines", "International Council of Ophthalmology Guidelines"),
        ),

        # Actual retrieved passages remain available internally.
        # The PDF generator displays only filenames and page numbers.
        "evidence": evidence,
    }


# ---------------------------------------------------------
# FINAL PDF GENERATION
# ---------------------------------------------------------

def generate_analysis_report(
    analysis: Any,
    original_image_path: str | Path,
    gradcam_paths: dict[str, Path] | None = None,
    patient_name: str | None = None,
    generator: RAGAnswerGenerator | None = None,
) -> bytes:
    """
    Build structured report data and generate the final PDF.
    """

    report_data = build_report_data(
        analysis=analysis,
        original_image_path=original_image_path,
        gradcam_paths=gradcam_paths,
        patient_name=patient_name,
        generator=generator,
    )

    return generate_patient_report(report_data)