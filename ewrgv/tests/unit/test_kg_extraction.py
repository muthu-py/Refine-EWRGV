"""
tests/unit/test_kg_extraction.py
----------------------------------
Entity tagging and gap-statement extraction for the knowledge graph.
"""

from __future__ import annotations

import pytest

from app.knowledge_graph.statements import find_gap_statements, section_weight
from app.knowledge_graph.tagger import EntityTagger, normalise_term, query_terms


class TestTagger:
    def test_lexicon_and_synonyms_map_to_canonical(self) -> None:
        tagger = EntityTagger()
        mentions = tagger.tag("We use a Vision Transformer (ViT) and CNNs on OCT scans for diabetic retinopathy.")
        found = {(m.entity_type, m.canonical) for m in mentions}
        assert ("METHOD", "vision transformer") in found
        assert ("METHOD", "convolutional neural network") in found
        assert ("MODALITY", "optical coherence tomography") in found
        assert ("DISEASE", "diabetic retinopathy") in found

    def test_short_acronyms_are_case_sensitive(self) -> None:
        tagger = EntityTagger()
        assert any(m.canonical == "diabetic retinopathy" for m in tagger.tag("Screening for DR in adults."))
        assert not any(m.canonical == "diabetic retinopathy" for m in tagger.tag("Dr. Smith reviewed the images."))

    def test_longest_match_wins_and_no_overlap(self) -> None:
        tagger = EntityTagger()
        mentions = tagger.tag("multi-label classification of fundus images")
        names = [m.canonical for m in mentions]
        assert "multi-label classification" in names
        assert "classification" not in names
        spans = sorted((m.start, m.end) for m in mentions)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))

    def test_dataset_regex(self) -> None:
        tagger = EntityTagger()
        names = {m.canonical for m in tagger.tag("Results on the RetinaX dataset and the public dataset.") if m.entity_type == "DATASET"}
        assert names == {"retinax"}

    def test_query_understanding_terms_extend_vocabulary(self) -> None:
        terms = query_terms({"methods": ["capsule network"], "datasets": ["ODIR-5K"], "task": "glaucoma grading",
                             "concepts": ["domain generalisation"]})
        tagger = EntityTagger(terms)
        found = {(m.entity_type, m.canonical) for m in tagger.tag("A capsule network for glaucoma grading.")}
        assert ("METHOD", "capsule network") in found
        assert ("TASK", "glaucoma grading") in found

    def test_normalise_term(self) -> None:
        assert normalise_term("Vision-Transformers") == normalise_term("vision transformer")


class TestGapStatements:
    def test_cue_classes_and_strength(self) -> None:
        text = ("The generalisation of OCT models to unseen scanners remains largely unexplored. "
                "Our evaluation is limited to a single dataset from one hospital in this setting. "
                "Future work should validate the model on multi-centre cohorts with diverse patients.")
        hits = find_gap_statements(text, "LIMITATIONS")
        assert [h.cue_class for h in hits] == ["OPEN_PROBLEM", "LIMITATION", "FUTURE_WORK"]
        assert hits[0].strength == pytest.approx(1.0)
        assert hits[2].strength == pytest.approx(0.6)

    def test_section_weight_scales_strength(self) -> None:
        text = "The generalisation of OCT models to unseen scanners remains largely unexplored in practice."
        (intro,) = find_gap_statements(text, "INTRODUCTION")
        assert intro.strength == pytest.approx(1.0 * section_weight("INTRODUCTION"))
        assert section_weight("OTHER", "2.1 Research Gap and Novelty Analysis") >= 0.9

    @pytest.mark.parametrize("sentence", [
        "To overcome the limitation of independent CNN and ViT models, we implement a hybrid model here.",
        "The proposed system provides several advantages over conventional single-modality approaches in screening.",
        "Considerable advances in SSL resolve the sample limitations of medical data for detection tasks.",
        "Traditional retinal examination methods rely heavily on manual expert interpretation of the images.",
        "The app offers a user-friendly interface but lacks offline support for clinics in the region.",
        "[3] CNN-based model Fundus images High accuracy Limited interpretability [4] CNN glaucoma detection.",
        "Volume 8, Issue 2, March 2026 low-resource settings remain limited for such screening systems today.",
    ])
    def test_non_gap_sentences_are_dropped(self, sentence: str) -> None:
        assert find_gap_statements(sentence, "DISCUSSION") == []

    def test_length_limits(self) -> None:
        assert find_gap_statements("This remains unexplored.", "LIMITATIONS") == []


