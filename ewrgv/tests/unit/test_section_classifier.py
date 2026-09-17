"""
tests/unit/test_section_classifier.py
--------------------------------------
Unit tests for app/ingestion/sectioning/classifier.py
"""
import pytest
from app.domain.enums import SectionType
from app.ingestion.sectioning.classifier import classify


class TestSectionClassifier:
    """Tests for the rule-based section type classifier."""

    def test_abstract(self):
        assert classify("Abstract") == SectionType.ABSTRACT

    def test_abstract_case_insensitive(self):
        assert classify("ABSTRACT") == SectionType.ABSTRACT

    def test_introduction(self):
        assert classify("Introduction") == SectionType.INTRODUCTION

    def test_background(self):
        assert classify("Background") == SectionType.BACKGROUND

    def test_related_work(self):
        assert classify("Related Work") == SectionType.RELATED_WORK

    def test_related_studies_variant(self):
        assert classify("Related Studies") == SectionType.RELATED_WORK

    def test_literature_review(self):
        assert classify("Literature Review") == SectionType.LITERATURE_REVIEW

    def test_methodology(self):
        assert classify("Methodology") == SectionType.METHODOLOGY

    def test_proposed_method(self):
        assert classify("Proposed Method") == SectionType.METHODOLOGY

    def test_methods(self):
        assert classify("Methods") == SectionType.METHODS

    def test_materials_and_methods(self):
        assert classify("Materials and Methods") == SectionType.METHODS

    def test_dataset(self):
        assert classify("Dataset") == SectionType.DATASET

    def test_datasets(self):
        assert classify("Datasets") == SectionType.DATASET

    def test_experiments(self):
        assert classify("Experiments") == SectionType.EXPERIMENTS

    def test_ablation_study(self):
        assert classify("Ablation Study") == SectionType.EXPERIMENTS

    def test_experimental_setup(self):
        assert classify("Experimental Setup") == SectionType.EXPERIMENTAL_SETUP

    def test_results(self):
        assert classify("Results") == SectionType.RESULTS

    def test_results_and_discussion(self):
        assert classify("Results and Discussion") == SectionType.DISCUSSION

    def test_discussion(self):
        assert classify("Discussion") == SectionType.DISCUSSION

    def test_evaluation(self):
        assert classify("Evaluation") == SectionType.EVALUATION

    def test_limitations(self):
        assert classify("Limitations") == SectionType.LIMITATIONS

    def test_future_work(self):
        assert classify("Future Work") == SectionType.FUTURE_WORK

    def test_conclusion(self):
        assert classify("Conclusion") == SectionType.CONCLUSION

    def test_conclusions(self):
        assert classify("Conclusions") == SectionType.CONCLUSION

    def test_concluding_remarks(self):
        assert classify("Concluding Remarks") == SectionType.CONCLUSION

    def test_references(self):
        assert classify("References") == SectionType.REFERENCES

    def test_bibliography(self):
        assert classify("Bibliography") == SectionType.REFERENCES

    def test_acknowledgments(self):
        assert classify("Acknowledgments") == SectionType.ACKNOWLEDGMENTS

    def test_appendix(self):
        assert classify("Appendix") == SectionType.APPENDIX

    def test_supplementary_material(self):
        assert classify("Supplementary Material") == SectionType.APPENDIX

    def test_empty_string(self):
        assert classify("") == SectionType.UNKNOWN

    def test_unknown_heading(self):
        result = classify("Some Completely Unrecognized Heading Title")
        assert result == SectionType.OTHER

    def test_model_architecture(self):
        assert classify("Model Architecture") == SectionType.MODEL

    def test_implementation_details(self):
        assert classify("Implementation Details") == SectionType.IMPLEMENTATION

    def test_system_design(self):
        assert classify("System Design") == SectionType.SYSTEM_DESIGN
