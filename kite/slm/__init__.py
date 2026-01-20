"""SLM Specialists module."""
from .sql_generator_slm import SQLGeneratorSLM
from .classifier_slm import ClassifierSLM
from .code_reviewer_slm import CodeReviewerSLM

__all__ = ['SQLGeneratorSLM', 'ClassifierSLM', 'CodeReviewerSLM']
