"""SLM Specialists Manager"""

class SLMSpecialists:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        # Handle dict or Config object
        self.config_dict = config if isinstance(config, dict) else getattr(config, '__dict__', {})
        self._init_specialists()
    
    def _init_specialists(self):
        from .slm_providers import SLMFactory
        
        provider = None
        slm_provider = self.config_dict.get('slm_provider')
        
        try:
            # Extract SLM model settings
            slm_kwargs = {}
            if 'slm_model' in self.config_dict:
                slm_kwargs['model'] = self.config_dict['slm_model']
            if 'slm_sql_model' in self.config_dict:
                slm_kwargs['sql_model'] = self.config_dict['slm_sql_model']
            if 'slm_classifier_model' in self.config_dict:
                slm_kwargs['classifier_model'] = self.config_dict['slm_classifier_model']
            if 'slm_code_review_model' in self.config_dict:
                slm_kwargs['code_review_model'] = self.config_dict['slm_code_review_model']
            if 'slm_base_url' in self.config_dict:
                slm_kwargs['base_url'] = self.config_dict['slm_base_url']

            # Try configured provider
            if slm_provider:
                provider = SLMFactory.create(slm_provider, **slm_kwargs)
            else:
                provider = SLMFactory.auto_detect(**slm_kwargs)
        except Exception as e:
            self.logger.warning(f"    SLM Provider {slm_provider}: {e}")
            try:
                # Fallback to auto-detect
                provider = SLMFactory.auto_detect()
            except Exception as e2:
                self.logger.warning(f"    SLM Auto-detect failed: {e2}")
        
        if provider:
            self.provider = provider
            # Create wrappers for compatibility
            self.sql_generator = SQLGeneratorWrapper(self.provider)
            self.classifier = ClassifierWrapper(self.provider)
            self.code_reviewer = CodeReviewerWrapper(self.provider)
            self.logger.info(f"  [OK] SLM: {self.provider.name}")
        else:
            # Ultimate fallback to original implementations or mocks
            try:
                from .slm import SQLGeneratorSLM, ClassifierSLM, CodeReviewerSLM
                self.sql_generator = SQLGeneratorSLM()
                self.classifier = ClassifierSLM()
                self.code_reviewer = CodeReviewerSLM()
                
                # Add aliases for compatibility
                if not hasattr(self.classifier, 'classify'):
                    self.classifier.classify = getattr(self.classifier, 'classify_intent', None)
                if not hasattr(self.code_reviewer, 'review'):
                    self.code_reviewer.review = getattr(self.code_reviewer, 'review_code', None)
                    
                self.logger.info("  [OK] SLM (original)")
            except Exception as e:
                self.logger.warning(f"    Using absolute mocks: {e}")
                from .mocks import MockSQLGenerator, MockClassifier, MockCodeReviewer
                self.sql_generator = MockSQLGenerator()
                self.classifier = MockClassifier()
                self.code_reviewer = MockCodeReviewer()


class SQLGeneratorWrapper:
    """Wrapper to make provider compatible with old interface."""
    def __init__(self, provider):
        self.provider = provider
    
    def generate(self, query: str, schema=None):
        return self.provider.generate_sql(query, schema)


class ClassifierWrapper:
    """Wrapper to make provider compatible with old interface."""
    def __init__(self, provider):
        self.provider = provider
    
    def classify(self, text: str, categories=None):
        return self.provider.classify(text, categories)


class CodeReviewerWrapper:
    """Wrapper to make provider compatible with old interface."""
    def __init__(self, provider):
        self.provider = provider
    
    def review(self, code: str, language="python"):
        return self.provider.review_code(code, language)
