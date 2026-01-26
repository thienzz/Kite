"""Pipeline Manager"""

class PipelineManager:
    def __init__(self, pipeline_class, logger):
        self.pipeline_class = pipeline_class
        self.logger = logger
        self.pipelines = {}
    
    def create(self, name: str, event_bus = None):
        pipeline = self.pipeline_class(name, event_bus=event_bus)
        self.pipelines[name] = pipeline
        return pipeline
    
    def get(self, name: str):
        return self.pipelines.get(name)
