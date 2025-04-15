"""Weak value pipeline"""


class WvPipeline:
    def __init__(self, params):
        self.params = params
        self.results = None

    def run(self):
        raise NotImplementedError

    def set_params(self, params):
        self.params = params
