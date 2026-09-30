from lithops import FunctionExecutor

from vectordb.config import SvlessVectorDBParams

from .indexing.indexator import check_payload, initialize_database, initialize_from_plan
from .orchestration.orchestrator import Orchestrator

class ServerlessVectorDB():
    
    def __init__(self, wait_timeout=None, **parameters):
        self.params: SvlessVectorDBParams = SvlessVectorDBParams(**parameters)
        self.wait_timeout = wait_timeout
        self.indexing_executor = FunctionExecutor()
        self.orchestrator = Orchestrator(self.params, wait_timeout=wait_timeout)
        
    def indexing(self, filename, num_workers):
        if not self.params.skip_init:
            return initialize_database(filename, self.params, self.indexing_executor, num_workers, self.wait_timeout)
        return {}

    def check_plan(self, plan):
        """Refuse a plan whose tasks the executor would refuse to send."""
        check_payload(plan, self.params, self.indexing_executor)

    def indexing_from_plan(self, plan):
        """Build the blocks of a parquet plan, one task each."""
        return initialize_from_plan(plan, self.params, self.indexing_executor, self.wait_timeout)
        
    def search(self, id, query_vector, filter_tags=None):
        return self.orchestrator.search(id, query_vector, self.params.num_centroids_search, self.params.k_search, self.params.k_result, filter_tags=filter_tags)