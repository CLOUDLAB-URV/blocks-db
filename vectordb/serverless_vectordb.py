from lithops import FunctionExecutor

from vectordb.config import SvlessVectorDBParams

from .indexing.indexator import check_payload, initialize_database, initialize_from_plan
from .indexing.prepare import check_ephemeral_storage
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
        """Refuse a plan the functions could not run: a largest block that
        would not fit the disk of a function, or task arguments the executor
        would refuse to send. The disk is sized from the backend section of
        the Lithops configuration when the runtime is deployed; a backend
        without the setting, like localhost, sets no limit."""
        executor = self.indexing_executor
        limit_mb = executor.config.get(executor.backend, {}).get("ephemeral_storage")
        if limit_mb is not None:
            check_ephemeral_storage(plan, limit_mb)
        check_payload(plan, self.params, executor)

    def indexing_from_plan(self, plan):
        """Build the blocks of a parquet plan, one task each."""
        return initialize_from_plan(plan, self.params, self.indexing_executor, self.wait_timeout)
        
    def search(self, id, query_vector, filter_tags=None):
        return self.orchestrator.search(id, query_vector, self.params.num_centroids_search, self.params.k_search, self.params.k_result, filter_tags=filter_tags)