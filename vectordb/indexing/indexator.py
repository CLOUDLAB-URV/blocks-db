import time
import logging
import importlib
import dataclasses

import cloudpickle
from lithops.constants import MAX_AGG_DATA_SIZE
from lithops.utils import verify_args

from vectordb.indexing.planner import PlanError
from vectordb.utils.waiting import collect


def initialize_database(filename, params, fexec, num_workers=16, wait_timeout=None):

    init = time.time()

    if not params.skip_kmeans:
        try:
            module = importlib.import_module(
                f"vectordb.implementations.{params.implementation}.preprocess"
            )
            preprocessor_cls = module.IMPLEMENTATION_PREPROCESSOR
            preprocessor = preprocessor_cls(params, fexec.storage)
            preprocessor.run(filename, num_workers)
        except ModuleNotFoundError:
            logging.info(f"No preprocess step for implementation '{params.implementation}'")

    vectors_key = f"{params.storage_bucket}/{filename}"

    distribute_time = None
    try:
        dist_module = importlib.import_module(
            f"vectordb.implementations.{params.implementation}.distribute"
        )
        distribute_fn = dist_module.IMPLEMENTATION_DISTRIBUTOR

        logging.info("Starting distribute phase")
        futures = fexec.map(
            distribute_fn,
            vectors_key,
            extra_args=[params],
            obj_chunk_number=num_workers,
            runtime_memory=params.index_mem
        )
        distribute_time = collect(fexec, futures, wait_timeout)
        logging.info("Distribute phase complete")

    except ModuleNotFoundError:
        logging.info(f"No distribute step for implementation '{params.implementation}'")

    logging.info("Starting indexing")

    index_module = importlib.import_module(
        f"vectordb.implementations.{params.implementation}.initialize"
    )
    indexing_function = index_module.get_index_builder()

    if distribute_time is not None:
        all_index = list(range(params.num_index))
        n_per_worker = max(1, params.num_index // num_workers)
        index_batches = [
            all_index[i:i + n_per_worker]
            for i in range(0, len(all_index), n_per_worker)
        ]
        futures = fexec.map(
            indexing_function,
            index_batches,
            extra_args=[params],
            runtime_memory=params.index_mem,
        )
    else:
        obj_chunk = num_workers
        n_blocks_per_function = max(1, int(params.num_index / obj_chunk))
        futures = fexec.map(
            indexing_function,
            vectors_key,
            extra_args=[params, n_blocks_per_function],
            obj_chunk_number=obj_chunk,
            runtime_memory=params.index_mem,
        )

    indexing_function_time = collect(fexec, futures, wait_timeout)

    lambda_invocation_indexing = [
        f.stats["worker_func_start_tstamp"] - f.stats["host_job_create_tstamp"]
        for f in futures
    ]

    end = time.time()

    timers = {
        f"distribute_{params.implementation}": distribute_time,
        f"indexing_function_{params.implementation}": indexing_function_time,
        f"indexing_function_invocation_{params.implementation}": lambda_invocation_indexing,
        f"total_indexing_{params.implementation}": end - init,
    }

    return timers


def task_params(params):
    """The parameters every build task carries.

    The plan already names the file and the rows of each block, so the
    lists of source files and id ranges sealed for config.json stay
    behind: repeated in every task, they are most of the payload of a
    build from hundreds of files.
    """
    return dataclasses.replace(params, source_keys=None, block_ranges=None)


def check_payload(plan, params, fexec):
    """Refuse a plan whose task arguments Lithops would refuse, before the
    build has any side effect.

    The size is measured as Lithops measures it: the arguments of every
    task pickled on their own and summed, against ``data_limit`` in the
    ``lithops`` section of the executor's configuration, in MiB (4 when
    it is not set; a false value disables the check, in Lithops and here).
    """
    from vectordb.implementations.blocks.initialize import build_block_from_parquet

    limit = fexec.config["lithops"].get("data_limit", MAX_AGG_DATA_SIZE)
    if not limit:
        return
    tasks = verify_args(build_block_from_parquet, list(plan.blocks), [task_params(params)])
    size = sum(len(cloudpickle.dumps(task)) for task in tasks)
    if size > limit * 1024 ** 2:
        raise PlanError(
            f"the arguments of the {plan.num_index} build tasks weigh"
            f" {size / 1024 ** 2:.2f} MiB ({len(plan.sources)} source files,"
            f" {sum(len(block.ranges) for block in plan.blocks):,} row ranges),"
            f" more than the {limit} MiB Lithops sends to the functions."
            " Raise data_limit in the lithops section of the Lithops"
            " configuration, or build from fewer files"
        )


def initialize_from_plan(plan, params, fexec, wait_timeout=None):
    """Build every block of a parquet plan: one task per block.

    The plan already fixes the block count and the ids, so the number of
    functions is the number of blocks whatever the executor's
    concurrency; nothing is split by bytes. Returns the indexing timers of
    the CSV path (there is no distribute phase) plus the per-block reports
    (rows kept, rows rejected), so a build can be audited from its output
    alone.
    """
    from vectordb.implementations.blocks.initialize import build_block_from_parquet

    init = time.time()
    futures = fexec.map(
        build_block_from_parquet,
        list(plan.blocks),
        extra_args=[task_params(params)],
        runtime_memory=params.index_mem,
    )
    reports = collect(fexec, futures, wait_timeout)
    invocation = [
        f.stats["worker_func_start_tstamp"] - f.stats["host_job_create_tstamp"]
        for f in futures
    ]
    end = time.time()
    return {
        f"indexing_function_{params.implementation}": [r["seconds"] for r in reports],
        f"indexing_function_invocation_{params.implementation}": invocation,
        f"total_indexing_{params.implementation}": end - init,
        "blocks": sorted(reports, key=lambda r: r["block"]),
        "rows": sum(r["rows"] for r in reports),
        "rejected": sum(r["rejected"] for r in reports),
    }
