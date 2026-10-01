"""From declared sources to a sealed build: the client half of the parquet path.

:func:`prepare_build` inspects the declared files, plans the blocks and
returns the plan with the index configuration sealed: what was read
(``source_format``, ``source_keys``), how many rows the source holds
(``source_rows``), the dimension the files actually have (``features``)
and the block count. Everything a query later needs comes from that
sealed ``config.json``; nothing is probed afterwards.

Nothing here uploads, writes or invokes a function; it only reads the
sources, over the network when they are on S3.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path
from typing import Callable, Sequence

from vectordb.config import SvlessVectorDBParams
from vectordb.indexing.planner import Plan, PlanError, largest_nlist, plan, suggested_nlist
from vectordb.utils.parquet import OWI_V2, EmptyParquetFile, inspect

# a FAISS IVF,Flat block on disk costs about one float32 per value plus a
# small header; the function writes it to its ephemeral disk before
# uploading, so the largest block has to fit there with room to spare
_BYTES_PER_VALUE = 4
_EPHEMERAL_HEADROOM = 1.3


def expand_sources(
    source: str,
    list_s3: Callable[[str, str], list[str]] | None = None,
    files: str = "*.parquet",
) -> list[str]:
    """The files a declared source stands for, in sorted order.

    A local directory means the files under it, at any depth, whose name
    matches ``files``; an ``s3://`` URI ending in ``/`` means the matching
    keys under that prefix, listed once through ``list_s3(bucket, prefix)``;
    anything else is one file.

    ``files`` is a shell pattern on the file name, such as
    ``*_embeddings.parquet`` to leave out other parquet files kept beside
    the vector files.
    """
    if source.startswith("s3://"):
        if not source.endswith("/"):
            return [source]
        if list_s3 is None:
            raise PlanError(f"{source}: listing an s3 prefix needs a lister")
        bucket, prefix = source[len("s3://"):].split("/", 1)
        keys = [key for key in list_s3(bucket, prefix) if fnmatchcase(key.rsplit("/", 1)[-1], files)]
        if not keys:
            raise PlanError(f"{source}: no files matching '{files}' under the prefix")
        return sorted(f"s3://{bucket}/{key}" for key in keys)
    path = Path(source).expanduser()
    if path.is_dir():
        found = sorted(
            str(item.resolve()) for item in path.rglob("*")
            if item.is_file() and fnmatchcase(item.name, files)
        )
        if not found:
            raise PlanError(f"{source}: no files matching '{files}' in the directory")
        return found
    if not path.exists():
        raise PlanError(f"{source}: not found")
    return [str(path.resolve())]


def prepare_build(sources: Sequence[str], config: dict) -> tuple[Plan, dict]:
    """Returns (plan, config to save): the planned blocks and the sealed configuration."""
    if not sources:
        raise PlanError("no sources declared")
    repeated = [uri for uri in set(sources) if list(sources).count(uri) > 1]
    if repeated:
        raise PlanError(
            "the same source is declared more than once, which would index"
            f" its rows twice under different ids: {sorted(repeated)[0]}"
        )
    num_index = config.get("num_index")
    if not isinstance(num_index, int) or isinstance(num_index, bool):
        raise PlanError("num_index (number of blocks) must be declared as an integer")
    implementation = config.get("implementation", "blocks")
    if implementation != "blocks":
        raise PlanError(
            f"implementation '{implementation}': the parquet path builds"
            " blocks; no other implementation reads a plan"
        )
    features = config.get("features")
    if not isinstance(features, int) or isinstance(features, bool):
        # the dimension decides which rows are rejected, so it is declared
        # and checked against the files, never inferred from a first row
        raise PlanError(
            "features (vector dimension) must be declared as an integer;"
            " it is what tells a malformed row from a short one"
        )

    files, empty = [], 0
    for uri in sources:
        try:
            files.append(inspect(uri))
        except EmptyParquetFile:
            # a file with no rows adds nothing to the build; it is counted
            # in source_files_skipped
            empty += 1
    if not files:
        raise PlanError(f"every declared source is empty ({empty} files)")
    if "k" not in config:
        # the default k of 4096 lists is more than a small block can train,
        # so a parquet build declares its list count
        draft = plan(files, num_index, features=features)
        raise PlanError(
            "k (IVF lists per block) must be declared; the smallest block"
            f" holds {draft.min_block_rows} rows, suggested k:"
            f" {suggested_nlist(draft.min_block_rows)}"
            f" (at most {largest_nlist(draft.min_block_rows)})"
        )
    result = plan(files, num_index, k=config["k"], features=features)

    sealed = dict(config)
    sealed.update(
        source_format="parquet",
        source_keys=list(result.sources),
        source_rows=result.total_vectors,
        total_vectors=result.total_vectors,
        num_vectors=result.total_vectors,
        features=result.dimension,
        num_index=num_index,
        implementation=implementation,
        block_ranges=[[b.block, b.first_id, b.last_id] for b in result.blocks],
        source_files_skipped=empty,
        unit_norm=result.dialect == OWI_V2,
    )
    # building the parameters here turns an unknown or misspelled key into
    # a failure now rather than a TypeError after the build, and a missing
    # one into a default the sealed config records
    SvlessVectorDBParams(**sealed)
    return result, sealed


def check_ephemeral_storage(result: Plan, limit_mb: int) -> None:
    """The largest block must fit on the disk of a function.

    ``limit_mb`` is the size Lithops gives that disk when it deploys the
    runtime, ``aws_lambda.ephemeral_storage`` in its configuration, so
    the caller reads it from the executor that runs the build.
    """
    needed = result.max_block_rows * result.dimension * _BYTES_PER_VALUE
    needed_mb = needed * _EPHEMERAL_HEADROOM / (1024 * 1024)
    if needed_mb > limit_mb:
        raise PlanError(
            f"the largest block holds {result.max_block_rows} vectors of"
            f" {result.dimension} values, about {needed / (1024 * 1024):.0f} MB"
            f" written to the function's disk, which does not fit in its"
            f" ephemeral storage of {limit_mb} MB. Raise aws_lambda.ephemeral_storage"
            " in the Lithops configuration (up to 10240) and deploy the runtime"
            " again, or use more blocks"
        )
