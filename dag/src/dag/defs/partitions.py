import os

import dagster as dg

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "50"))

chunk_partitions = dg.DynamicPartitionsDefinition(name="chunk")
