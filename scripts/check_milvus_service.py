#!/usr/bin/env python3
"""Validate only the owned loopback CI Milvus service, not the application store stub."""

import argparse
import uuid


def check_service(host: str, port: int) -> None:
    if host not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("The CI service contract requires a loopback host")
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

    alias = "ci_" + uuid.uuid4().hex
    name = "ci_contract_" + uuid.uuid4().hex
    created = False
    try:
        connections.connect(alias=alias, host=host, port=str(port), timeout=15)
        version = utility.get_server_version(using=alias, timeout=15)
        if not version.startswith("v2.3.") and not version.startswith("2.3."):
            raise ValueError(f"Expected the declared CI Milvus 2.3 service, got {version}")
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=2),
            ]
        )
        collection = Collection(name=name, schema=schema, using=alias, consistency_level="Strong")
        created = True
        collection.insert([[11, 12], [[1.0, 0.0], [0.0, 1.0]]], timeout=15)
        collection.flush(timeout=30)
        collection.create_index("vector", {"index_type": "FLAT", "metric_type": "L2", "params": {}}, timeout=30)
        collection.load(timeout=30)
        results = collection.search([[1.0, 0.0]], "vector", {"metric_type": "L2", "params": {}}, limit=1, timeout=30)
        assert len(results) == 1 and len(results[0]) == 1 and results[0][0].id == 11
        assert abs(results[0][0].distance) < 1e-6
        rows = collection.query("pk in [11, 12]", output_fields=["pk"], timeout=15)
        assert {row["pk"] for row in rows} == {11, 12}
    finally:
        try:
            if created:
                utility.drop_collection(name, using=alias, timeout=30)
        finally:
            connections.disconnect(alias)
    print("Owned loopback Milvus service: real SDK vectors/ranking passed; application backend remains unavailable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19530)
    args = parser.parse_args()
    check_service(args.host, args.port)


if __name__ == "__main__":
    main()
