import glob
import os
from typing import Any, Dict, List, Optional, Set
import duckdb
import numpy as np


class TreeCatalog:
    """SQL-enabled Catalog interface over one or more Parquet tree files."""

    def __init__(self, parquet_glob: str):
        self.parquet_glob = parquet_glob
        self.conn = duckdb.connect(database=":memory:")

    def has_files(self) -> bool:
        """Check if any files match the parquet glob."""
        files = glob.glob(self.parquet_glob, recursive=True)
        return len(files) > 0

    def get_existing_x0_hashes(self) -> Set[str]:
        """Return set of all unique x0_hashes across all catalog files for fast deduplication."""
        if not self.has_files():
            return set()
        try:
            sql = f"""
                SELECT DISTINCT x0_hash 
                FROM read_parquet('{self.parquet_glob}') 
                WHERE x0_hash IS NOT NULL AND x0_hash != ''
            """
            rows = self.conn.execute(sql).fetchall()
            return {r[0] for r in rows}
        except Exception:
            # If x0_hash column doesn't exist in older files
            return set()

    def query(
        self,
        depth_min: Optional[int] = None,
        depth_max: Optional[int] = None,
        leaf_min: Optional[int] = None,
        leaf_max: Optional[int] = None,
        dimension: Optional[str] = None,
        rule_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Filter trees with SQL speed and return parsed metadata and byte arrays."""
        if not self.has_files():
            return []

        conditions = []
        if depth_min is not None:
            conditions.append(f"depth >= {depth_min}")
        if depth_max is not None:
            conditions.append(f"depth <= {depth_max}")
        if leaf_min is not None:
            conditions.append(f"leaf_count >= {leaf_min}")
        if leaf_max is not None:
            conditions.append(f"leaf_count <= {leaf_max}")
        if dimension is not None:
            conditions.append(f"dimension = '{dimension}'")
        if rule_name is not None:
            conditions.append(f"rule_name = '{rule_name}'")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {limit}" if limit is not None else ""

        sql = f"""
            SELECT *
            FROM read_parquet('{self.parquet_glob}')
            {where_clause}
            {limit_clause}
        """

        table = self.conn.execute(sql).to_arrow_table()
        return table.to_pylist()

    def count(self) -> int:
        """Count total trees in catalog."""
        if not self.has_files():
            return 0
        sql = f"SELECT count(*) FROM read_parquet('{self.parquet_glob}')"
        return int(self.conn.execute(sql).fetchone()[0])

    def summary(self) -> Dict[str, Any]:
        """Aggregate depth and leaf statistics."""
        if not self.has_files():
            return {"total_trees": 0}
        sql = f"""
            SELECT 
                count(*) as total_trees,
                min(depth) as min_depth,
                max(depth) as max_depth,
                avg(depth) as avg_depth,
                min(leaf_count) as min_leaves,
                max(leaf_count) as max_leaves,
                avg(leaf_count) as avg_leaves,
                count(DISTINCT x0_hash) as unique_x0_states
            FROM read_parquet('{self.parquet_glob}')
        """
        row = self.conn.execute(sql).fetchone()
        return {
            "total_trees": int(row[0]),
            "min_depth": int(row[1]) if row[1] is not None else 0,
            "max_depth": int(row[2]) if row[2] is not None else 0,
            "avg_depth": float(row[3]) if row[3] is not None else 0.0,
            "min_leaves": int(row[4]) if row[4] is not None else 0,
            "max_leaves": int(row[5]) if row[5] is not None else 0,
            "avg_leaves": float(row[6]) if row[6] is not None else 0.0,
            "unique_x0_states": int(row[7]) if row[7] is not None else 0,
        }
