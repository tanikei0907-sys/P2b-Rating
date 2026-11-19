from __future__ import annotations

import gc
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pyodbc

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from forward_ops import copy_database, load_config
from viewer_data import cleanup_access_lock_files

OUT = Path(__file__).resolve().parent


def main() -> None:
    config = load_config(BASE / "forward_config.yaml")
    work = OUT / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as temporary:
        db = copy_database(BASE / "stage2_input/BaoZ_2_snapshot.mdb", Path(temporary), config["database"].get("linked_tables", {}))
        connection = f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db};ReadOnly=1;"
        try:
            with pyodbc.connect(connection, readonly=True, autocommit=True) as con:
                schemas = {}
                for table in ["出走馬T", "出走馬マスタ", "レースT", "レースマスタ"]:
                    cursor = con.cursor()
                    try:
                        schemas[table] = [{"name": column[0], "type": str(column[1])} for column in cursor.execute(f"SELECT * FROM [{table}] WHERE 1=0").description]
                    finally:
                        cursor.close()
                runner_names = [column["name"] for column in schemas["出走馬T"]]
                race_names = [column["name"] for column in schemas["レースT"]]
                runner_scale = [name for name in runner_names if "タイム" in name or "着差" in name]
                race_scale = [name for name in race_names if "タイム" in name or "着差" in name or "距離" in name or "馬場" in name]
                select_runner = ", ".join(f"r.[{name}]" for name in runner_scale)
                select_race = ", ".join(f"s.[{name}]" for name in race_scale)
                sample = pd.read_sql(
                    f"""
                    SELECT TOP 50 r.[競走コード] AS race_id, r.[馬番] AS horse_number,
                           r.[確定着順] AS finish_position, r.[異常区分コード] AS abnormal_code,
                           {select_runner}, {select_race}
                    FROM [レースT] AS s INNER JOIN [出走馬T] AS r ON s.[競走コード]=r.[競走コード]
                    WHERE s.[月日] >= #2024-12-01# AND s.[主催者コード]=1 AND s.[トラック種別コード] IN (0,1)
                    """,
                    con,
                )
                con.close()
        finally:
            gc.collect()
            cleanup_access_lock_files(temporary)
    (OUT / "time_db_schema.json").write_text(json.dumps(schemas, ensure_ascii=False, indent=2), encoding="utf-8")
    sample.to_csv(OUT / "time_schema_samples.csv", index=False)
    print(json.dumps({
        "runner_time_columns": [(column["name"], column["type"]) for column in schemas["出走馬T"] if column["name"] in runner_scale],
        "race_scale_columns": [(column["name"], column["type"]) for column in schemas["レースT"] if column["name"] in race_scale],
        "sample": sample.head(16).to_dict("records"),
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
