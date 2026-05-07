# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import sqlite3


def read_face_clusters(db_path: str) -> list[dict]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:

        def resolve_table(preferred: str, fallback: str) -> str:
            for table_name in (preferred, fallback):
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,),
                ).fetchone()
                if row:
                    return table_name
            raise sqlite3.OperationalError(
                f"Neither {preferred} nor {fallback} exists in Photos database"
            )

        person_table = resolve_table("ZPERSON", "ZGENERICPERSON")
        asset_table = resolve_table("ZASSET", "ZGENERICASSET")

        people = conn.execute(
            f"""
            SELECT Z_PK, ZFULLNAME FROM {person_table}
            WHERE ZFULLNAME IS NOT NULL AND ZFULLNAME != ''
            AND ZMERGEDINTO IS NULL
            """
        ).fetchall()

        clusters: list[dict] = []
        for person_pk, name in people:
            count_row = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM ZDETECTEDFACE
                JOIN {asset_table} ON ZDETECTEDFACE.ZASSET = {asset_table}.Z_PK
                WHERE ZDETECTEDFACE.ZPERSON = ?
                """,
                (person_pk,),
            ).fetchone()
            face_count = int(count_row[0] or 0) if count_row else 0
            if face_count == 0:
                continue

            day_rows = conn.execute(
                f"""
                SELECT DISTINCT date({asset_table}.ZDATECREATED + 978307200, 'unixepoch', 'localtime')
                FROM ZDETECTEDFACE
                JOIN {asset_table} ON ZDETECTEDFACE.ZASSET = {asset_table}.Z_PK
                WHERE ZDETECTEDFACE.ZPERSON = ?
                """,
                (person_pk,),
            ).fetchall()
            days = sorted(day.replace("-", "") for (day,) in day_rows if day)

            clusters.append(
                {
                    "person_pk": int(person_pk),
                    "name": name,
                    "face_count": face_count,
                    "days": days,
                }
            )

        return clusters
    finally:
        conn.close()
