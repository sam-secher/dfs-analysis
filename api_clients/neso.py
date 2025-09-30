import textwrap
from datetime import datetime

import pandas as pd

from api_clients.base import BaseClient
from utils.datetime_helpers import to_iso_z_ms


class NESOClient(BaseClient):
    def __init__(self):
        self.base_url = "https://api.neso.energy/"
        super().__init__(self.base_url)

        self.ckan_url = "api/3/action/datastore_search_sql"
        self.interconnector_requirements_table_id = "6a928369-bed3-445f-af8a-69cdb2cc5089"

    def get_interconnector_requirements(self, start_dt: datetime, end_dt: datetime, ascending: bool = True, limit: int = 100) -> pd.DataFrame:

        start_formatted = to_iso_z_ms(start_dt)
        end_formatted = to_iso_z_ms(end_dt)

        order_by = "ASC" if ascending else "DESC"

        sql_query = textwrap.dedent(f"""
            SELECT COUNT(*) OVER () AS _count, * FROM "{self.interconnector_requirements_table_id}"
            WHERE "Published DateTime" >= '{start_formatted}'
            AND "Published DateTime" <= '{end_formatted}'
            ORDER BY "_id" {order_by} LIMIT {limit}
        """).replace("\n", " ").strip() # noqa: S608

        params = {
            "sql": sql_query,
        }

        response = self._get(self.ckan_url, params=params)
        data = response.json()

        return pd.DataFrame(data["result"]["records"])

