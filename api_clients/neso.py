import textwrap
from datetime import datetime
from io import StringIO
from typing import Any, cast

import pandas as pd

from api_clients.base import BaseClient
from api_clients.objects.dfs_resources import DFSResources
from utils.datetime_helpers import to_iso_z_ms


class NESOClient(BaseClient):
    def __init__(self, rate_limit: int):
        base_url = "https://api.neso.energy/"
        super().__init__(base_url, rate_limit)

        self.ckan_url = "api/3/action/"
        self.interconnector_requirements_table_id = "6a928369-bed3-445f-af8a-69cdb2cc5089"

    def get_dfs_resources(self) -> DFSResources:

        url = self.ckan_url + "datapackage_show?id=demand-flexibility"

        response = self._get(url)
        resources = cast("list[dict[str, Any]]", response.json()["result"]["resources"])

        resource_data: list[pd.DataFrame] = []
        for resource in resources:
            name = resource["name"]
            package_id = resource["package_id"]
            table_id = resource["id"]
            url = f"dataset/{package_id}/resource/{table_id}/download/{name}.csv"

            response = self._get(url)
            resource_data.append(pd.read_csv(StringIO(response.text)))

        if len(resource_data) != 4:
            err_msg = f"Expected 4 DFS resources, got {len(resource_data)}"
            raise ValueError(err_msg)

        return DFSResources(resource_data[0], resource_data[1], resource_data[2], resource_data[3])

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

        url = self.ckan_url + "datastore_search_sql"

        response = self._get(url, params=params)
        data = response.json()

        return pd.DataFrame(data["result"]["records"])

