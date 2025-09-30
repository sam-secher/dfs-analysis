import pandas as pd


class DFSResources:
    def __init__(self, utilisation_report_summary: pd.DataFrame, utilisation_report: pd.DataFrame,
        service_requirement: pd.DataFrame, industry_notification: pd.DataFrame) -> None:

        self.utilisation_report_summary = utilisation_report_summary
        self.utilisation_report = utilisation_report
        self.service_requirement = service_requirement
        self.industry_notification = industry_notification
