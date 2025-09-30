import logging
from abc import ABC
from typing import Any, Optional
from urllib import parse

import requests


class BaseClient(ABC):
    def __init__(self, base_url: str, rate_limit: int) -> None:
        self.base_url = base_url
        self.request_delay = 1 / rate_limit # delay between requests in seconds
        self.logger = logging.getLogger(__name__)

    def _get(self, url: str, params: Optional[dict[str, Any]] = None, timeout: int = 30) -> requests.Response:
        """Perform HTTP GET request with exception handling.

        Args:
            `url`: The URL to make the request to
            `params`: Optional query parameters
            `timeout`: Request timeout in seconds (default: 30)

        Returns:
            `requests.Response`: The response object

        Raises:
            `requests.exceptions.Timeout`: When request times out
            `requests.exceptions.ConnectionError`: When connection fails
            `requests.exceptions.HTTPError`: When HTTP error occurs (4xx, 5xx)
            `requests.exceptions.RequestException`: For other request-related errors

        """
        url = f"{self.base_url}{url}"

        try:
            self.logger.debug(f"Making GET request to: {url}")
            kwargs = { "url": url }
            if params:
                self.logger.debug(f"Request params: {params}")
                params_encoded = parse.urlencode(params)
                kwargs["params"] = params_encoded

            response = requests.get(**kwargs, timeout=timeout) # type: ignore[arg-type]
        except requests.exceptions.RequestException:
            self.logger.exception(f"Request exception for URL: {url}")
            raise
        except Exception as e:
            self.logger.exception(f"Unexpected error for URL: {url}")
            err_msg = f"Unexpected error: {e!s}"
            raise requests.exceptions.RequestException(err_msg) from e
        else:
            response.raise_for_status()
            self.logger.debug(f"Request successful: {response.status_code}")
            return response
