from __future__ import annotations

import json
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from avito.constants import (
    HTTP_CONTENT_TYPE_JSON,
    HTTP_METHOD_POST,
    PIPELINE_API_CLIENT_DEFAULT_TIMEOUT_SEC,
    SHOULD_SPLIT_API_INFER_PATH,
)
from avito.should_split.api.schemas import AvitoApiRequest, AvitoApiResponse


class PipelineApiClientError(Exception):
    """Ошибка обращения к серверу пайплайна."""


class PipelineApiClient:
    """HTTP-клиент для shouldSplit FastAPI сервера."""

    def __init__(
        self,
        base_url: str,
        timeout_sec: int = PIPELINE_API_CLIENT_DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def infer(
        self,
        item_id: int | str,
        mc_id: int,
        mc_title: str,
        description: str,
    ) -> AvitoApiResponse:
        payload = AvitoApiRequest(
            itemId=item_id,
            mcId=mc_id,
            mcTitle=mc_title,
            description=description,
        )
        data = payload.model_dump_json().encode("utf-8")

        req = urllib_request.Request(
            url=f"{self.base_url}{SHOULD_SPLIT_API_INFER_PATH}",
            data=data,
            headers={"Content-Type": HTTP_CONTENT_TYPE_JSON},
            method=HTTP_METHOD_POST,
        )

        try:
            with urllib_request.urlopen(req, timeout=self.timeout_sec) as response:
                body = response.read().decode("utf-8")
        except HTTPError as error:
            raise PipelineApiClientError(f"HTTP error: {error.code}") from error
        except URLError as error:
            raise PipelineApiClientError(f"Connection error: {error.reason}") from error
        except Exception as error:
            raise PipelineApiClientError(f"Request failed: {error}") from error

        try:
            parsed = json.loads(body)
            return AvitoApiResponse.model_validate(parsed)
        except Exception as error:
            raise PipelineApiClientError(f"Invalid response: {error}") from error

