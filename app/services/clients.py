from __future__ import annotations

from typing import Any, TypeVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ValidationError

from app.models import (
    CameraCaptureResponse,
    DefectDetectionResponse,
    FruitDetectionResponse,
    ScanResult,
)

T = TypeVar("T", bound=BaseModel)


class ServiceCallError(RuntimeError):
    def __init__(self, service: str, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.service = service
        self.status_code = status_code


class ServiceValidationError(RuntimeError):
    def __init__(self, service: str, message: str) -> None:
        super().__init__(message)
        self.service = service


class BaseHttpClient:
    def __init__(self, *, service_name: str, base_url: str, timeout_seconds: float, retries: int) -> None:
        self._service_name = service_name
        self._retries = max(retries, 0)
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def _request_with_retries(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                response = await self._client.request(method, url, **kwargs)
                if response.status_code >= 500 and attempt < self._retries:
                    continue
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code
                if status >= 500 and attempt < self._retries:
                    continue
                raise ServiceCallError(
                    self._service_name,
                    f"{self._service_name} returned HTTP {status}",
                    status_code=status,
                ) from exc
            except httpx.RequestError as exc:
                last_error = exc
                if attempt < self._retries:
                    continue
                raise ServiceCallError(
                    self._service_name,
                    f"{self._service_name} request failed: {exc}",
                ) from exc

        raise ServiceCallError(self._service_name, f"{self._service_name} failed after retries: {last_error}")

    def _validate(self, model_type: type[T], payload: Any) -> T:
        try:
            return model_type.model_validate(payload)
        except ValidationError as exc:
            raise ServiceValidationError(
                self._service_name,
                f"Invalid response schema from {self._service_name}: {exc}",
            ) from exc


class CameraServiceClient(BaseHttpClient):
    async def capture(self) -> CameraCaptureResponse:
        response = await self._request_with_retries("POST", "/capture", json={})
        return self._validate(CameraCaptureResponse, response.json())

    async def fetch_image_bytes(self, image_url_or_path: str) -> bytes:
        parsed = urlparse(image_url_or_path)
        if parsed.scheme and parsed.netloc:
            response = await self._request_with_retries("GET", image_url_or_path)
        else:
            normalized_path = f"/{image_url_or_path.lstrip('/')}"
            response = await self._request_with_retries("GET", normalized_path)
        return response.content


class FruitDetectorClient(BaseHttpClient):
    def __init__(
        self,
        *,
        service_name: str,
        base_url: str,
        timeout_seconds: float,
        retries: int,
        detect_path: str = "/detect-fruits",
    ) -> None:
        super().__init__(
            service_name=service_name,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
        self._detect_path = detect_path

    async def detect(self, *, image_bytes: bytes, imgsz: int) -> FruitDetectionResponse:
        files = {
            "file": ("scan.jpg", image_bytes, "image/jpeg"),
        }
        response = await self._request_with_retries(
            "POST",
            self._detect_path,
            params={"imgsz": imgsz},
            files=files,
        )
        return self._validate(FruitDetectionResponse, response.json())


class DefectDetectorClient(BaseHttpClient):
    async def detect(
        self,
        *,
        image_bytes: bytes,
        image_id: str,
        fruit_id: str,
    ) -> DefectDetectionResponse:
        files = {
            "image": ("crop.jpg", image_bytes, "image/jpeg"),
        }
        data = {
            "image_id": image_id,
            "fruit_id": fruit_id,
        }
        response = await self._request_with_retries("POST", "/detect-defects", files=files, data=data)
        return self._validate(DefectDetectionResponse, response.json())


class PublisherClient(BaseHttpClient):
    async def publish(self, *, payload: ScanResult, path: str) -> None:
        await self._request_with_retries(
            "POST",
            path,
            json=payload.model_dump(mode="json"),
        )
