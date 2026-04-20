"""Tests for the Atropos HTTP client."""

from unittest.mock import MagicMock, patch

import pytest
import requests
from recipe.atropos.atropos_client import AtroposClient


class TestAtroposClient:
    def test_init_strips_trailing_slash(self):
        client = AtroposClient(api_url="http://localhost:8000/")
        assert client.api_url == "http://localhost:8000"

    @patch("recipe.atropos.atropos_client.requests")
    def test_register_trainer(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"uuid": "test-uuid"}
        mock_resp.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_resp

        client = AtroposClient(api_url="http://localhost:8000")
        result = client.register_trainer(
            batch_size=8,
            max_token_len=2048,
            num_steps=30,
            checkpoint_dir="/tmp/ckpt",
        )

        mock_requests.post.assert_called_once()
        call_args = mock_requests.post.call_args
        assert "/register" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["batch_size"] == 8
        assert payload["num_steps"] == 30
        assert result == {"uuid": "test-uuid"}

    @patch("recipe.atropos.atropos_client.requests")
    def test_get_batch_returns_data(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"batch": [{"tokens": [[1, 2, 3]]}]}
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        client = AtroposClient(api_url="http://localhost:8000")
        result = client.get_batch()

        assert result == [{"tokens": [[1, 2, 3]]}]

    @patch("recipe.atropos.atropos_client.requests")
    def test_get_batch_returns_none_when_empty(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"batch": None}
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        client = AtroposClient(api_url="http://localhost:8000")
        result = client.get_batch()

        assert result is None

    @patch("recipe.atropos.atropos_client.requests")
    def test_retry_on_connection_error(self, mock_requests):
        mock_requests.ConnectionError = requests.ConnectionError
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"batch": []}
        mock_resp.raise_for_status = MagicMock()
        # fail twice, succeed on third
        mock_requests.get.side_effect = [
            requests.ConnectionError("refused"),
            requests.ConnectionError("refused"),
            mock_resp,
        ]

        client = AtroposClient(api_url="http://localhost:8000", max_retries=3, retry_delay=0.01)
        result = client.get_batch()

        assert result == []
        assert mock_requests.get.call_count == 3

    @patch("recipe.atropos.atropos_client.requests")
    def test_retry_exhausted_raises(self, mock_requests):
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.get.side_effect = requests.ConnectionError("refused")

        client = AtroposClient(api_url="http://localhost:8000", max_retries=2, retry_delay=0.01)

        with pytest.raises(requests.ConnectionError):
            client.get_batch()

    @patch("recipe.atropos.atropos_client.requests")
    def test_poll_batch_timeout(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"batch": None}
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        client = AtroposClient(api_url="http://localhost:8000")

        with pytest.raises(TimeoutError, match="No batch received"):
            client.poll_batch(timeout=0.1, poll_interval=0.05)

    @patch("recipe.atropos.atropos_client.requests")
    def test_http_error_4xx_raises_immediately(self, mock_requests):
        """4xx errors should not be retried."""
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.HTTPError = requests.HTTPError
        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.text = "Unprocessable Entity"
        mock_resp.raise_for_status.side_effect = requests.HTTPError(response=mock_resp)
        mock_requests.get.return_value = mock_resp

        client = AtroposClient(api_url="http://localhost:8000", max_retries=3, retry_delay=0.01)

        with pytest.raises(requests.HTTPError, match="422"):
            client.get_batch()
        assert mock_requests.get.call_count == 1  # no retry

    @patch("recipe.atropos.atropos_client.requests")
    def test_http_error_5xx_retries(self, mock_requests):
        """5xx errors should be retried with backoff."""
        mock_requests.HTTPError = requests.HTTPError
        mock_requests.ConnectionError = requests.ConnectionError

        mock_resp_500 = MagicMock()
        mock_resp_500.status_code = 500
        mock_resp_500.text = "Internal Server Error"
        mock_resp_500.raise_for_status.side_effect = requests.HTTPError(response=mock_resp_500)

        mock_resp_ok = MagicMock()
        mock_resp_ok.json.return_value = {"batch": []}
        mock_resp_ok.raise_for_status = MagicMock()

        mock_requests.get.side_effect = [mock_resp_500, mock_resp_ok]

        client = AtroposClient(api_url="http://localhost:8000", max_retries=3, retry_delay=0.01)
        result = client.get_batch()

        assert result == []
        assert mock_requests.get.call_count == 2

    @patch("recipe.atropos.atropos_client.requests")
    def test_http_error_no_response_object(self, mock_requests):
        """HTTPError with response=None should not crash."""
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.HTTPError = requests.HTTPError
        err = requests.HTTPError("connection reset")
        err.response = None
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = err
        mock_requests.get.return_value = mock_resp

        client = AtroposClient(api_url="http://localhost:8000", retry_delay=0.01)

        with pytest.raises(requests.HTTPError, match="connection reset"):
            client.get_batch()

    @patch("recipe.atropos.atropos_client.requests")
    def test_poll_batch_returns_single_batch(self, mock_requests):
        """poll_batch should return exactly one server-side batch."""
        batch1 = [{"tokens": [[1]]}, {"tokens": [[2]]}]

        mock_resp_1 = MagicMock()
        mock_resp_1.json.return_value = {"batch": batch1}
        mock_resp_1.raise_for_status = MagicMock()

        mock_requests.get.return_value = mock_resp_1

        client = AtroposClient(api_url="http://localhost:8000")
        result = client.poll_batch(timeout=5)

        assert result == batch1
        assert mock_requests.get.call_count == 1
