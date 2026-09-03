"""Keep externally managed units intact when automatic refresh is disabled."""

from types import SimpleNamespace

import pytest
import yaml

import hermes_cli.config as config_cli
import hermes_cli.gateway as gateway_cli


@pytest.mark.parametrize(
    "config, expected_unit",
    [
        ({"gateway": {"auto_refresh_service": False}}, "managed unit\n"),
        ({"gateway": {"auto_refresh_service": True}}, "generated unit\n"),
        ({}, "generated unit\n"),
    ],
)
def test_gateway_boot_respects_service_refresh_setting(
    tmp_path, monkeypatch, config, expected_unit
):
    unit_path = tmp_path / "hermes-gateway.service"
    unit_path.write_text("managed unit\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(config_cli, "get_config_path", lambda: config_path)
    monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(
        gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path
    )
    monkeypatch.setattr(
        gateway_cli,
        "generate_systemd_unit",
        lambda **kwargs: "generated unit\n",
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

    async def fake_start_gateway(**kwargs):
        return True

    monkeypatch.setattr("gateway.run.start_gateway", fake_start_gateway)

    gateway_cli.run_gateway()

    assert unit_path.read_text(encoding="utf-8") == expected_unit
    assert (["systemctl", "--user", "daemon-reload"] in calls) == (
        expected_unit == "generated unit\n"
    )
