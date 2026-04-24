"""Unit tests for the Snowflake connector."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import snowflake.sqlalchemy.custom_types as sct
import sqlalchemy as sa

from target_snowflake.connector import SnowflakeConnector, SnowflakeTimestampType
from target_snowflake.sinks import SnowflakeSink
from target_snowflake.snowflake_types import NUMBER, VARIANT
from target_snowflake.target import TargetSnowflake

MINIMAL_CONFIG = {"user": "u", "account": "a", "database": "db"}


def make_sink(config: dict) -> SnowflakeSink:
    target = MagicMock()
    target.name = "test"
    target.config = {**MINIMAL_CONFIG, **config}
    target.logger = MagicMock()
    return SnowflakeSink(
        target=target,
        stream_name="test_stream",
        schema={"properties": {"id": {"type": "integer"}}},
        key_properties=["id"],
    )


@pytest.fixture
def connector():
    return SnowflakeConnector()


@pytest.mark.parametrize(
    ("schema", "expected_type"),
    [
        pytest.param({"type": "object"}, VARIANT, id="object"),
        pytest.param({"type": ["array", "null"]}, VARIANT, id="array"),
        pytest.param({"type": ["array", "object", "string"]}, VARIANT, id="array_object_string"),
        pytest.param({"type": ["integer", "null"]}, NUMBER, id="integer"),
        pytest.param({"type": ["number", "null"]}, sct.DOUBLE, id="number"),
        pytest.param({"type": ["string", "null"], "format": "date-time"}, sct.TIMESTAMP_NTZ, id="date-time"),
        # Upstream types
        pytest.param({"type": ["string", "null"]}, sa.types.VARCHAR, id="string"),
        pytest.param({"type": ["boolean", "null"]}, sa.types.BOOLEAN, id="boolean"),
        pytest.param({"type": "string", "format": "time"}, sa.types.TIME, id="time"),
        pytest.param({"type": "string", "format": "date"}, sa.types.DATE, id="date"),
        pytest.param({"type": "string", "format": "uuid"}, sa.types.UUID, id="uuid"),
    ],
)
def test_jsonschema_to_sql(connector: SnowflakeConnector, schema: dict, expected_type: type[sa.types.TypeEngine]):
    sql_type = connector.to_sql_type(schema)
    assert isinstance(sql_type, expected_type)


@pytest.mark.parametrize(
    ("config", "expected_type"),
    [
        ({"timestamp_type": SnowflakeTimestampType.TIMESTAMP_TZ}, sct.TIMESTAMP_TZ),
        ({"timestamp_type": SnowflakeTimestampType.TIMESTAMP_LTZ}, sct.TIMESTAMP_LTZ),
        ({"timestamp_type": SnowflakeTimestampType.TIMESTAMP_NTZ}, sct.TIMESTAMP_NTZ),
    ],
)
def test_datetime_to_sql(connector: SnowflakeConnector, config: dict, expected_type: type[sa.types.TypeEngine]):
    connector.config.update(config)
    schema = {"type": ["string", "null"], "format": "date-time"}
    sql_type = connector.to_sql_type(schema)
    assert isinstance(sql_type, expected_type)


def test_to_sql_type_with_max_varchar_length(connector: SnowflakeConnector):
    sql_type = connector.to_sql_type({"type": "string", "maxLength": 1_000_000})
    assert isinstance(sql_type, sa.types.VARCHAR)
    assert sql_type.length == 1_000_000

    sql_type = connector.to_sql_type({"type": "string", "maxLength": SnowflakeConnector.max_varchar_length + 1})
    assert isinstance(sql_type, sa.types.VARCHAR)
    assert sql_type.length == SnowflakeConnector.max_varchar_length


def test_email_format(connector: SnowflakeConnector):
    sql_type = connector.to_sql_type({"type": "string", "format": "email"})
    assert isinstance(sql_type, sa.types.VARCHAR)
    assert sql_type.length == 254


def test_uri_format(connector: SnowflakeConnector):
    sql_type = connector.to_sql_type({"type": "string", "format": "uri"})
    assert isinstance(sql_type, sa.types.VARCHAR)
    assert sql_type.length == 2083


def test_hostname_format(connector: SnowflakeConnector):
    sql_type = connector.to_sql_type({"type": "string", "format": "hostname"})
    assert isinstance(sql_type, sa.types.VARCHAR)
    assert sql_type.length == 253


def test_ipv4_format(connector: SnowflakeConnector):
    sql_type = connector.to_sql_type({"type": "string", "format": "ipv4"})
    assert isinstance(sql_type, sa.types.VARCHAR)
    assert sql_type.length == 15


def test_ipv6_format(connector: SnowflakeConnector):
    sql_type = connector.to_sql_type({"type": "string", "format": "ipv6"})
    assert isinstance(sql_type, sa.types.VARCHAR)
    assert sql_type.length == 45


def test_batch_timeout_default():
    target = TargetSnowflake(config=MINIMAL_CONFIG)
    assert target._MAX_RECORD_AGE_IN_MINUTES == 5.0


def test_batch_timeout_custom():
    target = TargetSnowflake(config={**MINIMAL_CONFIG, "batch_timeout_minutes": 30})
    assert target._MAX_RECORD_AGE_IN_MINUTES == 30


def test_batch_size_bytes_not_full_when_unset():
    sink = make_sink({})
    context: dict = {}
    sink.start_batch(context)
    for i in range(5):
        sink.process_record({"id": i, "data": "x" * 100}, context)
    assert not sink.is_full


def test_batch_size_bytes_not_full_below_limit():
    sink = make_sink({"batch_size_bytes": 10_000})
    context: dict = {}
    sink.start_batch(context)
    sink.process_record({"id": 1}, context)
    assert sink._batch_bytes < 10_000
    assert not sink.is_full


def test_batch_size_bytes_full_at_limit():
    # serialize_json produces 8 bytes per {"id": N} record; 6 records = 48 bytes
    sink = make_sink({"batch_size_bytes": 40})
    context: dict = {}
    sink.start_batch(context)
    for i in range(6):
        sink.process_record({"id": i}, context)
    assert sink._batch_bytes >= 40
    assert sink.is_full


def test_batch_size_bytes_resets_on_new_batch():
    sink = make_sink({"batch_size_bytes": 10_000})
    context: dict = {}
    sink.start_batch(context)
    sink.process_record({"id": 1, "data": "x" * 200}, context)
    assert sink._batch_bytes > 0
    sink.start_batch({})
    assert sink._batch_bytes == 0


def test_singer_decimal(connector: SnowflakeConnector):
    sql_type = connector.to_sql_type(
        {
            "type": "string",
            "format": "x-singer.decimal",
            "precision": 38,
            "scale": 18,
        },
    )
    assert isinstance(sql_type, sa.types.DECIMAL)
    assert sql_type.precision == 38
    assert sql_type.scale == 18
