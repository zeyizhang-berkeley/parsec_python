"""CPU-only checks that optional MPI tracing observes, rather than changes, exchange.

The helper and real wrapper bodies are loaded without importing a GPU backend.
Buffer doubles reject payload inspection and record existing transport operations.
"""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "Eigensolvers"


def load_trace():
    name = "parsec_mpi_trace_cpu_test"
    spec = importlib.util.spec_from_file_location(name, SOURCE / "mpi_trace.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Counts(list):
    def __mul__(self, scale):
        return Counts(value * scale for value in self)

    def astype(self, dtype):
        return self

    def sum(self):
        return sum(self)


class Buffer:
    def __init__(self, name, count, dtype, operations):
        self.name, self.size, self.dtype = name, count, dtype
        self.shape = (count,)
        self.ndim = 1
        self.itemsize = 16 if dtype == "complex128" else 8
        self.nbytes = self.size * self.itemsize
        self.flags = SimpleNamespace(c_contiguous=True, f_contiguous=True)
        self.device = SimpleNamespace(id=0)
        self.operations = operations

    def __array__(self, *args, **kwargs):
        raise AssertionError("Tracing must not inspect an orbital buffer")

    def __iter__(self):
        raise AssertionError("Tracing must not iterate an orbital buffer")

    def view(self, dtype):
        self.operations.append((self.name, "view", dtype))
        size = self.nbytes // (16 if dtype == "complex128" else 8)
        return Buffer(self.name + ".view", size, dtype, self.operations)

    def set(self, source):
        self.operations.append((self.name, "upload", source.name))

    def __setitem__(self, key, source):
        self.operations.append((self.name, "assign", source.name))


class ArrayNamespace:
    float64 = "float64"
    complex128 = "complex128"
    int64 = "int64"

    def __init__(self, operations):
        self.operations = operations

    def asarray(self, values, dtype=None):
        return Counts(values)

    def cumsum(self, values):
        result, total = [], 0
        for value in values:
            total += value
            result.append(total)
        return Counts(result)

    def concatenate(self, pieces):
        return Counts(item for piece in pieces for item in piece)

    def empty(self, count, dtype):
        self.operations.append(("allocate_host", count, dtype))
        return Buffer("host_receive", count, dtype, self.operations)

    def ascontiguousarray(self, array):
        self.operations.append(("contiguous_host", array.name))
        return array


def wrapper_body(filename, class_name, namespace):
    module = ast.parse((SOURCE / filename).read_text())
    group = next(node for node in module.body
                 if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in group.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_exchange")
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE / filename), "exec"), namespace)
    return namespace["_exchange"]


class MPITraceTests(unittest.TestCase):
    def setUp(self):
        self.trace = load_trace()

    def test_default_off_does_not_inspect_group_or_buffers(self):
        class Untouchable:
            def __getattribute__(self, name):
                raise AssertionError("Disabled tracing accessed metadata: " + name)

        item = Untouchable()
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertIsNone(self.trace.start_exchange(item, item, item, item, item))
        self.assertEqual(stream.getvalue(), "")

    def test_all_rank_metadata_counts_cache_and_sequence(self):
        operations = []
        group = SimpleNamespace(rank=1, size=2, device=0, transport="host",
                                comm=SimpleNamespace(rank=1, size=2))
        send = Buffer("send", 5, "float64", operations)
        receive = Buffer("receive", 5, "float64", operations)
        with patch.dict(os.environ, {"PARSEC_MPI_TRACE": "on", "PARSEC_DEBUG": "off",
                                     "PARSEC_DEBUG_RANKS": "leaders", "PMI_RANK": "17"}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                first = self.trace.start_exchange(group, send, [2, 3], receive, [3, 2], label="ritz_rows")
                first.emit("mpi_enter")
                first.emit("mpi_exit")
                second = self.trace.start_exchange(group, send, [2, 3], receive, [3, 2], label="ritz_rows")
                second.emit("mpi_enter")
        records = [json.loads(line.split(" ", 1)[1]) for line in output.getvalue().splitlines()]
        layouts = [record for record in records if record["stage"] == "layout"]
        self.assertEqual(len(layouts), 1)
        layout = layouts[0]
        self.assertEqual(layout["logical_send_counts"], [2, 3])
        self.assertEqual(layout["logical_receive_counts"], [3, 2])
        self.assertEqual(layout["send_counts"], [2, 3])
        self.assertEqual(layout["receive_counts"], [3, 2])
        self.assertEqual(layout["send_displacements"], [0, 2])
        self.assertEqual(layout["receive_displacements"], [0, 3])
        self.assertEqual((layout["send_bytes"], layout["receive_bytes"]), (40, 40))
        self.assertEqual(layout["send"]["shape"], [5])
        self.assertEqual(layout["send"]["dtype"], "float64")
        self.assertTrue(layout["counts_valid"])
        self.assertTrue(layout["send_bounds_valid"])
        self.assertTrue(layout["receive_bounds_valid"])
        entries = [record for record in records if record["stage"] == "mpi_enter"]
        self.assertEqual([record["group_sequence"] for record in entries], [1, 2])
        self.assertEqual(len({record["layout_signature"] for record in records}), 1)
        self.assertTrue(all(record["rank"] == 17 and record["local_rank"] == 1
                            and record["comm_size"] == 2 for record in records))
        self.assertTrue(all(record["time"] > 0 and record["monotonic_ns"] > 0 for record in records))
        self.assertTrue(all("send_counts" not in record for record in records
                            if record["stage"] != "layout"))
        event_order = [record["event_sequence"] for record in records]
        self.assertEqual(event_order, sorted(set(event_order)))
        self.assertEqual(operations, [])  # Metadata observation did not touch the buffers.

    def test_complex_counts_are_reported_in_logical_and_wire_units(self):
        operations = []
        group = SimpleNamespace(rank=0, size=2, device=0, transport="cuda",
                                comm=SimpleNamespace(rank=0, size=2))
        with patch.dict(os.environ, {"PARSEC_MPI_TRACE": "on"}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                trace = self.trace.start_exchange(group,
                    Buffer("send", 5, "complex128", operations), [2, 3],
                    Buffer("receive", 5, "complex128", operations), [3, 2], wire_multiplier=2)
                trace.emit("mpi_enter")
        records = [json.loads(line.split(" ", 1)[1]) for line in output.getvalue().splitlines()]
        layout = next(record for record in records if record["stage"] == "layout")
        self.assertEqual(layout["send"]["dtype"], "complex128")
        self.assertEqual(layout["wire_dtype"], "float64")
        self.assertEqual(layout["wire_itemsize"], 8)
        self.assertEqual(layout["wire_multiplier"], 2)
        self.assertEqual(layout["logical_send_counts"], [2, 3])
        self.assertEqual(layout["send_counts"], [4, 6])
        self.assertEqual(layout["receive_counts"], [6, 4])
        self.assertEqual(layout["send_displacements"], [0, 4])
        self.assertEqual(layout["receive_displacements"], [0, 6])
        self.assertEqual((layout["send_bytes"], layout["receive_bytes"]), (80, 80))
        self.assertTrue(layout["send_bounds_valid"] and layout["receive_bounds_valid"])
        self.assertEqual(operations, [])

    def test_invalid_bounds_are_observed_without_changing_execution(self):
        operations = []
        group = SimpleNamespace(rank=0, size=2, device=0, transport="host",
                                comm=SimpleNamespace(rank=0, size=2))
        with patch.dict(os.environ, {"PARSEC_MPI_TRACE": "on"}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                trace = self.trace.start_exchange(group,
                    Buffer("send", 5, "float64", operations), [2, 4],
                    Buffer("receive", 5, "float64", operations), [3, 2])
                trace.emit("mpi_enter")
        records = [json.loads(line.split(" ", 1)[1]) for line in output.getvalue().splitlines()]
        layout = next(record for record in records if record["stage"] == "layout")
        self.assertFalse(layout["send_bounds_valid"])
        self.assertTrue(layout["receive_bounds_valid"])
        self.assertEqual(operations, [])

    def test_metadata_failure_is_reported_once_without_changing_transport(self):
        off, _ = self.run_wrapper(False, "host", False)
        with patch.object(self.trace, "_buffer", side_effect=RuntimeError("private payload")):
            first, records = self.run_wrapper(False, "host", True)
            second, repeated = self.run_wrapper(False, "host", True)
        self.assertEqual(first, off)
        self.assertEqual(second, off)
        self.assertEqual(repeated, [])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["stage"], "trace_error")
        self.assertEqual(records[0]["error_type"], "RuntimeError")
        self.assertEqual(records[0]["rank"], 17)
        self.assertEqual(set(records[0]), {"stage", "rank", "time", "monotonic_ns", "error_type"})
        self.assertNotIn("private payload", json.dumps(records))

    def test_serialization_failure_is_reported_once_without_exception_payload(self):
        trace = self.trace.ExchangeTrace(dict(rank=17, unsupported=object()))
        with patch.dict(os.environ, {"PARSEC_MPI_TRACE": "on", "PMI_RANK": "17"}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                trace.emit("mpi_enter")
                trace.emit("mpi_exit")
        records = [json.loads(line.split(" ", 1)[1]) for line in output.getvalue().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["stage"], "trace_error")
        self.assertEqual(records[0]["error_type"], "TypeError")
        self.assertEqual(set(records[0]), {"stage", "rank", "time", "monotonic_ns", "error_type"})

    def test_unsupported_group_device_is_reported_but_does_not_disable_trace(self):
        class DeviceHandle:
            def __int__(self):
                raise AssertionError("Do not coerce an arbitrary device handle")

        operations = []
        group = SimpleNamespace(rank=1, size=2, device=DeviceHandle(), transport="host")
        send, receive = Buffer("send", 5, "float64", operations), Buffer("receive", 5, "float64", operations)
        with patch.dict(os.environ, {"PARSEC_MPI_TRACE": "on"}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                for _ in range(2):
                    trace = self.trace.start_exchange(group, send, [2, 3], receive, [3, 2])
                    trace.emit("mpi_enter")
        records = [json.loads(line.split(" ", 1)[1]) for line in output.getvalue().splitlines()]
        errors = [record for record in records if record["stage"] == "trace_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error_type"], "TypeError")
        stages = [record for record in records if record["stage"] != "trace_error"]
        self.assertEqual([record["stage"] for record in stages], ["layout", "mpi_enter", "mpi_enter"])
        self.assertTrue(all(record["group_device_id"] is None for record in stages))
        self.assertEqual(operations, [])

    def run_wrapper(self, complex_values, transport, tracing):
        operations = []
        np = ArrayNamespace(operations)
        cp = SimpleNamespace(float64="float64")

        def host(array):
            operations.append(("download", array.name))
            return Buffer("host_send", array.size, array.dtype, operations)

        cp.asnumpy = host
        group = SimpleNamespace(rank=1, size=2, device=0, transport=transport,
            _mpi=SimpleNamespace(DOUBLE="DOUBLE"), mpi=SimpleNamespace(DOUBLE="DOUBLE"),
            xp=cp, sent_bytes=0, _host=host,
            _sync=lambda: operations.append(("sync",)))

        def collective(send_spec, receive_spec):
            operations.append(("Alltoallv", tuple(send_spec[1][0]), tuple(receive_spec[1][0])))

        group.comm = SimpleNamespace(rank=1, size=2, Alltoallv=collective)

        def specs(send, send_counts, receive, receive_counts):
            operations.append(("specs",))
            return [send, (send_counts, [0, send_counts[0]]), "DOUBLE"], [
                receive, (receive_counts, [0, receive_counts[0]]), "DOUBLE"]

        group._specs = specs
        dtype = "complex128" if complex_values else "float64"
        send, receive = Buffer("send", 5, dtype, operations), Buffer("receive", 5, dtype, operations)
        namespace = dict(np=np, require_cupy=lambda: (cp, None),
                         start_exchange=self.trace.start_exchange)
        filename, name = ("chebff_columns.py", "ColumnGroup") if complex_values else (
            "mpi_shared_basis.py", "MPIShardGroup")
        function = wrapper_body(filename, name, namespace)
        with patch.dict(os.environ, {"PARSEC_MPI_TRACE": "on" if tracing else "off",
                                     "PARSEC_DEBUG": "off", "PMI_RANK": "17"}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                function(group, send, [2, 3], receive, [3, 2])
        records = [json.loads(line.split(" ", 1)[1]) for line in output.getvalue().splitlines()
                   if line.startswith("PARSEC_MPI_TRACE ")]
        return operations, records

    def test_tracing_preserves_real_and_complex_transport_operation_order(self):
        for complex_values in (False, True):
            for transport in ("host", "cuda"):
                with self.subTest(complex_values=complex_values, transport=transport):
                    off, no_records = self.run_wrapper(complex_values, transport, False)
                    on, records = self.run_wrapper(complex_values, transport, True)
                    self.assertEqual(on, off)
                    self.assertEqual(no_records, [])
                    self.assertTrue(records)
                    self.assertEqual(sum(op[0] == "Alltoallv" for op in on), 1)
                    self.assertEqual(sum(op[0] == "sync" for op in on),
                                     2 if transport == "host" and not complex_values else 1)
                    self.assertTrue(all(row["rank"] == 17 for row in records))
                    expected = ["layout", "device_sync_enter", "device_sync_done"]
                    if transport == "host":
                        expected += ["host_stage_enter", "host_stage_done"]
                    expected += ["mpi_enter", "mpi_exit"]
                    if transport == "host":
                        expected += ["upload_enter", "upload_done"]
                        if not complex_values:
                            expected += ["device_sync_enter", "device_sync_done"]
                    self.assertEqual([row["stage"] for row in records], expected)
                    if transport == "host":
                        upload = next(row for row in records if row["stage"] == "upload_done")
                        self.assertEqual(upload["completion"], "enqueued")
                    if transport == "host" and not complex_values:
                        self.assertEqual([row["phase"] for row in records[-2:]],
                                         ["post_upload", "post_upload"])


if __name__ == "__main__":
    unittest.main()
