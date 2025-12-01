import os
import io
import subprocess
from pathlib import Path
import pytest
from PIL import Image
from api.worker import process_file, preprocess_image_for_ocr, convert_image_to_pdf

# python
# api/test_worker.py
# We recommend installing an extension to run python tests.



# Absolute imports of the functions to test (and module for patching STORAGE_DIR)
import api.worker as worker_module


def _create_sample_image(path: Path, size=(100, 50), color=(128, 128, 128)):
    img = Image.new("RGB", size, color=color)
    img.save(path, format="PNG")


def test_preprocess_image_for_ocr_creates_output(tmp_path):
    src = tmp_path / "in.png"
    dst = tmp_path / "out_preproc.png"
    _create_sample_image(src)
    # call the function
    preprocess_image_for_ocr(str(src), str(dst), dpi=150)
    # assertions
    assert dst.exists() and dst.stat().st_size > 0
    # ensure Pillow can open the output
    img = Image.open(dst)
    img.load()
    assert img.size[0] > 0 and img.size[1] > 0


def test_convert_image_to_pdf_creates_pdf(tmp_path):
    src = tmp_path / "in.png"
    out_pdf = tmp_path / "out.pdf"
    _create_sample_image(src)
    convert_image_to_pdf(str(src), str(out_pdf))
    assert out_pdf.exists() and out_pdf.stat().st_size > 0
    # quick check for PDF header
    with open(out_pdf, "rb") as f:
        header = f.read(4)
    assert header == b"%PDF"


def test_process_file_image_calls_ocrmypdf_and_cleans_temp(tmp_path, monkeypatch):
    # prepare storage dir and monkeypatch in module
    storage = tmp_path / "storage"
    storage.mkdir()
    monkeypatch.setattr(worker_module, "STORAGE_DIR", str(storage))

    # create an image in storage (process_file will use basename)
    input_name = "sample.png"
    input_path = storage / input_name
    _create_sample_image(input_path)

    output_name = "result.pdf"
    output_path = storage / output_name

    calls = []

    def fake_run(cmd, check, stdout, stderr):
        calls.append({"cmd": cmd, "check": check})
        # simulate successful run
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # run process_file using basenames (module will join with STORAGE_DIR)
    result = process_file(input_name, output_name)
    # result should be full path in STORAGE_DIR
    assert result == str(output_path)

    # subprocess.run should have been called once
    assert len(calls) == 1
    cmd = calls[0]["cmd"]
    # command should start with ocrmypdf and include output path
    assert "ocrmypdf" in cmd[0]
    assert str(output_path) == cmd[-1]
    # input argument passed to ocrmypdf should be a temp pdf (endswith _temp.pdf)
    assert cmd[-2].endswith("_temp.pdf")
    # ensure temp pdf removed
    temp_pdf = storage / (input_name + "_temp.pdf")
    assert not temp_pdf.exists()


def test_process_file_pdf_calls_ocrmypdf_directly(tmp_path, monkeypatch):
    storage = tmp_path / "storage2"
    storage.mkdir()
    monkeypatch.setattr(worker_module, "STORAGE_DIR", str(storage))

    input_name = "doc.pdf"
    input_path = storage / input_name
    # create a minimal valid PDF header to make mimetypes/pillow happy if needed
    with open(input_path, "wb") as f:
        f.write(b"%PDF-1.4\n%EOF\n")
    output_name = "out_doc.pdf"
    output_path = storage / output_name

    recorded = []

    def fake_run(cmd, check, stdout, stderr):
        recorded.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = process_file(input_name, output_name)
    assert result == str(output_path)
    assert len(recorded) == 1
    cmd = recorded[0]
    # input should be original pdf path
    assert str(input_path) in cmd
    assert str(output_path) in cmd


def test_process_file_missing_raises(tmp_path, monkeypatch):
    storage = tmp_path / "empty_storage"
    storage.mkdir()
    monkeypatch.setattr(worker_module, "STORAGE_DIR", str(storage))

    with pytest.raises(FileNotFoundError):
        process_file("does_not_exist.pdf", "out.pdf")