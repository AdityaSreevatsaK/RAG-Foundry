"""Offline tests for the document preflight and local RAG utilities."""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paths import DATA_DIRECTORY, bundled_document
from utils import setup_logger, track_time


class BundledDocumentTests(unittest.TestCase):
    def test_text_sample_is_found_from_any_working_directory(self):
        expected = (ROOT / "data" / "Data Science, Machine Learning and Artificial Intelligence.txt").resolve()
        self.assertTrue(expected.is_file())
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            try:
                os.chdir(temp)
                actual = bundled_document(expected.name)
            finally:
                os.chdir(original)
        self.assertEqual(expected, actual.resolve())

    def test_pdf_sample_is_found(self):
        pdf = bundled_document("United Nations Universal Declaration of Human Rights 1948.pdf")
        self.assertTrue(pdf.is_file())
        self.assertEqual(DATA_DIRECTORY, pdf.parent)

    def test_missing_file_can_be_detected_without_model_download(self):
        missing = bundled_document("definitely-not-a-real-fixture.pdf")
        self.assertFalse(missing.is_file())

    def test_rejects_directory_traversal_and_absolute_paths(self):
        for value in ("", "../private.txt", "/etc/passwd", "subfolder/example.pdf", "subfolder\\example.pdf"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    bundled_document(value)


class LoggingTests(unittest.TestCase):
    def test_logger_setup_does_not_duplicate_own_console_handler(self):
        name = "rag-foundry-test-logger"
        logger = logging.getLogger(name)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        try:
            a = setup_logger(name)
            b = setup_logger(name)
            self.assertIs(a, b)
            self.assertEqual(1, len(a.handlers))
        finally:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()

    def test_timing_decorator_preserves_function_result_and_metadata(self):
        @track_time
        def add(a, b):
            """Add two values."""
            return a + b
        self.assertEqual(7, add(3, 4))
        self.assertEqual("add", add.__name__)
        self.assertEqual("Add two values.", add.__doc__)

    def test_timing_decorator_does_not_suppress_errors(self):
        @track_time
        def boom():
            raise RuntimeError("deliberate")
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            boom()


if __name__ == "__main__":
    unittest.main()
