"""Real Markdown parser contracts; no PDF renderer or model calls."""

import tempfile
import unittest
from xml.etree import ElementTree

import markdown

from models.document import Document
from processors.format_converter import FormatConverterProcessor, MathExtension


class FormatConverterTests(unittest.TestCase):
    def test_inline_math_uses_supported_element_api(self):
        html = markdown.markdown("Before $x + y$ after.", extensions=[MathExtension()])
        paragraph = ElementTree.fromstring(html)
        self.assertEqual(paragraph.tag, "p")
        span = paragraph.find("span")
        self.assertIsNotNone(span)
        self.assertEqual(span.attrib, {"class": "math"})
        self.assertEqual(span.text, "x + y")
        self.assertEqual(span.tail, " after.")

    def test_block_math_is_outside_paragraph_and_preserves_multiline(self):
        html = markdown.markdown("Before.\n\n$$\nx^2 +\ny^2\n$$\n\nAfter.", extensions=[MathExtension()])
        root = ElementTree.fromstring("<root>" + html + "</root>")
        self.assertEqual([node.tag for node in root], ["p", "div", "p"])
        self.assertEqual(root[1].attrib, {"class": "math-block"})
        self.assertEqual(root[1].text, "x^2 +\ny^2")

    def test_unclosed_math_and_code_are_preserved(self):
        html = markdown.markdown("Open $formula and `$literal$`.", extensions=[MathExtension()])
        self.assertIn("Open $formula", html)
        self.assertIn("<code>$literal$</code>", html)
        self.assertNotIn('class="math"', html)

    def test_content_rejects_nontext_and_markdown_entry_works(self):
        with tempfile.TemporaryDirectory() as directory:
            processor = FormatConverterProcessor({"output_dir": directory, "create_pdf": False})
            document = Document("fixture.pdf")
            document.store_content("ocr", {"text": ["invalid text"]})
            self.assertFalse(processor.process(document).is_successful())
            document.store_content("ocr", {"text": "Fixture title\n\nA paragraph."})
            result = processor.process(document)
            self.assertTrue(result.is_successful(), result.message)
