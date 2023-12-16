import numpy as np
import pytest
from docsaidkit import Document, Polygon, Polygons


def create_sample_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def create_sample_polygon():
    return Polygon(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))


def test_document_creation():
    doc = Document()
    assert doc is not None


def test_doc_polygon_setter_getter():
    doc = Document()
    polygon = create_sample_polygon()
    doc.doc_polygon = polygon
    assert np.array_equal(doc.doc_polygon.numpy(), polygon.numpy())


def test_doc_polygon_invalid_value():
    doc = Document()
    with pytest.raises(ValueError):
        doc.doc_polygon = Polygon(np.array([[0, 0], [1, 0], [1, 1]]))


def test_has_doc_polygon():
    doc = Document()
    assert not doc.has_doc_polygon
    doc.doc_polygon = create_sample_polygon()
    assert doc.has_doc_polygon


def test_has_ocr_polygons():
    doc = Document()
    assert not doc.has_ocr_polygons
    doc.ocr_polygons = Polygons([create_sample_polygon()])
    assert doc.has_ocr_polygons


def test_has_ocr_texts():
    doc = Document()
    assert not doc.has_ocr_texts
    doc.ocr_texts = ["Test"]
    assert doc.has_ocr_texts


def test_gen_doc_flat_img_without_polygon():
    doc = Document(image=create_sample_image())
    assert np.array_equal(doc.gen_doc_flat_img(), doc.image)
