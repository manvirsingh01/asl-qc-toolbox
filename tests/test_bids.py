"""Tests for BIDS metadata parsing."""

from asl_qc.bids import parse_asl_context, parse_asl_metadata, validate_metadata


def test_parse_asl_context_valid(aslcontext_tsv):
    ctx = parse_asl_context(aslcontext_tsv)
    assert ctx.n_volumes == 40
    assert ctx.n_controls == 20
    assert ctx.n_labels == 20
    assert not ctx.has_m0
    assert len(ctx.control_indices()) == 20
    assert len(ctx.label_indices()) == 20


def test_parse_asl_context_invalid(tmp_dir):
    bad = tmp_dir / "bad.tsv"
    bad.write_text("volume_type\ninvalid_type\n")
    try:
        parse_asl_context(bad)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unrecognised" in str(e)


def test_parse_asl_context_not_found(tmp_dir):
    try:
        parse_asl_context(tmp_dir / "nonexistent.tsv")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_parse_asl_metadata(asl_json):
    meta = parse_asl_metadata(asl_json)
    assert meta.labeling_type == "PCASL"
    assert meta.post_labeling_delay == 1.8
    assert meta.background_suppression is False
    assert meta.magnetic_field_strength == 3.0


def test_validate_metadata_complete(asl_json):
    meta = parse_asl_metadata(asl_json)
    missing = validate_metadata(meta)
    assert missing == []


def test_validate_metadata_missing(asl_json):
    import json
    data = json.loads(asl_json.read_text())
    del data["PostLabelingDelay"]
    asl_json.write_text(json.dumps(data))
    meta = parse_asl_metadata(asl_json)
    missing = validate_metadata(meta)
    assert "PostLabelingDelay" in missing
