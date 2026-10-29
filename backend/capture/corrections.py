"""Corrections: edits a user (or the review step) makes to the editable model, stored in report.json under "corrections" and applied by
model.rebuild(model, answers, corrections) with no model call.

Adapter hook: from_sentence() is where Muse Spark will turn a sentence such as "this part is flat like the bottom", said about one review
item, into exactly one correction. It takes the model call as an argument (`ask`, a function returning JSON text); nothing here calls a
model. Whatever comes back is validated against SCHEMA before it can touch the model.
"""
import json

REGION = {"type": "object", "required": ["polygon_xy"], "additionalProperties": False, "properties": {
    "polygon_xy": {"type": "array", "minItems": 3, "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}},
    "z_mm": {"type": ["array", "null"], "minItems": 2, "maxItems": 2, "items": {"type": "number"}}}}
PLANE = {"oneOf": [{"type": "string"}, {"type": "object", "required": ["n", "d"], "properties": {
    "n": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}}, "d": {"type": "number"}}}]}
SHAPES = ["box", "rounded", "hill"]
COMMON = {"op": {"type": "string"}, "reason": {"type": "string"}, "source": {"type": "string"}, "review_item": {"type": "string"}}

SCHEMA = {"oneOf": [
    {"type": "object", "required": ["op", "region", "to_plane"], "properties": {**COMMON, "op": {"const": "flatten_region"}, "region": REGION,
                                                                             "to_plane": PLANE, "thickness_mm": {"type": ["number", "null"], "exclusiveMinimum": 0},
                                                                             "thickness_answer": {"type": "string"}, "feature_id": {"type": "string"}}},
    {"type": "object", "required": ["op", "region"], "properties": {**COMMON, "op": {"const": "remove_region"}, "region": REGION}},
    {"type": "object", "required": ["op", "feature_id"], "properties": {**COMMON, "op": {"const": "move_feature_to_opposite_face"}, "feature_id": {"type": "string"}}},
    {"type": "object", "required": ["op", "feature_id", "shape"], "properties": {**COMMON, "op": {"const": "set_feature_shape"}, "feature_id": {"type": "string"},
                                                                              "shape": {"enum": SHAPES}, "top_radius_mm": {"type": ["number", "null"], "exclusiveMinimum": 0}}},
    {"type": "object", "required": ["op", "feature_id", "feature"], "properties": {**COMMON, "op": {"const": "add_feature"}, "feature_id": {"type": "string"},
                                                                                "feature": {"type": "object"}}},
]}
OPS = ("flatten_region", "remove_region", "move_feature_to_opposite_face", "set_feature_shape", "add_feature")


def validate(c, feature_ids=(), plane_names=()):
    """Raise ValueError unless c is one well-formed correction that names existing features / planes."""
    import jsonschema
    try:
        jsonschema.validate(c, SCHEMA)
    except jsonschema.ValidationError as e:
        raise ValueError(f"invalid correction: {e.message}") from None
    if c["op"] in ("move_feature_to_opposite_face", "set_feature_shape") and c["feature_id"] not in feature_ids:
        raise ValueError(f"unknown feature {c['feature_id']!r}")
    if c["op"] == "flatten_region" and isinstance(c["to_plane"], str) and c["to_plane"] not in plane_names:
        raise ValueError(f"unknown plane {c['to_plane']!r}")
    return c


PROMPT = ("You turn one user sentence about one flagged region of a 3D part model into exactly one JSON correction that matches this "
          "JSON schema:\n{schema}\nThe region the user is looking at: {item}\nNamed planes: {planes}. Features: {features}.\n"
          "User sentence: {sentence!r}\nAnswer with the JSON object only. If the sentence does not ask for one of these edits, answer {{\"op\": \"none\"}}.")


def from_sentence(sentence, item, feature_ids, plane_names, ask):
    """One correction (validated) from a user sentence about one review item, or None when the sentence asks for no edit. `ask(prompt)` is
    the language-model call (Muse Spark in the app), injected by the caller."""
    text = ask(PROMPT.format(schema=json.dumps(SCHEMA), item=json.dumps(item), planes=list(plane_names), features=list(feature_ids), sentence=sentence))
    c = json.loads(text)
    if c.get("op") == "none":
        return None
    c.setdefault("source", "user sentence")
    c.setdefault("review_item", item.get("id"))
    return validate(c, feature_ids, plane_names)
