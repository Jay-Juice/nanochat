"""Small helpers shared by supervised fine-tuning entry points."""


def resolve_output_model_tag(model_tag, output_model_tag, depth):
    return output_model_tag or model_tag or f"d{depth}"
