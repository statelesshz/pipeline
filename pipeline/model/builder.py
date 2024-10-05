from typing import Dict

from .base import BaseModel


OPENMIND_MODEL_MAPPING: Dict[str, BaseModel] = {}

MODEL_ID_2_OPENMIND_MODEL_MAPPING: Dict[str, str] = {}

def register_model(openmind_model_name: str, base_model: BaseModel):
    if openmind_model_name in OPENMIND_MODEL_MAPPING:
        raise Exception("duplicate openmind_model_name {}".format(openmind_model_name))
    OPENMIND_MODEL_MAPPING[openmind_model_name] = base_model
    MODEL_ID_2_OPENMIND_MODEL_MAPPING[base_model.model_id_or_path] = openmind_model_name


def get_openmind_model_by_id(model_id_or_path:str):
    if model_id_or_path in MODEL_ID_2_OPENMIND_MODEL_MAPPING:
        return OPENMIND_MODEL_MAPPING[MODEL_ID_2_OPENMIND_MODEL_MAPPING[model_id_or_path]]
    else:
        common_model = OPENMIND_MODEL_MAPPING["common"]
        common_model.model_id_or_path = model_id_or_path
        return common_model
