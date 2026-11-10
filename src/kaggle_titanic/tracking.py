# src/kaggle_titanic/tracking.py
from omegaconf import DictConfig, OmegaConf

def flatten_config(cfg: DictConfig) -> dict:
    """
    вложенный конфиг omega становится в плоский словарь для входа млфлоу
    чтобы не онли артефакты трекать , но фильтровать через ui по фичам 
    впример {'model': {'C': 1.0}} становится {'model.C': 1.0}
    """
    
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) # py dict на выходе
    
    
    result = {}

    def _walk(current_dict: dict, parent_key: str = ""):
        
        for k, v in current_dict.items():
            if parent_key:
                new_key = f"{parent_key}.{k}"
            else:
                new_key = k
            
            if isinstance(v, dict):
                _walk(v, new_key)
            elif isinstance(v, list):
                result[new_key] = ",".join(map(str, v))
            else:
                result[new_key] = v

    _walk(cfg_dict)
    
    return result