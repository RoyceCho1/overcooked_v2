# 평가 보조 유틸(예: 레시피 식별자 문자열 생성 등).
import os
from datetime import datetime
from pathlib import Path
from typing import List
import jax


def get_recipe_identifier(ingredients: List[int]) -> int:
    """
    Get the identifier for a recipe given the ingredients.
    """
    return f"{ingredients[0]}_{ingredients[1]}_{ingredients[2]}"
