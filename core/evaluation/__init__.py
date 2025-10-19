# -*- coding: utf-8 -*-
"""Funções de avaliação para IA."""

from .base_evaluator import BaseEvaluator
from .piece_count_evaluator import PieceCountEvaluator

__all__ = [
    'BaseEvaluator',
    'PieceCountEvaluator'
]