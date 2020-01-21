# --- built in ---
import os
import abc
import sys
import time
import logging

from typeing import Any, Union, Tuple

# --- 3rd party ---
import numpy as np

# --- my module ---
from cyaegha import logger

from cyaegha.common.route import Route
from cyaegha.common.utils import ParameterPack
from cyaegha.common.utils import counter

__all__ = [
    'BaseObject',
    'BaseGraph'
]


class BaseObject():


class BaseTrace(BaseObject):



graph[1, 1].add_trace()

class BaseSubplotHandler():
    @classmethod
    def add_func(cls, args):


class BaseSubplotHandler():
    def __init__(self, **kwargs):



class BaseGraph():
    # === built-in methods ===
    def __init__(self, rows: int=1, cols: int=1, **kwargs):
        '''
        Create Graph

        rows: (int)
        cols: (int)
        '''

        self._subplot = (rows, cols)
        self._kwargs = ParameterPack(**kwargs)


        self._subplot = (rows, cols)
        self._kwargs = ParameterPack(**kwargs)

        self._index_to_rowcol = {}

        # create a counter starting from 0, steping 1
        count = counter()

        # indexing subplot
        for r in range(self._subplot[0]):
            for c in range(self._subplot[1]):
                if 'specs' in self._kwargs.keys():
                    if self._kwargs.specs[r][c] is not None:
                        self._index_to_rowcol[count()] = (r, c)
                else:
                    self._index_to_rowcol[count()] = (r, c)
        

    def __call__(self, *args, **kwargs):
        return self.plot()

    def __getitem__(self, key: Union[int, Tuple(int, int)]) -> BaseSubplotHandler:
        return self.subplot(key)

    # === properties ===
    @property
    def rows(self) -> int:
        return self._subplot[0]

    @property
    def cols(self) -> int:
        return self._subplot[1]

    # === functions ===
    def plot(self):
        '''
        Plot graph
        '''
        raise NotImplementedError('Method not implemented')

    @polymethod()
    def subplot(self, row: int, col: int) -> BaseSubplotHandler:
        return BaseSubplotHandler(self, row, col)

    @polymethod()
    def subplot(self, subplot: Tuple[int, int]) -> BaseSubplotHandler:
        return BaseSubplotHandler(self, subplot[0], subplot[1])

    @polymethod()
    def subplot(self, index: int) -> BaseSubplotHandler:
        row, col = self._subplot_index[index]

        return BaseSubplotHandler(self, row, col)

    @polymethod()
    def subplot(self, *args, **kwargs):
        raise RuntimeError('subplot only accepts 1 or 2 arguments')


    def add_trace(self, trace):
        #TODO
        return self


graph[1, 1].add_trace()
