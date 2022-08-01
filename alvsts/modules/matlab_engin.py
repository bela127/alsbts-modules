from __future__ import annotations
from typing import TYPE_CHECKING

import matlab.engine

class MatLabEngin:

    def __init__(self):
        print("starting matlab engin")
        self.eng = matlab.engine.start_matlab()

    def __enter__(self) -> MatLabEngin:
        return self.eng

    def __exit__(self, type, value, traceback):
        print("stopping matlab engin")
        self.eng.quit()

    #ONLY FOR TYPE HINTING:

    def set_param(self, *args, nargout=0):
        ...
    
    def get_param(self, *args, nargout=0):
        ...
    
    def run(self, *args, nargout=0):
        ...

    def eval(self, *args, nargout=0):
        ...

    @property
    def workspace(self) -> dict[str, list[list[float]]]:
        ...