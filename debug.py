# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function


class Debug(object):
    def __gt__(self, other):
        if other.__class__.__name__ == "Variable" or other.__class__.__name__ == "Tensor":
            print(other.eval())
        else:
            print(other)


d = Debug()
