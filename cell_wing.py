#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:13:25 2024

@author: simonecicolini
"""

class cell_wing:
    def __init__(self, u, delta_tot, notch_tot, delta_free, notch_free, signal, notch_reporter, index, border):
        self.u =u
        self.delta_tot=delta_tot
        self.notch_tot=notch_tot
        self.delta_free=delta_free
        self.notch_free=notch_free
        self.notch_reporter=notch_reporter
        self.signal=signal
        self.index = index
        self.border=border #0 if is not in the border
        self.neighbours = []
