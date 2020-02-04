# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 21:06:11 2019

@author: Przemek
"""
import snap7
import snap7.client as c
from snap7.util import *

def WriteOutput(dev, byte, bit, cmd):
    data = dev.read_area(0x82,0,byte,1)
    set_bool(data,byte,bit,cmd)
    dev.write_area(0x82,0,byte,data)

