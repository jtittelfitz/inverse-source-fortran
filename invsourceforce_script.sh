#!/bin/bash
f2py -c --fcompiler=gnu95 --f90flags=-ffree-form -m PDE PDE.f90
ipython invsourceforce.py
