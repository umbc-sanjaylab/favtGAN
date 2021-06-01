#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

for filename in os.listdir("."):
    if filename.startswith("VIS_"):
        os.rename(filename, filename[4:])
        print("old filename:", filename, "new filename:", filename[4:])
