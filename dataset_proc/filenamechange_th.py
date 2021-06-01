#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

for filename in os.listdir("."):
    if filename.startswith("TH_"):
        os.rename(filename, filename[3:])
        print("old filename:", filename, "new filename:", filename[3:])
