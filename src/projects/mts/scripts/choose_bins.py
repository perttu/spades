
############################################################################
# Copyright (c) 2023-2024 SPAdes team
# All Rights Reserved
# See file LICENSE for details.
############################################################################

#!/usr/bin/env python3
from __future__ import (print_function)

import re
import sys

from common import contig_length
import numpy
import pandas
from pandas import DataFrame

min_len = int(sys.argv[1])
in_fn = sys.argv[2]
d = pandas.read_table(in_fn, names=["name", "bin"], dtype=str)
d["group"] = d.apply(lambda row: re.findall("\\w+\\d+", row["name"])[0], axis=1)
d["length"] = d.apply(lambda row: contig_length(row["name"]), axis=1)
del d["name"]
info = d.groupby(["bin", "group"], as_index=False).sum()
info = info.groupby("bin", as_index=False)["length"].max()
info = info[info["length"] > min_len].sort_values("length", ascending=False)
info.to_csv(sys.stdout, sep="\t", header=False, index=False)
