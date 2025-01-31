# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch

def normalize(
    V : torch.Tensor,
    F : torch.Tensor):
    """Normalize mesh vertices and query points to fit inside a unit cube [-1,1]^3."""
    V_max, _ = torch.max(V, dim=0)  # Get max along each axis
    V_min, _ = torch.min(V, dim=0)  # Get min along each axis
    V_center = (V_max + V_min) / 2.  # Compute center
    V = V - V_center  # Shift to origin

    # Scale uniformly so the longest axis fits within [-1,1]
    V_range = V_max - V_min  # Compute range along each axis
    max_range = torch.max(V_range)  # Find the largest range
    V_scale = 2.0 / max_range  # Scale so that the largest axis spans [-1,1]

    V *= V_scale

    return V, F
