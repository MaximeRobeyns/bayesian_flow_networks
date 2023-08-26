# Copyright (C) 2023 Maxime Robeyns <dev@maximerobeyns.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch as t

from math import isclose

from torch_bfn.utils import norm_denorm


def test_norm_denorm():
    B, D = 128, 4
    data = t.randn(B, D)
    data[:, 0] *= 8
    data[:, 1] *= 3
    data[:, 1] += 3
    norm_data, denorm = norm_denorm(data, -0.5, 0.5)
    assert norm_data.shape == data.shape
    for i in range(norm_data.size(1)):
        assert isclose(norm_data[:, i].min(), -0.5)
        assert isclose(norm_data[:, i].max(), 0.5)
    denorm_data = denorm(norm_data)
    assert t.allclose(data, denorm_data)
