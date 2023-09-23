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
"""PyTorch implementation of Bayesian Flow Networks, due to Alex Graves, Rupesh
Kumar Srivastava, Timothy Atkinson and Faustino Gomez:
https://arxiv.org/abs/2308.07037
"""

__version__ = "0.0.3"

from torch_bfn.bfn import ContinuousBFN, DiscreteBFN
from torch_bfn.networks import LinearNetwork, Unet
