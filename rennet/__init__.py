#  Copyright 2018 Fraunhofer IAIS. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""RENNET: Deep Learning Utilities, for speech and audio segmentation, for now."""
from __future__ import absolute_import

from .version import VERSION as __version__

# IDEA: Implement keras like `get` functions to identify functions and classes by strings,
# and 'serializing' and 'desirializing' them to and from h5 model files.
