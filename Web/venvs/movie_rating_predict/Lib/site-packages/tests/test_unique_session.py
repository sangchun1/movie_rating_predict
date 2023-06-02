# --------------------------------------------
import keyring as kr
import os
import random
import json
import re
import sys
import time
from collections import defaultdict
from functools import reduce

import codefast as cf
import joblib
import numpy as np
import pandas as pd
from rich import print
from typing import List, Union, Callable, Set, Dict, Tuple, Optional, Any

from pydantic import BaseModel

import asyncio
import aiohttp
import aioredis
import pytest
# —--------------------------------------------

from codefast.asyncio import UniqueSession


@pytest.mark.asyncio
async def test_code_bug():
    async with UniqueSession() as us:
        async with us.get('https://ipinfo.io/json') as resp:
            js = await resp.json()
            assert isinstance(js, dict)
            assert resp.status == 200

