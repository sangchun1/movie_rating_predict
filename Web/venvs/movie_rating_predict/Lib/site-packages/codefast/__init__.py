import ast
import builtins
import codefast.reader
import codefast.utils as utils
from codefast.base.format_print import FormatPrint as fp
from codefast.constants import constants
from codefast.ds import fpjson, fplist, nstr, pair_sample, fpdict
from codefast.functools.random import random_string, sample, sample_one, hex
from codefast.io import FastJson
from codefast.io import FileIO as io
from codefast.io import dblite
from codefast.io.osdb import osdb
from codefast.io._json import fpjson
from codefast.logger import error, info, warning
from codefast.math import Math as math
from codefast.network import Network as net
from codefast.network import urljoin
from codefast.utils import (b64decode, b64encode, cipher, decipher, retry,
                            shell, syscall, uuid, md5sum)

# ----------------------------


def date_file(prefix: str, file_ext: str) -> str:
    import datetime
    return f"{prefix}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.{file_ext}"


def eval(s: str):
    try:
        import json
        return json.loads(s)
    except json.decoder.JSONDecodeError as e:
        warning(e)
        return ast.literal_eval(s)


def generate_version():
    """ Generate package version based on date
    """
    import datetime
    return datetime.datetime.now().strftime('%y.%m.%d.%H')

def blocking():
    # block main thread from exiting
    import time
    while True:
        time.sleep(1<<10)

csv = utils.CSVIO
dic = fpdict
j = fpjson
js = FastJson()
lis = fplist
l = fplist
os = utils._os()
r = io.read
read = io.read

builtins.lis = fplist
builtins.dic = fpdict
