import hashlib
import os
import sys
import shutil
from functools import wraps, partial

import pytest

if os.path.dirname('src') not in sys.path:
    sys.path.append('src')

from src.utils import call_subprocess_onetask, makedirs


def get_inf_port():
    if os.getenv('HOST') is not None:
        inf_port = os.environ['HOST'].split(':')[-1]
    elif os.getenv('GRADIO_SERVER_PORT') is not None:
        inf_port = os.environ['GRADIO_SERVER_PORT']
    else:
        raise ValueError("Expect tests to set HOST or GRADIO_SERVER_PORT")
    return int(inf_port)


def get_inf_server():
    if os.getenv('HOST') is not None:
        inf_server = os.environ['HOST']
    elif os.getenv('GRADIO_SERVER_PORT') is not None:
        inf_server = "http://localhost:%s" % os.environ['GRADIO_SERVER_PORT']
    else:
        raise ValueError("Expect tests to set HOST or GRADIO_SERVER_PORT")
    return inf_server


def get_mods():
    testtotalmod = int(os.getenv('TESTMODULOTOTAL', '1'))
    testmod = int(os.getenv('TESTMODULO', '0'))
    return testtotalmod, testmod


def do_skip_test(name):
    """
    Control if skip test.  note that skipping all tests does not fail, doing no tests is what fails
    :param name:
    :return:
    """
    testtotalmod, testmod = get_mods()
    return int(get_sha(name), 16) % testtotalmod != testmod


def wrap_test_forked(func):
    """Decorate a function to test, call in subprocess"""

    @wraps(func)
    def f(*args, **kwargs):
        # automatically list or set, so can globally control server ports or host for all tests
        gradio_port = os.environ['GRADIO_SERVER_PORT'] = os.getenv('GRADIO_SERVER_PORT', str(7860))
        gradio_port = int(gradio_port)
        # testtotalmod, testmod = get_mods()
        # gradio_port += testmod
        os.environ['HOST'] = os.getenv('HOST', "http://localhost:%s" % gradio_port)

        pytest_name = get_test_name()
        if do_skip_test(pytest_name):
            # Skipping is based on raw name, so deterministic
            pytest.skip("[%s] TEST SKIPPED due to TESTMODULO" % pytest_name)
        func_new = partial(call_subprocess_onetask, func, args, kwargs)
        return run_test(func_new)

    return f


def run_test(func, *args, **kwargs):
    return func(*args, **kwargs)


def get_sha(value):
    return hashlib.md5(str(value).encode('utf-8')).hexdigest()


def sanitize_filename(name):
    """
    Sanitize file *base* names.  Also used to generation valid class names.
    :param name:
    :return:
    """
    bad_chars = ['[', ']', ',', '/', '\\', '\\w', '\\s', '-', '+', '\"', '\'', '>', '<', ' ', '=', ')', '(', ':', '^']
    for char in bad_chars:
        name = name.replace(char, "_")

    length = len(name)
    file_length_limit = 250  # bit smaller than 256 for safety
    sha_length = 32
    real_length_limit = file_length_limit - (sha_length + 2)
    if length > file_length_limit:
        sha = get_sha(name)
        half_real_length_limit = max(1, int(real_length_limit / 2))
        name = name[0:half_real_length_limit] + "_" + sha + "_" + name[length - half_real_length_limit:length]

    return name


def get_test_name():
    tn = os.environ['PYTEST_CURRENT_TEST'].split(':')[-1]
    tn = "_".join(tn.split(' ')[:-1])  # skip (call) at end
    return sanitize_filename(tn)


def make_user_path_test():
    import os
    import shutil
    user_path = 'user_path_test'
    if os.path.isdir(user_path):
        shutil.rmtree(user_path)
    os.makedirs(user_path)
    db_dir = "db_dir_UserData"
    if os.path.isdir(db_dir):
        shutil.rmtree(db_dir)
    shutil.copy('data/pexels-evg-kowalievska-1170986_small.jpg', user_path)
    shutil.copy('README.md', user_path)
    shutil.copy('docs/FAQ.md', user_path)
    return user_path


def get_llama(llama_type=2):
    from huggingface_hub import hf_hub_download

    # default should match .env_gpt4all
    if llama_type == 1:
        file = 'ggml-model-q4_0_7b.bin'
        dest = 'models/7B/'
        prompt_type = 'plain'
    elif llama_type == 2:
        file = 'WizardLM-7B-uncensored.ggmlv3.q8_0.bin'
        dest = './'
        prompt_type = 'wizard2'
    else:
        raise ValueError("unknown llama_type=%s" % llama_type)

    makedirs(dest, exist_ok=True)
    full_path = os.path.join(dest, file)

    if not os.path.isfile(full_path):
        # True for case when locally already logged in with correct token, so don't have to set key
        token = os.getenv('HUGGINGFACE_API_TOKEN', True)
        out_path = hf_hub_download('h2oai/ggml', file, token=token, repo_type='model')
        # out_path will look like '/home/jon/.cache/huggingface/hub/models--h2oai--ggml/snapshots/57e79c71bb0cee07e3e3ffdea507105cd669fa96/ggml-model-q4_0_7b.bin'
        shutil.copy(out_path, dest)
    return prompt_type


def kill_weaviate(db_type):
    """
    weaviate launches detatched server, which accumulates entries in db, but we want to start freshly
    """
    if db_type == 'weaviate':
        os.system('pkill --signal 9 -f weaviate-embedded/weaviate')
