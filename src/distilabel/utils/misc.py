import os
import json
from pdf2image import pdfinfo_from_path
from pathlib import Path as pth
import yaml
from contextlib import contextmanager
from io import StringIO
import sys
import re
from typing import Callable

from .image import get_image, downsample_image, b64_encode_image

@contextmanager
def suppress_output(debug: bool):
    """Stfu utility."""
    if debug:
        yield
    else:
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        class NullIO(StringIO):
            def write(self, txt: str) -> None:
                pass

        sys.stdout = NullIO()
        sys.stderr = NullIO()
        yield
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

def normalize_distribution(dist: list[float]) -> list[float]:
    '''Normalize a distribution so that the sum of the distribution is 1'''
    total = sum(dist)
    return [d / total for d in dist]

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def pdf_name(page):
    '''Return the pdf name from a page filename'''
    return page[:page.rfind('_page_')] + '.pdf'

def pdf_page(page):
    '''Return the page number from a page filename'''
    return int(page[page.rfind('_page_') + 6:page.rfind('.pdf')])

def path_as_page(path, page):
    '''Return the page filename from a path and page number'''
    return path[:path.rfind('_page_')] + f'_page_{page}.pdf'

def n_pages(path):
    '''Return the number of pages in a pdf'''
    return pdfinfo_from_path(path)['Pages']

def generate_idx_to_filename(ds):
    idx_to_filename = {
        idx: filename
        for idx, filename in enumerate(ds['image_filename'])
    }
    return idx_to_filename

def generate_field_to_idx(ds, field):
    field_to_idx = {
        field: idx
        for idx, field in enumerate(ds[field])
    }
    return field_to_idx

def clear_dir(directory):
    """Empty a directory."""
    if pth(directory).is_dir():
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                (pth(root) / name).unlink()
            for name in dirs:
                (pth(root) / name).rmdir()
        pth(directory).rmdir()

def load_pydantic(path, config_class):
    '''load yaml config and convert into pydantic config'''
    with open(path, 'r') as fin:
        config = yaml.safe_load(fin)
    config = {
        k: str(pth(v).expanduser().resolve()) if 'path' in k else v
        for k, v in config.items()
    }
    config = config_class.model_validate(config)
    return config

def is_openai_model_name(model_name: str) -> bool:
    """
    Check if a string is the name of an OpenAI model by matching 'gpt' or 'o' followed by a digit and anything else.
    """
    return bool(re.search(r'(gpt|o\d.*)', model_name, re.IGNORECASE))

def source_to_msg(source: str | list[str], max_dims: tuple[int, int], msg_content_img: Callable) -> dict:
    '''
    Convert a source into an openai message.
    
    A source is a string directly for input, or a list of paths to images or pdf pages.
    '''
    if isinstance(source, str):
            # Text source
        return {'role': 'user', 'content': source}
    else:
        # Image source (list of paths)
        content = []
        for path in source:
            img = get_image(None, path)
            img = downsample_image(img, max_dims)
            b64_img = b64_encode_image(img)
            content.append(msg_content_img(b64_img))
            
        return {'role': 'user', 'content': content}
