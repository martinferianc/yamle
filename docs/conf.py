import datetime
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath("../yamle/"))

import yamle


def run_apidoc(app):
    """Generate doc stubs using sphinx-apidoc."""
    module_dir = os.path.join(app.srcdir, "../yamle/")
    output_dir = os.path.join(app.srcdir, "_apidoc")
    excludes = []

    # Ensure that any stale apidoc files are cleaned up first.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cmd = [
        "--separate",
        "--module-first",
        "--doc-project=API Reference",
        "-o",
        output_dir,
        module_dir,
    ]
    cmd.extend(excludes)

    try:
        from sphinx.ext import apidoc  # Sphinx >= 1.7

        apidoc.main(cmd)
    except ImportError:
        from sphinx import apidoc  # Sphinx < 1.7

        cmd.insert(0, apidoc.__file__)
        apidoc.main(cmd)


def setup(app):
    """Register our sphinx-apidoc hook."""
    app.connect("builder-inited", run_apidoc)


# Sphinx configuration below.
project = "YAMLE"
version = yamle.__version__
release = yamle.__version__
athor = "Martin Ferianc"
copyright = f"2023-{datetime.datetime.now().year}, Martin"


extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints", 
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "hoverxref.extension",
    "sphinx_copybutton",
    "sphinxext.opengraph",
    "sphinx_paramlinks", 
]
coverage_show_missing_items = True

autosectionlabel_prefix_document = True

hoverxref_auto_ref = True
hoverxref_role_types = {"ref": "tooltip"}

source_suffix = [".rst", ".md"]

master_doc = "index"

autoclass_content = "class"
autodoc_member_order = "bysource"
default_role = "py:obj"

html_theme = "pydata_sphinx_theme"
html_sidebars = {"**": ["sidebar-nav-bs"]}
html_theme_options = {
    "primary_sidebar_end": [],
    "footer_start": ["copyright"],
    "footer_end": [],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/martinferianc/YAMLE",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ],
    "use_edit_page_button": True,
    "collapse_navigation": True,
}
html_context = {
    "github_user": "martinferianc",
    "github_repo": "YAMLE",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "light",
}

htmlhelp_basename = "{}doc".format(project)

napoleon_use_rtype = False

rst_prolog = """
.. role:: python(code)
    :language: python
    :class: highlight
"""