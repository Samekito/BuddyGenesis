"""Paints the chat UI's theme colours from the page's first bytes, so full page loads never flash white.

Chainlit applies public/theme.json only once its JS bundle runs, and its page first waits on
render-blocking external stylesheets (fonts, KaTeX); browsers show a blank white page meanwhile.
Consumed by app.py, which injects the snippet into every Chainlit page.
"""

import json
import logging
from pathlib import Path

# Chainlit's own localStorage key and default for the light/dark choice; must stay in sync with it.
THEME_STORAGE_KEY = "vite-ui-theme"
DEFAULT_MODE = "dark"
HEAD_TAG = "<head>"

logger = logging.getLogger(__name__)

_SCRIPT_TEMPLATE = """<script>
(function () {
  var variables = %(variables)s, saved = null;
  try { saved = localStorage.getItem(%(storage_key)s); } catch (storageBlocked) {}
  var choice = saved || %(default_mode)s;
  var mode = choice === "system" ? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light") : choice;
  var root = document.documentElement, colours = variables[mode] || {};
  root.classList.add(mode);
  root.style.colorScheme = mode;
  for (var name in colours) root.style.setProperty(name, colours[name]);
  if (colours["--background"]) root.style.backgroundColor = "hsl(" + colours["--background"] + ")";
})();
</script>"""


def load_theme_variables(theme_file: Path) -> dict:
    """Returns theme.json's {"light": {...}, "dark": {...}} CSS variables, or {} if the file is absent."""
    if not theme_file.exists():
        return {}
    return json.loads(theme_file.read_text(encoding="utf-8")).get("variables", {})


def early_theme_script(variables: dict) -> str:
    return _SCRIPT_TEMPLATE % {
        "variables": _script_safe_json(variables),
        "storage_key": _script_safe_json(THEME_STORAGE_KEY),
        "default_mode": _script_safe_json(DEFAULT_MODE),
    }


def inject_into_head(html: str, snippet: str) -> str:
    if HEAD_TAG not in html:
        # A Chainlit upgrade that writes <head ...> would otherwise bring the white flash back unnoticed.
        logger.warning("No plain <head> tag in the page; early theme snippet not injected")
        return html
    return html.replace(HEAD_TAG, HEAD_TAG + snippet, 1)


def _script_safe_json(value) -> str:
    # "<" escaped so a value can never close the <script> element early.
    return json.dumps(value).replace("<", "\\u003c")
