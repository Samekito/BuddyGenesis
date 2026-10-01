import json

from buddy.theme import early_theme_script, inject_into_head, load_theme_variables

VARIABLES = {"dark": {"--background": "40 9% 13%"}, "light": {"--background": "39 57% 88%"}}


def test_script_carries_both_palettes():
    script = early_theme_script(VARIABLES)

    assert "40 9% 13%" in script and "39 57% 88%" in script


def test_script_cannot_be_closed_early_by_a_theme_value():
    script = early_theme_script({"dark": {"--x": "</script><script>alert(1)"}})

    assert script.count("</script>") == 1


def test_snippet_goes_first_in_head():
    html = inject_into_head("<html><head><link rel=stylesheet></head></html>", "<script>x</script>")

    assert html == "<html><head><script>x</script><link rel=stylesheet></head></html>"


def test_theme_variables_are_read_from_theme_json(tmp_path):
    theme_file = tmp_path / "theme.json"
    theme_file.write_text(json.dumps({"custom_fonts": [], "variables": VARIABLES}))

    assert load_theme_variables(theme_file) == VARIABLES


def test_missing_theme_file_gives_no_variables(tmp_path):
    assert load_theme_variables(tmp_path / "absent.json") == {}


def test_page_without_a_plain_head_tag_is_left_alone_and_logged(caplog):
    html = '<html><head lang="en"></head></html>'

    assert inject_into_head(html, "<script>x</script>") == html
    assert "early theme snippet not injected" in caplog.text
