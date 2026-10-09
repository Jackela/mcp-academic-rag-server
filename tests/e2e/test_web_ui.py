"""Normal Chromium exercises production Web routes and JavaScript without paid services.

Haystack retrieval, prompts, sessions and HTTP are real. Only query embeddings and
model replies are fixed test inputs. No DOM injection, browser-security bypass or
success-on-missing-browser skip is used.
"""

import re
import threading

import pytest
from playwright.sync_api import expect, sync_playwright
from werkzeug.serving import make_server

from tests.fixtures.web_runtime import web_runtime


@pytest.fixture
def web_browser(tmp_path):
    with web_runtime(tmp_path) as (web, generator):
        server = make_server("127.0.0.1", 0, web.app, threaded=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as driver:
                browser = driver.chromium.launch(headless=True)
                context = browser.new_context(base_url=f"http://127.0.0.1:{server.server_port}")
                # External cosmetic resources are not part of the offline contract.
                context.route("https://**", lambda route: route.abort())
                page = context.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                try:
                    yield web, generator, context, page, errors
                finally:
                    context.close()
                    browser.close()
        finally:
            server.shutdown()
            thread.join(5)
            server.server_close()
            assert not thread.is_alive()


def ask(page, question):
    page.locator("#userInput").fill(question)
    with page.expect_response(
        lambda response: response.url.endswith("/api/chat") and response.request.method == "POST"
    ) as response:
        page.locator("#userInput").press("Enter")
    return response.value


def test_navigation_and_real_upload_preview(web_browser, tmp_path):
    _, _, _, page, errors = web_browser
    page.goto("/")
    expect(page).to_have_title("首页 - 学术文献RAG系统")
    page.locator(".navbar-nav").get_by_role("link", name="上传文档", exact=True).click()
    page.wait_for_load_state("domcontentloaded")
    expect(page.locator("#fileInput")).to_be_attached()
    file = tmp_path / "controlled.pdf"
    file.write_bytes(b"%PDF-1.4\nControlled upload preview fixture")
    page.locator("#fileInput").set_input_files(str(file))
    expect(page.locator("#previewContainer")).to_be_visible()
    expect(page.locator("#fileDetails")).to_contain_text("controlled.pdf")
    page.get_by_role("link", name="文档管理", exact=True).click()
    expect(page).to_have_url(re.compile(r"/documents$"))
    page.get_by_role("link", name="文献问答", exact=True).click()
    expect(page.locator("#userInput")).to_be_visible()
    assert errors == []


def test_actual_chat_renders_retrieved_table_code_and_reloads_history(web_browser):
    web, generator, _, page, errors = web_browser
    page.goto("/chat")
    assert web.app.config["TESTING"] is False
    assert ask(page, "Controlled browser question").status == 200
    expect(page.locator(".user-message")).to_have_count(1)
    expect(page.locator(".assistant-message").filter(has_text="Controlled Web fixture answer")).to_have_count(1)
    expect(page.locator(".content-table th")).to_have_text(["Parameter", "Value"])
    expect(page.locator(".content-table tbody")).to_contain_text("25.5")
    expect(page.locator(".code-snippet code")).to_contain_text("def fixture_average")
    assert len(generator.calls) == 1
    assert len(web.session_manager.sessions) == 1
    chat = next(iter(web.session_manager.sessions.values()))
    assert [message.role for message in chat.messages] == ["user", "assistant"]
    page.reload()
    expect(page.locator(".user-message")).to_have_count(1)
    expect(page.locator(".assistant-message").filter(has_text="Controlled Web fixture answer")).to_have_count(1)
    assert errors == []


def test_real_code_copy_and_clear_then_send_again(web_browser):
    web, _, context, page, errors = web_browser
    context.grant_permissions(["clipboard-read", "clipboard-write"])
    page.goto("/chat")
    assert ask(page, "Controlled code question").status == 200
    page.locator(".copy-code-btn").click()
    expect(page.locator(".copy-code-btn")).to_have_text("已复制!")
    assert "def fixture_average" in page.evaluate("navigator.clipboard.readText()")
    page.once("dialog", lambda dialog: dialog.accept())
    with page.expect_response("**/api/chat/reset") as response:
        page.locator("#clearChat").click()
    assert response.value.status == 200
    expect(page.locator(".user-message")).to_have_count(0)
    assert next(iter(web.session_manager.sessions.values())).messages == []
    assert ask(page, "Controlled question after reset").status == 200
    expect(page.locator(".user-message")).to_have_count(1)
    assert errors == []


def test_failure_is_visible_and_next_request_recovers(web_browser):
    _, generator, _, page, errors = web_browser
    page.goto("/chat")
    generator.fail = True
    assert ask(page, "Controlled failed question").status == 502
    expect(page.locator(".alert-danger")).to_contain_text("controlled Web generator failure")
    generator.fail = False
    assert ask(page, "Controlled recovered question").status == 200
    expect(page.locator(".assistant-message").filter(has_text="Controlled Web fixture answer")).to_have_count(1)
    assert errors == []


def test_distinct_browser_cookies_keep_histories_separate(web_browser):
    web, _, context, page, errors = web_browser
    page.goto("/chat")
    assert ask(page, "First browser private question").status == 200
    with context.browser.new_context(base_url=page.url.rsplit("/", 1)[0]) as second:
        second.route("https://**", lambda route: route.abort())
        other = second.new_page()
        other.goto("/chat")
        expect(other.locator(".user-message")).to_have_count(0)
        assert ask(other, "Second browser private question").status == 200
        expect(other.locator(".user-message")).to_have_text("Second browser private question刚刚")
        assert len(web.session_manager.sessions) == 2
    assert errors == []


@pytest.mark.parametrize("width,height", [(375, 667), (768, 1024)])
def test_chat_remains_interactive_at_existing_mobile_and_tablet_sizes(web_browser, width, height):
    _, _, _, page, errors = web_browser
    page.set_viewport_size({"width": width, "height": height})
    page.goto("/chat")
    expect(page.locator("#userInput")).to_be_visible()
    assert ask(page, "Controlled responsive question").status == 200
    expect(page.locator(".content-table")).to_be_visible()
    expect(page.locator(".code-snippet")).to_be_visible()
    assert errors == []
