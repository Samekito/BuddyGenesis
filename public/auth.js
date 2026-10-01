// Shared behaviour of SOC Buddy's own account pages (sign-up, email confirmation, password reset).
// Each page loads this first, then its own inline script, which calls these helpers.

const OFFLINE_MESSAGE = "Could not reach the server. Check your connection and try again.";

// The logo image differs per theme; the theme class was set on <html> by the injected snippet.
function showThemedLogo() {
  const mode = document.documentElement.classList.contains("light") ? "light" : "dark";
  document.getElementById("logo").src = `/logo?theme=${mode}`;
}

function enablePasswordReveal() {
  document.querySelectorAll(".reveal").forEach((button) => {
    button.addEventListener("click", () => {
      const input = document.getElementById(button.dataset.for);
      const show = input.type === "password";
      input.type = show ? "text" : "password";
      button.setAttribute("aria-label", show ? "Hide password" : "Show password");
    });
  });
}

function showError(message) {
  const errorBox = document.getElementById("error");
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function hideError() {
  document.getElementById("error").hidden = true;
}

// Replaces the form's fields with a message; used once an action has succeeded.
function showNotice(form, message) {
  form.querySelector(".fields").hidden = true;
  hideError();
  const notice = form.querySelector(".notice");
  notice.textContent = message;
  notice.hidden = false;
}

// POSTs JSON; returns { ok, detail }. Throws Error(OFFLINE_MESSAGE) when the server cannot be reached.
async function postJson(url, body) {
  let response;
  try {
    response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch (networkFailure) {
    throw new Error(OFFLINE_MESSAGE);
  }
  const payload = await response.json().catch(() => ({}));
  const detail = typeof payload.detail === "string" ? payload.detail : "Something went wrong. Please try again.";
  return { ok: response.ok, detail };
}

// Reads the one-use token from the link, then removes it from the address bar and history so it
// is not left in the browser history or shown on screen.
function takeTokenFromLink() {
  const token = new URLSearchParams(location.search).get("token") || "";
  history.replaceState(null, "", location.pathname);
  return token;
}

// Disables the button while `work` runs, showing `busyLabel`; restores it if `work` throws.
async function whileBusy(button, busyLabel, work) {
  const label = button.textContent;
  button.disabled = true;
  button.textContent = busyLabel;
  try {
    await work();
  } catch (failure) {
    showError(failure.message);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}
