// Small UI fixes for Chainlit's bundled frontend, loaded through custom_js in
// .chainlit/config.toml (Chainlit accepts only one custom_js file). Chainlit puts it in
// <head> as a blocking script, so it runs before the first paint and before <body> exists.

// Adjusts two auth responses before Chainlit's own code sees them:
// - Sign-out goes straight to the login page. Chainlit's own flow clears the user (blank
//   loading screen), reloads the chat page (the chat UI flashes), and only then redirects to
//   /login. Once the server has cleared the auth cookie we navigate away ourselves and never
//   hand the response back, so that flow never runs.
// - A refused sign-in (too many attempts, database down) gets a readable message. Chainlit's
//   login page names a failure only by the response's status text, looked up lower-cased under
//   auth.login.errors in .chainlit/translations, and HTTP/2 sends no status text at all, so the
//   response is handed over with the matching translation key as its status text.
(function adjustAuthResponses() {
  const originalFetch = window.fetch.bind(window);
  const SIGN_IN_ERROR_KEYS = { 429: "toomanysignins", 503: "signinunavailable" };

  // Returns the request's method and path, or null for input this check does not understand.
  function describe(input, init) {
    try {
      const url = new URL(input instanceof Request ? input.url : input, window.location.href);
      const method = (init && init.method) || (input instanceof Request && input.method) || "GET";
      return { method: method.toUpperCase(), path: url.pathname };
    } catch (unusualInput) {
      return null; // Never let this check break an ordinary request.
    }
  }

  window.fetch = async function (input, init) {
    const response = await originalFetch(input, init);
    const request = describe(input, init);
    if (!request || request.method !== "POST") return response;
    // Suffix matches: the app may be served under a root path, which /login and /logout share.
    if (response.ok && request.path.endsWith("/logout")) {
      window.location.replace(request.path.replace(/logout$/, "login"));
      return new Promise(() => {});
    }
    const errorKey = SIGN_IN_ERROR_KEYS[response.status];
    if (errorKey && request.path.endsWith("/login")) {
      return new Response(response.body, { status: response.status, statusText: errorKey, headers: response.headers });
    }
    return response;
  };
})();

function whenBodyReady(callback) {
  if (document.body) callback();
  else document.addEventListener("DOMContentLoaded", callback);
}

// Adds a "Privacy Policy" link to every page, including the login page. Google's OAuth
// brand review requires the app's home page (the login page for signed-out visitors)
// to link to the privacy policy.
whenBodyReady(function addPrivacyLink() {
  if (document.getElementById("privacy-policy-link")) return;
  const link = document.createElement("a");
  link.id = "privacy-policy-link";
  link.href = "/public/privacy.html";
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = "Privacy Policy";
  link.style.cssText =
    "position:fixed;left:12px;bottom:8px;z-index:50;font:12px system-ui,sans-serif;" +
    "color:inherit;opacity:.65;text-decoration:underline;";
  document.body.appendChild(link);
});

// Adds "Forgot your password?" and "Don't have an account? Sign up" under the login form. React
// renders the form after this script runs and may re-render it, so the links are (re)added
// whenever they go missing.
whenBodyReady(function addAccountLinks() {
  const LINKS_ID = "account-links";
  const LINE_STYLE = "margin:0;text-align:center;font-size:14px;color:hsl(var(--muted-foreground));";
  const LINK_STYLE = "color:hsl(var(--foreground));font-weight:500;text-decoration:underline;text-underline-offset:4px;";

  function line(prefix, href, label) {
    const paragraph = document.createElement("p");
    paragraph.style.cssText = LINE_STYLE;
    paragraph.append(prefix);
    const link = document.createElement("a");
    link.href = href;
    link.textContent = label;
    link.style.cssText = LINK_STYLE;
    paragraph.append(link);
    return paragraph;
  }

  const ensureLinks = () => {
    if (window.location.pathname !== "/login" || document.getElementById(LINKS_ID)) return;
    const form = document.querySelector("form");
    if (!form) return;
    const links = document.createElement("div");
    links.id = LINKS_ID;
    links.style.cssText = "margin-top:24px;display:grid;gap:12px;";
    links.append(line("", "/forgot-password", "Forgot your password?"), line("Don't have an account? ", "/signup", "Sign up"));
    form.after(links);
  };
  ensureLinks();
  new MutationObserver(ensureLinks).observe(document.body, { childList: true, subtree: true });
});
