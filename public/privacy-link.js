// Adds a "Privacy Policy" link to every page, including the login page.
// Loaded through custom_js in .chainlit/config.toml. Google's OAuth brand review requires
// the app's home page (the login page for signed-out visitors) to link to the privacy policy.
(function addPrivacyLink() {
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
})();
