function extractVisibleText() {
  const root = document.querySelector("article") || document.querySelector("main") || document.body;
  if (!root) {
    return "";
  }
  const clone = root.cloneNode(true);
  clone
    .querySelectorAll("script, style, noscript, svg, canvas, iframe, nav, aside, footer, form")
    .forEach((node) => node.remove());
  return clone.innerText.replace(/\s+/g, " ").trim();
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "GET_PAGE_CONTENT") {
    sendResponse({
      url: location.href,
      title: document.title,
      content: extractVisibleText(),
    });
  }
});

