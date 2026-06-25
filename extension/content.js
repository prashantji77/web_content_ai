function extractVisibleText() {
  const clone = document.body.cloneNode(true);
  clone.querySelectorAll("script, style, noscript, svg, canvas, iframe").forEach((node) => {
    node.remove();
  });
  return clone.innerText.replace(/\s+/g, " ").trim();
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "GET_PAGE_CONTENT") {
    sendResponse({
      url: location.href,
      title: document.title,
      text: extractVisibleText(),
    });
  }
});

