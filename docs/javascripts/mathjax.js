window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex|jp-RenderedHTMLCommon|jp-MarkdownOutput"
  }
};

// Re-typeset after Material instant navigation loads new content
document$.subscribe(function() {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.typeset();
});
