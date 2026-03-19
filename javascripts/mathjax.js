window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: "no-mathjax",
    processHtmlClass: "arithmatex|jupyter-wrapper|jp-Notebook|jp-RenderedMarkdown|jp-RenderedHTMLCommon|jp-Cell"
  }
};

// Re-typeset after Material instant navigation loads new content
document$.subscribe(function() {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.typeset();
});
