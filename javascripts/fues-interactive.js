// FUES interactive scan — preloads TikZ-compiled SVG frames,
// swaps on hover over invisible point targets overlaid on the image.
document.addEventListener('DOMContentLoaded', function() {
  var el = document.getElementById('fues-interactive');
  if (!el) return;

  var N = 9; // 9 points, frames 0 (initial) through 9

  // Preload all SVG frames as Image objects
  var imgs = [];
  for (var i = 0; i <= N; i++) {
    var img = new Image();
    img.src = 'images/scan-frames/step' + i + '.svg';
    imgs.push(img);
  }

  // Point positions as % of image dimensions (measured from tikzfig2 coordinates)
  // These map the 9 circle centres to relative positions within the SVG
  var ptsRel = [
    { xp: 0.11, yp: 0.82 }, // x1 (1.5, 2.25)
    { xp: 0.19, yp: 0.70 }, // x2 (2.55, 3.45)
    { xp: 0.31, yp: 0.58 }, // x3 (4.1, 4.8)
    { xp: 0.38, yp: 0.52 }, // x4 (5.0, 5.5)
    { xp: 0.42, yp: 0.66 }, // x5 (5.55, 3.8) sub-optimal
    { xp: 0.59, yp: 0.32 }, // x6 (7.65, 8.0)
    { xp: 0.70, yp: 0.42 }, // x7 (9.1, 7.4) sub-optimal
    { xp: 0.69, yp: 0.18 }, // x8 (9.05, 9.6)
    { xp: 0.81, yp: 0.38 }, // x9 (10.5, 7.725) sub-optimal
  ];

  var opt = [true, true, true, true, false, true, false, true, false];

  var labels = [
    'Hover over a point to step through the FUES scan',
    '\u0078\u0302\u2081: first point \u2014 always retained',
    '\u0078\u0302\u2082: no jump, left turn \u2014 retained',
    '\u0078\u0302\u2083: no jump, left turn \u2014 retained',
    '\u0078\u0302\u2084: no jump, left turn \u2014 retained',
    '\u0078\u0302\u2085: policy jump + right turn (g\u2082 < g\u2081) \u2014 removed',
    '\u0078\u0302\u2086: left turn at crossing \u2014 retained',
    '\u0078\u0302\u2087: policy jump + right turn \u2014 removed',
    '\u0078\u0302\u2088: left turn \u2014 retained',
    '\u0078\u0302\u2089: policy jump + right turn \u2014 removed',
  ];

  var currentStep = 0;

  function render() {
    // Container holds: the SVG image + invisible hover targets + label
    var wrapper = document.createElement('div');
    wrapper.style.cssText = 'position:relative;display:inline-block;width:100%;max-width:700px;margin:0 auto;';

    // Image
    var imgEl = document.createElement('img');
    imgEl.src = 'images/scan-frames/step' + currentStep + '.svg';
    imgEl.alt = 'FUES scan step ' + currentStep;
    imgEl.style.cssText = 'width:100%;height:auto;display:block;';
    wrapper.appendChild(imgEl);

    // Invisible hover targets over each point
    for (var i = 0; i < ptsRel.length; i++) {
      var target = document.createElement('div');
      target.style.cssText = 'position:absolute;width:6%;height:8%;border-radius:50%;cursor:pointer;'
        + 'left:' + ((ptsRel[i].xp - 0.03) * 100) + '%;'
        + 'top:' + ((ptsRel[i].yp - 0.04) * 100) + '%;';
      target.setAttribute('data-step', i + 1);
      target.addEventListener('mouseenter', function() {
        currentStep = parseInt(this.getAttribute('data-step'));
        doRender();
      });
      wrapper.appendChild(target);
    }

    // Label
    var label = document.createElement('p');
    label.style.cssText = 'text-align:center;font-size:0.9em;color:#4c566a;margin:6px 0 0;font-style:italic;';
    label.textContent = labels[currentStep];

    el.innerHTML = '';
    el.style.textAlign = 'center';
    el.appendChild(wrapper);
    el.appendChild(label);
  }

  function doRender() { render(); }

  // Reset on mouse leave
  el.addEventListener('mouseleave', function(e) {
    if (!el.contains(e.relatedTarget)) {
      currentStep = 0;
      render();
    }
  });

  render();
});
