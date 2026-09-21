"""The 3D structure viewer, as a string of JavaScript with no dependencies.

WHY THIS IS HAND-WRITTEN AND NOT JSmol
    The report's one hard rule is that it is a single self-contained file: it
    gets emailed, and it gets opened over a mounted filesystem from a laptop.
    A page that fetches anything renders as a blank box half the time.

    JSmol cannot satisfy that.  The copy in `web_mmi` is 102 MB, 48 MB of it in
    `j2s/` alone, loaded lazily over HTTP at runtime.  It is excellent and it is
    a web application, not an attachment.

    So the default viewer is this: an orthographic ball-and-stick renderer in a
    2D canvas, about 200 lines, that draws the cell edges, z-sorts the atoms,
    shades them by depth, and rotates under the mouse.  It is not a ray tracer.
    It is enough to see a coordination environment, a layer stacking, or a
    substituted sublattice, and it still works in five years with no network.

    `--jsmol-url` switches the same page to JSmol when the report is being
    served from somewhere that has it, e.g. the `/campaigns` portal.  The
    embedded data is identical either way -- the page writes a CIF from it in
    the browser -- so nothing about the science depends on which one you use.

WHAT IT DRAWS
    * atoms, radius from the covalent radius, colour from Jmol's element table
      or from the site's own magnetic moment
    * bonds, where two atoms are closer than 1.25 x the sum of their covalent
      radii
    * the unit cell, as a wireframe parallelepiped
    * a supercell, 1x1x1 to 3x3x3, because one cell of a layered structure is
      usually unreadable

INPUTS   window.CSP.structures (payload dicts), window.CSP.elements (colour/radius)
OUTPUTS  a drawn canvas per open card
"""

from __future__ import annotations

VIEWER_JS = r"""
(function () {
  var CSP = window.CSP || (window.CSP = {});

  // ---- small vector helpers -------------------------------------------
  function matmul(m, v) {
    return [m[0][0]*v[0] + m[1][0]*v[1] + m[2][0]*v[2],
            m[0][1]*v[0] + m[1][1]*v[1] + m[2][1]*v[2],
            m[0][2]*v[0] + m[1][2]*v[1] + m[2][2]*v[2]];
  }
  function rotate(p, yaw, pitch) {
    var cy = Math.cos(yaw), sy = Math.sin(yaw);
    var cp = Math.cos(pitch), sp = Math.sin(pitch);
    var x = p[0]*cy + p[2]*sy;
    var z = -p[0]*sy + p[2]*cy;
    var y = p[1]*cp - z*sp;
    z = p[1]*sp + z*cp;
    return [x, y, z];
  }
  function shade(hex, f) {
    var n = parseInt(hex.slice(1), 16);
    var r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
    r = Math.round(r*f); g = Math.round(g*f); b = Math.round(b*f);
    r = Math.min(255, r); g = Math.min(255, g); b = Math.min(255, b);
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }
  // Diverging scale for "colour by moment": blue is negative, red positive,
  // pale is zero. Symmetric about zero on purpose -- an antiferromagnetic
  // arrangement must be visible as two colours, not as a gradient.
  function momentColour(m, scale) {
    var t = Math.max(-1, Math.min(1, m / (scale || 1)));
    if (t >= 0) { return 'rgb(' + Math.round(235 - 40*t) + ',' +
                         Math.round(235 - 175*t) + ',' + Math.round(235 - 175*t) + ')'; }
    return 'rgb(' + Math.round(235 + 235*t) + ',' + Math.round(235 + 175*t) +
           ',' + Math.round(235 - 20*t) + ')';
  }

  // ---- expand to a supercell -------------------------------------------
  function expand(s, n) {
    var cell = s.cell, out = [];
    for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) for (var k = 0; k < n; k++) {
      for (var a = 0; a < s.symbols.length; a++) {
        var f = s.frac[a];
        var fx = f[0] + i, fy = f[1] + j, fz = f[2] + k;
        out.push({
          sym: s.symbols[a],
          m: s.moments.length ? s.moments[a] : null,
          p: [fx*cell[0][0] + fy*cell[1][0] + fz*cell[2][0],
              fx*cell[0][1] + fy*cell[1][1] + fz*cell[2][1],
              fx*cell[0][2] + fy*cell[1][2] + fz*cell[2][2]]
        });
      }
    }
    return out;
  }

  function cellEdges(cell, n) {
    var c = [[cell[0][0]*n, cell[0][1]*n, cell[0][2]*n],
             [cell[1][0]*n, cell[1][1]*n, cell[1][2]*n],
             [cell[2][0]*n, cell[2][1]*n, cell[2][2]*n]];
    var v = [];
    for (var i = 0; i < 8; i++) {
      var a = i & 1, b = (i >> 1) & 1, d = (i >> 2) & 1;
      v.push([a*c[0][0] + b*c[1][0] + d*c[2][0],
              a*c[0][1] + b*c[1][1] + d*c[2][1],
              a*c[0][2] + b*c[1][2] + d*c[2][2]]);
    }
    var e = [];
    for (var p = 0; p < 8; p++) for (var q = p + 1; q < 8; q++) {
      var diff = (p ^ q);
      if (diff === 1 || diff === 2 || diff === 4) { e.push([v[p], v[q]]); }
    }
    return e;
  }

  function bonds(atoms, radii) {
    var out = [], n = atoms.length;
    // O(n^2). At the 3x3x3 cap this is ~1.8M pairs for a 68-atom cell, which
    // a browser does in well under a second, and it only runs when the
    // supercell changes rather than on every frame.
    for (var i = 0; i < n; i++) for (var j = i + 1; j < n; j++) {
      var ri = radii[atoms[i].sym] || 1.5, rj = radii[atoms[j].sym] || 1.5;
      var cut = 1.25 * (ri + rj), c2 = cut * cut;
      var dx = atoms[i].p[0] - atoms[j].p[0];
      if (dx > cut || dx < -cut) continue;
      var dy = atoms[i].p[1] - atoms[j].p[1];
      if (dy > cut || dy < -cut) continue;
      var dz = atoms[i].p[2] - atoms[j].p[2];
      if (dz > cut || dz < -cut) continue;
      if (dx*dx + dy*dy + dz*dz <= c2) out.push([i, j]);
    }
    return out;
  }

  // ---- one viewer ------------------------------------------------------
  function Viewer(canvas, structure) {
    this.canvas = canvas;
    this.s = structure;
    this.yaw = -0.55; this.pitch = 0.32; this.zoom = 1.0;
    // One cell of a 12-atom structure shows nothing about the packing and a
    // 2x2x2 of a 68-atom one is a thicket, so the default follows the cell.
    this.n = structure.symbols.length > 40 ? 1 : 2;
    this.mode = 'element';
    this.rebuild();
    this.bind();
  }

  Viewer.prototype.rebuild = function () {
    var elements = CSP.elements || {};
    var radii = {};
    for (var k in elements) { radii[k] = elements[k].r; }
    this.atoms = expand(this.s, this.n);
    this.bondList = this.atoms.length <= 900 ? bonds(this.atoms, radii) : [];
    this.edges = cellEdges(this.s.cell, this.n);
    var cx = 0, cy = 0, cz = 0, i;
    for (i = 0; i < this.atoms.length; i++) {
      cx += this.atoms[i].p[0]; cy += this.atoms[i].p[1]; cz += this.atoms[i].p[2];
    }
    var n = this.atoms.length || 1;
    this.centre = [cx/n, cy/n, cz/n];
    var span = 1;
    for (i = 0; i < this.atoms.length; i++) {
      var d = Math.hypot(this.atoms[i].p[0]-this.centre[0],
                         this.atoms[i].p[1]-this.centre[1],
                         this.atoms[i].p[2]-this.centre[2]);
      if (d > span) span = d;
    }
    this.span = span;
    this.mscale = 0;
    for (i = 0; i < this.atoms.length; i++) {
      if (this.atoms[i].m !== null) {
        this.mscale = Math.max(this.mscale, Math.abs(this.atoms[i].m));
      }
    }
    this.draw();
  };

  Viewer.prototype.draw = function () {
    var cv = this.canvas, ctx = cv.getContext('2d');
    var dpr = window.devicePixelRatio || 1;
    var w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== w*dpr || cv.height !== h*dpr) {
      cv.width = w*dpr; cv.height = h*dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    var self = this;
    var scale = 0.42 * Math.min(w, h) / (this.span || 1) * this.zoom;
    function project(p) {
      var q = rotate([p[0]-self.centre[0], p[1]-self.centre[1], p[2]-self.centre[2]],
                     self.yaw, self.pitch);
      return [w/2 + q[0]*scale, h/2 - q[1]*scale, q[2]];
    }

    var e, pr;
    ctx.strokeStyle = 'rgba(140,150,165,0.55)';
    ctx.lineWidth = 1;
    for (e = 0; e < this.edges.length; e++) {
      var p1 = project(this.edges[e][0]), p2 = project(this.edges[e][1]);
      ctx.beginPath(); ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke();
    }

    pr = [];
    for (var i = 0; i < this.atoms.length; i++) {
      pr.push(project(this.atoms[i].p));
    }
    ctx.strokeStyle = 'rgba(120,128,140,0.75)';
    ctx.lineWidth = 2.2;
    for (var b = 0; b < this.bondList.length; b++) {
      var u = pr[this.bondList[b][0]], v = pr[this.bondList[b][1]];
      ctx.beginPath(); ctx.moveTo(u[0], u[1]); ctx.lineTo(v[0], v[1]); ctx.stroke();
    }

    var order = [];
    for (i = 0; i < this.atoms.length; i++) { order.push(i); }
    order.sort(function (a, b) { return pr[a][2] - pr[b][2]; });

    var elements = CSP.elements || {};
    var zmin = Infinity, zmax = -Infinity;
    for (i = 0; i < pr.length; i++) {
      if (pr[i][2] < zmin) zmin = pr[i][2];
      if (pr[i][2] > zmax) zmax = pr[i][2];
    }
    var zspan = (zmax - zmin) || 1;
    for (var o = 0; o < order.length; o++) {
      var a = this.atoms[order[o]], q = pr[order[o]];
      var info = elements[a.sym] || {c: '#ff1493', r: 1.5};
      var base = (this.mode === 'moment' && a.m !== null)
               ? momentColour(a.m, this.mscale) : info.c;
      var depth = 0.62 + 0.38 * (q[2] - zmin) / zspan;
      var radius = Math.max(2.2, info.r * scale * 0.42);
      var grad = ctx.createRadialGradient(q[0]-radius*0.35, q[1]-radius*0.35,
                                          radius*0.1, q[0], q[1], radius);
      grad.addColorStop(0, shade(base.charAt(0) === '#' ? base : '#888888',
                                 depth*1.35));
      grad.addColorStop(1, base.charAt(0) === '#' ? shade(base, depth*0.8) : base);
      ctx.beginPath();
      ctx.arc(q[0], q[1], radius, 0, 6.2832);
      ctx.fillStyle = base.charAt(0) === '#' ? grad : base;
      ctx.fill();
      ctx.lineWidth = 0.9;
      ctx.strokeStyle = 'rgba(20,22,26,0.55)';
      ctx.stroke();
    }
  };

  Viewer.prototype.bind = function () {
    var self = this, dragging = false, lx = 0, ly = 0;
    this.canvas.addEventListener('pointerdown', function (ev) {
      dragging = true; lx = ev.clientX; ly = ev.clientY;
      self.canvas.setPointerCapture(ev.pointerId);
    });
    this.canvas.addEventListener('pointermove', function (ev) {
      if (!dragging) return;
      self.yaw += (ev.clientX - lx) * 0.01;
      self.pitch += (ev.clientY - ly) * 0.01;
      self.pitch = Math.max(-1.5, Math.min(1.5, self.pitch));
      lx = ev.clientX; ly = ev.clientY;
      self.draw();
    });
    this.canvas.addEventListener('pointerup', function () { dragging = false; });
    this.canvas.addEventListener('wheel', function (ev) {
      ev.preventDefault();
      self.zoom *= ev.deltaY < 0 ? 1.12 : 0.89;
      self.zoom = Math.max(0.25, Math.min(6, self.zoom));
      self.draw();
    }, {passive: false});
    window.addEventListener('resize', function () { self.draw(); });
  };

  // ---- a CIF, written in the browser from the same payload -------------
  function cifOf(s) {
    function norm(v) { return Math.hypot(v[0], v[1], v[2]); }
    function ang(u, v) {
      var d = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) / (norm(u)*norm(v));
      return Math.acos(Math.max(-1, Math.min(1, d))) * 180 / Math.PI;
    }
    var c = s.cell;
    var out = ['data_cspflow_' + s.id,
      '_cell_length_a ' + norm(c[0]).toFixed(6),
      '_cell_length_b ' + norm(c[1]).toFixed(6),
      '_cell_length_c ' + norm(c[2]).toFixed(6),
      '_cell_angle_alpha ' + ang(c[1], c[2]).toFixed(4),
      '_cell_angle_beta ' + ang(c[0], c[2]).toFixed(4),
      '_cell_angle_gamma ' + ang(c[0], c[1]).toFixed(4),
      "_symmetry_space_group_name_H-M 'P 1'",
      'loop_', ' _symmetry_equiv_pos_as_xyz', "  'x, y, z'",
      'loop_', ' _atom_site_label', ' _atom_site_type_symbol',
      ' _atom_site_fract_x', ' _atom_site_fract_y', ' _atom_site_fract_z'];
    for (var i = 0; i < s.symbols.length; i++) {
      out.push('  ' + s.symbols[i] + (i+1) + ' ' + s.symbols[i] + ' ' +
               s.frac[i][0].toFixed(6) + ' ' + s.frac[i][1].toFixed(6) + ' ' +
               s.frac[i][2].toFixed(6));
    }
    return out.join('\n') + '\n';
  }

  CSP.cifOf = cifOf;

  CSP.mount = function (host, sid) {
    var s = (CSP.structures || {})[sid];
    if (!s) { host.innerHTML = '<p class=note>no geometry published for this structure</p>'; return; }
    if (host.dataset.mounted) { return; }
    host.dataset.mounted = '1';

    if (CSP.jsmolUrl) {
      // The portal path: hand the same payload to JSmol as a CIF.
      var div = document.createElement('div');
      div.id = 'jsmol-' + sid;
      host.appendChild(div);
      var info = {width: '100%', height: 420, use: 'HTML5',
                  j2sPath: CSP.jsmolUrl + '/j2s', script: 'set antialiasDisplay ON;',
                  disableInitialConsole: true, disableJ2SLoadMonitor: true};
      div.innerHTML = Jmol.getAppletHtml('jmol_' + sid, info);
      Jmol.script(window['jmol_' + sid],
                  'load inline "' + cifOf(s).replace(/\n/g, '\\n') +
                  '" {2 2 2}; set unitcell on; wireframe 0.15; spacefill 25%;');
      return;
    }

    var bar = document.createElement('div');
    bar.className = 'vbar';
    bar.innerHTML =
      '<select class="v-cells">' +
      '<option value="1">1 x 1 x 1</option>' +
      '<option value="2">2 x 2 x 2</option>' +
      '<option value="3">3 x 3 x 3</option></select>' +
      '<select class="v-mode">' +
      '<option value="element">colour: element</option>' +
      '<option value="moment">colour: site moment</option></select>' +
      '<button class="v-cif" type="button">download CIF</button>';
    var canvas = document.createElement('canvas');
    canvas.className = 'v-canvas';
    host.appendChild(bar);
    host.appendChild(canvas);

    var viewer = new Viewer(canvas, s);
    bar.querySelector('.v-cells').value = String(viewer.n);
    bar.querySelector('.v-cells').addEventListener('change', function (ev) {
      viewer.n = parseInt(ev.target.value, 10); viewer.rebuild();
    });
    bar.querySelector('.v-mode').addEventListener('change', function (ev) {
      viewer.mode = ev.target.value; viewer.draw();
    });
    bar.querySelector('.v-cif').addEventListener('click', function () {
      var blob = new Blob([cifOf(s)], {type: 'chemical/x-cif'});
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'cspflow-' + sid + '.cif';
      a.click();
      URL.revokeObjectURL(a.href);
    });
  };
})();
"""
