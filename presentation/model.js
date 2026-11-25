// Scores survey answers in the browser with the trees exported from the notebook's gradient boosting
// models, and explains each score with TreeSHAP (the same algorithm as shap.TreeExplainer).
(function (root) {
  function leafValue(tree, x) {
    let n = 0;
    while (tree.f[n] >= 0) {
      const v = x[tree.f[n]];
      const left = Number.isNaN(v) ? tree.m[n] === 1 : v <= tree.t[n];
      n = left ? tree.l[n] : tree.r[n];
    }
    return tree.v[n];
  }

  // Raw score (log-odds) of a HistGradientBoostingClassifier
  function rawScore(model, x) {
    let s = model.base;
    for (const t of model.trees) s += leafValue(t, x);
    return s;
  }

  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  // Isotonic regression with out_of_bounds="clip": linear interpolation between thresholds
  function isotonic(xs, ys, v) {
    if (v <= xs[0]) return ys[0];
    if (v >= xs[xs.length - 1]) return ys[ys.length - 1];
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] <= v) lo = mid; else hi = mid;
    }
    const w = (v - xs[lo]) / (xs[hi] - xs[lo] || 1);
    return ys[lo] + w * (ys[hi] - ys[lo]);
  }

  // CalibratedClassifierCV(method="isotonic"): average of the fold models' calibrated outputs
  function calibratedRisk(calib, x) {
    let s = 0;
    for (const c of calib) s += isotonic(c.iso_x, c.iso_y, rawScore(c, x));
    return s / calib.length;
  }

  // ---------------------------------------------------------------- TreeSHAP (path-dependent)
  function extendPath(p, depth, zf, of, fi) {
    p[depth] = { f: fi, z: zf, o: of, w: depth === 0 ? 1 : 0 };
    for (let i = depth - 1; i >= 0; i--) {
      p[i + 1].w += of * p[i].w * (i + 1) / (depth + 1);
      p[i].w = zf * p[i].w * (depth - i) / (depth + 1);
    }
  }

  function unwindPath(p, depth, idx) {
    const of = p[idx].o, zf = p[idx].z;
    let next = p[depth].w;
    for (let i = depth - 1; i >= 0; i--) {
      if (of !== 0) {
        const tmp = p[i].w;
        p[i].w = next * (depth + 1) / ((i + 1) * of);
        next = tmp - p[i].w * zf * (depth - i) / (depth + 1);
      } else {
        p[i].w = (p[i].w * (depth + 1)) / (zf * (depth - i));
      }
    }
    for (let i = idx; i < depth; i++) {
      p[i].f = p[i + 1].f; p[i].z = p[i + 1].z; p[i].o = p[i + 1].o;
    }
  }

  function unwoundSum(p, depth, idx) {
    const of = p[idx].o, zf = p[idx].z;
    let next = p[depth].w, total = 0;
    for (let i = depth - 1; i >= 0; i--) {
      if (of !== 0) {
        const tmp = next * (depth + 1) / ((i + 1) * of);
        total += tmp;
        next = p[i].w - tmp * zf * ((depth - i) / (depth + 1));
      } else {
        total += (p[i].w / zf) / ((depth - i) / (depth + 1));
      }
    }
    return total;
  }

  function treeShap(t, x, phi) {
    function rec(node, depth, parent, zf, of, fi) {
      const p = parent.slice(0, depth).map((e) => ({ ...e }));
      extendPath(p, depth, zf, of, fi);
      if (t.f[node] < 0) {
        for (let i = 1; i <= depth; i++) {
          const w = unwoundSum(p, depth, i);
          phi[p[i].f] += w * (p[i].o - p[i].z) * t.v[node];
        }
        return;
      }
      const f = t.f[node], v = x[f];
      const goLeft = Number.isNaN(v) ? t.m[node] === 1 : v <= t.t[node];
      const hot = goLeft ? t.l[node] : t.r[node], cold = goLeft ? t.r[node] : t.l[node];
      const c = t.c[node];
      let iz = 1, io = 1, d = depth;
      let k = 0;
      for (; k <= d; k++) if (p[k].f === f) break;
      if (k !== d + 1) {
        iz = p[k].z; io = p[k].o;
        unwindPath(p, d, k);
        d -= 1;
      }
      rec(hot, d + 1, p, (t.c[hot] / c) * iz, io, f);
      rec(cold, d + 1, p, (t.c[cold] / c) * iz, 0, f);
    }
    rec(0, 0, [], 1, 1, -1);
  }

  // Expected raw score over the training data (the SHAP base value)
  function expectedValue(model) {
    let s = model.base;
    for (const t of model.trees) {
      let num = 0, den = 0;
      for (let n = 0; n < t.f.length; n++) if (t.f[n] < 0) { num += t.v[n] * t.c[n]; den += t.c[n]; }
      s += num / den;
    }
    return s;
  }

  function shapValues(model, x) {
    const phi = new Array(x.length).fill(0);
    for (const t of model.trees) treeShap(t, x, phi);
    return phi;
  }

  // ---------------------------------------------------------------- RL interviewer (MaskablePPO policy)
  function dense(layer, v, act) {
    return layer.W.map((row, i) => {
      let s = layer.b[i];
      for (let j = 0; j < row.length; j++) s += row[j] * v[j];
      return act === "tanh" ? Math.tanh(s) : act === "relu" ? Math.max(0, s) : s;
    });
  }

  // Interview one person: the agent picks each next question (or stops); the partial-answer
  // classifier turns the answers so far into a risk. Returns every step for the demo.
  function rlInterview(rl, x) {
    const n = x.length, asked = new Array(n).fill(false);
    const partial = () => x.map((v, j) => (asked[j] ? v : NaN));
    let p = sigmoid(rawScore(rl.clf, partial()));
    const steps = [{ q: -1, p }];
    while (asked.some((a) => !a)) {
      const obs = [...x.map((v, j) => (asked[j] ? (v - rl.mu[j]) / rl.sd[j] : 0)), ...asked.map(Number), p];
      let h = obs;
      for (const layer of rl.hidden) h = dense(layer, h, rl.act);
      const logits = dense(rl.out, h, "none");
      let best = n, bestV = logits[n];                      // index n = stop (always allowed)
      for (let j = 0; j < n; j++) if (!asked[j] && logits[j] > bestV) { best = j; bestV = logits[j]; }
      if (best === n) break;
      asked[best] = true;
      p = sigmoid(rawScore(rl.clf, partial()));
      steps.push({ q: best, p });
    }
    return { steps, p, nAsked: steps.length - 1, fullP: sigmoid(rawScore(rl.clf, x)) };
  }

  root.DiabetesModel = { rawScore, sigmoid, calibratedRisk, shapValues, expectedValue, rlInterview };
})(typeof window !== "undefined" ? window : globalThis);
