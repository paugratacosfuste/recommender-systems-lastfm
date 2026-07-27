/**
 * Renders the recommender demo from precomputed JSON.
 *
 * Everything on this page was computed offline by scripts/build_static_site.py - the
 * eight models were fitted and evaluated in Python, and all 1,892 x 8 recommendation
 * lists were written to static files. Nothing runs on a server when you load this, so
 * there is no cold start to wait for.
 *
 * Load order: meta.json, artists.json and the user's shard are requested in parallel;
 * artwork hashes come later, off the critical path.
 */
"use strict";

// Must match BUILD in scripts/build_static_site.py. It busts the year-long `immutable`
// cache on /data/*, so a rebuild that forgets to bump it would serve stale JSON.
// tests/test_build_static_site.py asserts the two agree.
const BUILD = "20260727-1";
const SHARD_SIZE = 100;
// Used to request a shard before meta.json lands. Confirmed against meta.default_user.
const FALLBACK_USER = 2;

const IMG_PREFIX = "https://cdn-images.dzcdn.net/images/artist/";
const IMG_SUFFIX = "/250x250-000000-80-0-0.jpg";

let meta = null;
let artists = null;
let images = null;

const shardCache = new Map();
const inflight = new Map();
const state = { user: null, method: null };

const $ = (id) => document.getElementById(id);
const shardOf = (uid) => Math.floor(uid / SHARD_SIZE);
const pad2 = (n) => String(n).padStart(2, "0");
const methodIndex = (label) => meta.methods.findIndex((m) => m.label === label);

function getJSON(path) {
  return fetch(`data/${path}?v=${BUILD}`).then((r) => {
    if (!r.ok) throw new Error(`${path} -> HTTP ${r.status}`);
    return r.json();
  });
}

/** Fetch a user shard, de-duplicating concurrent requests and caching for the session. */
function getShard(n) {
  if (shardCache.has(n)) return Promise.resolve(shardCache.get(n));
  if (inflight.has(n)) return inflight.get(n);
  const p = getJSON(`users/${pad2(n)}.json`)
    .then((data) => {
      shardCache.set(n, data);
      inflight.delete(n);
      return data;
    })
    .catch((err) => {
      inflight.delete(n);
      throw err;
    });
  inflight.set(n, p);
  return p;
}

// --------------------------------------------------------------------------- elements

/**
 * Build an element. Text always goes through textContent - artist names in this
 * dataset contain &, < and quotes, which innerHTML would mangle or worse.
 */
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function imgURL(hash) {
  return hash[0] === "!" ? hash.slice(1) : IMG_PREFIX + hash + IMG_SUFFIX;
}

function initialTile(idx) {
  const span = el("span", "avatar", artists.init[idx]);
  span.style.setProperty("--h", artists.hue[idx]);
  span.dataset.ai = idx;
  return span;
}

/** Artist photo when we have one, coloured-initial tile otherwise. */
function avatar(idx) {
  const hash = images ? images.img[idx] : null;
  if (!hash) return initialTile(idx);
  const img = el("img", "avatar avatar-photo");
  img.src = imgURL(hash);
  img.alt = "";
  img.loading = "lazy";
  img.referrerPolicy = "no-referrer";
  img.dataset.ai = idx;
  // A dead CDN hash would otherwise leave an empty grey circle.
  img.addEventListener("error", () => img.replaceWith(initialTile(idx)), { once: true });
  return img;
}

function pills(tagIdxs) {
  if (!tagIdxs || !tagIdxs.length) return null;
  const wrap = el("span", "pills");
  for (const t of tagIdxs) wrap.appendChild(el("span", "pill", artists.tags[t]));
  return wrap;
}

// ---------------------------------------------------------------------------- renders

function renderTaste(entry) {
  $("taste-title").textContent = `What user ${state.user} listens to`;

  const ul = $("top-artists");
  ul.replaceChildren();
  if (!entry.t.length) {
    ul.appendChild(el("li", "muted", "No listening history."));
  } else {
    for (const [idx] of entry.t) {
      const li = document.createElement("li");
      li.appendChild(avatar(idx));
      const text = el("span", "chip-text");
      text.appendChild(el("span", "chip-name", artists.name[idx]));
      const p = pills(artists.tag[idx]);
      if (p) text.appendChild(p);
      li.appendChild(text);
      ul.appendChild(li);
    }
  }

  const block = $("genre-block");
  block.hidden = !entry.g.length;
  if (!entry.g.length) return;

  const max = Math.max(...entry.g.map((g) => g[1])) || 1;
  const bars = $("genre-bars");
  bars.replaceChildren();
  for (const [tagIdx, count] of entry.g) {
    const row = el("div", "bar-row");
    row.appendChild(el("span", "bar-label", artists.tags[tagIdx]));
    const track = el("span", "bar-track");
    const fill = el("span", "bar-fill");
    fill.style.width = `${Math.floor((100 * count) / max)}%`;
    track.appendChild(fill);
    row.appendChild(track);
    bars.appendChild(row);
  }
}

function renderRecs(entry) {
  const ids = entry.r[methodIndex(state.method)] || [];

  $("recs-title").replaceChildren(
    document.createTextNode("Recommended by "),
    el("span", "accent", state.method)
  );
  $("recs-sub").textContent = `Top ${ids.length} artists this user has not played yet.`;

  const ol = $("recs");
  ol.replaceChildren();
  if (!ids.length) {
    ol.appendChild(el("li", "muted", "No recommendations."));
    return;
  }
  ids.forEach((idx, i) => {
    const li = el("li", "rec-card");
    li.appendChild(el("span", "rank", String(i + 1)));
    li.appendChild(avatar(idx));
    const body = el("span", "rec-body");
    body.appendChild(el("span", "rec-name", artists.name[idx]));
    const p = pills(artists.tag[idx]);
    if (p) body.appendChild(p);
    li.appendChild(body);
    ol.appendChild(li);
  });
}

function renderCards(host, cards, beyond) {
  host.replaceChildren();
  for (const c of cards) {
    const card = el("div", beyond ? "metric-card beyond" : "metric-card");
    card.title = c.hint;
    card.appendChild(el("span", "metric-value", c.value));
    card.appendChild(el("span", "metric-label", c.label));
    card.appendChild(el("span", "metric-hint", c.hint));
    host.appendChild(card);
  }
}

function renderComparison() {
  const host = $("comparison");
  host.replaceChildren();
  for (const row of meta.comparison) {
    const active = row.label === state.method;
    const line = el("div", active ? "bar-row active" : "bar-row");
    line.appendChild(el("span", "bar-label wide", row.label));
    const track = el("span", "bar-track");
    const fill = el("span", active ? "bar-fill hot" : "bar-fill");
    fill.style.width = `${row.pct}%`; // precomputed in Python - no float maths here
    track.appendChild(fill);
    line.appendChild(track);
    line.appendChild(el("span", "bar-val", row.value));
    host.appendChild(line);
  }
}

/** Everything that depends only on the chosen method - no user data needed. */
function renderMethod() {
  const m = meta.methods[methodIndex(state.method)];

  const badge = $("method-badge");
  badge.className = `badge ${m.personalised ? "personal" : "generic"}`;
  badge.textContent = m.personalised ? "Personalised" : "Non-personalised";
  $("method-desc").textContent = m.description;

  $("scores-title").replaceChildren(
    document.createTextNode("How good is "),
    el("span", "accent", m.label),
    document.createTextNode("?")
  );
  $("scores-sub").textContent =
    `Measured offline on held-out listening for ${meta.n_eval_users} users - hover any ` +
    "card for what it means. Higher is better for all of these.";

  renderCards($("accuracy-cards"), m.accuracy_cards, false);
  const beyondBlock = $("beyond-block");
  beyondBlock.hidden = !m.beyond_cards.length;
  if (m.beyond_cards.length) renderCards($("beyond-cards"), m.beyond_cards, true);

  renderComparison();
}

function renderUser(shard) {
  const entry = shard[String(state.user)];
  if (!entry) return;
  renderTaste(entry);
  renderRecs(entry);
}

// ------------------------------------------------------------------------ URL + state

function parseQuery() {
  const params = new URLSearchParams(location.search);
  const raw = params.get("user_id") || "";
  // /^\d+$/ mirrors Python's str.isdigit() check in app/app.py:31-32.
  return { user: /^\d+$/.test(raw) ? Number(raw) : null, method: params.get("method") };
}

function syncURL() {
  const q = new URLSearchParams({ user_id: String(state.user), method: state.method });
  history.replaceState(null, "", `${location.pathname}?${q}`);
}

async function onUserChange(event) {
  state.user = Number(event.target.value);
  syncURL();
  renderUser(await getShard(shardOf(state.user)));
}

function onMethodChange(event) {
  state.method = event.target.value;
  syncURL();
  renderMethod();
  // All eight lists live in the shard we already hold - switching costs no network.
  const shard = shardCache.get(shardOf(state.user));
  if (shard) renderRecs(shard[String(state.user)]);
}

function populateControls() {
  const users = $("user_id");
  // Safe as markup: these values are integers straight from the build.
  users.innerHTML = meta.users.map((u) => `<option value="${u}">User ${u}</option>`).join("");
  users.value = String(state.user);
  users.disabled = false;
  users.addEventListener("change", onUserChange);

  const methods = $("method");
  methods.value = state.method;
  methods.addEventListener("change", onMethodChange);
}

/** Artwork is a nicety, so it loads last and upgrades the tiles already on screen. */
function loadImages() {
  const go = () =>
    getJSON("images.json")
      .then((data) => {
        images = data;
        for (const node of document.querySelectorAll("span.avatar[data-ai]")) {
          const idx = Number(node.dataset.ai);
          if (images.img[idx]) node.replaceWith(avatar(idx));
        }
      })
      .catch(() => {}); // coloured initials are a perfectly good fallback
  if ("requestIdleCallback" in window) requestIdleCallback(go, { timeout: 2000 });
  else setTimeout(go, 200);
}

function fail(err) {
  console.error(err);
  $("recs-sub").textContent = "";
  $("recs").replaceChildren(
    el("li", "muted error-note", "Could not load the recommendation data. Please reload the page.")
  );
}

async function boot() {
  try {
    const query = parseQuery();
    const guess = query.user != null ? query.user : FALLBACK_USER;

    // All three go out together: the shard filename is derived from the user id, so it
    // does not have to wait for meta.json to arrive first.
    const metaP = getJSON("meta.json");
    const artistsP = getJSON("artists.json");
    let shardP = getShard(shardOf(guess));

    meta = await metaP;
    // Fall back exactly like app/service.py:141-143 and app/app.py:31-33 do.
    state.user = meta.users.includes(query.user) ? query.user : meta.default_user;
    state.method = meta.methods.some((m) => m.label === query.method)
      ? query.method
      : meta.methods[0].label;

    populateControls();
    $("chart-title").textContent = `Precision@${meta.k} across all methods`;
    renderMethod();
    syncURL();

    artists = await artistsP;
    if (shardOf(state.user) !== shardOf(guess)) shardP = getShard(shardOf(state.user));
    renderUser(await shardP);

    loadImages();
  } catch (err) {
    fail(err);
  }
}

boot();
