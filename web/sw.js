/* CurbIQ service worker — installable PWA + offline app-shell & artifact cache. */
const CACHE = "curbiq-v2";
const SHELL = ["/", "/index.html", "/app.js", "/styles.css", "/manifest.webmanifest", "/icon.svg"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.origin !== self.location.origin) return;       // let CDN libs go to network
  if (url.pathname.startsWith("/api/")) {
    // network-first for data; fall back to last cached response when offline
    e.respondWith(
      fetch(e.request).then((r) => {
        const copy = r.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy));
        return r;
      }).catch(() => caches.match(e.request)));
  } else {
    // cache-first for the app shell
    e.respondWith(caches.match(e.request).then((r) => r || fetch(e.request)));
  }
});
