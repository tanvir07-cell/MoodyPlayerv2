globalThis.addEventListener("install", async (event) => {
  const assets = [
    "/",
    "./main.js",
    "./style.css",
    "https://www.youtube.com",

    "https://www.youtube.com/embed",

    "https://www.youtube.com/embed/LjhCEhWiKXk",

    "https://www.youtube.com/embed/Y66j_BUCBMY",

    "https://www.youtube.com/embed/iPUmE-tne5U",

    "https://www.youtube.com/embed/nfWlot6h_JM",

    "https://www.youtube.com/embed/wsdy_rct6uo",

    "https://www.youtube.com/embed/ru0K8uYEZWw",

    "https://www.youtube.com/embed/Pw-0pbY9JeU",

    "https://www.youtube.com/embed/hT_nvWreIhg",
    "https://www.youtube.com/embed/HCjNJDNzw8Y",

    "https://www.youtube.com/embed/YQHsXMglC9A",
    "https://www.youtube.com/embed/hLQl3WQQoQ0",
    "https://www.youtube.com/embed/RBumgq5yVrA",
    "https://www.youtube.com/embed/6EEW-9NDM5k",
    "https://www.youtube.com/embed/0G3_kG5FFfQ",

    "https://www.youtube.com/embed/VT1-sitWRtY",

    "https://www.youtube.com/embed/HLphrgQFHUQ",

    "https://www.youtube.com/embed/koJlIGDImiU",

    "https://www.youtube.com/embed/My2FRPA3Gf8",

    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json",
  ];
  const cache = await caches.open("moody-assets");
  cache.addAll(assets);
});

self.addEventListener("fetch", (event) => {
  event.respondWith(
    (async () => {
      const cache = await caches.open("moody-assets");

      // from the cache;

      const cachedResponse = await cache.match(event.request);

      // Fetch the latest resource from the network
      const fetchPromise = fetch(event.request)
        .then((networkResponse) => {
          // Update the cache with the latest version
          cache.put(event.request, networkResponse.clone());
          return networkResponse;
        })
        .catch(() => cachedResponse); // In case of network failure, use cached response

      // return cached immediately and update cache in the background
      return cachedResponse || fetchPromise;
    })()
  );
});
