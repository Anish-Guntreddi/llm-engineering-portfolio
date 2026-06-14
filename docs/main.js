// Research-telemetry showcase — progressive enhancement only (page is readable without JS).
(() => {
  "use strict";
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Scroll progress hairline
  const bar = document.getElementById("progress");
  const onScroll = () => {
    const h = document.documentElement;
    const p = h.scrollTop / (h.scrollHeight - h.clientHeight || 1);
    bar.style.width = (p * 100).toFixed(2) + "%";
  };
  document.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // Fill a block's metric bars from their data-w (animates via CSS transition)
  const fillBars = (root) => {
    root.querySelectorAll(".fill[data-w]").forEach((el) => {
      const w = el.getAttribute("data-w") + "%";
      if (reduce) { el.style.width = w; return; }
      requestAnimationFrame(() => { el.style.width = w; });
    });
  };

  // Reveal on scroll + trigger bar fills
  const items = document.querySelectorAll(".reveal");
  if (reduce || !("IntersectionObserver" in window)) {
    items.forEach((el) => { el.classList.add("in"); fillBars(el); });
  } else {
    const io = new IntersectionObserver((entries, obs) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("in");
          fillBars(e.target);
          obs.unobserve(e.target);
        }
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -8% 0px" });
    items.forEach((el) => io.observe(el));
  }
})();
