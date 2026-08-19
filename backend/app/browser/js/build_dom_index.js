/**
 * build_dom_index.js -- the agent's eyes.
 *
 * Injected into every frame of the page. Walks the DOM (descending through
 * open shadow roots), decides which elements a human could actually interact
 * with right now, assigns each a small integer index, and returns both a
 * machine-readable element list and a compact text rendering of the page in
 * document order.
 *
 * The integer index is the entire interaction contract with the LLM: the model
 * never writes a CSS selector or an XPath, it says `click(index=12)`. That is
 * what makes the agent resilient to obfuscated class names and to sites that
 * rewrite their markup between visits.
 *
 * Iframes are deliberately NOT traversed here. Playwright already exposes every
 * frame (including cross-origin ones, which JS cannot reach), so the Python
 * side runs this script once per frame and merges the results. That keeps this
 * file same-document and lets element handles be resolved in their own frame.
 *
 * Exposed globals:
 *   window.__agentBuildIndex(options) -> {elements, content, scroll, meta}
 *   window.__agentElements            -> Element[] indexed by LOCAL index
 *   window.__agentHighlight(items)    -> draw numbered boxes (set-of-marks)
 *   window.__agentClearHighlights()
 */
(() => {
  "use strict";

  // Re-injection is normal (every navigation re-runs this); make it idempotent.
  if (window.__agentBuildIndex) return;

  /** Tags that are interactive without needing any ARIA help. */
  const INTERACTIVE_TAGS = new Set([
    "a", "button", "input", "select", "textarea", "summary", "details",
    "label", "option", "video", "audio",
  ]);

  /** ARIA roles that imply the element responds to clicks or typing. */
  const INTERACTIVE_ROLES = new Set([
    "button", "link", "checkbox", "radio", "menuitem", "menuitemcheckbox",
    "menuitemradio", "option", "switch", "tab", "textbox", "searchbox",
    "combobox", "slider", "spinbutton", "treeitem", "listbox",
  ]);

  /** Never index or read these -- they carry no user-visible content. */
  const SKIPPED_TAGS = new Set([
    "script", "style", "noscript", "template", "meta", "link", "head", "svg",
    "path", "br", "hr",
  ]);

  /** Attributes worth showing the model, in the order they should be rendered. */
  const DESCRIPTIVE_ATTRS = [
    "aria-label", "placeholder", "name", "title", "alt", "value", "type",
    "role", "aria-expanded", "aria-checked", "aria-selected", "data-testid",
  ];

  const MAX_TEXT = 160;            // per-element text cap
  const MAX_CONTENT_CHARS = 24000; // total page rendering cap
  const HIGHLIGHT_CONTAINER_ID = "__agent_highlight_layer__";

  // -------------------------------------------------------------- helpers --

  const clip = (s, n = MAX_TEXT) => {
    if (!s) return "";
    const flat = s.replace(/\s+/g, " ").trim();
    return flat.length > n ? flat.slice(0, n - 1) + "…" : flat;
  };

  /** Root node an element lives in: its shadow root, or the document. */
  const rootOf = (el) => el.getRootNode() || document;

  /**
   * Is this element rendered and non-transparent?
   *
   * Cheap checks first (zero-size rect) because getComputedStyle is the
   * expensive call and we run this over every node on the page.
   */
  function isVisible(el, rect, style) {
    // Zero-size wrappers around a real control are common; they are rejected
    // here and picked up instead when the walker reaches the real child.
    if (rect.width <= 1 || rect.height <= 1) return false;
    if (style.visibility === "hidden" || style.display === "none") return false;
    if (parseFloat(style.opacity) === 0) return false;
    // `inert` and `hidden` remove an element from interaction entirely.
    if (el.hasAttribute("inert") || el.hasAttribute("hidden")) return false;
    if (el.getAttribute("aria-hidden") === "true") return false;
    return true;
  }

  /** Would a real click at the element centre actually land on it? */
  function isTopmost(el, rect) {
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    // Outside the viewport there is nothing to hit-test against; assume yes
    // and let the scroll-into-view at action time sort it out.
    if (cx < 0 || cy < 0 || cx > window.innerWidth || cy > window.innerHeight) {
      return true;
    }
    const root = rootOf(el);
    const hit = (root.elementFromPoint ? root : document).elementFromPoint(cx, cy);
    if (!hit) return false;
    // A hit on a descendant (the <span> inside a <button>) still counts, and so
    // does a hit on an ancestor that merely wraps us.
    return hit === el || el.contains(hit) || hit.contains(el);
  }

  /** Does this element accept clicks or typing? */
  function isInteractive(el, style) {
    const tag = el.tagName.toLowerCase();

    if (el.disabled === true || el.getAttribute("aria-disabled") === "true") {
      return false;
    }
    if (INTERACTIVE_TAGS.has(tag)) {
      // A bare <label> is only useful when it drives a control.
      if (tag === "label") return el.control != null || el.hasAttribute("for");
      // Hidden inputs are real inputs but not human-operable.
      if (tag === "input" && el.type === "hidden") return false;
      return true;
    }
    const role = el.getAttribute("role");
    if (role && INTERACTIVE_ROLES.has(role)) return true;
    if (el.isContentEditable) return true;
    if (el.hasAttribute("onclick")) return true;

    const tabindex = el.getAttribute("tabindex");
    if (tabindex !== null && tabindex !== "-1") return true;

    // Last resort: sites frequently build buttons out of divs and betray it
    // only through the cursor. Restrict to leaf-ish nodes so we do not index
    // an entire clickable card and every element inside it.
    if (style.cursor === "pointer" && el.childElementCount <= 3) return true;

    return false;
  }

  /** Visible text directly owned by an element, excluding nested controls. */
  function ownText(el) {
    let out = "";
    for (const node of el.childNodes) {
      if (node.nodeType === Node.TEXT_NODE) out += node.nodeValue;
    }
    return clip(out);
  }

  /**
   * Does this element hold something that must never reach the model?
   *
   * Checked in one place and consulted by every path that could echo a value,
   * because a single unguarded fallback is enough to leak a credential.
   */
  function isSecret(el) {
    if (el.type === "password") return true;
    const hint = [
      el.getAttribute("name"), el.getAttribute("id"),
      el.getAttribute("autocomplete"), el.getAttribute("aria-label"),
    ].join(" ").toLowerCase();
    return /pass|cvv|cvc|otp|one-?time|secur|\bpin\b|card-?num/.test(hint);
  }

  /** Best human-readable label, in decreasing order of trustworthiness. */
  function labelFor(el) {
    const secret = isSecret(el);
    const aria = el.getAttribute("aria-label");
    if (aria) return clip(aria);

    const labelledBy = el.getAttribute("aria-labelledby");
    if (labelledBy) {
      const parts = labelledBy.split(/\s+/)
        .map((id) => rootOf(el).getElementById?.(id)?.textContent || "")
        .filter(Boolean);
      if (parts.length) return clip(parts.join(" "));
    }
    const text = clip(el.innerText || el.textContent || "");
    if (text) return text;

    // `value` is last, and never consulted for a secret field: a password
    // input with no label would otherwise be described by its own contents.
    const fallbacks = secret
      ? ["placeholder", "title", "alt", "name"]
      : ["placeholder", "title", "alt", "name", "value"];
    for (const attr of fallbacks) {
      const v = el.getAttribute?.(attr);
      if (v) return clip(v);
    }
    return secret ? "password field" : "";
  }

  /** Attribute subset rendered into the text view, as key="value" pairs. */
  function describeAttrs(el, label) {
    const out = [];
    const tag = el.tagName.toLowerCase();

    const secret = isSecret(el);

    for (const attr of DESCRIPTIVE_ATTRS) {
      let v = el.getAttribute(attr);
      if (!v) continue;
      // Never echo a credential back to the model.
      if (attr === "value" && secret) v = "•••";
      v = clip(v, 60);
      // Skip attributes that merely repeat the label we already show.
      if (v && v !== label) out.push(attr + '="' + v + '"');
    }
    if (tag === "a" && el.getAttribute("href")) {
      out.push('href="' + clip(el.getAttribute("href"), 80) + '"');
    }
    if (tag === "input" && el.value && !secret) {
      out.push('current="' + clip(el.value, 40) + '"');
    }
    if (el.checked === true) out.push("checked");
    return out.join(" ");
  }

  // --------------------------------------------------------------- walker --

  /**
   * Depth-first walk in document order, collecting interactive elements and
   * emitting the text rendering as we go.
   */
  function walk(ctx, node) {
    if (ctx.chars > MAX_CONTENT_CHARS) return;

    if (node.nodeType === Node.TEXT_NODE) {
      const text = clip(node.nodeValue, 300);
      // Length > 1 filters out stray punctuation and layout whitespace.
      if (text.length > 1) ctx.push(text);
      return;
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return;

    const el = node;
    const tag = el.tagName.toLowerCase();
    if (SKIPPED_TAGS.has(tag)) return;
    if (el.id === HIGHLIGHT_CONTAINER_ID) return; // our own overlay

    const style = window.getComputedStyle(el);
    // display:none prunes the whole subtree, so bail before recursing.
    if (style.display === "none") return;

    const rect = el.getBoundingClientRect();
    const visible = isVisible(el, rect, style);

    if (visible && isInteractive(el, style) && isTopmost(el, rect)) {
      const index = ctx.startIndex + ctx.elements.length;
      const label = labelFor(el);
      const attrs = describeAttrs(el, label);

      ctx.elements.push({
        index,
        localIndex: ctx.elements.length,
        tag,
        label,
        attrs,
        inViewport:
          rect.bottom > 0 && rect.right > 0 &&
          rect.top < window.innerHeight && rect.left < window.innerWidth,
        box: {
          x: Math.round(rect.left),
          y: Math.round(rect.top),
          width: Math.round(rect.width),
          height: Math.round(rect.height),
        },
      });
      ctx.handles.push(el);
      ctx.push("[" + index + "]<" + tag + (attrs ? " " + attrs : "") + ">" + label + "</" + tag + ">");

      // Containers such as <select> still need their children listed, but for
      // an ordinary leaf control the label already captured everything inside.
      if (tag !== "select" && el.childElementCount === 0) return;
    } else if (visible) {
      const text = ownText(el);
      if (text.length > 1) ctx.push(text);
    }

    // Recurse: shadow root first (it replaces light-DOM rendering), then
    // ordinary children.
    if (el.shadowRoot) {
      for (const child of el.shadowRoot.childNodes) walk(ctx, child);
    }
    for (const child of el.childNodes) walk(ctx, child);
  }

  // ------------------------------------------------------------- entry pt --

  /**
   * @param {{startIndex?: number}} options
   *        startIndex -- global index to start numbering at, so indices stay
   *        unique when the Python side merges several frames.
   */
  window.__agentBuildIndex = (options) => {
    const opts = options || {};
    const ctx = {
      startIndex: opts.startIndex || 0,
      elements: [],
      handles: [],
      lines: [],
      chars: 0,
      push(line) {
        // Collapse consecutive duplicates: repeated product cards and nav
        // items would otherwise flood the model context with noise.
        if (this.lines[this.lines.length - 1] === line) return;
        this.lines.push(line);
        this.chars += line.length + 1;
      },
    };

    const root = document.body || document.documentElement;
    if (root) walk(ctx, root);

    window.__agentElements = ctx.handles;

    let content = ctx.lines.join("\n");
    const truncated = content.length > MAX_CONTENT_CHARS;
    if (truncated) content = content.slice(0, MAX_CONTENT_CHARS);

    const doc = document.documentElement;
    return {
      elements: ctx.elements,
      content,
      truncated,
      scroll: {
        x: Math.round(window.scrollX),
        y: Math.round(window.scrollY),
        // How much page remains below the fold, in pixels. The model uses this
        // to decide whether scrolling could reveal anything new.
        pixelsBelow: Math.max(
          0, Math.round(doc.scrollHeight - window.scrollY - window.innerHeight)
        ),
        pixelsAbove: Math.round(window.scrollY),
        viewportHeight: window.innerHeight,
        documentHeight: doc.scrollHeight,
      },
      meta: {
        url: location.href,
        title: document.title,
        readyState: document.readyState,
      },
    };
  };

  // --------------------------------------------------------- set-of-marks --

  /**
   * Draw numbered boxes over the given elements so that a screenshot carries
   * the same indices the text rendering uses. This is the vision fallback: the
   * model sees a picture in which every clickable thing is labelled with the
   * number it must pass to `click`.
   *
   * @param {{localIndex:number, index:number}[]} items
   */
  window.__agentHighlight = (items) => {
    window.__agentClearHighlights();

    const layer = document.createElement("div");
    layer.id = HIGHLIGHT_CONTAINER_ID;
    Object.assign(layer.style, {
      position: "fixed", inset: "0", pointerEvents: "none", zIndex: "2147483647",
    });

    // Cycled so that adjacent boxes stay visually separable.
    const palette = ["#FF3B30", "#007AFF", "#34C759", "#FF9500", "#AF52DE", "#00C7BE"];

    (items || []).forEach((item, n) => {
      const el = (window.__agentElements || [])[item.localIndex];
      if (!el) return;
      const rect = el.getBoundingClientRect();
      if (rect.width < 1 || rect.height < 1) return;

      const color = palette[n % palette.length];

      const box = document.createElement("div");
      Object.assign(box.style, {
        position: "fixed",
        left: rect.left + "px", top: rect.top + "px",
        width: rect.width + "px", height: rect.height + "px",
        border: "2px solid " + color,
        boxSizing: "border-box",
      });

      const tag = document.createElement("div");
      tag.textContent = String(item.index);
      // Tuck the label inside the box when the element sits at the very top of
      // the viewport, otherwise it would be clipped off-screen.
      const above = rect.top > 18;
      Object.assign(tag.style, {
        position: "fixed",
        left: rect.left + "px",
        top: (above ? rect.top - 17 : rect.top) + "px",
        background: color, color: "#fff",
        font: '600 12px/16px ui-sans-serif, system-ui, sans-serif',
        padding: "0 4px", borderRadius: "3px", whiteSpace: "nowrap",
      });

      layer.append(box, tag);
    });

    (document.body || document.documentElement).appendChild(layer);
  };

  window.__agentClearHighlights = () => {
    const existing = document.getElementById(HIGHLIGHT_CONTAINER_ID);
    if (existing) existing.remove();
  };
})();
