"use strict";
var e = Object.defineProperty, t = (t, s, n) => ((t, s, n) => s in t ? e(t, s, {
    enumerable: !0,
    configurable: !0,
    writable: !0,
    value: n
}) : t[s] = n)(t, "symbol" != typeof s ? s + "" : s, n);

class s extends Error {
    constructor(e = {}) {
        super(e.message), t(this, "exceptionId"), t(this, "message"), t(this, "source"), t(this, "type"), this.message = e.message ?? "", this.source = e.source, this.type = this.constructor.name, this.exceptionId = e.exceptionId
    }
}

class n extends s {
}

class o extends s {
}

class r extends s {
}

const i = e => {
    "string" != typeof e ? e && "function" == typeof e.toString ? console.warn(`[nlux] ${e.toString()}`) : console.warn("[nlux]") : console.warn(`[nlux] ${e}`)
}, a = [], c = e => {
    a.includes(e) || (a.push(e), i(e))
}, l = class e {
    static register(t) {
        const s = t.__compId;
        s ? void 0 === e.componentDefs.get(s) && (t.__renderer && t.__updater ? e.componentDefs.set(s, {
            id: s,
            model: t,
            render: t.__renderer,
            update: t.__updater
        }) : i(`Component with id "${s}" missing renderer or updater`)) : i("Component definition missing valid id")
    }

    static retrieve(t) {
        const s = e.componentDefs.get(t);
        if (s) return s;
        i(`Component with id "${t}" not registered`)
    }
};
l.componentDefs = new Map;
let h = l;
const d = {version: "{versions.nlux}", [btoa("sectionsRegistered")]: !1}, u = e => {
    const t = requestAnimationFrame((() => {
        e()
    }));
    return () => {
        cancelAnimationFrame(t)
    }
}, p = e => {
    e.replaceChildren()
}, m = () => "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (e => {
    const t = 16 * Math.random() | 0;
    return ("x" == e ? t : 3 & t | 8).toString(16)
}));

class g {
    constructor(e, t) {
        this.subComponentElementIds = new Map, this.subComponents = new Map, this.__context = null, this.__destroyed = !1, this.__status = "unmounted", this.actionsOnDomReady = [], this.compEventGetter = e => {
            if (this.destroyed) return () => {
            };
            const t = this.rendererEventListeners.get(e);
            if (!t) throw new s({
                source: this.constructor.name,
                message: `Unable to call renderer event "${e}" because no matching event listener was found. Make sure that the event listener is registered using @CompEventListener() decorator in the component model class, and use class methods instead of arrow function attributes.`
            });
            return t
        };
        const o = Object.getPrototypeOf(this).constructor.__compId;
        if (!o) throw new n({
            source: this.constructor.name,
            message: "Unable to instantiate component: missing compId in implementation. Component should be annotated using @Model() to set compId before iy can be instantiated."
        });
        if (this.def = h.retrieve(o) ?? null, !this.def) throw new n({
            source: this.constructor.name,
            message: `Unable to instantiate component "${o}" because it's not registered. Component should be registered using CompRegistry.register(ComponentClass) before instantiating a component.`
        });
        this.__instanceId = m(), this.__destroyed = !1, this.__context = e, this.renderedDom = null, this.renderingRoot = null, this.props = t;
        const r = t ? Object.entries(t) : [];
        this.elementProps = new Map(r), this.rendererEventListeners = new Map;
        const a = this.constructor.__compEventListeners;
        a && a.forEach(((e, t) => {
            e.forEach((e => {
                const s = Object.getPrototypeOf(this)[e];
                "function" == typeof s ? this.addRendererEventListener(t, s.bind(this)) : i(`Unable to set event listener "${t}" because method "${e}" cannot be found on component "${this.constructor.name} at runtime!"`)
            }))
        })), this.rendererProps = Object.freeze(t)
    }

    get destroyed() {
        return this.__destroyed
    }

    get id() {
        return this.__instanceId
    }

    get rendered() {
        return null !== this.renderedDom
    }

    get root() {
        return this.throwIfDestroyed(), this.renderedDom && this.renderingRoot ? this.renderingRoot : null
    }

    get status() {
        return this.__status
    }

    get context() {
        if (!this.__context) throw new n({
            source: this.constructor.name,
            message: "Unable to get context because it's not set"
        });
        return this.__context
    }

    destroy() {
        this.destroyComponent()
    }

    destroyListItemComponent() {
        this.destroyComponent(!0)
    }

    getProp(e) {
        return this.throwIfDestroyed(), this.elementProps.get(e) ?? null
    }

    render(e, t) {
        if (!this.def) return;
        if (this.destroyed) return void i(`Unable to render component "${this.def?.id}" because it is already destroyed`);
        if (this.rendered || this.renderedDom) return void i(`Unable to render component "${this.def.id}" because it is already rendered`);
        const n = document.createDocumentFragment(), o = Object.getPrototypeOf(this).constructor.__compId,
            r = this.executeRenderer(n);
        if (!r) throw new s({
            source: this.constructor.name,
            message: `Unable to render component "${o}" because renderer returned null`
        });
        this.renderedDom = r;
        for (const [, e] of this.subComponents) {
            const t = this.getSubComponentPortal(e.id);
            t && this.mountSubComponentToPortal(e.id, t)
        }
        u((() => {
            this.destroyed || (t ? e.insertBefore(n, t) : e.append(n), this.renderingRoot = e)
        }))
    }

    updateSubComponent(e, t, s) {
        this.throwIfDestroyed();
        const n = this.subComponents.get(e);
        n && !n.destroyed && n.setProp(t, s)
    }

    addSubComponent(e, t, s) {
        if (this.throwIfDestroyed(), this.subComponents.has(e)) throw new n({
            source: this.constructor.name,
            message: `Unable to add sub-component "${e}" because it already exists`
        });
        if (this.subComponents.set(e, t), s && this.subComponentElementIds.set(e, s), this.renderedDom) {
            const t = this.getSubComponentPortal(e);
            t && this.mountSubComponentToPortal(e, t)
        }
    }

    executeDomAction(e, ...t) {
        if (this.throwIfDestroyed(), !this.renderedDom) return void this.actionsOnDomReady.push((() => this.executeDomAction(e, ...t)));
        if (!this.renderingRoot) throw new s({
            source: this.constructor.name,
            message: "Unable to execute DOM action because renderingRoot is not set"
        });
        const n = this.renderedDom.actions[e];
        if (!n) throw new s({
            source: this.constructor.name,
            message: `Unable to execute DOM action "${String(e)}" because it does not exist`
        });
        return u((() => n(...t)))
    }

    executeRenderer(e) {
        const t = this.def?.render;
        if (!t) return null;
        if (this.renderingRoot) throw new s({
            source: this.constructor.name,
            message: "Unable to render component because renderingRoot is already set"
        });
        const n = t({
            appendToRoot: t => {
                e.append(t), this.runDomActionsQueue()
            }, compEvent: this.compEventGetter, props: this.rendererProps, context: this.context
        });
        return n && (this.renderingRoot = e), n
    }

    removeSubComponent(e) {
        this.throwIfDestroyed(), u((() => {
            const t = this.subComponents.get(e);
            t && (t.renderingRoot = null, t.destroy(), this.subComponents.delete(e))
        }))
    }

    runDomActionsQueue() {
        if (this.actionsOnDomReady.length > 0 && this.rendered) {
            const e = this.actionsOnDomReady;
            this.actionsOnDomReady = [];
            for (const t of e) u((() => t()))
        }
    }

    setProp(e, t) {
        this.destroyed ? i(`Unable to set prop "${String(e)}" because component "${this.constructor.name}" is destroyed`) : this.elementProps.has(e) ? (this.schedulePropUpdate(e, this.elementProps.get(e), t), this.props = Object.freeze(Object.fromEntries(this.elementProps)), this.elementProps.set(e, t)) : i(`Unable to set prop "${String(e)}" because it does not exist in the component props`)
    }

    throwIfDestroyed() {
        if (this.__destroyed) throw new n({
            source: this.constructor.name,
            message: "Unable to call method on destroyed component"
        })
    }

    addRendererEventListener(e, t) {
        if (this.throwIfDestroyed(), this.rendererEventListeners.has(e)) throw new n({
            source: this.constructor.name,
            message: `Unable to add event listener to rendererEvents "${e}" because it already exists`
        });
        this.rendererEventListeners.set(e, t)
    }

    destroyComponent(e = !1) {
        if (this.throwIfDestroyed(), this.subComponents.forEach((e => {
            e.destroy()
        })), this.renderedDom) {
            this.renderedDom.elements && (this.renderedDom.elements = void 0), this.renderedDom.actions && (this.renderedDom.actions = void 0), this.renderedDom.onDestroy && this.renderedDom.onDestroy();
            const t = this.renderingRoot;
            u((() => {
                if (t) if (t instanceof DocumentFragment) for (; t.firstChild;) t.removeChild(t.firstChild); else e ? t.parentElement?.removeChild(t) : p(t)
            })), this.renderedDom = null, this.renderingRoot = null
        }
        this.__destroyed = !0, this.__context = null, this.props = void 0, this.elementProps.clear(), this.rendererEventListeners.clear(), this.subComponents.clear()
    }

    getSubComponentPortal(e) {
        const t = this.subComponents.get(e), s = this.subComponentElementIds.get(e);
        if (!t || !s) return null;
        const n = (this.renderedDom?.elements)[s];
        return n instanceof HTMLElement ? n : null
    }

    mountSubComponentToPortal(e, t) {
        const s = this.subComponents.get(e);
        s?.render(t)
    }

    schedulePropUpdate(e, t, s) {
        if (!this.renderedDom || !this.def?.update) return;
        const n = this.renderedDom, o = this.renderingRoot, r = this.def.update;
        o && u((() => {
            r({
                propName: e,
                currentValue: t,
                newValue: s,
                dom: {root: o, elements: n.elements, actions: n.actions},
                updateSubComponent: this.updateSubComponent
            })
        }))
    }
}

g.__compEventListeners = null, g.__compId = null, g.__renderer = null, g.__updater = null;
const f = (e, t, s) => n => {
    n.__compId = e, n.__renderer = t, n.__updater = s
}, y = e => (t, s) => {
    const o = t;
    if ("function" != typeof o.constructor) throw new n({
        source: "CallbackFor",
        message: "@CallbackFor can only be used on methods of a class!"
    });
    o.constructor.hasOwnProperty("__compEventListeners") && null !== o.constructor.__compEventListeners || (o.constructor.__compEventListeners = new Map);
    const r = o.constructor.__compEventListeners, i = r.get(e);
    i ? i.push(s) : r.set(e, [s])
}, w = (e, t) => {
    const s = document.createElement("div");
    if (s.classList.add("nlux-comp-avatarContainer"), e) {
        const t = document.createElement("div");
        t.classList.add("nlux-comp-avatarPicture"), t.style.backgroundImage = `url("${encodeURI(e)}")`, s.append(t)
    }
    return s
}, k = "nlux-comp-avatar", x = e => {
    const t = document.createElement("div");
    return t.classList.add(k), e.avatar || e.name ? (e.name && (t.title = e.name), e.avatar && e.avatar instanceof HTMLElement ? (t.append(e.avatar.cloneNode(!0)), t) : (t.append(w(e.avatar)), t)) : t
}, v = {received: "nlux_msg_received", sent: "nlux_msg_sent"}, b = (e, t) => {
    Object.keys(v).forEach((t => {
        e.classList.remove(v[t])
    })), v[t] && e.classList.add(v[t])
}, C = {streaming: "nlux_msg_streaming", complete: "nlux_msg_complete"}, S = (e, t) => {
    Object.keys(C).forEach((t => {
        e.classList.remove(C[t])
    })), C[t] && e.classList.add(C[t])
}, I = e => {
    if (!(e instanceof HTMLButtonElement)) return;
    if ("true" === e.dataset.clickListenerSet) return;
    let t = !1;
    const s = e.nextElementSibling;
    e.addEventListener("click", (() => {
        if (t || !s) return;
        const n = s.innerText;
        navigator.clipboard.writeText(n ?? ""), t = !0, e.classList.add("clicked"), setTimeout((() => {
            t = !1, e.classList.remove("clicked")
        }), 1e3)
    })), e.dataset.clickListenerSet = "true"
}, A = e => {
    const t = "nlux-comp-copyButton";
    e instanceof HTMLButtonElement && e.classList.contains(t) ? I(e) : e.querySelectorAll(`.${t}`).forEach(I)
};

function P() {
    return {
        async: !1,
        breaks: !1,
        extensions: null,
        gfm: !0,
        hooks: null,
        pedantic: !1,
        renderer: null,
        silent: !1,
        tokenizer: null,
        walkTokens: null
    }
}

let E = {
    async: !1,
    breaks: !1,
    extensions: null,
    gfm: !0,
    hooks: null,
    pedantic: !1,
    renderer: null,
    silent: !1,
    tokenizer: null,
    walkTokens: null
};

function T(e) {
    E = e
}

var O = Object.defineProperty, B = (e, t, s) => ((e, t, s) => t in e ? O(e, t, {
    enumerable: !0,
    configurable: !0,
    writable: !0,
    value: s
}) : e[t] = s)(e, "symbol" != typeof t ? t + "" : t, s);

class L {
    constructor(e) {
        B(this, "options"), this.options = e || E
    }

    postprocess(e) {
        return e
    }

    preprocess(e) {
        return e
    }

    processAllTokens(e) {
        return e
    }
}

B(L, "passThroughHooks", new Set(["preprocess", "postprocess", "processAllTokens"]));
const D = /[&<>"']/, R = new RegExp(D.source, "g"), z = /[<>"']|&(?!(#\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\w+);)/,
    M = new RegExp(z.source, "g"), H = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"},
    U = e => H[e];

function Q(e, t) {
    if (t) {
        if (D.test(e)) return e.replace(R, U)
    } else if (z.test(e)) return e.replace(M, U);
    return e
}

const N = /&(#(?:\d+)|(?:#x[0-9A-Fa-f]+)|(?:\w+));?/gi;

function F(e) {
    return e.replace(N, ((e, t) => "colon" === (t = t.toLowerCase()) ? ":" : "#" === t.charAt(0) ? "x" === t.charAt(1) ? String.fromCharCode(parseInt(t.substring(2), 16)) : String.fromCharCode(+t.substring(1)) : ""))
}

const q = /(^|[^\[])\^/g;

function j(e, t) {
    let s = "string" == typeof e ? e : e.source;
    t = t || "";
    const n = {
        replace: (e, t) => {
            let o = "string" == typeof t ? t : t.source;
            return o = o.replace(q, "$1"), s = s.replace(e, o), n
        }, getRegex: () => new RegExp(s, t)
    };
    return n
}

function Y(e) {
    try {
        e = encodeURI(e).replace(/%25/g, "%")
    } catch (e) {
        return null
    }
    return e
}

const $ = {exec: () => null};

function W(e, t) {
    const s = e.replace(/\|/g, ((e, t, s) => {
        let n = !1, o = t;
        for (; --o >= 0 && "\\" === s[o];) n = !n;
        return n ? "|" : " |"
    })).split(/ \|/);
    let n = 0;
    if (s[0].trim() || s.shift(), s.length > 0 && !s[s.length - 1].trim() && s.pop(), t) if (s.length > t) s.splice(t); else for (; s.length < t;) s.push("");
    for (; n < s.length; n++) s[n] = s[n].trim().replace(/\\\|/g, "|");
    return s
}

function X(e, t, s) {
    const n = e.length;
    if (0 === n) return "";
    let o = 0;
    for (; o < n;) {
        const r = e.charAt(n - o - 1);
        if (r !== t || s) {
            if (r === t || !s) break;
            o++
        } else o++
    }
    return e.slice(0, n - o)
}

const Z = /^ {0,3}((?:-[\t ]*){3,}|(?:_[ \t]*){3,}|(?:\*[ \t]*){3,})(?:\n+|$)/, G = /(?:[*+-]|\d{1,9}[.)])/,
    J = j(/^(?!bull |blockCode|fences|blockquote|heading|html)((?:.|\n(?!\s*?\n|bull |blockCode|fences|blockquote|heading|html))+?)\n {0,3}(=+|-+) *(?:\n+|$)/).replace(/bull/g, G).replace(/blockCode/g, / {4}/).replace(/fences/g, / {0,3}(?:`{3,}|~{3,})/).replace(/blockquote/g, / {0,3}>/).replace(/heading/g, / {0,3}#{1,6}/).replace(/html/g, / {0,3}<[^\n>]+>\n/).getRegex(),
    K = /^([^\n]+(?:\n(?!hr|heading|lheading|blockquote|fences|list|html|table| +\n)[^\n]+)*)/,
    V = /(?!\s*\])(?:\\.|[^\[\]\\])+/,
    _ = j(/^ {0,3}\[(label)\]: *(?:\n *)?([^<\s][^\s]*|<.*?>)(?:(?: +(?:\n *)?| *\n *)(title))? *(?:\n+|$)/).replace("label", V).replace("title", /(?:"(?:\\"?|[^"\\])*"|'[^'\n]*(?:\n[^'\n]+)*\n?'|\([^()]*\))/).getRegex(),
    ee = j(/^( {0,3}bull)([ \t][^\n]+?)?(?:\n|$)/).replace(/bull/g, G).getRegex(),
    te = "address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|search|section|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul",
    se = /<!--(?:-?>|[\s\S]*?(?:-->|$))/,
    ne = j("^ {0,3}(?:<(script|pre|style|textarea)[\\s>][\\s\\S]*?(?:</\\1>[^\\n]*\\n+|$)|comment[^\\n]*(\\n+|$)|<\\?[\\s\\S]*?(?:\\?>\\n*|$)|<![A-Z][\\s\\S]*?(?:>\\n*|$)|<!\\[CDATA\\[[\\s\\S]*?(?:\\]\\]>\\n*|$)|</?(tag)(?: +|\\n|/?>)[\\s\\S]*?(?:(?:\\n *)+\\n|$)|<(?!script|pre|style|textarea)([a-z][\\w-]*)(?:attribute)*? */?>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n *)+\\n|$)|</(?!script|pre|style|textarea)[a-z][\\w-]*\\s*>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n *)+\\n|$))", "i").replace("comment", se).replace("tag", te).replace("attribute", / +[a-zA-Z:_][\w.:-]*(?: *= *"[^"\n]*"| *= *'[^'\n]*'| *= *[^\s"'=<>`]+)?/).getRegex(),
    oe = j(K).replace("hr", Z).replace("heading", " {0,3}#{1,6}(?:\\s|$)").replace("|lheading", "").replace("|table", "").replace("blockquote", " {0,3}>").replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list", " {0,3}(?:[*+-]|1[.)]) ").replace("html", "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag", te).getRegex(),
    re = {
        blockquote: j(/^( {0,3}> ?(paragraph|[^\n]*)(?:\n|$))+/).replace("paragraph", oe).getRegex(),
        code: /^( {4}[^\n]+(?:\n(?: *(?:\n|$))*)?)+/,
        def: _,
        fences: /^ {0,3}(`{3,}(?=[^`\n]*(?:\n|$))|~{3,})([^\n]*)(?:\n|$)(?:|([\s\S]*?)(?:\n|$))(?: {0,3}\1[~`]* *(?=\n|$)|$)/,
        heading: /^ {0,3}(#{1,6})(?=\s|$)(.*)(?:\n+|$)/,
        hr: Z,
        html: ne,
        lheading: J,
        list: ee,
        newline: /^(?: *(?:\n|$))+/,
        paragraph: oe,
        table: $,
        text: /^[^\n]+/
    },
    ie = j("^ *([^\\n ].*)\\n {0,3}((?:\\| *)?:?-+:? *(?:\\| *:?-+:? *)*(?:\\| *)?)(?:\\n((?:(?! *\\n|hr|heading|blockquote|code|fences|list|html).*(?:\\n|$))*)\\n*|$)").replace("hr", Z).replace("heading", " {0,3}#{1,6}(?:\\s|$)").replace("blockquote", " {0,3}>").replace("code", " {4}[^\\n]").replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list", " {0,3}(?:[*+-]|1[.)]) ").replace("html", "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag", te).getRegex(),
    ae = {
        ...re,
        table: ie,
        paragraph: j(K).replace("hr", Z).replace("heading", " {0,3}#{1,6}(?:\\s|$)").replace("|lheading", "").replace("table", ie).replace("blockquote", " {0,3}>").replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list", " {0,3}(?:[*+-]|1[.)]) ").replace("html", "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag", te).getRegex()
    }, ce = {
        ...re,
        html: j("^ *(?:comment *(?:\\n|\\s*$)|<(tag)[\\s\\S]+?</\\1> *(?:\\n{2,}|\\s*$)|<tag(?:\"[^\"]*\"|'[^']*'|\\s[^'\"/>\\s]*)*?/?> *(?:\\n{2,}|\\s*$))").replace("comment", se).replace(/tag/g, "(?!(?:a|em|strong|small|s|cite|q|dfn|abbr|data|time|code|var|samp|kbd|sub|sup|i|b|u|mark|ruby|rt|rp|bdi|bdo|span|br|wbr|ins|del|img)\\b)\\w+(?!:|[^\\w\\s@]*@)\\b").getRegex(),
        def: /^ *\[([^\]]+)\]: *<?([^\s>]+)>?(?: +(["(][^\n]+[")]))? *(?:\n+|$)/,
        heading: /^(#{1,6})(.*)(?:\n+|$)/,
        fences: $,
        lheading: /^(.+?)\n {0,3}(=+|-+) *(?:\n+|$)/,
        paragraph: j(K).replace("hr", Z).replace("heading", " *#{1,6} *[^\n]").replace("lheading", J).replace("|table", "").replace("blockquote", " {0,3}>").replace("|fences", "").replace("|list", "").replace("|html", "").replace("|tag", "").getRegex()
    }, le = /^\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])/, he = /^( {2,}|\\)\n(?!\s*$)/, de = "\\p{P}\\p{S}",
    ue = j(/^((?![*_])[\spunctuation])/, "u").replace(/punctuation/g, de).getRegex(),
    pe = j(/^(?:\*+(?:((?!\*)[punct])|[^\s*]))|^_+(?:((?!_)[punct])|([^\s_]))/, "u").replace(/punct/g, de).getRegex(),
    me = j("^[^_*]*?__[^_*]*?\\*[^_*]*?(?=__)|[^*]+(?=[^*])|(?!\\*)[punct](\\*+)(?=[\\s]|$)|[^punct\\s](\\*+)(?!\\*)(?=[punct\\s]|$)|(?!\\*)[punct\\s](\\*+)(?=[^punct\\s])|[\\s](\\*+)(?!\\*)(?=[punct])|(?!\\*)[punct](\\*+)(?!\\*)(?=[punct])|[^punct\\s](\\*+)(?=[^punct\\s])", "gu").replace(/punct/g, de).getRegex(),
    ge = j("^[^_*]*?\\*\\*[^_*]*?_[^_*]*?(?=\\*\\*)|[^_]+(?=[^_])|(?!_)[punct](_+)(?=[\\s]|$)|[^punct\\s](_+)(?!_)(?=[punct\\s]|$)|(?!_)[punct\\s](_+)(?=[^punct\\s])|[\\s](_+)(?!_)(?=[punct])|(?!_)[punct](_+)(?!_)(?=[punct])", "gu").replace(/punct/g, de).getRegex(),
    fe = j(/\\([punct])/, "gu").replace(/punct/g, de).getRegex(),
    ye = j(/^<(scheme:[^\s\x00-\x1f<>]*|email)>/).replace("scheme", /[a-zA-Z][a-zA-Z0-9+.-]{1,31}/).replace("email", /[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+(@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?![-_])/).getRegex(),
    we = j(se).replace("(?:--\x3e|$)", "--\x3e").getRegex(),
    ke = j("^comment|^</[a-zA-Z][\\w:-]*\\s*>|^<[a-zA-Z][\\w-]*(?:attribute)*?\\s*/?>|^<\\?[\\s\\S]*?\\?>|^<![a-zA-Z]+\\s[\\s\\S]*?>|^<!\\[CDATA\\[[\\s\\S]*?\\]\\]>").replace("comment", we).replace("attribute", /\s+[a-zA-Z:_][\w.:-]*(?:\s*=\s*"[^"]*"|\s*=\s*'[^']*'|\s*=\s*[^\s"'=<>`]+)?/).getRegex(),
    xe = /(?:\[(?:\\.|[^\[\]\\])*\]|\\.|`[^`]*`|[^\[\]\\`])*?/,
    ve = j(/^!?\[(label)\]\(\s*(href)(?:\s+(title))?\s*\)/).replace("label", xe).replace("href", /<(?:\\.|[^\n<>\\])+>|[^\s\x00-\x1f]*/).replace("title", /"(?:\\"?|[^"\\])*"|'(?:\\'?|[^'\\])*'|\((?:\\\)?|[^)\\])*\)/).getRegex(),
    be = j(/^!?\[(label)\]\[(ref)\]/).replace("label", xe).replace("ref", V).getRegex(),
    Ce = j(/^!?\[(ref)\](?:\[\])?/).replace("ref", V).getRegex(), Se = {
        _backpedal: $,
        anyPunctuation: fe,
        autolink: ye,
        blockSkip: /\[[^[\]]*?\]\([^\(\)]*?\)|`[^`]*?`|<[^<>]*?>/g,
        br: he,
        code: /^(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/,
        del: $,
        emStrongLDelim: pe,
        emStrongRDelimAst: me,
        emStrongRDelimUnd: ge,
        escape: le,
        link: ve,
        nolink: Ce,
        punctuation: ue,
        reflink: be,
        reflinkSearch: j("reflink|nolink(?!\\()", "g").replace("reflink", be).replace("nolink", Ce).getRegex(),
        tag: ke,
        text: /^(`+|[^`])(?:(?= {2,}\n)|[\s\S]*?(?:(?=[\\<!\[`*_]|\b_|$)|[^ ](?= {2,}\n)))/,
        url: $
    }, Ie = {
        ...Se,
        link: j(/^!?\[(label)\]\((.*?)\)/).replace("label", xe).getRegex(),
        reflink: j(/^!?\[(label)\]\s*\[([^\]]*)\]/).replace("label", xe).getRegex()
    }, Ae = {
        ...Se,
        escape: j(le).replace("])", "~|])").getRegex(),
        url: j(/^((?:ftp|https?):\/\/|www\.)(?:[a-zA-Z0-9\-]+\.?)+[^\s<]*|^email/, "i").replace("email", /[A-Za-z0-9._+-]+(@)[a-zA-Z0-9-_]+(?:\.[a-zA-Z0-9-_]*[a-zA-Z0-9])+(?![-_])/).getRegex(),
        _backpedal: /(?:[^?!.,:;*_'"~()&]+|\([^)]*\)|&(?![a-zA-Z0-9]+;$)|[?!.,:;*_'"~)]+(?!$))+/,
        del: /^(~~?)(?=[^\s~])([\s\S]*?[^\s~])\1(?=[^~]|$)/,
        text: /^([`~]+|[^`~])(?:(?= {2,}\n)|(?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)|[\s\S]*?(?:(?=[\\<!\[`*~_]|\b_|https?:\/\/|ftp:\/\/|www\.|$)|[^ ](?= {2,}\n)|[^a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-](?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)))/
    }, Pe = {
        ...Ae,
        br: j(he).replace("{2,}", "*").getRegex(),
        text: j(Ae.text).replace("\\b_", "\\b_| {2,}\\n").replace(/\{2,\}/g, "*").getRegex()
    }, Ee = {normal: re, gfm: ae, pedantic: ce}, Te = {normal: Se, gfm: Ae, breaks: Pe, pedantic: Ie};
var Oe = Object.defineProperty, Be = (e, t, s) => ((e, t, s) => t in e ? Oe(e, t, {
    enumerable: !0,
    configurable: !0,
    writable: !0,
    value: s
}) : e[t] = s)(e, "symbol" != typeof t ? t + "" : t, s);

function Le(e, t, s, n) {
    const o = t.href, r = t.title ? Q(t.title) : null, i = e[1].replace(/\\([\[\]])/g, "$1");
    if ("!" !== e[0].charAt(0)) {
        n.state.inLink = !0;
        const e = {type: "link", raw: s, href: o, title: r, text: i, tokens: n.inlineTokens(i)};
        return n.state.inLink = !1, e
    }
    return {type: "image", raw: s, href: o, title: r, text: Q(i)}
}

class De {
    constructor(e) {
        Be(this, "lexer"), Be(this, "options"), Be(this, "rules"), this.options = e || E
    }

    autolink(e) {
        const t = this.rules.inline.autolink.exec(e);
        if (t) {
            let e, s;
            return "@" === t[2] ? (e = Q(t[1]), s = "mailto:" + e) : (e = Q(t[1]), s = e), {
                type: "link",
                raw: t[0],
                text: e,
                href: s,
                tokens: [{type: "text", raw: e, text: e}]
            }
        }
    }

    blockquote(e) {
        const t = this.rules.block.blockquote.exec(e);
        if (t) {
            let e = t[0].replace(/\n {0,3}((?:=+|-+) *)(?=\n|$)/g, "\n    $1");
            e = X(e.replace(/^ *>[ \t]?/gm, ""), "\n");
            const s = this.lexer.state.top;
            this.lexer.state.top = !0;
            const n = this.lexer.blockTokens(e);
            return this.lexer.state.top = s, {type: "blockquote", raw: t[0], tokens: n, text: e}
        }
    }

    br(e) {
        const t = this.rules.inline.br.exec(e);
        if (t) return {type: "br", raw: t[0]}
    }

    code(e) {
        const t = this.rules.block.code.exec(e);
        if (t) {
            const e = t[0].replace(/^ {1,4}/gm, "");
            return {type: "code", raw: t[0], codeBlockStyle: "indented", text: this.options.pedantic ? e : X(e, "\n")}
        }
    }

    codespan(e) {
        const t = this.rules.inline.code.exec(e);
        if (t) {
            let e = t[2].replace(/\n/g, " ");
            const s = /[^ ]/.test(e), n = /^ /.test(e) && / $/.test(e);
            return s && n && (e = e.substring(1, e.length - 1)), e = Q(e, !0), {type: "codespan", raw: t[0], text: e}
        }
    }

    def(e) {
        const t = this.rules.block.def.exec(e);
        if (t) {
            const e = t[1].toLowerCase().replace(/\s+/g, " "),
                s = t[2] ? t[2].replace(/^<(.*)>$/, "$1").replace(this.rules.inline.anyPunctuation, "$1") : "",
                n = t[3] ? t[3].substring(1, t[3].length - 1).replace(this.rules.inline.anyPunctuation, "$1") : t[3];
            return {type: "def", tag: e, raw: t[0], href: s, title: n}
        }
    }

    del(e) {
        const t = this.rules.inline.del.exec(e);
        if (t) return {type: "del", raw: t[0], text: t[2], tokens: this.lexer.inlineTokens(t[2])}
    }

    emStrong(e, t, s = "") {
        let n = this.rules.inline.emStrongLDelim.exec(e);
        if (!n) return;
        if (n[3] && s.match(/[\p{L}\p{N}]/u)) return;
        if (!(n[1] || n[2] || "") || !s || this.rules.inline.punctuation.exec(s)) {
            const s = [...n[0]].length - 1;
            let o, r, i = s, a = 0;
            const c = "*" === n[0][0] ? this.rules.inline.emStrongRDelimAst : this.rules.inline.emStrongRDelimUnd;
            for (c.lastIndex = 0, t = t.slice(-1 * e.length + s); null != (n = c.exec(t));) {
                if (o = n[1] || n[2] || n[3] || n[4] || n[5] || n[6], !o) continue;
                if (r = [...o].length, n[3] || n[4]) {
                    i += r;
                    continue
                }
                if ((n[5] || n[6]) && s % 3 && !((s + r) % 3)) {
                    a += r;
                    continue
                }
                if (i -= r, i > 0) continue;
                r = Math.min(r, r + i + a);
                const t = [...n[0]][0].length, c = e.slice(0, s + n.index + t + r);
                if (Math.min(s, r) % 2) {
                    const e = c.slice(1, -1);
                    return {type: "em", raw: c, text: e, tokens: this.lexer.inlineTokens(e)}
                }
                const l = c.slice(2, -2);
                return {type: "strong", raw: c, text: l, tokens: this.lexer.inlineTokens(l)}
            }
        }
    }

    escape(e) {
        const t = this.rules.inline.escape.exec(e);
        if (t) return {type: "escape", raw: t[0], text: Q(t[1])}
    }

    fences(e) {
        const t = this.rules.block.fences.exec(e);
        if (t) {
            const e = t[0], s = function (e, t) {
                const s = e.match(/^(\s+)(?:```)/);
                if (null === s) return t;
                const n = s[1];
                return t.split("\n").map((e => {
                    const t = e.match(/^\s+/);
                    if (null === t) return e;
                    const [s] = t;
                    return s.length >= n.length ? e.slice(n.length) : e
                })).join("\n")
            }(e, t[3] || "");
            return {
                type: "code",
                raw: e,
                lang: t[2] ? t[2].trim().replace(this.rules.inline.anyPunctuation, "$1") : t[2],
                text: s
            }
        }
    }

    heading(e) {
        const t = this.rules.block.heading.exec(e);
        if (t) {
            let e = t[2].trim();
            if (/#$/.test(e)) {
                const t = X(e, "#");
                this.options.pedantic ? e = t.trim() : t && !/ $/.test(t) || (e = t.trim())
            }
            return {type: "heading", raw: t[0], depth: t[1].length, text: e, tokens: this.lexer.inline(e)}
        }
    }

    hr(e) {
        const t = this.rules.block.hr.exec(e);
        if (t) return {type: "hr", raw: t[0]}
    }

    html(e) {
        const t = this.rules.block.html.exec(e);
        if (t) {
            return {
                type: "html",
                block: !0,
                raw: t[0],
                pre: "pre" === t[1] || "script" === t[1] || "style" === t[1],
                text: t[0]
            }
        }
    }

    inlineText(e) {
        const t = this.rules.inline.text.exec(e);
        if (t) {
            let e;
            return e = this.lexer.state.inRawBlock ? t[0] : Q(t[0]), {type: "text", raw: t[0], text: e}
        }
    }

    lheading(e) {
        const t = this.rules.block.lheading.exec(e);
        if (t) return {
            type: "heading",
            raw: t[0],
            depth: "=" === t[2].charAt(0) ? 1 : 2,
            text: t[1],
            tokens: this.lexer.inline(t[1])
        }
    }

    link(e) {
        const t = this.rules.inline.link.exec(e);
        if (t) {
            const e = t[2].trim();
            if (!this.options.pedantic && /^</.test(e)) {
                if (!/>$/.test(e)) return;
                const t = X(e.slice(0, -1), "\\");
                if ((e.length - t.length) % 2 == 0) return
            } else {
                const e = function (e, t) {
                    if (-1 === e.indexOf(t[1])) return -1;
                    let s = 0;
                    for (let n = 0; n < e.length; n++) if ("\\" === e[n]) n++; else if (e[n] === t[0]) s++; else if (e[n] === t[1] && (s--, s < 0)) return n;
                    return -1
                }(t[2], "()");
                if (e > -1) {
                    const s = (0 === t[0].indexOf("!") ? 5 : 4) + t[1].length + e;
                    t[2] = t[2].substring(0, e), t[0] = t[0].substring(0, s).trim(), t[3] = ""
                }
            }
            let s = t[2], n = "";
            if (this.options.pedantic) {
                const e = /^([^'"]*[^\s])\s+(['"])(.*)\2/.exec(s);
                e && (s = e[1], n = e[3])
            } else n = t[3] ? t[3].slice(1, -1) : "";
            return s = s.trim(), /^</.test(s) && (s = this.options.pedantic && !/>$/.test(e) ? s.slice(1) : s.slice(1, -1)), Le(t, {
                href: s ? s.replace(this.rules.inline.anyPunctuation, "$1") : s,
                title: n ? n.replace(this.rules.inline.anyPunctuation, "$1") : n
            }, t[0], this.lexer)
        }
    }

    list(e) {
        let t = this.rules.block.list.exec(e);
        if (t) {
            let s = t[1].trim();
            const n = s.length > 1,
                o = {type: "list", raw: "", ordered: n, start: n ? +s.slice(0, -1) : "", loose: !1, items: []};
            s = n ? `\\d{1,9}\\${s.slice(-1)}` : `\\${s}`, this.options.pedantic && (s = n ? s : "[*+-]");
            const r = new RegExp(`^( {0,3}${s})((?:[\t ][^\\n]*)?(?:\\n|$))`);
            let i = "", a = "", c = !1;
            for (; e;) {
                let s = !1;
                if (!(t = r.exec(e))) break;
                if (this.rules.block.hr.test(e)) break;
                i = t[0], e = e.substring(i.length);
                let n = t[2].split("\n", 1)[0].replace(/^\t+/, (e => " ".repeat(3 * e.length))),
                    l = e.split("\n", 1)[0], h = 0;
                this.options.pedantic ? (h = 2, a = n.trimStart()) : (h = t[2].search(/[^ ]/), h = h > 4 ? 1 : h, a = n.slice(h), h += t[1].length);
                let d = !1;
                if (!n && /^ *$/.test(l) && (i += l + "\n", e = e.substring(l.length + 1), s = !0), !s) {
                    const t = new RegExp(`^ {0,${Math.min(3, h - 1)}}(?:[*+-]|\\d{1,9}[.)])((?:[ \t][^\\n]*)?(?:\\n|$))`),
                        s = new RegExp(`^ {0,${Math.min(3, h - 1)}}((?:- *){3,}|(?:_ *){3,}|(?:\\* *){3,})(?:\\n+|$)`),
                        o = new RegExp(`^ {0,${Math.min(3, h - 1)}}(?:\`\`\`|~~~)`),
                        r = new RegExp(`^ {0,${Math.min(3, h - 1)}}#`);
                    for (; e;) {
                        const c = e.split("\n", 1)[0];
                        if (l = c, this.options.pedantic && (l = l.replace(/^ {1,4}(?=( {4})*[^ ])/g, "  ")), o.test(l)) break;
                        if (r.test(l)) break;
                        if (t.test(l)) break;
                        if (s.test(e)) break;
                        if (l.search(/[^ ]/) >= h || !l.trim()) a += "\n" + l.slice(h); else {
                            if (d) break;
                            if (n.search(/[^ ]/) >= 4) break;
                            if (o.test(n)) break;
                            if (r.test(n)) break;
                            if (s.test(n)) break;
                            a += "\n" + l
                        }
                        d || l.trim() || (d = !0), i += c + "\n", e = e.substring(c.length + 1), n = l.slice(h)
                    }
                }
                o.loose || (c ? o.loose = !0 : /\n *\n *$/.test(i) && (c = !0));
                let u, p = null;
                this.options.gfm && (p = /^\[[ xX]\] /.exec(a), p && (u = "[ ] " !== p[0], a = a.replace(/^\[[ xX]\] +/, ""))), o.items.push({
                    type: "list_item",
                    raw: i,
                    task: !!p,
                    checked: u,
                    loose: !1,
                    text: a,
                    tokens: []
                }), o.raw += i
            }
            o.items[o.items.length - 1].raw = i.trimEnd(), o.items[o.items.length - 1].text = a.trimEnd(), o.raw = o.raw.trimEnd();
            for (let e = 0; e < o.items.length; e++) if (this.lexer.state.top = !1, o.items[e].tokens = this.lexer.blockTokens(o.items[e].text, []), !o.loose) {
                const t = o.items[e].tokens.filter((e => "space" === e.type)),
                    s = t.length > 0 && t.some((e => /\n.*\n/.test(e.raw)));
                o.loose = s
            }
            if (o.loose) for (let e = 0; e < o.items.length; e++) o.items[e].loose = !0;
            return o
        }
    }

    paragraph(e) {
        const t = this.rules.block.paragraph.exec(e);
        if (t) {
            const e = "\n" === t[1].charAt(t[1].length - 1) ? t[1].slice(0, -1) : t[1];
            return {type: "paragraph", raw: t[0], text: e, tokens: this.lexer.inline(e)}
        }
    }

    reflink(e, t) {
        let s;
        if ((s = this.rules.inline.reflink.exec(e)) || (s = this.rules.inline.nolink.exec(e))) {
            const e = t[(s[2] || s[1]).replace(/\s+/g, " ").toLowerCase()];
            if (!e) {
                const e = s[0].charAt(0);
                return {type: "text", raw: e, text: e}
            }
            return Le(s, e, s[0], this.lexer)
        }
    }

    space(e) {
        const t = this.rules.block.newline.exec(e);
        if (t && t[0].length > 0) return {type: "space", raw: t[0]}
    }

    table(e) {
        const t = this.rules.block.table.exec(e);
        if (!t) return;
        if (!/[:|]/.test(t[2])) return;
        const s = W(t[1]), n = t[2].replace(/^\||\| *$/g, "").split("|"),
            o = t[3] && t[3].trim() ? t[3].replace(/\n[ \t]*$/, "").split("\n") : [],
            r = {type: "table", raw: t[0], header: [], align: [], rows: []};
        if (s.length === n.length) {
            for (const e of n) /^ *-+: *$/.test(e) ? r.align.push("right") : /^ *:-+: *$/.test(e) ? r.align.push("center") : /^ *:-+ *$/.test(e) ? r.align.push("left") : r.align.push(null);
            for (const e of s) r.header.push({text: e, tokens: this.lexer.inline(e)});
            for (const e of o) r.rows.push(W(e, r.header.length).map((e => ({text: e, tokens: this.lexer.inline(e)}))));
            return r
        }
    }

    tag(e) {
        const t = this.rules.inline.tag.exec(e);
        if (t) return !this.lexer.state.inLink && /^<a /i.test(t[0]) ? this.lexer.state.inLink = !0 : this.lexer.state.inLink && /^<\/a>/i.test(t[0]) && (this.lexer.state.inLink = !1), !this.lexer.state.inRawBlock && /^<(pre|code|kbd|script)(\s|>)/i.test(t[0]) ? this.lexer.state.inRawBlock = !0 : this.lexer.state.inRawBlock && /^<\/(pre|code|kbd|script)(\s|>)/i.test(t[0]) && (this.lexer.state.inRawBlock = !1), {
            type: "html",
            raw: t[0],
            inLink: this.lexer.state.inLink,
            inRawBlock: this.lexer.state.inRawBlock,
            block: !1,
            text: t[0]
        }
    }

    text(e) {
        const t = this.rules.block.text.exec(e);
        if (t) return {type: "text", raw: t[0], text: t[0], tokens: this.lexer.inline(t[0])}
    }

    url(e) {
        let t;
        if (t = this.rules.inline.url.exec(e)) {
            let e, s;
            if ("@" === t[2]) e = Q(t[0]), s = "mailto:" + e; else {
                let n;
                do {
                    n = t[0], t[0] = this.rules.inline._backpedal.exec(t[0])?.[0] ?? ""
                } while (n !== t[0]);
                e = Q(t[0]), s = "www." === t[1] ? "http://" + t[0] : t[0]
            }
            return {type: "link", raw: t[0], text: e, href: s, tokens: [{type: "text", raw: e, text: e}]}
        }
    }
}

var Re = Object.defineProperty, ze = (e, t, s) => ((e, t, s) => t in e ? Re(e, t, {
    enumerable: !0,
    configurable: !0,
    writable: !0,
    value: s
}) : e[t] = s)(e, "symbol" != typeof t ? t + "" : t, s);

class Me {
    constructor(e) {
        ze(this, "options"), ze(this, "state"), ze(this, "tokens"), ze(this, "inlineQueue"), ze(this, "tokenizer"), this.tokens = [], this.tokens.links = Object.create(null), this.options = e || E, this.options.tokenizer = this.options.tokenizer || new De, this.tokenizer = this.options.tokenizer, this.tokenizer.options = this.options, this.tokenizer.lexer = this, this.inlineQueue = [], this.state = {
            inLink: !1,
            inRawBlock: !1,
            top: !0
        };
        const t = {block: Ee.normal, inline: Te.normal};
        this.options.pedantic ? (t.block = Ee.pedantic, t.inline = Te.pedantic) : this.options.gfm && (t.block = Ee.gfm, this.options.breaks ? t.inline = Te.breaks : t.inline = Te.gfm), this.tokenizer.rules = t
    }

    static get rules() {
        return {block: Ee, inline: Te}
    }

    static lex(e, t) {
        return new Me(t).lex(e)
    }

    static lexInline(e, t) {
        return new Me(t).inlineTokens(e)
    }

    blockTokens(e, t = []) {
        let s, n, o, r;
        for (e = this.options.pedantic ? e.replace(/\t/g, "    ").replace(/^ +$/gm, "") : e.replace(/^( *)(\t+)/gm, ((e, t, s) => t + "    ".repeat(s.length))); e;) if (!(this.options.extensions && this.options.extensions.block && this.options.extensions.block.some((n => !!(s = n.call({lexer: this}, e, t)) && (e = e.substring(s.raw.length), t.push(s), !0))))) if (s = this.tokenizer.space(e)) e = e.substring(s.raw.length), 1 === s.raw.length && t.length > 0 ? t[t.length - 1].raw += "\n" : t.push(s); else if (s = this.tokenizer.code(e)) e = e.substring(s.raw.length), n = t[t.length - 1], !n || "paragraph" !== n.type && "text" !== n.type ? t.push(s) : (n.raw += "\n" + s.raw, n.text += "\n" + s.text, this.inlineQueue[this.inlineQueue.length - 1].src = n.text); else if (s = this.tokenizer.fences(e)) e = e.substring(s.raw.length), t.push(s); else if (s = this.tokenizer.heading(e)) e = e.substring(s.raw.length), t.push(s); else if (s = this.tokenizer.hr(e)) e = e.substring(s.raw.length), t.push(s); else if (s = this.tokenizer.blockquote(e)) e = e.substring(s.raw.length), t.push(s); else if (s = this.tokenizer.list(e)) e = e.substring(s.raw.length), t.push(s); else if (s = this.tokenizer.html(e)) e = e.substring(s.raw.length), t.push(s); else if (s = this.tokenizer.def(e)) e = e.substring(s.raw.length), n = t[t.length - 1], !n || "paragraph" !== n.type && "text" !== n.type ? this.tokens.links[s.tag] || (this.tokens.links[s.tag] = {
            href: s.href,
            title: s.title
        }) : (n.raw += "\n" + s.raw, n.text += "\n" + s.raw, this.inlineQueue[this.inlineQueue.length - 1].src = n.text); else if (s = this.tokenizer.table(e)) e = e.substring(s.raw.length), t.push(s); else if (s = this.tokenizer.lheading(e)) e = e.substring(s.raw.length), t.push(s); else {
            if (o = e, this.options.extensions && this.options.extensions.startBlock) {
                let t = 1 / 0;
                const s = e.slice(1);
                let n;
                this.options.extensions.startBlock.forEach((e => {
                    n = e.call({lexer: this}, s), "number" == typeof n && n >= 0 && (t = Math.min(t, n))
                })), t < 1 / 0 && t >= 0 && (o = e.substring(0, t + 1))
            }
            if (this.state.top && (s = this.tokenizer.paragraph(o))) n = t[t.length - 1], r && "paragraph" === n.type ? (n.raw += "\n" + s.raw, n.text += "\n" + s.text, this.inlineQueue.pop(), this.inlineQueue[this.inlineQueue.length - 1].src = n.text) : t.push(s), r = o.length !== e.length, e = e.substring(s.raw.length); else if (s = this.tokenizer.text(e)) e = e.substring(s.raw.length), n = t[t.length - 1], n && "text" === n.type ? (n.raw += "\n" + s.raw, n.text += "\n" + s.text, this.inlineQueue.pop(), this.inlineQueue[this.inlineQueue.length - 1].src = n.text) : t.push(s); else if (e) {
                const t = "Infinite loop on byte: " + e.charCodeAt(0);
                if (this.options.silent) {
                    console.error(t);
                    break
                }
                throw new Error(t)
            }
        }
        return this.state.top = !0, t
    }

    inline(e, t = []) {
        return this.inlineQueue.push({src: e, tokens: t}), t
    }

    inlineTokens(e, t = []) {
        let s, n, o, r, i, a, c = e;
        if (this.tokens.links) {
            const e = Object.keys(this.tokens.links);
            if (e.length > 0) for (; null != (r = this.tokenizer.rules.inline.reflinkSearch.exec(c));) e.includes(r[0].slice(r[0].lastIndexOf("[") + 1, -1)) && (c = c.slice(0, r.index) + "[" + "a".repeat(r[0].length - 2) + "]" + c.slice(this.tokenizer.rules.inline.reflinkSearch.lastIndex))
        }
        for (; null != (r = this.tokenizer.rules.inline.blockSkip.exec(c));) c = c.slice(0, r.index) + "[" + "a".repeat(r[0].length - 2) + "]" + c.slice(this.tokenizer.rules.inline.blockSkip.lastIndex);
        for (; null != (r = this.tokenizer.rules.inline.anyPunctuation.exec(c));) c = c.slice(0, r.index) + "++" + c.slice(this.tokenizer.rules.inline.anyPunctuation.lastIndex);
        for (; e;) if (i || (a = ""), i = !1, !(this.options.extensions && this.options.extensions.inline && this.options.extensions.inline.some((n => !!(s = n.call({lexer: this}, e, t)) && (e = e.substring(s.raw.length), t.push(s), !0))))) if (s = this.tokenizer.escape(e)) e = e.substring(s.raw.length), t.push(s); else if (s = this.tokenizer.tag(e)) e = e.substring(s.raw.length), n = t[t.length - 1], n && "text" === s.type && "text" === n.type ? (n.raw += s.raw, n.text += s.text) : t.push(s); else if (s = this.tokenizer.link(e)) e = e.substring(s.raw.length), t.push(s); else if (s = this.tokenizer.reflink(e, this.tokens.links)) e = e.substring(s.raw.length), n = t[t.length - 1], n && "text" === s.type && "text" === n.type ? (n.raw += s.raw, n.text += s.text) : t.push(s); else if (s = this.tokenizer.emStrong(e, c, a)) e = e.substring(s.raw.length), t.push(s); else if (s = this.tokenizer.codespan(e)) e = e.substring(s.raw.length), t.push(s); else if (s = this.tokenizer.br(e)) e = e.substring(s.raw.length), t.push(s); else if (s = this.tokenizer.del(e)) e = e.substring(s.raw.length), t.push(s); else if (s = this.tokenizer.autolink(e)) e = e.substring(s.raw.length), t.push(s); else if (this.state.inLink || !(s = this.tokenizer.url(e))) {
            if (o = e, this.options.extensions && this.options.extensions.startInline) {
                let t = 1 / 0;
                const s = e.slice(1);
                let n;
                this.options.extensions.startInline.forEach((e => {
                    n = e.call({lexer: this}, s), "number" == typeof n && n >= 0 && (t = Math.min(t, n))
                })), t < 1 / 0 && t >= 0 && (o = e.substring(0, t + 1))
            }
            if (s = this.tokenizer.inlineText(o)) e = e.substring(s.raw.length), "_" !== s.raw.slice(-1) && (a = s.raw.slice(-1)), i = !0, n = t[t.length - 1], n && "text" === n.type ? (n.raw += s.raw, n.text += s.text) : t.push(s); else if (e) {
                const t = "Infinite loop on byte: " + e.charCodeAt(0);
                if (this.options.silent) {
                    console.error(t);
                    break
                }
                throw new Error(t)
            }
        } else e = e.substring(s.raw.length), t.push(s);
        return t
    }

    lex(e) {
        e = e.replace(/\r\n|\r/g, "\n"), this.blockTokens(e, this.tokens);
        for (let e = 0; e < this.inlineQueue.length; e++) {
            const t = this.inlineQueue[e];
            this.inlineTokens(t.src, t.tokens)
        }
        return this.inlineQueue = [], this.tokens
    }
}

var He = Object.defineProperty, Ue = (e, t, s) => ((e, t, s) => t in e ? He(e, t, {
    enumerable: !0,
    configurable: !0,
    writable: !0,
    value: s
}) : e[t] = s)(e, t + "", s);

class Qe {
    constructor(e) {
        Ue(this, "options"), this.options = e || E
    }

    blockquote(e) {
        return `<blockquote>\n${e}</blockquote>\n`
    }

    br() {
        return "<br>"
    }

    checkbox(e) {
        return "<input " + (e ? 'checked="" ' : "") + 'disabled="" type="checkbox">'
    }

    code(e, t, s) {
        const n = (t || "").match(/^\S*/)?.[0];
        return e = e.replace(/\n$/, "") + "\n", n ? '<pre><code class="language-' + Q(n) + '">' + (s ? e : Q(e, !0)) + "</code></pre>\n" : "<pre><code>" + (s ? e : Q(e, !0)) + "</code></pre>\n"
    }

    codespan(e) {
        return `<code>${e}</code>`
    }

    del(e) {
        return `<del>${e}</del>`
    }

    em(e) {
        return `<em>${e}</em>`
    }

    heading(e, t, s) {
        return `<h${t}>${e}</h${t}>\n`
    }

    hr() {
        return "<hr>\n"
    }

    html(e, t) {
        return e
    }

    image(e, t, s) {
        const n = Y(e);
        if (null === n) return s;
        let o = `<img src="${e = n}" alt="${s}"`;
        return t && (o += ` title="${t}"`), o += ">", o
    }

    link(e, t, s) {
        const n = Y(e);
        if (null === n) return s;
        let o = '<a href="' + (e = n) + '"';
        return t && (o += ' title="' + t + '"'), o += ">" + s + "</a>", o
    }

    list(e, t, s) {
        const n = t ? "ol" : "ul";
        return "<" + n + (t && 1 !== s ? ' start="' + s + '"' : "") + ">\n" + e + "</" + n + ">\n"
    }

    listitem(e, t, s) {
        return `<li>${e}</li>\n`
    }

    paragraph(e) {
        return `<p>${e}</p>\n`
    }

    strong(e) {
        return `<strong>${e}</strong>`
    }

    table(e, t) {
        return t && (t = `<tbody>${t}</tbody>`), "<table>\n<thead>\n" + e + "</thead>\n" + t + "</table>\n"
    }

    tablecell(e, t) {
        const s = t.header ? "th" : "td";
        return (t.align ? `<${s} align="${t.align}">` : `<${s}>`) + e + `</${s}>\n`
    }

    tablerow(e) {
        return `<tr>\n${e}</tr>\n`
    }

    text(e) {
        return e
    }
}

class Ne {
    br() {
        return ""
    }

    codespan(e) {
        return e
    }

    del(e) {
        return e
    }

    em(e) {
        return e
    }

    html(e) {
        return e
    }

    image(e, t, s) {
        return "" + s
    }

    link(e, t, s) {
        return "" + s
    }

    strong(e) {
        return e
    }

    text(e) {
        return e
    }
}

var Fe = Object.defineProperty, qe = (e, t, s) => ((e, t, s) => t in e ? Fe(e, t, {
    enumerable: !0,
    configurable: !0,
    writable: !0,
    value: s
}) : e[t] = s)(e, "symbol" != typeof t ? t + "" : t, s);

class je {
    constructor(e) {
        qe(this, "options"), qe(this, "renderer"), qe(this, "textRenderer"), this.options = e || E, this.options.renderer = this.options.renderer || new Qe, this.renderer = this.options.renderer, this.renderer.options = this.options, this.textRenderer = new Ne
    }

    static parse(e, t) {
        return new je(t).parse(e)
    }

    static parseInline(e, t) {
        return new je(t).parseInline(e)
    }

    parse(e, t = !0) {
        let s = "";
        for (let n = 0; n < e.length; n++) {
            const o = e[n];
            if (this.options.extensions && this.options.extensions.renderers && this.options.extensions.renderers[o.type]) {
                const e = o, t = this.options.extensions.renderers[e.type].call({parser: this}, e);
                if (!1 !== t || !["space", "hr", "heading", "code", "table", "blockquote", "list", "html", "paragraph", "text"].includes(e.type)) {
                    s += t || "";
                    continue
                }
            }
            switch (o.type) {
                case"space":
                    continue;
                case"hr":
                    s += this.renderer.hr();
                    continue;
                case"heading": {
                    const e = o;
                    s += this.renderer.heading(this.parseInline(e.tokens), e.depth, F(this.parseInline(e.tokens, this.textRenderer)));
                    continue
                }
                case"code": {
                    const e = o;
                    s += this.renderer.code(e.text, e.lang, !!e.escaped);
                    continue
                }
                case"table": {
                    const e = o;
                    let t = "", n = "";
                    for (let t = 0; t < e.header.length; t++) n += this.renderer.tablecell(this.parseInline(e.header[t].tokens), {
                        header: !0,
                        align: e.align[t]
                    });
                    t += this.renderer.tablerow(n);
                    let r = "";
                    for (let t = 0; t < e.rows.length; t++) {
                        const s = e.rows[t];
                        n = "";
                        for (let t = 0; t < s.length; t++) n += this.renderer.tablecell(this.parseInline(s[t].tokens), {
                            header: !1,
                            align: e.align[t]
                        });
                        r += this.renderer.tablerow(n)
                    }
                    s += this.renderer.table(t, r);
                    continue
                }
                case"blockquote": {
                    const e = o, t = this.parse(e.tokens);
                    s += this.renderer.blockquote(t);
                    continue
                }
                case"list": {
                    const e = o, t = e.ordered, n = e.start, r = e.loose;
                    let i = "";
                    for (let t = 0; t < e.items.length; t++) {
                        const s = e.items[t], n = s.checked, o = s.task;
                        let a = "";
                        if (s.task) {
                            const e = this.renderer.checkbox(!!n);
                            r ? s.tokens.length > 0 && "paragraph" === s.tokens[0].type ? (s.tokens[0].text = e + " " + s.tokens[0].text, s.tokens[0].tokens && s.tokens[0].tokens.length > 0 && "text" === s.tokens[0].tokens[0].type && (s.tokens[0].tokens[0].text = e + " " + s.tokens[0].tokens[0].text)) : s.tokens.unshift({
                                type: "text",
                                text: e + " "
                            }) : a += e + " "
                        }
                        a += this.parse(s.tokens, r), i += this.renderer.listitem(a, o, !!n)
                    }
                    s += this.renderer.list(i, t, n);
                    continue
                }
                case"html": {
                    const e = o;
                    s += this.renderer.html(e.text, e.block);
                    continue
                }
                case"paragraph": {
                    const e = o;
                    s += this.renderer.paragraph(this.parseInline(e.tokens));
                    continue
                }
                case"text": {
                    let r = o, i = r.tokens ? this.parseInline(r.tokens) : r.text;
                    for (; n + 1 < e.length && "text" === e[n + 1].type;) r = e[++n], i += "\n" + (r.tokens ? this.parseInline(r.tokens) : r.text);
                    s += t ? this.renderer.paragraph(i) : i;
                    continue
                }
                default: {
                    const e = 'Token with "' + o.type + '" type was not found.';
                    if (this.options.silent) return console.error(e), "";
                    throw new Error(e)
                }
            }
        }
        return s
    }

    parseInline(e, t) {
        t = t || this.renderer;
        let s = "";
        for (let n = 0; n < e.length; n++) {
            const o = e[n];
            if (this.options.extensions && this.options.extensions.renderers && this.options.extensions.renderers[o.type]) {
                const e = this.options.extensions.renderers[o.type].call({parser: this}, o);
                if (!1 !== e || !["escape", "html", "link", "image", "strong", "em", "codespan", "br", "del", "text"].includes(o.type)) {
                    s += e || "";
                    continue
                }
            }
            switch (o.type) {
                case"escape": {
                    const e = o;
                    s += t.text(e.text);
                    break
                }
                case"html": {
                    const e = o;
                    s += t.html(e.text);
                    break
                }
                case"link": {
                    const e = o;
                    s += t.link(e.href, e.title, this.parseInline(e.tokens, t));
                    break
                }
                case"image": {
                    const e = o;
                    s += t.image(e.href, e.title, e.text);
                    break
                }
                case"strong": {
                    const e = o;
                    s += t.strong(this.parseInline(e.tokens, t));
                    break
                }
                case"em": {
                    const e = o;
                    s += t.em(this.parseInline(e.tokens, t));
                    break
                }
                case"codespan": {
                    const e = o;
                    s += t.codespan(e.text);
                    break
                }
                case"br":
                    s += t.br();
                    break;
                case"del": {
                    const e = o;
                    s += t.del(this.parseInline(e.tokens, t));
                    break
                }
                case"text": {
                    const e = o;
                    s += t.text(e.text);
                    break
                }
                default: {
                    const e = 'Token with "' + o.type + '" type was not found.';
                    if (this.options.silent) return console.error(e), "";
                    throw new Error(e)
                }
            }
        }
        return s
    }
}

var Ye, $e, We, Xe = Object.defineProperty, Ze = e => {
    throw TypeError(e)
}, Ge = (e, t, s) => ((e, t, s) => t in e ? Xe(e, t, {
    enumerable: !0,
    configurable: !0,
    writable: !0,
    value: s
}) : e[t] = s)(e, "symbol" != typeof t ? t + "" : t, s), Je = (e, t, s) => (((e, t, s) => {
    t.has(e) || Ze("Cannot " + s)
})(e, t, "access private method"), s);
Ye = new WeakSet, $e = function (e, t) {
    return s => {
        if (s.message += "\nPlease report this to https://github.com/markedjs/marked.", e) {
            const e = "<p>An error occurred:</p><pre>" + Q(s.message + "", !0) + "</pre>";
            return t ? Promise.resolve(e) : e
        }
        if (t) return Promise.reject(s);
        throw s
    }
}, We = function (e, t) {
    return (s, n) => {
        const o = {...n}, r = {...this.defaults, ...o};
        !0 === this.defaults.async && !1 === o.async && (r.silent || console.warn("marked(): The async option was set to true by an extension. The async: false option sent to parse will be ignored."), r.async = !0);
        const i = Je(this, Ye, $e).call(this, !!r.silent, !!r.async);
        if (null == s) return i(new Error("marked(): input parameter is undefined or null"));
        if ("string" != typeof s) return i(new Error("marked(): input parameter is of type " + Object.prototype.toString.call(s) + ", string expected"));
        if (r.hooks && (r.hooks.options = r), r.async) return Promise.resolve(r.hooks ? r.hooks.preprocess(s) : s).then((t => e(t, r))).then((e => r.hooks ? r.hooks.processAllTokens(e) : e)).then((e => r.walkTokens ? Promise.all(this.walkTokens(e, r.walkTokens)).then((() => e)) : e)).then((e => t(e, r))).then((e => r.hooks ? r.hooks.postprocess(e) : e)).catch(i);
        try {
            r.hooks && (s = r.hooks.preprocess(s));
            let n = e(s, r);
            r.hooks && (n = r.hooks.processAllTokens(n)), r.walkTokens && this.walkTokens(n, r.walkTokens);
            let o = t(n, r);
            return r.hooks && (o = r.hooks.postprocess(o)), o
        } catch (e) {
            return i(e)
        }
    }
};
const Ke = new class {
    constructor(...e) {
        var t, s, n;
        t = this, (s = Ye).has(t) ? Ze("Cannot add the same private member more than once") : s instanceof WeakSet ? s.add(t) : s.set(t, n), Ge(this, "Hooks", L), Ge(this, "Lexer", Me), Ge(this, "Parser", je), Ge(this, "Renderer", Qe), Ge(this, "TextRenderer", Ne), Ge(this, "Tokenizer", De), Ge(this, "defaults", {
            async: !1,
            breaks: !1,
            extensions: null,
            gfm: !0,
            hooks: null,
            pedantic: !1,
            renderer: null,
            silent: !1,
            tokenizer: null,
            walkTokens: null
        }), Ge(this, "options", this.setOptions), Ge(this, "parse", Je(this, Ye, We).call(this, Me.lex, je.parse)), Ge(this, "parseInline", Je(this, Ye, We).call(this, Me.lexInline, je.parseInline)), this.use(...e)
    }

    lexer(e, t) {
        return Me.lex(e, t ?? this.defaults)
    }

    parser(e, t) {
        return je.parse(e, t ?? this.defaults)
    }

    setOptions(e) {
        return this.defaults = {...this.defaults, ...e}, this
    }

    use(...e) {
        const t = this.defaults.extensions || {renderers: {}, childTokens: {}};
        return e.forEach((e => {
            const s = {...e};
            if (s.async = this.defaults.async || s.async || !1, e.extensions && (e.extensions.forEach((e => {
                if (!e.name) throw new Error("extension name required");
                if ("renderer" in e) {
                    const s = t.renderers[e.name];
                    t.renderers[e.name] = s ? function (...t) {
                        let n = e.renderer.apply(this, t);
                        return !1 === n && (n = s.apply(this, t)), n
                    } : e.renderer
                }
                if ("tokenizer" in e) {
                    if (!e.level || "block" !== e.level && "inline" !== e.level) throw new Error("extension level must be 'block' or 'inline'");
                    const s = t[e.level];
                    s ? s.unshift(e.tokenizer) : t[e.level] = [e.tokenizer], e.start && ("block" === e.level ? t.startBlock ? t.startBlock.push(e.start) : t.startBlock = [e.start] : "inline" === e.level && (t.startInline ? t.startInline.push(e.start) : t.startInline = [e.start]))
                }
                "childTokens" in e && e.childTokens && (t.childTokens[e.name] = e.childTokens)
            })), s.extensions = t), e.renderer) {
                const t = this.defaults.renderer || new Qe(this.defaults);
                for (const s in e.renderer) {
                    if (!(s in t)) throw new Error(`renderer '${s}' does not exist`);
                    if ("options" === s) continue;
                    const n = s, o = e.renderer[n], r = t[n];
                    t[n] = (...e) => {
                        let s = o.apply(t, e);
                        return !1 === s && (s = r.apply(t, e)), s || ""
                    }
                }
                s.renderer = t
            }
            if (e.tokenizer) {
                const t = this.defaults.tokenizer || new De(this.defaults);
                for (const s in e.tokenizer) {
                    if (!(s in t)) throw new Error(`tokenizer '${s}' does not exist`);
                    if (["options", "rules", "lexer"].includes(s)) continue;
                    const n = s, o = e.tokenizer[n], r = t[n];
                    t[n] = (...e) => {
                        let s = o.apply(t, e);
                        return !1 === s && (s = r.apply(t, e)), s
                    }
                }
                s.tokenizer = t
            }
            if (e.hooks) {
                const t = this.defaults.hooks || new L;
                for (const s in e.hooks) {
                    if (!(s in t)) throw new Error(`hook '${s}' does not exist`);
                    if ("options" === s) continue;
                    const n = s, o = e.hooks[n], r = t[n];
                    L.passThroughHooks.has(s) ? t[n] = e => {
                        if (this.defaults.async) return Promise.resolve(o.call(t, e)).then((e => r.call(t, e)));
                        const s = o.call(t, e);
                        return r.call(t, s)
                    } : t[n] = (...e) => {
                        let s = o.apply(t, e);
                        return !1 === s && (s = r.apply(t, e)), s
                    }
                }
                s.hooks = t
            }
            if (e.walkTokens) {
                const t = this.defaults.walkTokens, n = e.walkTokens;
                s.walkTokens = function (e) {
                    let s = [];
                    return s.push(n.call(this, e)), t && (s = s.concat(t.call(this, e))), s
                }
            }
            this.defaults = {...this.defaults, ...s}
        })), this
    }

    walkTokens(e, t) {
        let s = [];
        for (const n of e) switch (s = s.concat(t.call(this, n)), n.type) {
            case"table": {
                const e = n;
                for (const n of e.header) s = s.concat(this.walkTokens(n.tokens, t));
                for (const n of e.rows) for (const e of n) s = s.concat(this.walkTokens(e.tokens, t));
                break
            }
            case"list": {
                const e = n;
                s = s.concat(this.walkTokens(e.items, t));
                break
            }
            default: {
                const e = n;
                this.defaults.extensions?.childTokens?.[e.type] ? this.defaults.extensions.childTokens[e.type].forEach((n => {
                    const o = e[n].flat(1 / 0);
                    s = s.concat(this.walkTokens(o, t))
                })) : e.tokens && (s = s.concat(this.walkTokens(e.tokens, t)))
            }
        }
        return s
    }
};

function Ve(e, t) {
    return Ke.parse(e, t)
}

Ve.options = Ve.setOptions = function (e) {
    return Ke.setOptions(e), Ve.defaults = Ke.defaults, T(Ve.defaults), Ve
}, Ve.getDefaults = P, Ve.defaults = E, Ve.use = function (...e) {
    return Ke.use(...e), Ve.defaults = Ke.defaults, T(Ve.defaults), Ve
}, Ve.walkTokens = function (e, t) {
    return Ke.walkTokens(e, t)
}, Ve.parseInline = Ke.parseInline, Ve.Parser = je, Ve.parser = je.parse, Ve.Renderer = Qe, Ve.TextRenderer = Ne, Ve.Lexer = Me, Ve.lexer = Me.lex, Ve.Tokenizer = De, Ve.Hooks = L, Ve.parse = Ve;
const _e = (e, t) => {
        const {showCodeBlockCopyButton: s, markdownLinkTarget: n, syntaxHighlighter: o, htmlSanitizer: r} = t || {},
            i = Ve(e, {async: !1, breaks: !0});
        if ("string" != typeof i) throw new Error("Markdown parsing failed");
        const a = document.createElement("div");
        return a.innerHTML = r ? r(i) : i, a.querySelectorAll("pre").forEach((e => {
            const s = document.createElement("div");
            s.className = "code-block";
            const n = e.querySelector("code");
            if (!n) {
                const t = e.innerHTML;
                return s.innerHTML = r ? r(t) : t, void e.replaceWith(s)
            }
            let i;
            for (let e = 0; e < n.classList.length; e++) {
                const t = n.classList[e];
                if (t.startsWith("language-")) {
                    i = t.slice(9);
                    break
                }
            }
            const a = document.createElement("pre"), c = "<div>" + n.innerHTML + "</div>";
            if (a.innerHTML = t?.htmlSanitizer ? t.htmlSanitizer(c) : c, i && (a.setAttribute("data-language", i), o)) {
                const e = "<div>" + o.createHighlighter()(n.textContent || "", i) + "</div>";
                a.innerHTML = r ? r(e) : e, a.className = "highlighter-dark"
            }
            p(s), s.appendChild(a), e.replaceWith(s)
        })), !1 !== s && a.querySelectorAll(".code-block").forEach((e => {
            if (!e.querySelector("pre")) return;
            if (e.previousElementSibling?.classList.contains("nlux-comp-copyButton")) return;
            const t = "Copy code block to clipboard", s = document.createElement("button");
            s.classList.add("nlux-comp-copyButton"), s.setAttribute("aria-label", t), s.setAttribute("title", t);
            const n = document.createElement("span");
            n.classList.add("icon-copy"), s.appendChild(n), e.appendChild(s)
        })), "self" !== n && a.querySelectorAll("a").forEach((e => {
            e.setAttribute("target", "_blank")
        })), a.innerHTML
    }, et = (e, t = "text", s) => {
        if ("markdown" === t) {
            const t = document.createElement("div"), n = _e(e, s);
            t.innerHTML = s?.htmlSanitizer ? s.htmlSanitizer(n) : n, A(t);
            const o = document.createDocumentFragment();
            for (; t.firstChild;) o.appendChild(t.firstChild);
            return o
        }
        return document.createTextNode(e)
    }, tt = "nlux-comp-message", st = {received: "nlux-comp-chatItem--received", sent: "nlux-comp-chatItem--sent"},
    nt = (e, t) => {
        Object.keys(st).forEach((t => {
            e.classList.remove(st[t])
        })), st[t] && e.classList.add(st[t])
    }, ot = {bubbles: "nlux-comp-chatItem--bubblesLayout", list: "nlux-comp-chatItem--listLayout"}, rt = (e, t) => {
        Object.keys(ot).forEach((t => {
            e.classList.remove(ot[t])
        })), ot[t] && e.classList.add(ot[t])
    }, it = "nlux-comp-chatItem-participantInfo", at = "nlux-comp-chatItem-participantName", ct = e => {
        const t = document.createElement("div");
        t.classList.add("nlux-comp-chatItem");
        const s = {direction: e.direction, status: e.status, message: e.message, htmlSanitizer: e.htmlSanitizer};
        let n;
        if (void 0 !== e.avatar) {
            const t = {name: e.name, avatar: e.avatar};
            n = x(t)
        }
        const o = document.createElement("span");
        o.classList.add(at), o.textContent = e.name;
        {
            const e = document.createElement("div");
            e.classList.add(it), void 0 !== n && e.append(n), e.append(o), t.append(e)
        }
        nt(t, e.direction), rt(t, e.layout);
        const r = (e => {
            const t = document.createElement("div");
            t.classList.add(tt);
            const s = e.status ? e.status : "complete";
            return S(t, s), b(t, e.direction), t
        })(s);
        return t.append(r), t
    }, lt = (e, t, s) => {
        if (t.name !== s.name && "string" == typeof s.avatar) {
            const t = s.name && s.name.length > 0 ? s.name[0].toUpperCase() : "",
                n = e.querySelector("* > .nlux-comp-avatarContainer > .avtr_ltr");
            n?.replaceChildren(t)
        }
    }, ht = (e, t, s) => {
        t.avatar === s.avatar && t.name === s.name || (t.avatar !== s.avatar && ((e, t, s) => {
            if (t.avatar !== s.avatar) if ("string" == typeof s.avatar && "string" == typeof t.avatar) {
                const t = e.querySelector("* > .nlux-comp-avatarContainer > .nlux-comp-avatarPicture");
                null !== t && (t.style.backgroundImage = `url("${encodeURI(s.avatar)}")`)
            } else if ("string" == typeof s.avatar) {
                const t = w(s.avatar);
                e.replaceChildren(t)
            } else s.avatar ? e.replaceChildren(s.avatar.cloneNode(!0)) : p(e)
        })(e, t, s), s.name ? t.name !== s.name && (e.title = s.name, lt(e, t, s)) : (e.title = "", lt(e, t, s)))
    }, dt = (e, t, s) => {
        if (t.message === s.message && t.status === s.status && t.direction === s.direction) return;
        if (!s || !s.hasOwnProperty("message") && !s.hasOwnProperty("status") && !s.hasOwnProperty("direction")) return;
        t.direction !== s.direction && b(e, s.direction);
        const n = s.status;
        if (t.status !== n) return S(e, s.status), void ((e, t, s) => {
            const n = s.status;
            if ("streaming" !== n && "complete" === n) {
                const t = s.message ? s.message : "", o = document.createTextNode(t);
                e.classList.add(C[n]), p(e), e.append(o)
            }
        })(e, 0, s);
        "complete" === n && (t.message === s.message && t.format === s.format || ((e, t, s) => {
            t.message === s.message && t.format === s.format || (p(e), e.append(et(s.message ?? "", s.format, {htmlSanitizer: s.htmlSanitizer})))
        })(e, t, s))
    }, ut = (e, t) => {
        let s = !1;
        const {onComplete: n} = t || {}, o = [],
            r = "timeout" === (t?.skipStreamingAnimation ? "timeout" : "animationFrame") ? e => setTimeout(e, 0) : e => requestAnimationFrame(e);
        const a = document.createElement("div");
        a.classList.add("md-in-progress"), e.append(a);
        const c = () => {
                for (; a.firstChild;) {
                    const e = a.firstChild;
                    e instanceof HTMLElement && A(e), a.before(e)
                }
            },
            l = !t?.skipStreamingAnimation && t?.streamingAnimationSpeed && t.streamingAnimationSpeed >= 0 ? t.streamingAnimationSpeed : t?.skipStreamingAnimation ? 0 : 8,
            h = {timeSinceLastProcessing: (new Date).getTime(), currentMarkdown: "", previousHtml: void 0};
        let d = setInterval((() => {
            const e = (new Date).getTime(), l = "never" !== t?.waitTimeBeforeStreamCompletion;
            if (0 === o.length && l) {
                const o = "number" == typeof t?.waitTimeBeforeStreamCompletion ? t.waitTimeBeforeStreamCompletion : 2e3;
                return void ((s || e - h.timeSinceLastProcessing > o) && (s = !0, d && (clearInterval(d), d = void 0), c(), a.remove(), n?.()))
            }
            h.timeSinceLastProcessing = e;
            const u = o.shift();
            void 0 !== u && "string" == typeof u && r((() => {
                const e = h.currentMarkdown + u, s = _e(e, t).trim();
                if ("string" != typeof s) return h.currentMarkdown = h.currentMarkdown.slice(0, -u.length), void i("Markdown parsing failed");
                if (h.previousHtml && s.length > h.previousHtml.length && s.startsWith(h.previousHtml)) {
                    c();
                    const e = s.slice(h.previousHtml.length).trim();
                    a.innerHTML = t?.htmlSanitizer ? t.htmlSanitizer(e) : e, h.currentMarkdown = u, h.previousHtml = void 0
                } else a.innerHTML = t?.htmlSanitizer ? t.htmlSanitizer(s) : s, h.currentMarkdown = e, h.previousHtml = s
            }))
        }), l);
        return {
            next: e => {
                if (s) i("Stream is already complete. No more chunks can be added"); else for (const t of e) o.push(t)
            }, complete: () => {
                o.push("\n"), s = !0
            }, cancel: () => {
                d && (clearInterval(d), d = void 0), s = !0, a.remove()
            }, error: () => {
                s = !0
            }
        }
    }, pt = "dom/getElement";
var mt = Object.defineProperty, gt = Object.getOwnPropertyDescriptor, ft = (e, t, s, n) => {
    for (var o, r = n > 1 ? void 0 : n ? gt(t, s) : t, i = e.length - 1; i >= 0; i--) (o = e[i]) && (r = (n ? o(t, s, r) : o(r)) || r);
    return n && r && mt(t, s, r), r
};
let yt = class extends g {
    constructor(e, t) {
        super(e, t), this.serverResponse = [], this.stringContent = "", void 0 !== t.domProps.message && (this.stringContent = t.domProps.message)
    }

    addChunk(e, t) {
        this.throwIfDestroyed(), this.executeDomAction("processStreamedChunk", e), "string" == typeof e && (this.stringContent += e), this.serverResponse.push(t)
    }

    commitChunks() {
        this.throwIfDestroyed(), this.executeDomAction("commitStreamedChunks")
    }

    getChatSegmentItem() {
        return "received" === this.getProp("domProps").direction ? {
            uid: this.props.uid,
            participantRole: "assistant",
            content: this.getItemContent(),
            contentType: "text",
            serverResponse: this.serverResponse,
            status: "complete",
            dataTransferMode: "batch",
            time: new Date
        } : {
            uid: this.props.uid,
            participantRole: "user",
            content: this.getItemContent(),
            contentType: "text",
            status: "complete",
            time: new Date
        }
    }

    getItemContent() {
        return this.aiMessageContent ?? this.stringContent
    }

    updateDomProps(e) {
        const t = this.getProp("domProps"), s = {...t, ...e};
        this.setProp("domProps", s), this.executeDomAction("updateDomProps", t, s)
    }

    updateMarkdownStreamRenderer(e, t) {
        if (this.setProp(e, t), "syntaxHighlighter" === e) {
            const e = t;
            this.executeDomAction("updateMarkdownStreamRenderer", {syntaxHighlighter: e})
        }
        if ("htmlSanitizer" === e) {
            const e = t;
            this.executeDomAction("updateMarkdownStreamRenderer", {htmlSanitizer: e})
        }
    }

    onMarkdownStreamComplete(e) {
        this.context.emit("messageRendered", {uid: this.props.uid})
    }
};
ft([y("markdown-stream-complete")], yt.prototype, "onMarkdownStreamComplete", 1), yt = ft([f("chatItem", (({
                                                                                                               props: e,
                                                                                                               appendToRoot: t,
                                                                                                               compEvent: s
                                                                                                           }) => {
    const n = ct({...e.domProps, htmlSanitizer: e.htmlSanitizer, message: void 0}), o = ((e, t) => {
        const s = e.querySelector(t);
        if (!s) throw new r({
            source: pt,
            message: `Could not find element with query "${t}". Make sure the query provided matches an element that exists in the root element.`
        });
        if (!(s instanceof HTMLElement)) throw new r({
            source: pt,
            message: `Element with query "${t}" is not a valid HTMLElement.`
        });
        return s
    })(n, ".nlux-comp-message");
    if (!o) throw new Error("Message container not found");
    const i = document.createElement("div");
    i.classList.add("nlux-markdownStream-root");
    const a = document.createElement("div");
    if (a.classList.add("nlux-markdown-container"), a.setAttribute("nlux-message-id", e.uid), i.append(a), o.append(i), e.domProps.message) {
        const t = et(e.domProps.message ?? "", "markdown", {
            markdownLinkTarget: e.markdownLinkTarget,
            syntaxHighlighter: e.syntaxHighlighter,
            htmlSanitizer: e.htmlSanitizer
        });
        a.append(t)
    }
    let c;
    t(n);
    let l = {...e};
    const h = e => ((e, t) => {
        const s = ut(e, {
            syntaxHighlighter: t?.syntaxHighlighter,
            htmlSanitizer: t?.htmlSanitizer,
            markdownLinkTarget: t?.markdownLinkTarget,
            showCodeBlockCopyButton: t?.showCodeBlockCopyButton,
            skipStreamingAnimation: t?.skipStreamingAnimation,
            streamingAnimationSpeed: t?.streamingAnimationSpeed,
            waitTimeBeforeStreamCompletion: t?.waitTimeBeforeStreamCompletion,
            onComplete: t?.onComplete
        });
        return {
            next(e) {
                s.next(e)
            }, complete() {
                s.complete()
            }
        }
    })(a, {
        syntaxHighlighter: e.syntaxHighlighter,
        htmlSanitizer: e.htmlSanitizer,
        markdownLinkTarget: e.markdownLinkTarget,
        showCodeBlockCopyButton: e.showCodeBlockCopyButton,
        skipStreamingAnimation: e.skipStreamingAnimation,
        streamingAnimationSpeed: e.streamingAnimationSpeed,
        onComplete: () => s("markdown-stream-complete")
    });
    return {
        elements: {chatItemContainer: n}, actions: {
            focus: () => {
                n.focus()
            }, processStreamedChunk: e => {
                c || (c = h(l)), c.next(e)
            }, commitStreamedChunks: () => {
                c && c.complete()
            }, updateMarkdownStreamRenderer: e => {
                l = {...l, ...e}, h(l)
            }, updateDomProps: (e, t) => {
                ((e, t, s) => {
                    if ((t.direction !== s.direction || t.layout !== s.layout || t.status !== s.status || t.message !== s.message || t.name !== s.name || t.avatar !== s.avatar) && s && (s.hasOwnProperty("direction") || s.hasOwnProperty("layout") || s.hasOwnProperty("status") || s.hasOwnProperty("message") || s.hasOwnProperty("loader") || s.hasOwnProperty("name") || s.hasOwnProperty("avatar"))) {
                        if (t.direction !== s.direction && nt(e, s.direction), t.layout !== s.layout && rt(e, s.layout), t.direction !== s.direction || t.status !== s.status || t.message !== s.message) {
                            const n = e.querySelector(`.${tt}`);
                            n && dt(n, {
                                direction: t.direction,
                                status: t.status,
                                message: t.message,
                                htmlSanitizer: t.htmlSanitizer
                            }, {
                                direction: s.direction,
                                status: s.status,
                                message: s.message,
                                htmlSanitizer: t.htmlSanitizer
                            })
                        }
                        if (t.name !== s.name || t.avatar !== s.avatar) {
                            const n = e.querySelector(`.${k}`);
                            if (!s.name && !s.avatar) return void n?.remove();
                            if (n) ht(n, {name: t.name, avatar: t.avatar}, {
                                name: s.name,
                                avatar: s.avatar
                            }); else if (void 0 !== s.name || void 0 !== s.avatar) {
                                const t = {name: s.name, avatar: s.avatar}, n = x(t), o = e.querySelector(`.${it}`);
                                o && o.prepend(n)
                            }
                        }
                        if (t.name !== s.name) {
                            const t = e.querySelector(`.${at}`);
                            t && (t.textContent = s.name || "")
                        }
                    }
                })(n, e, t)
            }
        }, onDestroy: () => {
            n.remove(), c = void 0
        }
    }
}), (({propName: e, newValue: t, dom: s}) => {
    switch (e) {
        case"markdownLinkTarget":
        case"skipStreamingAnimation":
        case"syntaxHighlighter":
        case"htmlSanitizer":
        case"showCodeBlockCopyButton":
        case"streamingAnimationSpeed":
            s.actions?.updateMarkdownStreamRenderer({[e]: t})
    }
}))], yt);
const wt = (e, t) => {
    let n, o = t, r = e, i = !0;
    const a = ((e, t) => {
        let n = !1;
        if ("function" != typeof e) throw new s({source: "x/throttle", message: "Callback must be a function"});
        return (...s) => {
            n || (e.apply(void 0, s), n = !0, setTimeout((function () {
                n = !1
            }), t))
        }
    })((e => {
        let t, s;
        return n => {
            const o = n.target;
            if (!(o instanceof HTMLElement)) return;
            const {scrollTop: r, clientHeight: i, scrollHeight: a} = o, c = a - 30, l = Math.ceil(r + i) >= c,
                h = void 0 === t || void 0 === s ? void 0 : r > t && s === a ? "down" : r < t && s === a ? "up" : void 0;
            s = a, t = r, e({scrolledToBottom: l, scrollDirection: h})
        }
    })((({scrolledToBottom: e, scrollDirection: t}) => {
        i ? "up" === t && (i = !1) : "down" === t && e && (i = !0)
    })), 50), c = e => {
        e.addEventListener("scroll", a)
    }, l = e => {
        e?.removeEventListener("scroll", a)
    }, h = e => {
        n?.uid === e && (d?.disconnect(), u?.disconnect(), n = void 0, d = void 0, u = void 0)
    };
    let d, u;
    const p = () => {
        r?.scrollTo({top: 5e4, behavior: "instant"})
    }, m = () => {
        r && o && i && p()
    }, g = () => {
        m()
    };
    return c(r), {
        updateConversationContainer: e => {
            l(r), c(e), r = e
        }, updateProps: ({autoScroll: e}) => {
            o = e
        }, handleNewChatSegmentAdded: (e, t) => {
            n && (d?.disconnect(), u?.disconnect()), n = {
                uid: e,
                container: t
            }, d = new ResizeObserver(m), d.observe(t), u = new MutationObserver(g), u.observe(t, {
                childList: !0,
                subtree: !0,
                characterData: !0
            }), o && p()
        }, handleChatSegmentRemoved: e => h(e), handleChatSegmentComplete: e => h(e), destroy: () => {
            n && (h(n.uid), n = void 0), l(r), r = void 0
        }
    }
}, kt = e => {
    const t = "function" == typeof e ? e.__compId : void 0;
    if (!t) throw new Error("Invalid compClass! Component should be registered before using");
    const s = h.retrieve(t)?.model;
    if (!s || "function" != typeof s) throw new Error(`Component "${t}" is not registered`);
    return {withContext: e => ({create: () => new s(e, {}), withProps: t => ({create: () => new s(e, t)})})}
}, xt = e => {
    const t = ["adapter", "events"], s = Object.keys(e).filter((e => !t.includes(e))), n = {};
    for (let t = 0; t < s.length; t++) {
        const o = s[t];
        n[o] = e[o]
    }
    return n
}, vt = () => {
    const e = document.createElement("div");
    e.classList.add("nlux-comp-messageLoader");
    const t = document.createElement("span");
    t.classList.add("spinning-loader");
    const s = document.createElement("div");
    return s.classList.add("nlux-comp-loaderContainer"), s.appendChild(t), e.appendChild(s), e
}, bt = {
    typing: "nlux-composer--typing",
    "submitting-prompt": "nlux-composer--submitting",
    "submitting-conversation-starter": "nlux-composer--submitting",
    waiting: "nlux-composer--waiting"
}, Ct = (e, t) => {
    Object.keys(bt).forEach((t => {
        e.classList.remove(bt[t])
    })), e.classList.add(bt[t])
}, St = e => {
    const t = document.createElement("div");
    t.classList.add("nlux-comp-composer");
    const s = document.createElement("textarea");
    s.placeholder = e.placeholder ?? "", s.value = e.message ?? "", e.autoFocus && (s.autofocus = !0);
    const n = document.createElement("button");
    return n.append((() => {
        const e = document.createElement("div");
        e.classList.add("nlux-comp-sendIcon");
        const t = document.createElement("div");
        return t.classList.add("nlux-comp-sendIcon-container"), e.appendChild(t), e
    })()), n.append(vt()), t.append(s), t.append(n), Ct(t, e.status), "submitting-conversation-starter" !== e.status && "submitting-prompt" !== e.status || (s.disabled = !0, n.disabled = !0), "waiting" === e.status && (n.disabled = !0), "typing" === e.status && (n.disabled = e.disableSubmitButton ?? "" === s.value), t
}, It = (e, t) => {
    let s = !1;
    const n = t ? e.querySelector(t) : e, o = n instanceof HTMLElement ? n : void 0;
    if (!o) throw new r({
        source: "dom/listenTo",
        message: `Could not find element with query "${t}". Make sure the query provided matches an element that exists in the root element.`
    });
    const i = new Map, a = new Map, c = () => {
        o && (i.forEach(((e, t) => {
            o.removeEventListener(t, e)
        })), i.clear(), a.clear())
    }, l = {
        on: (e, t) => {
            if (!t || !o) return l;
            if (!i.has(e)) {
                const t = t => {
                    a.get(e)?.forEach((e => e(t)))
                };
                i.set(e, t), o.addEventListener(e, t)
            }
            a.has(e) || a.set(e, new Set);
            return a.get(e).add(t), l
        }, get: () => {
            if (s) throw new Error("listenTo().get() can only be used once!");
            return s = !0, [o, c]
        }
    };
    return l
}, At = e => {
    e.style.height = "auto", e.style.height = `${e.scrollHeight}px`
};
var Pt = Object.defineProperty, Et = Object.getOwnPropertyDescriptor, Tt = (e, t, s, n) => {
    for (var o, r = n > 1 ? void 0 : n ? Et(t, s) : t, i = e.length - 1; i >= 0; i--) (o = e[i]) && (r = (n ? o(t, s, r) : o(r)) || r);
    return n && r && Pt(t, s, r), r
};
let Ot = class extends g {
    constructor(e, {props: t, eventListeners: s}) {
        super(e, t), this.userEventListeners = s
    }

    focusTextInput() {
        this.executeDomAction("focusTextInput")
    }

    handleCommandEnterKeyPressed(e) {
        const t = this.getProp("domCompProps")?.submitShortcut;
        "CommandEnter" === t && (this.handleSendButtonClick(), e?.preventDefault())
    }

    handleEnterKeyPressed(e) {
        const t = this.getProp("domCompProps")?.submitShortcut;
        t && ("Enter" !== t || e?.isComposing) || (this.handleSendButtonClick(), e?.preventDefault())
    }

    handleSendButtonClick() {
        const e = this.getProp("domCompProps");
        if (e?.disableSubmitButton) return;
        const t = e?.message;
        if (!t) return;
        const s = this.userEventListeners?.onSubmit;
        s && s()
    }

    handleTextChange(e) {
        const t = this.userEventListeners?.onTextUpdated;
        t && t(e);
        const s = this.getProp("domCompProps");
        this.setDomProps({...s, message: e})
    }

    handleTextInputUpdated(e) {
        const t = e.target;
        t instanceof HTMLTextAreaElement && this.handleTextChange(t.value)
    }

    setDomProps(e) {
        this.setProp("domCompProps", e)
    }
};
Tt([y("command-enter-key-pressed")], Ot.prototype, "handleCommandEnterKeyPressed", 1), Tt([y("enter-key-pressed")], Ot.prototype, "handleEnterKeyPressed", 1), Tt([y("send-message-clicked")], Ot.prototype, "handleSendButtonClick", 1), Tt([y("text-updated")], Ot.prototype, "handleTextInputUpdated", 1), Ot = Tt([f("composer", (({
                                                                                                                                                                                                                                                                                                                                           appendToRoot: e,
                                                                                                                                                                                                                                                                                                                                           props: t,
                                                                                                                                                                                                                                                                                                                                           compEvent: s
                                                                                                                                                                                                                                                                                                                                       }) => {
    const n = St(t.domCompProps);
    e(n);
    const [o, i] = It(n, ":scope > textarea").on("input", s("text-updated")).on("keydown", (e => {
        const t = "Enter" === e.key && !e.isComposing, n = e.altKey || e.ctrlKey || e.metaKey || e.shiftKey;
        if (t && !n) return void s("enter-key-pressed")(e);
        const o = e.getModifierState("Meta") && "Enter" === e.key,
            r = e.getModifierState("Control") && "Enter" === e.key;
        (o || r) && s("command-enter-key-pressed")(e)
    })).get(), [a, c] = It(n, ":scope > button").on("click", s("send-message-clicked")).get();
    if (!(a instanceof HTMLButtonElement)) throw new Error("Expected a button element");
    if (!(o instanceof HTMLTextAreaElement)) throw new r({
        source: (l = "composer", h = "render", `#${l}/${h}`),
        message: "Expected a textarea element"
    });
    var l, h;
    return {
        elements: {root: n, textInput: o, sendButton: a}, actions: {
            focusTextInput: () => u((() => {
                o.focus(), o.setSelectionRange(o.value.length, o.value.length)
            }))
        }, onDestroy: () => {
            i(), c()
        }
    }
}), (({propName: e, currentValue: t, newValue: s, dom: n}) => {
    "domCompProps" === e && n.elements?.root && ((e, t, s) => {
        if (t.status === s.status && t.message === s.message && t.placeholder === s.placeholder && t.autoFocus === s.autoFocus && t.disableSubmitButton === s.disableSubmitButton) return;
        const n = e.querySelector("* > textarea");
        if (t.status !== s.status) return Ct(e, s.status), ((e, t, s) => {
            if (t.status === s.status) return;
            const n = e.querySelector("* > textarea");
            "typing" !== s.status && "waiting" !== s.status || !n.disabled ? "submitting-prompt" !== s.status && "submitting-conversation-starter" !== s.status || n.disabled || (n.disabled = !0) : n.disabled = !1;
            const o = e.querySelector("* > button");
            if ("typing" === s.status) {
                const e = (t.disableSubmitButton !== s.disableSubmitButton ? s.disableSubmitButton : t.disableSubmitButton) ?? "" === n.value;
                o.disabled !== e && (o.disabled = e)
            } else "waiting" !== s.status && "submitting-prompt" !== s.status && "submitting-conversation-starter" !== s.status || o.disabled || (o.disabled = !0);
            t.placeholder !== s.placeholder && (n.placeholder = s.placeholder ?? ""), t.message !== s.message && (n.value = s.message ?? ""), t.autoFocus !== s.autoFocus && (n.autofocus = s.autoFocus ?? !1)
        })(e, t, s), void At(n);
        if (t.placeholder !== s.placeholder && (n.placeholder = s.placeholder ?? ""), t.autoFocus !== s.autoFocus && (n.autofocus = s.autoFocus ?? !1), t.message !== s.message && (n.value = s.message ?? "", At(n)), "typing" === t.status) {
            const o = e.querySelector("* > button"),
                r = (t.disableSubmitButton !== s.disableSubmitButton ? s.disableSubmitButton : t.disableSubmitButton) ?? "" === n.value;
            o.disabled !== r && (o.disabled = r)
        }
    })(n.elements.root, t, s)
}))], Ot);
const Bt = e => {
        const t = "nlux-chatSegment";
        return "complete" === e ? `${t} nlux-chatSegment--complete` : "error" === e ? `${t} nlux-chatSegment--error` : `${t} nlux-chatSegment--active`
    }, Lt = (e, t) => "assistant" === e ? t?.assistant?.name ?? "Assistant" : "user" === e ? t?.user?.name ?? "User" : "",
    Dt = "bubbles", Rt = e => e ?? Dt;
var zt = Object.defineProperty, Mt = Object.getOwnPropertyDescriptor, Ht = (e, t, s, n) => {
    for (var o, r = n > 1 ? void 0 : n ? Mt(t, s) : t, i = e.length - 1; i >= 0; i--) (o = e[i]) && (r = (n ? o(t, s, r) : o(r)) || r);
    return n && r && zt(t, s, r), r
};
let Ut = class extends g {
    constructor(e, t) {
        super(e, t), this.chatItemCompIdsByIndex = [], this.chatItemComponentsById = new Map
    }

    addChatItem(e) {
        if (this.throwIfDestroyed(), this.chatItemComponentsById.has(e.uid)) throw new Error(`CompChatSegment: chat item with id "${e.uid}" already exists`);
        const t = ((e, t, s, n) => {
            const o = t ?? Dt;
            if ("assistant" === e.participantRole) {
                const t = "complete" === e.status ? "complete" : "streaming";
                return "stream" === e.dataTransferMode ? {
                    status: t,
                    layout: o,
                    direction: "received",
                    name: Lt("assistant", {assistant: n}),
                    avatar: n?.avatar
                } : "complete" === e.status ? {
                    status: t,
                    layout: o,
                    direction: "received",
                    name: Lt("assistant", {assistant: n}),
                    avatar: n?.avatar,
                    message: (r = e.content, "string" == typeof r ? r : "object" == typeof r ? `${r}` : null == r ? "" : "function" == typeof r.toString ? r.toString() : JSON.stringify(r))
                } : {
                    status: t,
                    layout: o,
                    direction: "received",
                    name: Lt("assistant", {assistant: n}),
                    avatar: n?.avatar
                }
            }
            var r;
            return {
                status: "complete",
                layout: o,
                direction: "sent",
                message: e.content,
                name: Lt("user", {user: s}),
                avatar: s?.avatar
            }
        })(e, this.getProp("conversationLayout"), this.getProp("userPersona"), this.getProp("assistantPersona"));
        if (!t) throw new Error(`CompChatSegment: chat item with id "${e.uid}" has invalid props`);
        const s = kt(yt).withContext(this.context).withProps({
            uid: e.uid,
            domProps: t,
            markdownLinkTarget: this.getProp("markdownLinkTarget"),
            showCodeBlockCopyButton: this.getProp("showCodeBlockCopyButton"),
            skipStreamingAnimation: this.getProp("skipStreamingAnimation"),
            syntaxHighlighter: this.getProp("syntaxHighlighter"),
            htmlSanitizer: this.getProp("htmlSanitizer"),
            streamingAnimationSpeed: this.getProp("streamingAnimationSpeed")
        }).create();
        this.chatItemComponentsById.set(e.uid, s), this.chatItemCompIdsByIndex.push(e.uid), this.rendered && (this.renderedDom?.elements?.chatSegmentContainer ? s.render(this.renderedDom.elements.chatSegmentContainer, this.renderedDom.elements.loaderContainer) : c("CompChatSegment: chatSegmentContainer is not available"))
    }

    addChunk(e, t, s) {
        if (this.destroyed) return;
        const n = this.chatItemComponentsById.get(e);
        if (!n) throw new Error(`CompChatSegment: chat item with id "${e}" not found`);
        n.addChunk(t, s)
    }

    complete() {
        this.throwIfDestroyed(), this.chatItemComponentsById.forEach((e => e.commitChunks())), this.setProp("status", "complete")
    }

    destroy() {
        this.chatItemComponentsById.forEach((e => e.destroy())), this.chatItemComponentsById.clear(), this.chatItemCompIdsByIndex = [], super.destroy()
    }

    getChatItems() {
        return this.chatItemCompIdsByIndex.map((e => this.chatItemComponentsById.get(e))).filter((e => !!e))
    }

    onLoaderShown(e) {
        this.renderedDom?.elements && (this.renderedDom.elements.loaderContainer = e)
    }

    setAssistantPersona(e) {
        this.setProp("assistantPersona", e);
        const t = {name: e?.name, avatar: e?.avatar};
        this.chatItemComponentsById.forEach((e => {
            "assistant" === e.getChatSegmentItem().participantRole && e.updateDomProps(t)
        }))
    }

    setLayout(e) {
        this.setProp("conversationLayout", e), this.chatItemComponentsById.forEach((t => {
            t.updateDomProps({layout: e})
        }))
    }

    setUserPersona(e) {
        this.setProp("userPersona", e);
        const t = {name: e?.name, avatar: e?.avatar};
        this.chatItemComponentsById.forEach((e => {
            "user" === e.getChatSegmentItem().participantRole && e.updateDomProps(t)
        }))
    }

    updateMarkdownStreamRenderer(e, t) {
        this.setProp(e, t)
    }

    setProp(e, t) {
        super.setProp(e, t), "markdownLinkTarget" !== e && "syntaxHighlighter" !== e && "htmlSanitizer" !== e && "skipStreamingAnimation" !== e && "streamingAnimationSpeed" !== e || this.chatItemComponentsById.forEach((s => {
            s.updateMarkdownStreamRenderer(e, t)
        }))
    }

    onChatSegmentReady() {
        u((() => {
            if (!this.renderedDom?.elements?.chatSegmentContainer) return;
            const e = this.renderedDom?.elements?.chatSegmentContainer;
            this.chatItemComponentsById.forEach((t => {
                t.rendered || t.render(e)
            }))
        }))
    }

    onLoaderHidden() {
        this.renderedDom?.elements && (this.renderedDom.elements.loaderContainer = void 0)
    }
};
Ht([y("loader-shown")], Ut.prototype, "onLoaderShown", 1), Ht([y("chat-segment-ready")], Ut.prototype, "onChatSegmentReady", 1), Ht([y("loader-hidden")], Ut.prototype, "onLoaderHidden", 1), Ut = Ht([f("chatSegment", (({
                                                                                                                                                                                                                              props: e,
                                                                                                                                                                                                                              compEvent: t,
                                                                                                                                                                                                                              appendToRoot: s
                                                                                                                                                                                                                          }) => {
    let n;
    const o = document.createElement("div");
    o.className = Bt(e.status);
    const r = () => {
        if (!n) {
            n = document.createElement("div"), n.className = "nlux-chatSegment-loader-container";
            const e = vt();
            n.appendChild(e), o.appendChild(n), t("loader-shown")(n)
        }
    };
    return "active" === e.status && r(), s(o), t("chat-segment-ready")(), {
        elements: {
            chatSegmentContainer: o,
            loaderContainer: n
        }, actions: {
            showLoader: r, hideLoader: () => {
                n && (n.remove(), n = void 0, t("loader-hidden")())
            }
        }, onDestroy: () => {
            o.remove()
        }
    }
}), (({propName: e, newValue: t, dom: s}) => {
    if ("status" === e) {
        const e = s.elements?.chatSegmentContainer;
        if (!e) return;
        const n = t;
        e.className = Bt(n), "active" === n ? s.actions?.showLoader() : s.actions?.hideLoader()
    }
}))], Ut);
var Qt = Object.defineProperty, Nt = Object.getOwnPropertyDescriptor;
let Ft = class extends g {
    constructor(e, t) {
        super(e, t), this.chatSegmentCompIdsByIndex = [], this.chatSegmentComponentsById = new Map, t.messages && t.messages.length > 0 && this.addChatSegment("complete", t.messages)
    }

    addChatItem(e, t) {
        const s = this.chatSegmentComponentsById.get(e);
        if (!s) throw new Error(`CompConversation: chat segment with id "${e}" not found`);
        s.destroyed ? c(`CompConversation: chat segment with id "${e}" is destroyed and cannot be used`) : s.addChatItem(t)
    }

    addChatSegment(e = "active", t) {
        this.throwIfDestroyed();
        const s = m(), n = kt(Ut).withContext(this.context).withProps({
            uid: s,
            status: e,
            conversationLayout: this.getProp("conversationLayout"),
            userPersona: this.getProp("userPersona"),
            assistantPersona: this.getProp("assistantPersona"),
            markdownLinkTarget: this.getProp("markdownLinkTarget"),
            showCodeBlockCopyButton: this.getProp("showCodeBlockCopyButton"),
            skipStreamingAnimation: this.getProp("skipStreamingAnimation"),
            syntaxHighlighter: this.getProp("syntaxHighlighter"),
            htmlSanitizer: this.getProp("htmlSanitizer"),
            streamingAnimationSpeed: this.getProp("streamingAnimationSpeed")
        }).create();
        if (t) for (const e of t) "assistant" === e.role ? n.addChatItem({
            uid: m(),
            participantRole: "assistant",
            time: new Date,
            dataTransferMode: "batch",
            status: "complete",
            content: e.message,
            serverResponse: e.serverResponse,
            contentType: "text"
        }) : "user" === e.role && n.addChatItem({
            uid: m(),
            participantRole: "user",
            time: new Date,
            status: "complete",
            content: e.message,
            contentType: "text"
        });
        this.chatSegmentComponentsById.set(s, n), this.chatSegmentCompIdsByIndex.push(s);
        const o = n.id;
        return this.addSubComponent(o, n, "segmentsContainer"), this.notifyAboutSegmentCountChange(this.chatSegmentCompIdsByIndex.length), s
    }

    addChunk(e, t, s, n) {
        const o = this.chatSegmentComponentsById.get(e);
        if (!o) throw new Error(`CompConversation: chat segment with id "${e}" not found`);
        o.addChunk(t, s, n)
    }

    completeChatSegment(e) {
        const t = this.chatSegmentComponentsById.get(e);
        if (!t) throw new Error(`CompConversation: chat segment with id "${e}" not found`);
        t.destroyed || t.complete()
    }

    getChatSegmentContainer(e) {
        const t = this.chatSegmentComponentsById.get(e);
        if (t?.root instanceof HTMLElement) return t.root
    }

    getConversationContentForAdapter(e = "max") {
        if ("number" == typeof e && e < 0) return void c(`Invalid value provided for 'historyPayloadSize' : "${e}"! Value must be a positive integer or 'max'.`);
        if (0 === e) return;
        const t = (e => {
            const t = [];
            return e.forEach((e => {
                e.items.forEach((e => {
                    if ("complete" === e.status) if ("assistant" === e.participantRole) t.push({
                        role: "assistant",
                        message: e.content
                    }); else if ("user" === e.participantRole) return t.push({role: "user", message: e.content})
                }))
            })), t
        })(this.chatSegmentCompIdsByIndex.map((e => this.chatSegmentComponentsById.get(e))).filter((e => void 0 !== e)).map((e => ({
            uid: e.id,
            status: "complete",
            items: e.getChatItems().map((e => e.getChatSegmentItem()))
        }))));
        return "max" === e ? t : t.slice(-e)
    }

    removeChatSegment(e) {
        const t = this.chatSegmentComponentsById.get(e);
        if (!t) return;
        const s = t.id;
        this.subComponents.has(s) && this.removeSubComponent(s), this.chatSegmentComponentsById.delete(t.id);
        const n = this.chatSegmentCompIdsByIndex.indexOf(e);
        n >= 0 && this.chatSegmentCompIdsByIndex.splice(n, 1), this.notifyAboutSegmentCountChange(this.chatSegmentCompIdsByIndex.length)
    }

    setAssistantPersona(e) {
        this.setProp("assistantPersona", e), this.chatSegmentComponentsById.forEach((t => {
            t.setAssistantPersona(e)
        }))
    }

    setConversationLayout(e) {
        this.setProp("conversationLayout", e), this.chatSegmentComponentsById.forEach((t => {
            t.setLayout(e)
        }))
    }

    setUserPersona(e) {
        this.setProp("userPersona", e), this.chatSegmentComponentsById.forEach((t => {
            t.setUserPersona(e)
        }))
    }

    updateMarkdownStreamRenderer(e, t) {
        this.setProp(e, t)
    }

    setProp(e, t) {
        if (super.setProp(e, t), "markdownLinkTarget" === e || "syntaxHighlighter" === e || "htmlSanitizer" === e || "skipStreamingAnimation" === e || "streamingAnimationSpeed" === e || "showCodeBlockCopyButton" === e) {
            const s = e, n = t;
            this.chatSegmentComponentsById.forEach((e => {
                e.updateMarkdownStreamRenderer(s, n)
            }))
        }
    }

    notifyAboutSegmentCountChange(e) {
        const t = this.getProp("onSegmentCountChange");
        t && t(e)
    }
};
Ft = ((e, t, s, n) => {
    for (var o, r = n > 1 ? void 0 : n ? Nt(t, s) : t, i = e.length - 1; i >= 0; i--) (o = e[i]) && (r = (n ? o(t, s, r) : o(r)) || r);
    return n && r && Qt(t, s, r), r
})([f("conversation", (({appendToRoot: e}) => {
    const t = document.createElement("div");
    return t.classList.add("nlux-chatSegments-container"), e(t), {elements: {segmentsContainer: t}, actions: {}}
}), (() => {
}))], Ft);
var qt = Object.defineProperty, jt = Object.getOwnPropertyDescriptor, Yt = (e, t, s, n) => {
    for (var o, r = n > 1 ? void 0 : n ? jt(t, s) : t, i = e.length - 1; i >= 0; i--) (o = e[i]) && (r = (n ? o(t, s, r) : o(r)) || r);
    return n && r && qt(t, s, r), r
};
let $t = class extends g {
    constructor(e, t) {
        super(e, t), this.updateConversationStarters = e => {
        }
    }

    conversationStarterClicked(e) {
        this.getProp("onConversationStarterSelected")(e)
    }
};
Yt([y("conversation-starter-selected")], $t.prototype, "conversationStarterClicked", 1), $t = Yt([f("conversationStarters", (({
                                                                                                                                  appendToRoot: e,
                                                                                                                                  props: t,
                                                                                                                                  compEvent: s
                                                                                                                              }) => {
    const n = (e => {
        const t = document.createElement("div");
        return t.classList.add("nlux-comp-conversationStarters"), e.forEach(((e, s) => {
            const n = document.createElement("button");
            n.classList.add("nlux-comp-conversationStarter");
            let o = document.createElement("div");
            e.icon && ("string" == typeof e.icon ? (o = document.createElement("img"), o.setAttribute("src", e.icon), o.setAttribute("width", "16px")) : (o.className = "nlux-comp-conversationStarter-icon-container", o.appendChild(e.icon)));
            const r = document.createElement("span");
            r.classList.add("nlux-comp-conversationStarter-prompt"), r.textContent = e.label ?? e.prompt, n.appendChild(o), n.appendChild(r), t.appendChild(n)
        })), t
    })(t.conversationStarters);
    e(n);
    let o = [];
    return t.conversationStarters.forEach(((e, t) => {
        const [r, i] = It(n, `:scope > :nth-child(${t + 1})`).on("click", (() => {
            s("conversation-starter-selected")(e)
        })).get();
        o.push(i)
    })), {
        elements: {}, actions: {}, onDestroy: () => {
            o.forEach((e => e())), o = [], n.remove()
        }
    }
}), (({}) => {
}))], $t);
const Wt = "nlux-comp-welcomeMessage-text", Xt = (e, t) => {
    const s = e.querySelector(`.${Wt}`);
    if ("" !== t && void 0 !== t) if (s) s.textContent = t; else {
        const s = document.createElement("div");
        s.classList.add(Wt), s.textContent = t, e.appendChild(s)
    } else s?.remove()
}, Zt = "nlux-comp-welcomeMessage-personaName", Gt = e => {
    const t = document.createElement("div");
    t.classList.add("nlux-comp-welcomeMessage");
    const s = x({name: e.name, avatar: e.avatar});
    t.append(s);
    const n = document.createElement("div"), o = document.createTextNode(e.name);
    return n.append(o), n.classList.add(Zt), t.append(n), Xt(t, e.message), t
}, Jt = () => Gt({
    name: "",
    avatar: "data:image/png;base64,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"
});
var Kt = Object.defineProperty, Vt = Object.getOwnPropertyDescriptor;
let _t = class extends g {
    constructor(e, t) {
        super(e, t), this.setConversationStarters(t.conversationStarters)
    }

    setAssistantPersona(e) {
        this.setProp("assistantPersona", e), this.executeDomAction("updateAssistantPersona", e)
    }

    setConversationStarters(e) {
        if (e || this.conversationStartersComp) {
            if (e && !this.conversationStartersComp) {
                const t = this.getProp("onConversationStarterSelected");
                return this.conversationStartersComp = kt($t).withContext(this.context).withProps({
                    conversationStarters: e,
                    onConversationStarterSelected: t
                }).create(), void this.addSubComponent(this.conversationStartersComp.id, this.conversationStartersComp, "conversationStartersContainer")
            }
            !e && this.conversationStartersComp ? (this.removeSubComponent(this.conversationStartersComp.id), this.conversationStartersComp = void 0) : this.conversationStartersComp?.updateConversationStarters(e)
        }
    }

    setShowGreeting(e) {
        this.setProp("showGreeting", e), this.executeDomAction("resetGreeting", e)
    }

    resetConversationStarters() {
        const e = this.getProp("conversationStarters");
        this.setConversationStarters(e)
    }
};
_t = ((e, t, s, n) => {
    for (var o, r = n > 1 ? void 0 : n ? Vt(t, s) : t, i = e.length - 1; i >= 0; i--) (o = e[i]) && (r = (n ? o(t, s, r) : o(r)) || r);
    return n && r && Kt(t, s, r), r
})([f("launchPad", (({appendToRoot: e, props: t}) => {
    const s = {
        assistantPersona: t.assistantPersona,
        conversationStarters: t.conversationStarters,
        showGreeting: !1 !== t.showGreeting
    };
    let n;
    if (s.showGreeting) if (t.assistantPersona) {
        const e = t.assistantPersona;
        n = Gt({name: e.name, avatar: e.avatar, message: e.tagline})
    } else n = Jt();
    n && (e(n), s.greetingElement = n);
    const o = document.createElement("div");
    o.classList.add("nlux-conversationStarters-container"), e(o);
    const r = (e = !0) => {
        s.showGreeting = e, s.greetingElement && (s.greetingElement.remove(), s.greetingElement = void 0), e && (s.greetingElement = s.assistantPersona ? Gt({
            name: s.assistantPersona.name,
            avatar: s.assistantPersona.avatar,
            message: s.assistantPersona.tagline
        }) : Jt(), s.greetingElement && o.insertAdjacentElement("beforebegin", s.greetingElement))
    };
    return {
        elements: {conversationStartersContainer: o}, actions: {
            resetGreeting: (e = !0) => {
                r(e)
            }, updateAssistantPersona: e => {
                const t = s.assistantPersona;
                s.assistantPersona = e, (t || e) && s.showGreeting && (e && t ? ((e, t, s) => {
                    if (t.message !== s.message || t.name !== s.name || t.avatar !== s.avatar) {
                        if (t.message !== s.message && Xt(e, s.message), t.name !== s.name) {
                            const t = e.querySelector(`.${Zt}`);
                            if (t) {
                                const e = document.createTextNode(s.name);
                                t.replaceChildren(e)
                            }
                        }
                        if (t.avatar !== s.avatar || t.name !== s.name) {
                            const n = e.querySelector(`.${k}`);
                            n && ht(n, {name: t.name, avatar: t.avatar}, {name: s.name, avatar: s.avatar})
                        }
                    }
                })(s.greetingElement, {name: t?.name, avatar: t?.avatar, message: t?.tagline}, {
                    name: e.name,
                    avatar: e.avatar,
                    message: e.tagline
                }) : r())
            }
        }, onDestroy: () => {
            s.greetingElement?.remove(), o.remove()
        }
    }
}), (({}) => {
}))], _t);
const es = e => {
    if ("object" != typeof e || null === e) return !1;
    const t = e;
    return ("function" == typeof t.streamText || "function" == typeof t.batchText) && ["stream", "batch"].includes(t.dataTransferMode) && "string" == typeof t.id && "object" == typeof t.info && null !== t.info && "function" == typeof t.preProcessAiBatchedMessage && "function" == typeof t.preProcessAiStreamedChunk
}, ts = "undefined" != typeof navigator && navigator?.userAgent?.includes("Safari") ? 100 : 1, ss = (e, t = ts) => {
    setTimeout((() => {
        e()
    }), t)
}, ns = (e, t, s, n, o, r, a, c, l) => new Promise((h => {
    const d = m(), u = [], p = [];
    let g = !1, f = !1, y = !1;
    const w = es(s), k = {
        next: e => {
            if (f || y) return;
            let t, r;
            if (w) {
                const o = e, i = s.preProcessAiStreamedChunk(o, n);
                null != i && (t = i, r = o, u.push(t), p.push(r))
            } else t = e, u.push(t);
            null != t ? (g || g || (g = !0, ss((() => {
                o.forEach((e => {
                    e({
                        uid: d,
                        status: "streaming",
                        time: new Date,
                        participantRole: "assistant",
                        dataTransferMode: "stream"
                    })
                }))
            }))), ss((() => {
                a.forEach((e => {
                    e({chunk: t, messageId: d, serverResponse: r})
                }))
            }))) : i("Adapter returned an undefined or null value from streamText. This chunk will be ignored.")
        }, complete: () => {
            f || y || (y = !0, ss((() => {
                const e = {
                    uid: d,
                    status: "complete",
                    content: u,
                    contentType: "text",
                    serverResponse: void 0,
                    time: new Date,
                    participantRole: "assistant",
                    dataTransferMode: "stream"
                };
                r.forEach((t => {
                    t(e)
                })), h()
            })), ss((() => {
                const s = {
                    uid: e,
                    status: "complete",
                    items: [t, {
                        uid: d,
                        status: "complete",
                        contentType: "text",
                        content: u,
                        serverResponse: p,
                        time: new Date,
                        participantRole: "assistant",
                        dataTransferMode: "stream"
                    }]
                };
                c.forEach((e => {
                    e(s)
                }))
            })))
        }, error: e => {
            f || y || (f = !0, ss((() => {
                l.forEach((t => {
                    t("failed-to-stream-content", e)
                })), h()
            })))
        }
    };
    s.streamText(t.content, k, n)
})), os = (e, t, s) => {
    if (!e) return (() => {
        const e = new Set, t = m(), s = {uid: t, status: "complete", items: []};
        return ss((() => {
            e.forEach((e => {
                e(s)
            })), e.clear()
        })), {
            segment: s, observable: {
                on: (t, s) => {
                    "complete" === t && e.add(s)
                }, removeListener: (t, s) => {
                    e.delete(s)
                }, destroy: () => {
                    e.clear()
                }, get segmentId() {
                    return t
                }
            }, dataTransferMode: "batch"
        }
    })();
    const n = t;
    if (void 0 === n.streamText && void 0 === n.batchText && void 0 === n.streamServerComponent) return (e => {
        const t = new Set, s = m(), n = {uid: s, status: "error", items: []};
        return ss((() => {
            t.forEach((t => t(e))), t.clear()
        })), {
            segment: n, dataTransferMode: "stream", observable: {
                on: (e, s) => {
                    "error" === e && t.add(s)
                }, removeListener: (e, s) => {
                    t.delete(s)
                }, destroy: () => {
                    t.clear()
                }, get segmentId() {
                    return s
                }
            }
        }
    })("no-data-transfer-mode-supported");
    const o = m(), r = (e => ({
        uid: m(),
        time: new Date,
        status: "complete",
        participantRole: "user",
        content: e,
        contentType: "text"
    }))(e);
    let a, c, l, h, d, u, p = new Set, g = new Set, f = new Set;
    ss((() => {
        p?.size && (p.forEach((e => {
            e(r)
        })), p.clear(), p = void 0)
    }));
    const y = (e => {
        const t = [], s = e, n = e;
        void 0 === s?.streamText && void 0 === n?.streamServerComponent || t.push("stream"), void 0 !== s?.batchText && t.push("batch");
        const o = es(e) ? e : void 0, r = o?.dataTransferMode ?? void 0, i = 1 === t.length ? t[0] : "stream";
        return r ?? i
    })(t), w = (e => "streamServerComponent" in e ? "server-component" : "text")(t);
    "server-component" === w ? (l = new Set, c = new Set, ((e, t, s, n, o, r, a, c) => new Promise(((l, h) => {
        try {
            const i = t.content, h = m(), d = "assistant", u = "streaming", p = new Date, g = "stream";
            let f, y;
            const w = () => {
                ss((() => {
                    r.forEach((e => {
                        f && y && e({...f, content: y, status: "complete"})
                    }))
                }), 20);
                const s = {uid: e, status: "complete", items: [t, f]};
                ss((() => {
                    a.forEach((e => {
                        e(s)
                    })), l()
                }), 20)
            }, k = () => {
                c.forEach((e => {
                    e("failed-to-stream-server-component", new Error("Failed to load content"))
                }))
            };
            y = s.streamServerComponent(i, n, {onServerComponentReceived: w, onError: k}), f = {
                uid: h,
                content: y,
                contentType: "server-component",
                participantRole: d,
                status: u,
                time: p,
                dataTransferMode: g
            }, ss((() => {
                o.forEach((e => {
                    e(f)
                }))
            }), 10)
        } catch (e) {
            i(e);
            const t = e instanceof Error ? e : "string" == typeof e ? new Error(e) : new Error("Unknown error");
            ss((() => {
                c.forEach((e => e("failed-to-load-content", t)))
            }))
        }
    })))(o, r, t, s, c, l, g, f).finally((() => {
        ss((() => k()))
    }))) : "batch" === y ? (a = new Set, (async (e, t, s, n, o, r, a) => {
        try {
            const i = t.content, a = es(s) ? s : void 0, c = void 0 !== a, l = m(), h = "assistant", d = "complete",
                u = new Date, p = "batch";
            let g;
            if (c) {
                const e = await a.batchText(i, n), t = a.preProcessAiBatchedMessage(e, n);
                null != t && (g = {
                    uid: l,
                    content: t,
                    contentType: "text",
                    serverResponse: e,
                    participantRole: h,
                    status: d,
                    time: u,
                    dataTransferMode: p
                })
            } else g = {
                uid: l,
                content: await s.batchText(i, n),
                contentType: "text",
                serverResponse: void 0,
                participantRole: h,
                status: d,
                time: u,
                dataTransferMode: p
            };
            if (!g) throw new Error("Response from adapter was undefined or cannot be processed");
            const f = g;
            ss((() => {
                o.forEach((e => {
                    e(f)
                }))
            }));
            const y = {uid: e, status: "complete", items: [t, g]};
            ss((() => {
                r.forEach((e => {
                    e(y)
                }))
            }))
        } catch (e) {
            i(e);
            const t = e instanceof Error ? e : "string" == typeof e ? new Error(e) : new Error("Unknown error");
            ss((() => {
                a.forEach((e => e("failed-to-load-content", t)))
            }))
        }
    })(o, r, t, s, a, g, f).finally((() => {
        ss((() => k()))
    }))) : (h = new Set, d = new Set, u = new Set, ns(o, r, t, s, h, d, u, g, f).finally((() => {
        ss((() => k()))
    })));
    const k = () => {
        p?.clear(), p = void 0, a?.clear(), a = void 0, c?.clear(), c = void 0, l?.clear(), l = void 0, h?.clear(), h = void 0, d?.clear(), d = void 0, u?.clear(), u = void 0, g?.clear(), g = void 0, f?.clear(), f = void 0
    };
    return {
        segment: {status: "active", uid: o, items: []}, dataTransferMode: y, observable: {
            get segmentId() {
                return o
            }, on: (e, t) => {
                "userMessageReceived" === e && p ? p.add(t) : "aiMessageReceived" === e && a ? a.add(t) : "aiServerComponentStreamStarted" === e && c ? c.add(t) : "aiServerComponentStreamed" === e && l ? l.add(t) : "aiMessageStreamStarted" === e && h ? h.add(t) : "aiMessageStreamed" === e && d ? d.add(t) : "aiChunkReceived" === e && u ? u.add(t) : "complete" === e && g ? g.add(t) : "error" === e && f && f.add(t)
            }, removeListener: (e, t) => {
                "userMessageReceived" !== e ? "aiMessageReceived" !== e ? "aiMessageStreamStarted" !== e ? "aiMessageStreamed" !== e ? "aiChunkReceived" !== e ? "complete" !== e ? "error" !== e || f?.delete(t) : g?.delete(t) : u?.delete(t) : d?.delete(t) : h?.delete(t) : a?.delete(t) : p?.delete(t)
            }, destroy: () => k()
        }
    }
}, rs = {
    "data-transfer-mode-not-supported": "Requested data transfer mode is not supported",
    "no-data-transfer-mode-supported": "Adapter does not support any valid data transfer modes",
    "connection-error": "Connection error",
    "invalid-credentials": "Invalid credentials",
    "invalid-api-key": "Invalid API key",
    "http-server-side-error": "HTTP server side error",
    "http-client-side-error": "HTTP client side error",
    "failed-to-load-content": "Failed to load content",
    "failed-to-stream-content": "Failed to stream content",
    "failed-to-stream-server-component": "Failed to stream server component",
    "failed-to-render-content": "Failed to display content"
}, is = (e, t) => void 0 !== e && e.length > 0 || void 0 !== t && t > 0 ? "active" : "starting";
var as = Object.defineProperty, cs = Object.getOwnPropertyDescriptor, ls = (e, t, s, n) => {
    for (var o, r = n > 1 ? void 0 : n ? cs(t, s) : t, i = e.length - 1; i >= 0; i--) (o = e[i]) && (r = (n ? o(t, s, r) : o(r)) || r);
    return n && r && as(t, s, r), r
};
let hs = class extends g {
    constructor(e, {
        conversationLayout: t,
        autoScroll: s,
        streamingAnimationSpeed: n,
        visible: o = !0,
        composer: r,
        assistantPersona: i,
        userPersona: a,
        showGreeting: c,
        conversationStarters: l,
        initialConversationContent: h,
        syntaxHighlighter: d,
        htmlSanitizer: u,
        markdownLinkTarget: p,
        showCodeBlockCopyButton: m,
        skipStreamingAnimation: g
    }) {
        if (super(e, {
            conversationLayout: t,
            visible: o,
            autoScroll: s,
            streamingAnimationSpeed: n,
            assistantPersona: i,
            userPersona: a,
            conversationStarters: l,
            showGreeting: c,
            initialConversationContent: h,
            composer: r,
            syntaxHighlighter: d,
            htmlSanitizer: u,
            markdownLinkTarget: p,
            showCodeBlockCopyButton: m,
            skipStreamingAnimation: g
        }), this.handleConversationStarterClick = e => {
            this.composer.setDomProps({status: "submitting-conversation-starter"}), this.composer.handleTextChange(e.prompt), this.composer.handleSendButtonClick()
        }, this.handleSegmentCountChange = e => {
            if (this.segmentCount === e) return;
            this.segmentCount = e;
            const t = is(this.getProp("initialConversationContent") || void 0, this.segmentCount);
            this.chatRoomStatus !== t && (this.chatRoomStatus = t, this.executeDomAction("updateChatRoomStatus", this.chatRoomStatus), "active" === this.chatRoomStatus ? this.launchPad?.id && (this.removeSubComponent(this.launchPad?.id), this.launchPad = void 0) : this.addLaunchPad(this.getProp("showGreeting") ?? !0, this.getProp("assistantPersona"), this.getProp("conversationStarters"), this.handleConversationStarterClick))
        }, this.prompt = "", this.segmentCount = h && h.length > 0 ? 1 : 0, this.chatRoomStatus = is(h, this.segmentCount), "starting" === this.chatRoomStatus && this.addLaunchPad(c, i, l, this.handleConversationStarterClick), this.addConversation(i, a, h), this.addComposer(r?.placeholder, r?.autoFocus, r?.disableSubmitButton, r?.submitShortcut), !this.conversation || !this.composer) throw new Error("Chat room not initialized — An error occurred while initializing key components.")
    }

    getConversationContentForAdapter(e = "max") {
        return this.conversation.getConversationContentForAdapter(e)
    }

    hide() {
        this.setProp("visible", !1)
    }

    messagesContainerClicked() {
        this.composer?.focusTextInput()
    }

    onChatRoomReady() {
        u((() => {
            const e = this.renderedDom?.elements?.conversationContainer;
            if (e instanceof HTMLElement) {
                this.autoScrollController = wt(e, this.getProp("autoScroll") ?? !0);
                let t = 0;
                const s = 20, n = () => {
                    e.scrollHeight > e.clientHeight && (e.scrollTo({behavior: "smooth", top: 5e4}), clearInterval(o))
                }, o = setInterval((() => {
                    t >= s ? clearInterval(o) : (n(), t++)
                }), 10)
            }
            this.context.emit("ready", {aiChatProps: xt(this.context.aiChatProps)})
        }))
    }

    setProps(e) {
        if (e.hasOwnProperty("autoScroll")) {
            const t = e.autoScroll;
            this.autoScrollController?.updateProps({autoScroll: t})
        }
        if (e.hasOwnProperty("conversationLayout") && this.conversation?.setConversationLayout(e.conversationLayout), e.hasOwnProperty("syntaxHighlighter") && this.setProp("syntaxHighlighter", e.syntaxHighlighter), e.hasOwnProperty("htmlSanitizer") && this.setProp("htmlSanitizer", e.htmlSanitizer), e.hasOwnProperty("markdownLinkTarget") && this.setProp("markdownLinkTarget", e.markdownLinkTarget), e.hasOwnProperty("skipStreamingAnimation") && this.setProp("skipStreamingAnimation", e.skipStreamingAnimation), e.hasOwnProperty("streamingAnimationSpeed") && this.setProp("streamingAnimationSpeed", e.streamingAnimationSpeed), e.hasOwnProperty("assistantPersona") && (this.conversation?.setAssistantPersona(e.assistantPersona ?? void 0), this.launchPad?.setAssistantPersona(e.assistantPersona ?? void 0)), e.hasOwnProperty("userPersona") && this.conversation?.setUserPersona(e.userPersona ?? void 0), e.hasOwnProperty("showGreeting") && this.launchPad?.setShowGreeting(e.showGreeting ?? !0), e.hasOwnProperty("conversationStarters") && this.launchPad?.setConversationStarters(e.conversationStarters), e.hasOwnProperty("composer") && this.composer) {
            const t = {...this.composer.getProp("domCompProps"), ...e.composer};
            this.composer.setDomProps(t)
        }
    }

    show() {
        this.setProp("visible", !0)
    }

    setProp(e, t) {
        if (super.setProp(e, t), "markdownLinkTarget" === e || "syntaxHighlighter" === e || "htmlSanitizer" === e || "skipStreamingAnimation" === e || "streamingAnimationSpeed" === e) {
            const s = e, n = t;
            this.conversation.updateMarkdownStreamRenderer(s, n)
        }
    }

    addComposer(e, t, s, n) {
        this.composer = kt(Ot).withContext(this.context).withProps({
            props: {
                domCompProps: {
                    status: "typing",
                    placeholder: e,
                    autoFocus: t,
                    disableSubmitButton: s,
                    submitShortcut: n
                }
            },
            eventListeners: {
                onTextUpdated: e => this.handleComposerTextChange(e),
                onSubmit: () => this.handleComposerSubmit()
            }
        }).create(), this.addSubComponent(this.composer.id, this.composer, "composerContainer")
    }

    addConversation(e, t, s) {
        this.conversation = kt(Ft).withContext(this.context).withProps({
            assistantPersona: e,
            userPersona: t,
            messages: s,
            conversationLayout: this.getProp("conversationLayout"),
            markdownLinkTarget: this.getProp("markdownLinkTarget"),
            showCodeBlockCopyButton: this.getProp("showCodeBlockCopyButton"),
            skipStreamingAnimation: this.getProp("skipStreamingAnimation"),
            streamingAnimationSpeed: this.getProp("streamingAnimationSpeed"),
            syntaxHighlighter: this.getProp("syntaxHighlighter"),
            htmlSanitizer: this.getProp("htmlSanitizer"),
            onSegmentCountChange: this.handleSegmentCountChange
        }).create(), this.addSubComponent(this.conversation.id, this.conversation, "conversationContainer")
    }

    addLaunchPad(e, t, s, n) {
        this.launchPad = kt(_t).withContext(this.context).withProps({
            showGreeting: e,
            assistantPersona: t,
            conversationStarters: s,
            onConversationStarterSelected: n
        }).create(), this.addSubComponent(this.launchPad.id, this.launchPad, "launchPadContainer")
    }

    handleComposerSubmit() {
        const e = this.props.composer;
        (({
              context: e,
              composerInstance: t,
              conversation: s,
              autoScrollController: n,
              messageToSend: o,
              resetComposer: r,
              setComposerAsWaiting: a
          }) => () => {
            const c = s.addChatSegment();
            try {
                const i = t.getProp("domCompProps");
                t.setDomProps({...i, status: "submitting-prompt"});
                const l = {
                    aiChatProps: e.aiChatProps,
                    conversationHistory: s.getConversationContentForAdapter(e.aiChatProps?.conversationOptions?.historyPayloadSize)
                }, h = os(o, e.adapter, l);
                h.observable.on("error", ((t, o) => {
                    s.removeChatSegment(c), n?.handleChatSegmentRemoved(c), r(!1), e.exception(t), e.emit("error", {
                        errorId: t,
                        message: rs[t],
                        errorObject: o
                    })
                })), h.observable.on("userMessageReceived", (t => {
                    s.addChatItem(c, t), e.emit("messageSent", {uid: t.uid, message: t.content}), u((() => {
                        if (n) {
                            const e = s.getChatSegmentContainer(c);
                            e && n.handleNewChatSegmentAdded(c, e)
                        }
                    }))
                })), "batch" === h.dataTransferMode ? h.observable.on("aiMessageReceived", (t => {
                    const n = "string" == typeof t.content, o = {...t, content: n ? "" : t.content};
                    s.addChatItem(c, o), n && s.addChunk(c, o.uid, t.content, t.serverResponse), s.completeChatSegment(c), e.emit("messageReceived", {
                        uid: t.uid,
                        message: t.content
                    }), r(!0)
                })) : (h.observable.on("aiMessageStreamStarted", (t => {
                    s.addChatItem(c, t), a(), e.emit("messageStreamStarted", {uid: t.uid})
                })), h.observable.on("aiChunkReceived", (e => {
                    const {messageId: t, chunk: n, serverResponse: o} = e;
                    s.addChunk(c, t, n, o)
                })), h.observable.on("aiMessageStreamed", (t => {
                    "complete" === t.status && e.emit("messageReceived", {uid: t.uid, message: t.content})
                })), h.observable.on("aiServerComponentStreamStarted", (t => {
                    s.addChatItem(c, t), a(), e.emit("messageStreamStarted", {uid: t.uid})
                })), h.observable.on("aiServerComponentStreamed", (t => {
                    "complete" === t.status && e.emit("messageReceived", {uid: t.uid, message: t.content})
                })), h.observable.on("complete", (() => {
                    s.completeChatSegment(c), r(!1)
                })))
            } catch (e) {
                i(e), r(!1)
            }
        })({
            context: this.context,
            composerInstance: this.composer,
            conversation: this.conversation,
            messageToSend: this.prompt,
            autoScrollController: this.autoScrollController,
            resetComposer: t => {
                this.destroyed || this.resetComposer(t, e?.autoFocus)
            },
            setComposerAsWaiting: () => {
                this.destroyed || this.composer.setDomProps({status: "waiting"})
            }
        })()
    }

    handleComposerTextChange(e) {
        this.prompt = e
    }

    resetComposer(e = !1, t = !1) {
        if (!this.composer) return;
        const s = {...this.composer.getProp("domCompProps"), status: "typing"};
        e && (s.message = ""), this.composer.setDomProps(s), t && this.composer.focusTextInput()
    }
};
ls([y("conversation-container-clicked")], hs.prototype, "messagesContainerClicked", 1), ls([y("chat-room-ready")], hs.prototype, "onChatRoomReady", 1), hs = ls([f("chatRoom", (({
                                                                                                                                                                                     appendToRoot: e,
                                                                                                                                                                                     compEvent: t,
                                                                                                                                                                                     props: s
                                                                                                                                                                                 }) => {
    const n = document.createElement("div");
    n.classList.add("nlux-conversation-container");
    const o = document.createElement("div");
    o.classList.add("nlux-composer-container");
    const r = document.createElement("div");
    r.classList.add("nlux-launchPad-container");
    const i = document.createDocumentFragment();
    i.appendChild(r), i.appendChild(n), i.appendChild(o);
    const a = s.visible ?? !0, c = document.createElement("div"), l = e => {
        c.classList.remove("nlux-chatRoom-starting"), c.classList.remove("nlux-chatRoom-active"), c.classList.add(`nlux-chatRoom-${e}`)
    };
    c.classList.add("nlux-chatRoom-container"), l(is(s.initialConversationContent)), c.append(i), c.style.display = a ? "" : "none";
    const [h, d] = It(c, ":scope > .nlux-conversation-container").on("click", t("conversation-container-clicked")).get();
    return e(c), t("chat-room-ready")(), {
        elements: {
            composerContainer: o,
            conversationContainer: n,
            launchPadContainer: r
        }, actions: {
            updateChatRoomStatus: e => {
                l(e)
            }
        }, onDestroy: () => {
            d()
        }
    }
}), (({propName: e, newValue: t, dom: {elements: s}}) => {
}))], hs);
const ds = e => {
    const t = new Set;
    let s = !1, n = null, o = null;
    const r = () => {
        if (s) return;
        if (0 === t.size) return;
        s = !0;
        const r = t.values().next().value;
        t.delete(r), n = (({message: e}) => {
            const t = document.createElement("div");
            t.classList.add("nlux-comp-exceptionItem");
            const s = document.createElement("span");
            return s.classList.add("nlux-comp-exp_itm_msg"), s.append(document.createTextNode(e)), t.append(s), t
        })(r), e.append(n), o = setTimeout(i, 3e3)
    }, i = () => {
        n?.classList.add("nlux-comp-exceptionItem--hiding"), o = setTimeout((() => {
            s = !1, n?.remove(), n = null, r()
        }), 500)
    };
    return {
        displayException: e => {
            t.add({message: e}), s || r()
        }, destroy: () => {
            t.clear(), n?.remove(), o && clearTimeout(o)
        }
    }
};
var us = Object.defineProperty, ps = Object.getOwnPropertyDescriptor;
let ms = class extends g {
    constructor(e, t) {
        super(e, t)
    }

    destroy() {
        super.destroy()
    }

    showAlert(e, t) {
        this.executeDomAction("displayException", t)
    }
};
ms = ((e, t, s, n) => {
    for (var o, r = n > 1 ? void 0 : n ? ps(t, s) : t, i = e.length - 1; i >= 0; i--) (o = e[i]) && (r = (n ? o(t, s, r) : o(r)) || r);
    return n && r && us(t, s, r), r
})([f("exceptionsBox", (({props: e, appendToRoot: t}) => {
    const s = (() => {
        const e = document.createElement("div");
        return e.classList.add("nlux-comp-exceptionBox"), e
    })();
    t(s);
    let n = ds(s);
    return {
        elements: {root: s}, actions: {
            displayException: e => {
                n?.displayException(e)
            }
        }, onDestroy: () => {
            n?.destroy(), s.remove(), n = void 0
        }
    }
}), (() => {
}))], ms);
const gs = () => {
    const e = (() => {
        if ("undefined" == typeof window) return;
        const e = window;
        return "object" == typeof e.NLUX && "string" == typeof e.NLUX.version ? e.NLUX : (e.NLUX = d, d)
    })(), t = btoa("sectionsRegistered");
    e && !0 === e[t] || (Object.entries({
        chatRoom: hs,
        launchPad: _t,
        conversation: Ft,
        composer: Ot,
        conversationStarters: $t,
        chatSegment: Ut,
        chatItem: yt,
        exceptionsBox: ms
    }).forEach((([, e]) => {
        h.register(e)
    })), "object" == typeof e && (e[t] = !0))
};

class fs {
    constructor() {
        this.emit = (e, ...t) => {
            this.eventListeners.has(e) && this.eventListeners.get(e)?.forEach((e => {
                "function" == typeof e && e(...t)
            }))
        }, this.on = (e, t) => {
            this.eventListeners.has(e) || this.eventListeners.set(e, new Set), this.eventListeners.get(e)?.add(t)
        }, this.removeAllEventListeners = e => this.eventListeners.delete(e), this.removeAllEventListenersForAllEvent = () => this.eventListeners.clear(), this.removeEventListener = (e, t) => {
            this.eventListeners.has(e) && (this.eventListeners.get(e)?.delete(t), this.eventListeners.get(e)?.size || this.eventListeners.delete(e))
        }, this.updateEventListeners = e => {
            const t = Object.keys(e);
            for (const s of t) this.eventListeners.set(s, new Set([e[s]]))
        }, this.eventListeners = new Map
    }
}

class ys {
    constructor(e, t, s, n = null) {
        if (this.chatRoom = null, this.exceptionsBox = null, this.isDestroyed = !1, this.isMounted = !1, this.rootCompId = null, this.rootElement = null, this.theClassName = null, this.theComposerOptions = {}, this.theConversationOptions = {}, this.theDisplayOptions = {}, this.theInitialConversationContent = null, this.theMessageOptions = {}, this.thePersonasOptions = {}, !t) throw new r({
            source: this.constructor.name,
            message: "Root component ID is not a valid component name"
        });
        this.__context = e, this.rootElement = s, this.rootElementInitialClassName = s.className, this.rootCompId = t, this.chatRoom = null, this.theClassName = n?.className ?? null, this.theDisplayOptions = n?.displayOptions ?? {}, this.theConversationOptions = n?.conversationOptions ?? {}, this.theMessageOptions = n?.messageOptions ?? {}, this.theInitialConversationContent = n?.initialConversation ?? null, this.theComposerOptions = n?.composerOptions ?? {}, this.thePersonasOptions = n?.personaOptions ?? {}
    }

    get className() {
        return this.theClassName ?? void 0
    }

    get colorScheme() {
        return this.theDisplayOptions.colorScheme
    }

    get context() {
        return this.__context
    }

    get mounted() {
        return this.isMounted
    }

    get themeId() {
        return this.theDisplayOptions.themeId
    }

    destroy() {
        this.mounted && this.unmount(), this.chatRoom && this.chatRoom.destroy(), this.rootElement = null, this.rootElementInitialClassName = null, this.rootCompId = null, this.chatRoom = null, this.isMounted = !1, this.isDestroyed = !0, this.theDisplayOptions = {}, this.theConversationOptions = {}, this.theComposerOptions = {}
    }

    hide() {
        if (!this.mounted) throw new r({
            source: this.constructor.name,
            message: "Renderer is not mounted and cannot be hidden"
        });
        this.chatRoom?.hide()
    }

    mount() {
        if (this.isDestroyed) throw new r({
            source: this.constructor.name,
            message: "Renderer is destroyed and cannot be mounted"
        });
        if (!this.rootCompId || !this.rootElement) throw new r({
            source: this.constructor.name,
            message: "Root component or root class is not set"
        });
        let e = null, t = null;
        try {
            if ("chatRoom" !== this.rootCompId) throw new r({
                source: this.constructor.name,
                message: "Root component is not a chat room"
            });
            e = kt(hs).withContext(this.context).withProps({
                visible: !0,
                conversationLayout: Rt(this.theConversationOptions.layout),
                assistantPersona: this.thePersonasOptions?.assistant ?? void 0,
                userPersona: this.thePersonasOptions?.user ?? void 0,
                conversationStarters: this.theConversationOptions?.conversationStarters ?? void 0,
                showGreeting: this.theConversationOptions?.showWelcomeMessage,
                initialConversationContent: this.theInitialConversationContent ?? void 0,
                autoScroll: this.theConversationOptions?.autoScroll,
                syntaxHighlighter: this.context.syntaxHighlighter,
                htmlSanitizer: this.context.htmlSanitizer,
                markdownLinkTarget: this.theMessageOptions?.markdownLinkTarget,
                showCodeBlockCopyButton: this.theMessageOptions?.showCodeBlockCopyButton,
                streamingAnimationSpeed: this.theMessageOptions?.streamingAnimationSpeed,
                skipStreamingAnimation: this.theMessageOptions?.skipStreamingAnimation,
                composer: {
                    placeholder: this.theComposerOptions?.placeholder,
                    autoFocus: this.theComposerOptions?.autoFocus,
                    disableSubmitButton: this.theComposerOptions?.disableSubmitButton,
                    submitShortcut: this.theComposerOptions?.submitShortcut
                }
            }).create();
            const s = h.retrieve("exceptionsBox")?.model;
            if (s ? t = kt(ms).withContext(this.context).create() : i("Exception alert component is not registered! No exceptions will be shown."), !e) throw new r({
                source: this.constructor.name,
                message: "Root component failed to instantiate"
            });
            if (this.setRootElementClassNames(), this.setRoomElementDimensions(this.theDisplayOptions), t && t.render(this.rootElement), e.render(this.rootElement), e && !e.rendered) throw this.rootElement.className = this.rootElementInitialClassName || "", new r({
                source: this.constructor.name,
                message: "Root component did not render"
            });
            this.chatRoom = e, this.exceptionsBox = t ?? null, this.isMounted = !0
        } catch (e) {
            this.rootElement.className = this.rootElementInitialClassName || "", this.renderEx("failed-to-render-content", e?.toString() ?? "Unknown error")
        }
    }

    renderEx(e, t) {
        if (!this.mounted) throw i("Renderer is not mounted and cannot render exceptions"), new r({
            source: this.constructor.name,
            message: t
        });
        this.exceptionsBox?.showAlert(e, t)
    }

    show() {
        if (!this.mounted) throw new r({
            source: this.constructor.name,
            message: "Renderer is not mounted and cannot be shown"
        });
        this.chatRoom?.show()
    }

    unmount() {
        if (this.isDestroyed) i("Renderer is destroyed and cannot be unmounted"); else {
            if (!this.chatRoom || !this.rootElement) throw new r({
                source: this.constructor.name,
                message: "Root component or root element is not set"
            });
            this.context.emit("preDestroy", {
                aiChatProps: xt(this.context.aiChatProps),
                conversationHistory: this.chatRoom.getConversationContentForAdapter(this.context.aiChatProps.conversationOptions?.historyPayloadSize) ?? []
            }), this.exceptionsBox && this.exceptionsBox.destroy(), this.chatRoom.destroy(), this.rootElement && (p(this.rootElement), this.rootElement.className = this.rootElementInitialClassName || ""), this.chatRoom = null, this.exceptionsBox = null, this.isMounted = !1
        }
    }

    updateProps(e) {
        e.hasOwnProperty("adapter") && e.adapter && this.context.update({adapter: e.adapter});
        let t = !1;
        e.hasOwnProperty("className") && e.className !== this.className && (this.theClassName = e.className ?? null, t = !0);
        let s = !1, n = !1;
        if (e.hasOwnProperty("displayOptions")) {
            const t = {};
            if (e.displayOptions?.themeId !== this.theDisplayOptions.themeId && (t.themeId = e.displayOptions?.themeId, s = !0), e.displayOptions?.colorScheme !== this.theDisplayOptions.colorScheme && (t.colorScheme = e.displayOptions?.colorScheme, n = !0), e.displayOptions?.height !== this.theDisplayOptions.height && (t.height = e.displayOptions?.height), e.displayOptions?.width !== this.theDisplayOptions.width && (t.width = e.displayOptions?.width), Object.keys(t).length > 0) {
                const e = {...this.theDisplayOptions, ...t};
                this.theDisplayOptions = e, this.setRoomElementDimensions(e)
            }
        }
        if ((s || n || t) && this.setRootElementClassNames(), e.hasOwnProperty("conversationOptions")) {
            const t = {}, s = {};
            e.conversationOptions?.layout !== this.theConversationOptions.layout && (t.layout = e.conversationOptions?.layout, s.conversationLayout = Rt(e.conversationOptions?.layout)), e.conversationOptions?.autoScroll !== this.theConversationOptions.autoScroll && (t.autoScroll = e.conversationOptions?.autoScroll, s.autoScroll = e.conversationOptions?.autoScroll), e.conversationOptions?.showWelcomeMessage !== this.theConversationOptions.showWelcomeMessage && (t.showWelcomeMessage = e.conversationOptions?.showWelcomeMessage, s.showGreeting = e.conversationOptions?.showWelcomeMessage), e.conversationOptions?.conversationStarters !== this.theConversationOptions.conversationStarters && (t.conversationStarters = e.conversationOptions?.conversationStarters, s.conversationStarters = e.conversationOptions?.conversationStarters), Object.keys(t).length > 0 && (this.theConversationOptions = {...this.theConversationOptions, ...t}, this.chatRoom?.setProps(s))
        }
        if (e.hasOwnProperty("messageOptions") && (this.theMessageOptions = e.messageOptions ?? {}, this.chatRoom?.setProps({
            streamingAnimationSpeed: e.messageOptions?.streamingAnimationSpeed ?? void 0,
            markdownLinkTarget: e.messageOptions?.markdownLinkTarget ?? void 0,
            syntaxHighlighter: e.messageOptions?.syntaxHighlighter,
            htmlSanitizer: e.messageOptions?.htmlSanitizer
        }), this.context.update({
            syntaxHighlighter: e.messageOptions?.syntaxHighlighter,
            htmlSanitizer: e.messageOptions?.htmlSanitizer
        })), e.hasOwnProperty("composerOptions")) {
            const t = {};
            e.composerOptions?.hasOwnProperty("placeholder") && e.composerOptions.placeholder !== this.theComposerOptions.placeholder && (t.placeholder = e.composerOptions.placeholder), e.composerOptions?.hasOwnProperty("autoFocus") && e.composerOptions.autoFocus !== this.theComposerOptions.autoFocus && (t.autoFocus = e.composerOptions.autoFocus), e.composerOptions?.hasOwnProperty("disableSubmitButton") && e.composerOptions.disableSubmitButton !== this.theComposerOptions.disableSubmitButton && (t.disableSubmitButton = e.composerOptions.disableSubmitButton), e.composerOptions?.hasOwnProperty("submitShortcut") && e.composerOptions.submitShortcut !== this.theComposerOptions.submitShortcut && (t.submitShortcut = e.composerOptions.submitShortcut), Object.keys(t).length > 0 && (this.theComposerOptions = {...this.theComposerOptions, ...t}, this.chatRoom?.setProps({composer: this.theComposerOptions}))
        }
        if (e.hasOwnProperty("personaOptions")) {
            const t = {};
            e.personaOptions?.assistant !== this.thePersonasOptions?.assistant && (t.assistantPersona = e.personaOptions?.assistant ?? void 0), e.personaOptions?.user !== this.thePersonasOptions?.user && (t.userPersona = e.personaOptions?.user ?? void 0), this.thePersonasOptions = {...this.thePersonasOptions, ...e.personaOptions}, this.chatRoom?.setProps(t)
        }
    }

    setRoomElementDimensions(e) {
        this.rootElement && (e.hasOwnProperty("width") && (this.rootElement.style.width = "number" == typeof e.width ? `${e.width}px` : "string" == typeof e.width ? e.width : ""), e.hasOwnProperty("height") && (this.rootElement.style.height = "number" == typeof e.height ? `${e.height}px` : "string" == typeof e.height ? e.height : ""))
    }

    setRootElementClassNames() {
        if (!this.rootElement) return;
        const e = (e => {
            const t = ["nlux-AiChat-root"], s = `nlux-theme-${e.themeId || "nova"}`;
            return t.push(s), e.className && t.push(e.className), t
        })({themeId: this.themeId, className: this.className});
        this.rootElement.className = "", this.rootElement.classList.add(...e), this.rootElement.dataset.colorScheme = "auto" !== this.colorScheme && this.colorScheme ? this.colorScheme : void 0 !== typeof globalThis && globalThis.matchMedia && globalThis.matchMedia("(prefers-color-scheme: dark)")?.matches ? "dark" : "light"
    }
}

class ws {
    constructor(e, t) {
        this.eventManager = new fs, this.nluxInstanceId = m(), this.renderException = e => {
            if (!this.mounted || !this.renderer) return null;
            const t = rs[e];
            if (!t) return i(`Exception with id '${e}' is not defined`), null;
            this.renderer.renderEx(e, t)
        }, this.renderer = null, this.rootCompId = "chatRoom", this.rootElement = e, this.internalProps = t
    }

    get mounted() {
        return this.renderer?.mounted
    }

    hide() {
        this.mounted && this.renderer?.hide()
    }

    mount() {
        if (this.mounted) return;
        const e = ((e, t, s) => {
            const n = {
                ...e, update: e => {
                    Object.assign(n, e)
                }, emit: (e, ...t) => {
                    s(e, ...t)
                }, get aiChatProps() {
                    return t()
                }
            };
            return n
        })({
            instanceId: this.nluxInstanceId,
            exception: this.renderException,
            adapter: this.internalProps.adapter,
            syntaxHighlighter: this.internalProps.messageOptions.syntaxHighlighter,
            htmlSanitizer: this.internalProps.messageOptions.htmlSanitizer
        }, (() => this.getUpdatedAiChatPropsFromInternalProps(this.internalProps)), this.eventManager.emit);
        this.renderer = new ys(e, this.rootCompId, this.rootElement, this.internalProps), this.renderer.mount()
    }

    on(e, t) {
        this.eventManager.on(e, t)
    }

    removeAllEventListeners(e) {
        this.eventManager.removeAllEventListeners(e)
    }

    removeAllEventListenersForAllEvent() {
        this.eventManager.removeAllEventListenersForAllEvent()
    }

    removeEventListener(e, t) {
        this.eventManager.removeEventListener(e, t)
    }

    show() {
        this.mounted && this.renderer?.show()
    }

    unmount() {
        this.mounted && (this.renderer?.unmount(), this.renderer = null)
    }

    updateProps(e) {
        this.renderer?.updateProps(e), this.internalProps = {...this.internalProps, ...e}, e.events && (this.internalProps.events = e.events, this.eventManager.updateEventListeners(e.events))
    }

    getUpdatedAiChatPropsFromInternalProps(e) {
        const t = {...e};
        for (const e of Object.keys(t)) (void 0 === t[e] || null === t[e] || "object" == typeof t[e] && 0 === Object.keys(t[e]).length) && delete t[e];
        return t
    }
}

class ks {
    constructor() {
        this.theAdapter = null, this.theAdapterBuilder = null, this.theAdapterType = null, this.theClassName = null, this.theComposerOptions = null, this.theConversationOptions = null, this.theDisplayOptions = null, this.theInitialConversation = null, this.theMessageOptions = null, this.thePersonasOptions = null, this.aiChatStatus = "idle", this.controller = null, this.unregisteredEventListeners = new Map
    }

    get status() {
        return this.aiChatStatus
    }

    get isIdle() {
        return "idle" === this.status
    }

    hide() {
        if (!this.controller) throw new r({
            source: this.constructor.name,
            message: "Unable to hide. nlux is not mounted."
        });
        this.controller.hide()
    }

    mount(e) {
        if (!this.isIdle) throw new n({
            source: this.constructor.name,
            message: "Unable to create nlux instance. nlux is already or was previously mounted. You can only mount a nlux instance once, when the status is `idle`."
        });
        const t = this.theAdapter && "instance" === this.theAdapterType ? this.theAdapter : "builder" === this.theAdapterType && this.theAdapterBuilder ? this.theAdapterBuilder.create() : null;
        if (!t) throw new o({
            source: this.constructor.name,
            message: "Unable to create nlux instance. ChatAdapter is not properly set. You should call `withAdapter(adapter)` method before mounting nlux."
        });
        gs();
        const s = document.createElement("div");
        e.appendChild(s);
        const i = new ws(s, {
            adapter: t,
            className: this.theClassName ?? void 0,
            initialConversation: this.theInitialConversation ?? void 0,
            messageOptions: this.theMessageOptions ?? {},
            displayOptions: this.theDisplayOptions ?? {},
            conversationOptions: this.theConversationOptions ?? {},
            composerOptions: this.theComposerOptions ?? {},
            personaOptions: this.thePersonasOptions ?? {}
        });
        for (const [e, t] of this.unregisteredEventListeners.entries()) for (const s of t) i.on(e, s);
        if (i.mount(), !i.mounted) throw new r({
            source: this.constructor.name,
            message: "AiChat root component did not render."
        });
        this.aiChatStatus = "mounted", this.controller = i, this.unregisteredEventListeners.clear()
    }

    on(e, t) {
        if ("unmounted" === this.status) throw new n({
            source: this.constructor.name,
            message: "Unable to add event listener. nlux was previously unmounted."
        });
        return this.controller ? (this.controller.on(e, t), this) : (this.unregisteredEventListeners.has(e) || this.unregisteredEventListeners.set(e, new Set), this.unregisteredEventListeners.get(e)?.add(t), this)
    }

    removeAllEventListeners(e) {
        this.controller?.removeAllEventListeners(e), this.unregisteredEventListeners.get(e)?.clear()
    }

    removeEventListener(e, t) {
        this.controller?.removeEventListener(e, t), this.unregisteredEventListeners.get(e)?.delete(t)
    }

    show() {
        if (!this.controller) throw new r({
            source: this.constructor.name,
            message: "Unable to show. nlux is not mounted."
        });
        this.controller.show()
    }

    unmount() {
        if (this.controller) {
            if (this.controller.unmount(), this.controller.mounted) throw new r({
                source: this.constructor.name,
                message: "Unable to unmount. Root component did not unmount."
            });
            this.controller = null, this.unregisteredEventListeners.clear(), this.aiChatStatus = "unmounted"
        }
    }

    updateProps(e) {
        if (!this.controller) throw new r({
            source: this.constructor.name,
            message: "Unable to update props. nlux is not mounted."
        });
        if (e.hasOwnProperty("adapter") && (this.theAdapter = e.adapter ?? null), e.hasOwnProperty("events")) {
            this.clearEventListeners();
            for (const [t, s] of Object.entries(e.events ?? {})) this.on(t, s)
        }
        e.hasOwnProperty("className") && (this.theClassName = e.className ?? null), e.hasOwnProperty("displayOptions") && (this.theDisplayOptions = e.displayOptions ?? null), e.hasOwnProperty("composerOptions") && (this.theComposerOptions = e.composerOptions ?? null), e.hasOwnProperty("personaOptions") && (this.thePersonasOptions = e.personaOptions ?? null), e.hasOwnProperty("conversationOptions") && (this.theConversationOptions = e.conversationOptions ?? null), e.hasOwnProperty("messageOptions") && (this.theMessageOptions = e.messageOptions ?? null), this.controller.updateProps(e)
    }

    withAdapter(e) {
        if (!this.isIdle) throw new n({
            source: this.constructor.name,
            message: "Unable to set adapter. nlux is already or was previously mounted. You can only set the adapter once, when the status is `idle`."
        });
        if (this.theAdapterBuilder || this.theAdapter) throw new n({
            source: this.constructor.name,
            message: "Unable to change config. A adapter or adapter builder was already set."
        });
        if ("object" != typeof e || null === e) throw new n({
            source: this.constructor.name,
            message: "Unable to set adapter. Invalid adapter or adapter-builder type."
        });
        const t = e;
        if ("function" == typeof t.create) return this.theAdapterType = "builder", this.theAdapterBuilder = t, this;
        if ("function" == typeof t.batchText || "function" == typeof t.streamText) return this.theAdapterType = "instance", this.theAdapter = t, this;
        throw new n({
            source: this.constructor.name,
            message: "Unable to set adapter. Invalid adapter or adapter-builder implementation! When an `ChatAdapterBuilder` is provided, it must implement either `create()` method that returns an ChatAdapter instance. When an ChatAdapter instance is provided, must implement `batchText()` and/or `streamText()` methods. None of the above were found."
        })
    }

    withClassName(e) {
        if (!this.isIdle) throw new n({
            source: this.constructor.name,
            message: "Unable to set class name. nlux is already or was previously mounted. You can only set the class name once, when the status is `idle`."
        });
        if (this.theClassName) throw new n({
            source: this.constructor.name,
            message: "Unable to change config. A class name was already set."
        });
        return this.theClassName = e, this
    }

    withComposerOptions(e) {
        if (!this.isIdle) throw new n({
            source: this.constructor.name,
            message: "Unable to set composer options. nlux is already or was previously mounted. You can only set the composer options once, when the status is `idle`."
        });
        if (this.theComposerOptions) throw new n({
            source: this.constructor.name,
            message: "Unable to change config. Composer options were already set."
        });
        return this.theComposerOptions = {...e}, this
    }

    withConversationOptions(e) {
        if (!this.isIdle) throw new n({
            source: this.constructor.name,
            message: "Unable to set conversation options. nlux is already or was previously mounted. You can only set the conversation options once, when the status is `idle`."
        });
        if (this.theConversationOptions) throw new n({
            source: this.constructor.name,
            message: "Unable to change config. Conversation options were already set."
        });
        return this.theConversationOptions = {...e}, this
    }

    withDisplayOptions(e) {
        if (!this.isIdle) throw new n({
            source: this.constructor.name,
            message: "Unable to set display options. nlux is already or was previously mounted. You can only set the display options once, when the status is `idle`."
        });
        if (this.theDisplayOptions) throw new n({
            source: this.constructor.name,
            message: "Unable to change config. Display options were already set."
        });
        return this.theDisplayOptions = {...e}, this
    }

    withInitialConversation(e) {
        if (!this.isIdle) throw new n({
            source: this.constructor.name,
            message: "Unable to set initial conversation. nlux is already or was previously mounted. You can only set the initial conversation once, when the status is `idle`."
        });
        if (this.theInitialConversation) throw new n({
            source: this.constructor.name,
            message: "Unable to change config. Conversation history was already set."
        });
        return this.theInitialConversation = [...e], this
    }

    withMessageOptions(e) {
        if (!this.isIdle) throw new n({
            source: this.constructor.name,
            message: "Unable to set message options. nlux is already or was previously mounted. You can only set the message options once, when the status is `idle`."
        });
        if (this.theMessageOptions) throw new n({
            source: this.constructor.name,
            message: "Unable to change config. Message options were already set."
        });
        return this.theMessageOptions = {...e}, this
    }

    withPersonaOptions(e) {
        if (!this.isIdle) throw new n({
            source: this.constructor.name,
            message: "Unable to set persona options. nlux is already or was previously mounted. You can only set the persona options once, when the status is `idle`."
        });
        if (this.thePersonasOptions) throw new n({
            source: this.constructor.name,
            message: "Unable to change config. Personas were already set."
        });
        return this.thePersonasOptions = {...e}, this
    }

    clearEventListeners() {
        this.controller?.removeAllEventListenersForAllEvent(), this.unregisteredEventListeners.clear()
    }
}

class xs {
    constructor(e) {
        this.actionToPerformWhenIdle = "none", this.itemIds = new Set, this.status = "idle", this.theContextId = null, this.updateQueueByItemId = new Map, this.dataAdapter = e
    }

    get contextId() {
        return this.theContextId
    }

    async createContext(e) {
        if ("destroyed" === this.status) return {success: !1, error: "The context has been destroyed"};
        const t = e ?? null;
        if (this.itemIds.clear(), null !== t) {
            Object.keys(t).forEach((e => {
                this.itemIds.add(e)
            }))
        }
        this.actionToPerformWhenIdle = "none";
        try {
            const e = await this.dataAdapter.create(t ?? {});
            return e.success ? (this.theContextId = e.contextId, {success: !0, contextId: e.contextId}) : {
                success: !1,
                error: "Failed to set the context"
            }
        } catch (e) {
            return {success: !1, error: `${e}`}
        }
    }

    async destroy() {
        return "destroyed" === this.status ? (i("Context.DataSyncService.destroy() called on a state that has already been destroyed"), {success: !0}) : ("updating" !== this.status || this.contextId || i("Context.DataSyncService.destroy() called with no contextId!"), this.contextId && (this.status = "updating", await this.dataAdapter.discard(this.contextId)), this.status = "destroyed", this.theContextId = null, this.updateQueueByItemId.clear(), this.actionToPerformWhenIdle = "none", {success: !0})
    }

    async flush() {
        if (!this.contextId) throw new Error("Context not initialized");
        if ("updating" === this.status) return void (this.actionToPerformWhenIdle = "flush");
        this.status = "updating";
        const e = this.updateQueueByItemId.keys(), t = [], s = [];
        for (const n of e) {
            const e = this.updateQueueByItemId.get(n);
            e && ("delete" !== e.operation ? ["set", "update"].includes(e.operation) && t.push(n) : s.push(n))
        }
        const n = t.reduce(((e, t) => {
            const s = this.updateQueueByItemId.get(t);
            return s ? ("set" === s.operation && (e[t] = {
                value: s.data,
                description: s.description
            }), "update" !== s.operation || void 0 === s.data && void 0 === s.description || (e[t] = {
                value: s.data,
                description: s.description
            }), e) : e
        }), {});
        if (Object.keys(n).length > 0) {
            Object.keys(n).forEach((e => {
                this.updateQueueByItemId.delete(e)
            }));
            try {
                await this.dataAdapter.updateItems(this.contextId, n)
            } catch (e) {
                i(`Failed to update context data: ${e}`), Object.keys(n).forEach((e => {
                    const t = n[e];
                    t && this.updateQueueByItemId.set(e, {
                        operation: "update",
                        data: t.value,
                        description: t.description
                    })
                }))
            }
        }
        if (s.length > 0) {
            s.forEach((e => {
                this.itemIds.delete(e), this.updateQueueByItemId.delete(e)
            }));
            try {
                await this.dataAdapter.removeItems(this.contextId, s)
            } catch (e) {
                i(`Failed to delete context data: ${e}`), s.forEach((e => {
                    this.itemIds.add(e), this.updateQueueByItemId.set(e, {operation: "delete"})
                }))
            }
        }
        await this.backToIdle()
    }

    hasActiveItemWithId(e) {
        return this.itemIds.has(e) && (!this.updateQueueByItemId.has(e) || "delete" !== this.updateQueueByItemId.get(e)?.operation)
    }

    hasItemWithId(e) {
        return this.itemIds.has(e)
    }

    removeItem(e) {
        if ("destroyed" === this.status) throw new Error("The context has been destroyed");
        if (!this.contextId) throw new Error("Context not initialized");
        if (!this.itemIds.has(e)) throw new Error("Item not found");
        this.updateQueueByItemId.set(e, {operation: "delete"})
    }

    async resetContextData(e) {
        const t = this.contextId;
        if (this.itemIds.clear(), this.updateQueueByItemId.clear(), "updating" !== this.status) {
            if (!t) return i("resetContextData() called with no contextId!"), void await this.backToIdle();
            try {
                this.status = "updating", await this.dataAdapter.resetItems(t, e)
            } catch (e) {
                i(`Failed to reset context data: ${e}`)
            }
            if (this.updateQueueByItemId.clear(), e) {
                this.itemIds.clear();
                Object.keys(e).forEach((e => {
                    this.itemIds.add(e)
                }))
            } else this.itemIds.clear();
            await this.backToIdle()
        } else this.actionToPerformWhenIdle = "reset"
    }

    setItemData(e, t, s) {
        if ("destroyed" === this.status) throw new Error("The context has been destroyed");
        this.updateQueueByItemId.set(e, {operation: "set", description: t, data: s}), this.itemIds.add(e)
    }

    updateItemData(e, t, s) {
        if ("destroyed" === this.status) throw new Error("The context has been destroyed");
        if (void 0 === s && void 0 === t) return;
        const n = this.updateQueueByItemId.get(e);
        if ("delete" === n?.operation) throw new Error("Item has been deleted");
        const o = s ?? n?.data ?? void 0, r = t ?? n?.description ?? void 0;
        this.updateQueueByItemId.set(e, {operation: "update", data: o, description: r})
    }

    async backToIdle() {
        this.status = "idle";
        const e = this.actionToPerformWhenIdle;
        this.actionToPerformWhenIdle = "none", "flush" !== e ? "reset" !== e || await this.resetContextData() : await this.flush()
    }
}

class vs {
    constructor(e, t) {
        this.actionToPerformWhenIdle = "none", this.status = "idle", this.taskCallbacks = new Map, this.tasks = new Set, this.updateQueueByTaskId = new Map, this.contextId = e, this.adapter = t
    }

    canRunTask(e) {
        return this.taskCallbacks.has(e)
    }

    async destroy() {
        return "destroyed" === this.status ? (i("Context.TasksService.destroy() called on a state that has already been destroyed"), {success: !0}) : (this.status = "updating", await this.unregisterAllTasks(), this.status = "destroyed", this.updateQueueByTaskId.clear(), this.tasks.clear(), {success: !0})
    }

    async flush() {
        if ("updating" === this.status) return void (this.actionToPerformWhenIdle = "flush");
        const e = this.updateQueueByTaskId.keys(), t = [], s = [], n = [];
        for (const o of e) {
            const e = this.updateQueueByTaskId.get(o);
            e && ("delete" !== e.operation ? "set" !== e.operation ? "update" === e.operation && s.push(o) : t.push(o) : n.push(o))
        }
        if (0 === t.length && 0 === s.length && 0 === n.length) return;
        this.status = "updating";
        const o = {...this.buildUpdateObject(t), ...this.buildUpdateObject(s)};
        if (Object.keys(o).length > 0) try {
            const e = await this.adapter.updateTasks(this.contextId, o);
            if (e.success) for (const e of Object.keys(o)) {
                const t = this.updateQueueByTaskId.get(e);
                t && "set" === t.operation && (this.taskCallbacks.set(e, t.callback), this.updateQueueByTaskId.delete(e))
            } else i(`Context.TasksService.flush() failed to register tasks for context ID ${this.contextId}\nError: ${e.error}`)
        } catch (e) {
            i(`Context.TasksService.flush() failed to register tasks for context ID ${this.contextId}\nError: ${e}`)
        }
        if (n.length > 0) try {
            const e = await this.adapter.removeTasks(this.contextId, n);
            if (e.success) for (const e of n) this.taskCallbacks.delete(e), this.updateQueueByTaskId.delete(e); else i(`Context.TasksService.flush() failed to unregister tasks for context ID ${this.contextId}\nError: ${e.error}`)
        } catch (e) {
            i(`Context.TasksService.flush() failed to unregister tasks for context ID ${this.contextId}\nError: ${e}`)
        }
        await this.backToIdle()
    }

    hasTask(e) {
        return this.tasks.has(e)
    }

    async registerTask(e, t, s, n) {
        if ("destroyed" === this.status) throw new Error("Context has been destroyed");
        if (this.tasks.has(e)) throw new Error(`A task with ID '${e}' already exists. Task IDs must be unique.`);
        this.updateQueueByTaskId.set(e, {
            operation: "set",
            description: t,
            paramDescriptions: n || [],
            callback: s
        }), this.tasks.add(e)
    }

    async resetContextData() {
        const e = this.contextId;
        if (this.tasks.clear(), this.taskCallbacks.clear(), this.updateQueueByTaskId.clear(), "updating" !== this.status) {
            if (!e) return i("resetContextData() called with no contextId!"), void await this.backToIdle();
            try {
                this.status = "updating", await this.unregisterAllTasks()
            } catch (e) {
                i(`Failed to reset context data: ${e}`)
            }
            await this.backToIdle()
        } else this.actionToPerformWhenIdle = "reset"
    }

    async runTask(e, t) {
        if ("destroyed" === this.status) throw new Error("Context has been destroyed");
        if (!this.tasks.has(e)) return {success: !1, error: `Task with ID ${e} not found`};
        const s = this.taskCallbacks.get(e);
        if (!s) return {
            success: !1,
            error: `The task with ID '${e}' has no callback. This is potential due to failed registration.`
        };
        try {
            return {success: !0, result: s(...t ?? [])}
        } catch (e) {
            return {success: !1, error: `${e}`}
        }
    }

    async unregisterAllTasks() {
        if (0 === this.tasks.size) return {success: !0};
        const e = await this.adapter.resetTasks(this.contextId);
        return e.success && (this.tasks.clear(), this.taskCallbacks.clear(), this.updateQueueByTaskId.clear()), e
    }

    async unregisterTask(e) {
        if ("destroyed" === this.status) throw new Error("Context has been destroyed");
        return this.tasks.has(e) ? (this.tasks.delete(e), this.taskCallbacks.delete(e), this.updateQueueByTaskId.set(e, {operation: "delete"}), {success: !0}) : {success: !0}
    }

    async updateTaskCallback(e, t) {
        if ("destroyed" === this.status) throw new Error("The context has been destroyed");
        if (!this.tasks.has(e)) throw new Error(`Task with ID ${e} not found`);
        this.taskCallbacks.set(e, t)
    }

    async updateTaskDescription(e, t) {
        if ("destroyed" === this.status) throw new Error("The context has been destroyed");
        if (!this.tasks.has(e)) throw new Error(`Task with ID ${e} not found`);
        const s = this.updateQueueByTaskId.get(e);
        if (s) if ("update" !== s.operation) {
            const s = {operation: "update", description: t};
            this.updateQueueByTaskId.set(e, s)
        } else s.description = t; else this.updateQueueByTaskId.set(e, {operation: "update", description: t})
    }

    async updateTaskParamDescriptions(e, t) {
        if ("destroyed" === this.status) throw new Error("The context has been destroyed");
        if (!this.tasks.has(e)) throw new Error(`Task with ID ${e} not found`);
        const s = this.updateQueueByTaskId.get(e);
        if (s) if ("update" !== s.operation) {
            const s = {operation: "update", paramDescriptions: t};
            this.updateQueueByTaskId.set(e, s)
        } else s.paramDescriptions = t; else this.updateQueueByTaskId.set(e, {
            operation: "update",
            paramDescriptions: t
        })
    }

    async backToIdle() {
        this.status = "idle";
        const e = this.actionToPerformWhenIdle;
        this.actionToPerformWhenIdle = "none", "flush" !== e ? "reset" !== e || await this.unregisterAllTasks() : await this.flush()
    }

    buildUpdateObject(e) {
        return e.reduce(((e, t) => {
            const s = this.updateQueueByTaskId.get(t);
            if (!s) return e;
            if ("set" === s.operation && (e[t] = {
                description: s.description,
                paramDescriptions: s.paramDescriptions
            }), "update" === s.operation && (void 0 !== s.description || void 0 !== s.paramDescriptions)) {
                const n = {};
                void 0 !== s.description && (n.description = s.description), void 0 !== s.paramDescriptions && (n.paramDescriptions = s.paramDescriptions), e[t] = n
            }
            return e
        }), {})
    }
}

class bs {
    constructor() {
        this.destroy = async () => ("destroyed" === this.theStatus || (this.theStatus = "destroyed", await Promise.all([this.theDataSyncService?.destroy(), this.theTasksService?.destroy()]), this.theDataSyncService = null, this.theTasksService = null, this.theDataAdapter = null, this.theTasksAdapter = null), {success: !0}), this.flush = async () => {
            try {
                await (this.theDataSyncService?.flush())
            } catch (e) {
                return {success: !1, error: "Failed to flush context data"}
            }
            try {
                await (this.theTasksService?.flush())
            } catch (e) {
                return {success: !1, error: "Failed to flush context tasks"}
            }
            return {success: !0}
        }, this.initialize = async e => {
            if ("initializing" === this.theStatus) return i(`${this.constructor.name}.initialize() called while context is still initializing! You cannot initialize twice at the same time. Use ${this.constructor.name}.status or await ${this.constructor.name}.initialize() to make sure that the context is not initializing before calling this method.`), {
                success: !1,
                error: "Context is still initializing! Use AiContext.status to check the context status before calling this method."
            };
            if ("syncing" === this.theStatus) return i(`${this.constructor.name}.initialize() called on an already initialized context! Use ${this.constructor.name}.status to check the context status before calling this method. `), {
                success: !1,
                error: "Context already initialized! Use AiContext.status to check the context status before calling this method."
            };
            if ("destroyed" === this.theStatus) return i(`${this.constructor.name}.initialize() called on destroyed context! Use ${this.constructor.name}.status to check the context status before calling this method. When the context is destroyed, it cannot be used anymore and you should create a new context.`), {
                success: !1,
                error: "Context has been destroyed"
            };
            if ("error" === this.theStatus) return i(`${this.constructor.name}.initialize() called on a context in error state! Use ${this.constructor.name}.status to check the context status before calling this method. When the context is in error state, it cannot be used anymore and you should create a new context.`), {
                success: !1,
                error: "Context is in error state"
            };
            if (!this.theDataAdapter) return i(`Adapter not set! You must set the adapter before initializing the context. Use ${this.constructor.name}.withAdapter() to set the adapter before calling this method.`), {
                success: !1,
                error: "Adapter not set"
            };
            this.theStatus = "initializing", this.theDataSyncService = new xs(this.theDataAdapter);
            try {
                const t = await this.theDataSyncService.createContext(e);
                return "destroyed" === this.status ? (t.success && await (this.theDataSyncService?.resetContextData()), {
                    success: !1,
                    error: "Context has been destroyed"
                }) : t.success ? this.contextId ? (this.theStatus = "syncing", this.theTasksAdapter ? this.theTasksService = new vs(this.contextId, this.theTasksAdapter) : i("Initializing nlux AiContext without tasks adapter. The context will not handle registering and executing tasks by AI. If you want to use tasks triggered by AI, you should provide an adapter that implements ContextAdapter interface [type ContextAdapter = ContextDataAdapter & ContextTasksAdapter]"), {
                    success: !0,
                    contextId: t.contextId
                }) : (this.theStatus = "error", {
                    success: !1,
                    error: "Failed to obtain context ID"
                }) : (this.theStatus = "error", {success: !1, error: "Failed to initialize context"})
            } catch (e) {
                return this.theStatus = "error", {success: !1, error: `${e}`}
            }
        }, this.observeState = (e, t, s) => {
            if ("idle" !== this.theStatus) if ("initializing" !== this.theStatus) {
                if ("destroyed" !== this.theStatus) return this.theDataSyncService?.setItemData(e, t, s ?? null), {
                    setData: t => {
                        this.theDataSyncService?.updateItemData(e, void 0, t)
                    }, setDescription: t => {
                        this.theDataSyncService?.updateItemData(e, t, void 0)
                    }, discard: () => {
                        this.theDataSyncService?.removeItem(e)
                    }
                };
                i(`${this.constructor.name}.observeState() called on destroyed context! Use ${this.constructor.name}.status to check the context status before calling this method. When the context is destroyed, it cannot be used anymore and you should create a new context.`)
            } else i(`${this.constructor.name}.observeState() called while context is still initializing! You cannot observe state items while the context is initializing. Use ${this.constructor.name}.status or await ${this.constructor.name}.initialize() to make sure that the context is not initializing before calling this method.`); else i(`${this.constructor.name}.observeState() called on idle context! Use ${this.constructor.name}.status to check the context status before calling this method. Use ${this.constructor.name}.initialize() to initialize the context when it is not initialized.`)
        }, this.registerTask = (e, t, s, n) => {
            if ("idle" === this.theStatus) return void i(`${this.constructor.name}.registerTask() called on idle context! Use ${this.constructor.name}.status to check the context status before calling this method. Use ${this.constructor.name}.initialize() to initialize the context when it is not initialized.`);
            if (!this.theTasksService) return void i(`${this.constructor.name}.registerTask() called on a context that has does not have tasks service! You should use an adapter that implements ContextTasksAdapter interface in order to register tasks. Use ${this.constructor.name}.withAdapter() to set the right adapter before calling this method.`);
            if ("destroyed" === this.theStatus) return void i(`${this.constructor.name}.registerTask() called on destroyed context! Use ${this.constructor.name}.status to check the context status before calling this method. When the context is destroyed, it cannot be used anymore and you should create a new context.`);
            let o = "updating";
            if (!this.theTasksService.hasTask(e)) return this.theTasksService.registerTask(e, t, s, n).then((() => {
                "updating" === o && (o = "set")
            })).catch((() => {
                i(`${this.constructor.name}.registerTask() failed to register task '${e}'!\nThe task will be marked as deleted and will not be updated anymore.`), "updating" === o && (o = "deleted", this.unregisterTask(e))
            })), {
                discard: () => {
                    o = "deleted", this.unregisterTask(e)
                }, setDescription: t => {
                    if ("deleted" === o) throw new Error("Task has been deleted");
                    o = "updating", this.theTasksService?.updateTaskDescription(e, t).then((() => {
                        "updating" === o && (o = "set")
                    })).catch((() => {
                        "updating" === o && (o = "set")
                    }))
                }, setCallback: t => {
                    if ("deleted" === o) throw new Error("Task has been deleted");
                    o = "updating", this.theTasksService?.updateTaskCallback(e, t).then((() => {
                        "updating" === o && (o = "set")
                    })).catch((() => {
                        "updating" === o && (o = "set")
                    }))
                }, setParamDescriptions: t => {
                    if ("deleted" === o) throw new Error("Task has been deleted");
                    o = "updating", this.theTasksService?.updateTaskParamDescriptions(e, t).then((() => {
                        "updating" === o && (o = "set")
                    })).catch((() => {
                        "updating" === o && (o = "set")
                    }))
                }
            };
            console.warn(`${this.constructor.name}.registerTask() called with existing taskId: ${e}! It's only possible to register a task once. Use ${this.constructor.name}.hasTask() to check if the task already exists. Use ${this.constructor.name}.registerTask() with a different taskId if you want to register a different task.`)
        }, this.reset = async e => {
            if (!this.theDataSyncService) return i(`${this.constructor.name}.reset() called on a state that has not been initialized! Use ${this.constructor.name}.initialize() to initialize the context before attempting any reset.`), {
                success: !1,
                error: "Context has not been initialized"
            };
            try {
                return await (this.theDataSyncService?.resetContextData(e)), await (this.theTasksService?.resetContextData()), this.theStatus = "syncing", {success: !0}
            } catch (e) {
                return this.theStatus = "error", {success: !1, error: `${e}`}
            }
        }, this.runTask = async (e, t) => this.theTasksService ? this.theTasksService.runTask(e, t) : (i(`${this.constructor.name}.runTask() called on a state that has not been initialized! Use ${this.constructor.name}.initialize() to initialize the context before attempting any task execution.`), Promise.resolve({
            success: !1,
            error: "Context has not been initialized with tasks service. An adapter that implements ContextTasksAdapter interface should be provided to the context, and the context should be initialized before running any tasks."
        })), this.withAdapter = e => {
            if (this.theDataAdapter) throw new Error("Adapter already set");
            const t = "function" == typeof e?.build;
            this.theDataAdapter = t ? e.build() : e;
            const s = (e => {
                const t = e;
                return !(!t || "function" != typeof t.resetTasks || "function" != typeof t.updateTasks || "function" != typeof t.removeTasks) && t
            })(this.theDataAdapter);
            return s && (this.theTasksAdapter = s), this
        }, this.withDataSyncOptions = e => {
            if (this.theDataSyncOptions) throw new Error("Data sync options already set");
            return this.theDataSyncOptions = {...e}, this
        }, this.theDataAdapter = null, this.theDataSyncOptions = null, this.theDataSyncService = null, this.theStatus = "idle", this.theTasksAdapter = null, this.theTasksService = null, this.unregisterTask = e => this.theTasksService ? this.theTasksService.unregisterTask(e) : (i(`${this.constructor.name}.unregisterTask() called on a state that has not been initialized! Use ${this.constructor.name}.initialize() to initialize the context before attempting any task unregister.`), Promise.resolve({
            success: !1,
            error: "Context has not been initialized"
        }))
    }

    get contextId() {
        return this.theDataSyncService?.contextId ?? null
    }

    get status() {
        return this.theStatus
    }

    hasItem(e) {
        return this.theDataSyncService?.hasItemWithId(e) ?? !1
    }

    hasRunnableTask(e) {
        return this.theTasksService?.canRunTask(e) ?? !1
    }

    hasTask(e) {
        return this.theTasksService?.hasTask(e) ?? !1
    }
}

window.AiChat = ks, window.Observable = class {
    constructor({replay: e} = {}) {
        this.buffer = [], this.errorReceived = null, this.isCompleted = !1, this.isReplayObservable = !1, this.subscribers = new Set, this.isReplayObservable = e ?? !1
    }

    complete() {
        this.subscribers.forEach((e => e.complete?.())), this.isCompleted = !0
    }

    error(e) {
        this.subscribers.forEach((t => t.error?.(e))), this.isReplayObservable && (this.errorReceived = e)
    }

    next(e) {
        this.subscribers.forEach((t => t.next(e))), this.isReplayObservable && this.buffer.push(e)
    }

    reset() {
        this.subscribers.clear(), this.buffer = []
    }

    subscribe(e) {
        return this.subscribers.add(e), this.isReplayObservable && this.sendBufferToObserver(e), {unsubscribe: () => this.unsubscribe(e)}
    }

    unsubscribe(e) {
        this.subscribers.delete(e)
    }

    sendBufferToObserver(e) {
        this.buffer.forEach((t => e.next(t))), this.errorReceived ? e.error?.(this.errorReceived) : this.isCompleted && e.complete?.()
    }
}, window.createAiChat = () => new ks, window.createAiContext = () => new bs, window.predefinedContextSize = {
    "1k": 1e3,
    "10k": 1e4,
    "100k": 1e5,
    "1mb": 1e6,
    "10mb": 1e7
};
