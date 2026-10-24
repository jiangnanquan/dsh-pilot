/**
 * Browser engine for dsh-pilot: one shared browser (or persistent profile
 * context), a tab registry with console/failure capture, navigation staleness
 * tracking, and — when an origin policy is supplied — a network-layer origin
 * fence that blocks disallowed main-frame navigations however they start
 * (goto, redirects, link clicks, back/forward, meta refresh).
 *
 * Ported from dsh-preview's manager (same lifecycle discipline: lazy launch,
 * effect-owned disposal, seq-numbered console buffers) and extended with tab
 * limits, persistent profiles, popup containment, and the fence.
 * @module dsh-pilot/engine
 */

import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import type { Browser, BrowserContext, Frame, Page } from 'playwright-core'
import { chromium } from 'playwright-core'
import { withCancellation, type CancellationScope } from './cancellation.js'

export { raceAbort, throwIfAborted, withCancellation } from './cancellation.js'

/** Longest console message text retained; the tail is dropped, not the entry. */
const MAX_CONSOLE_TEXT = 2000

/** One captured console message. */
export interface ConsoleEntry {
  /** Monotonic capture id; survives buffer eviction, so diffs stay correct. */
  seq: number
  /** Console level as reported by the page (`log`, `warning`, `error`, ...). */
  level: string
  /** Message text, capped at {@link MAX_CONSOLE_TEXT} characters. */
  text: string
}

/** One failed network request. */
export interface RequestFailure {
  /** Request URL. */
  url: string
  /** Failure reason reported by the browser (e.g. `net::ERR_CONNECTION_REFUSED`). */
  reason: string
}

/** One completed download attempt. */
export interface DownloadEntry {
  /** Capture seq, so a tool reports only downloads since its own start. */
  seq: number
  /** Saved path on success. */
  path?: string
  /** Failure reason on error. */
  error?: string
}

/** A tracked tab with its capture buffers and snapshot state. */
export interface TrackedPage {
  /** Stable id handed to the model (`tab-1`, `tab-2`, ...). */
  id: string
  /** The playwright page. */
  page: Page
  /** Console messages captured since open, newest last, capped by the manager. */
  console: ConsoleEntry[]
  /** Failed requests captured since open, capped by the manager. */
  failures: RequestFailure[]
  /** Seq of the most recently captured console entry; 0 before the first. */
  lastSeq: number
  /** Downloads this tab produced, capped like the console buffer. */
  downloads: DownloadEntry[]
  /** True once a ref-bearing snapshot was taken for this tab. */
  snapshotTaken: boolean
  /** True once the main frame navigated after the latest snapshot. */
  refsStale: boolean
  /**
   * Exempts this tab from the origin fence for the rest of its life. Set only
   * for a navigation authorized under a session that opted out of prompts
   * (approval policy `never`), so the exemption dies with the tab instead of
   * becoming a cross-session grant.
   */
  unfenced: boolean
}

/** Launch/viewport options fixed per manager by plugin config. */
export interface EngineOptions {
  /** Preferred browser channels in order; each is tried until one launches. */
  channels: string[]
  /** Run without a visible window. */
  headless: boolean
  /** Default viewport width in px. */
  viewportWidth: number
  /** Default viewport height in px. */
  viewportHeight: number
  /** Maximum console messages retained per tab. */
  maxConsoleMessages: number
  /** Maximum simultaneously open tabs. */
  maxTabs: number
  /** Persistent profile directory; empty string = fresh isolated context. */
  profileDir: string
  /** Absolute directory downloads are saved into. */
  downloadDir: string
  /**
   * Origin fence: http(s) main-frame navigations whose URL fails this
   * predicate are aborted at the network layer, whatever initiated them.
   * Omit to run without a fence (deployment allows every origin).
   */
  originAllowed?: (url: URL) => boolean
}

/**
 * Owns the browser process and the tab registry. Dispose closes everything;
 * the owning plugin registers that disposal as a Cordis effect so unload/HMR
 * never leaks a browser process.
 */
export class Engine {
  private browser: Browser | undefined
  private persistent: BrowserContext | undefined
  private contextPromise: Promise<BrowserContext> | undefined
  private pages = new Map<string, TrackedPage>()
  private tracked = new WeakMap<Page, TrackedPage>()
  private counter = 0
  private downloadCounter = 0
  private lastId: string | undefined
  private disposed = false

  constructor(private options: EngineOptions) {}

  /**
   * Open a new tab (without navigating). Enforces the tab limit.
   *
   * A cold first call launches the browser, which takes seconds; passing the
   * caller's signal is what lets a cancellation during that window take effect
   * instead of being noticed only once the page already exists.
   * @param signal - cooperative cancellation from the tool execution.
   * @returns the tracked tab, already registered as the current tab.
   */
  async newTab(signal?: AbortSignal): Promise<TrackedPage> {
    // Read through a call, not a captured value: `aborted` flips while these
    // awaits are pending, which a narrowed property read would not observe.
    const cancelled = (): boolean => signal !== undefined && signal.aborted
    this.throwIfDisposed()
    if (cancelled()) throw new Error('cancelled before the tab was opened')
    if (this.pages.size >= this.options.maxTabs) {
      throw new Error(`tab limit reached (${this.options.maxTabs}); close a tab with pilot_close first`)
    }
    const context = await this.ensureContext()
    this.throwIfDisposed()
    if (cancelled()) throw new Error('cancelled while the browser was starting')
    const page = await context.newPage()
    if (this.disposed || cancelled()) {
      await page.close().catch(() => { /* dispose or abort raced the open; close is best-effort */ })
      throw new Error(this.disposed ? 'browser engine is disposed (plugin unloading)' : 'cancelled before the tab was opened')
    }
    const id = `tab-${++this.counter}`
    const tracked: TrackedPage = { id, page, console: [], failures: [], lastSeq: 0, downloads: [], snapshotTaken: false, refsStale: false, unfenced: false }
    this.tracked.set(page, tracked)
    page.on('console', (msg) => this.capture(tracked, msg.type(), msg.text()))
    page.on('pageerror', (err) => this.capture(tracked, 'error', String(err)))
    page.on('requestfailed', (req) => {
      if (tracked.failures.length >= this.options.maxConsoleMessages) tracked.failures.shift()
      tracked.failures.push({ url: req.url(), reason: req.failure()?.errorText ?? 'unknown' })
    })
    page.on('framenavigated', (frame) => {
      if (frame === page.mainFrame()) tracked.refsStale = true
    })
    page.on('download', (download) => {
      const seq = ++tracked.lastSeq
      void (async () => {
        await mkdir(this.options.downloadDir, { recursive: true })
        const path = join(this.options.downloadDir, `${Date.now()}-${++this.downloadCounter}-${download.suggestedFilename()}`)
        await download.saveAs(path)
        this.record(tracked, { seq, path })
      })().catch((err: unknown) => {
        this.record(tracked, { seq, error: err instanceof Error ? err.message : String(err) })
      })
    })
    this.pages.set(id, tracked)
    this.lastId = id
    return tracked
  }

  /**
   * Resolve a tab by id, defaulting to the most recently used one, and make
   * it current.
   * @param pageId - explicit id, or undefined for the current tab.
   * @returns the tracked tab.
   */
  get(pageId?: string): TrackedPage {
    const id = pageId ?? this.lastId
    if (!id) throw new Error('no tab is open; call pilot_navigate first')
    const tracked = this.pages.get(id)
    if (!tracked) {
      throw new Error(`unknown pageId ${JSON.stringify(id)}; open tabs: ${[...this.pages.keys()].join(', ') || 'none'}`)
    }
    this.lastId = id
    return tracked
  }

  /** List open tabs. */
  list(): { id: string; url: string; current: boolean }[] {
    return [...this.pages.values()].map(tracked => ({
      id: tracked.id,
      url: tracked.page.url(),
      current: tracked.id === this.lastId,
    }))
  }

  /**
   * Close one tab and forget it. Closing a non-current tab leaves the current
   * tab selection untouched.
   * @param pageId - explicit id, or undefined for the current tab.
   * @returns the id that was closed.
   */
  async close(pageId?: string): Promise<string> {
    const prevLast = this.lastId
    const id = pageId ?? prevLast
    if (!id) throw new Error('no tab is open; call pilot_navigate first')
    const tracked = this.pages.get(id)
    if (!tracked) {
      throw new Error(`unknown pageId ${JSON.stringify(id)}; open tabs: ${[...this.pages.keys()].join(', ') || 'none'}`)
    }
    this.pages.delete(id)
    this.lastId = id === prevLast ? [...this.pages.keys()].pop() : prevLast
    await tracked.page.close().catch(() => { /* already closed by the browser; state is what we wanted */ })
    return id
  }

  /**
   * Forget and close a tab that never finished its first navigation, without
   * reporting an error for the already-failed page.
   * @param pageId - the tab to discard.
   */
  async discard(pageId: string): Promise<void> {
    const tracked = this.pages.get(pageId)
    if (!tracked) return
    this.pages.delete(pageId)
    if (this.lastId === pageId) this.lastId = [...this.pages.keys()].pop()
    await tracked.page.close().catch(() => { /* already closed; nothing left to release */ })
  }

  /**
   * Close every tab and the browser process; safe to call twice and safe to
   * race an in-flight `newTab` (the launch is settled, then closed).
   */
  async dispose(): Promise<void> {
    this.disposed = true
    this.pages.clear()
    this.lastId = undefined
    const pending = this.contextPromise
    this.contextPromise = undefined
    if (pending) await pending.catch(() => undefined)
    const persistent = this.persistent
    this.persistent = undefined
    if (persistent) await persistent.close().catch(() => { /* context already gone; disposal is complete either way */ })
    const browser = this.browser
    this.browser = undefined
    if (browser) await browser.close().catch(() => { /* process already exited; disposal is complete either way */ })
  }

  private record(tracked: TrackedPage, entry: DownloadEntry): void {
    if (tracked.downloads.length >= this.options.maxConsoleMessages) tracked.downloads.shift()
    tracked.downloads.push(entry)
  }

  private capture(tracked: TrackedPage, level: string, text: string): void {
    if (tracked.console.length >= this.options.maxConsoleMessages) tracked.console.shift()
    tracked.lastSeq += 1
    tracked.console.push({ seq: tracked.lastSeq, level, text: text.length > MAX_CONSOLE_TEXT ? text.slice(0, MAX_CONSOLE_TEXT) : text })
  }

  private throwIfDisposed(): void {
    if (this.disposed) throw new Error('browser engine is disposed (plugin unloading)')
  }

  private ensureContext(): Promise<BrowserContext> {
    this.contextPromise ??= this.launchContext().catch((err: unknown) => {
      this.contextPromise = undefined
      throw err
    })
    return this.contextPromise
  }

  private async launchContext(): Promise<BrowserContext> {
    const viewport = { width: this.options.viewportWidth, height: this.options.viewportHeight }
    const errors: string[] = []
    let context: BrowserContext | undefined
    for (const channel of this.options.channels) {
      const launchOptions = channel === 'chromium'
        ? { headless: this.options.headless }
        : { headless: this.options.headless, channel }
      try {
        if (this.options.profileDir !== '') {
          context = await chromium.launchPersistentContext(this.options.profileDir, { ...launchOptions, viewport, acceptDownloads: true })
          if (this.disposed) {
            await context.close().catch(() => { /* dispose raced the launch; close is best-effort */ })
            throw new Error('browser engine is disposed (plugin unloading)')
          }
          this.persistent = context
        } else {
          const browser = await chromium.launch(launchOptions)
          if (this.disposed) {
            await browser.close().catch(() => { /* dispose raced the launch; close is best-effort */ })
            throw new Error('browser engine is disposed (plugin unloading)')
          }
          this.browser = browser
          context = await browser.newContext({ viewport, acceptDownloads: true })
        }
        break
      } catch (err) {
        if (this.disposed) throw err
        errors.push(`${channel}: ${err instanceof Error ? err.message.split('\n')[0] : String(err)}`)
      }
    }
    if (!context) {
      throw new Error(
        'no launchable browser found. Tried channels: '
        + errors.join('; ')
        + '. Install Google Chrome or Microsoft Edge, or run `npx playwright install chromium` and set browserChannels to chromium.',
      )
    }
    // Popup containment: pages opened by page scripts (window.open,
    // target=_blank) bypass the tab registry and the tools' origin gate, so
    // they are closed on arrival. Our own tabs come from context.newPage(),
    // which has no opener.
    context.on('page', (page) => {
      void page.opener().then((opener) => {
        if (opener) return page.close()
        return undefined
      }).catch(() => { /* popup already closed or crashed; containment achieved either way */ })
    })
    const originAllowed = this.options.originAllowed
    if (originAllowed) {
      // The fence blocks disallowed http(s) main-frame documents regardless of
      // what initiated the navigation: redirects, link clicks, history moves.
      await context.route('**/*', (route) => {
        const request = route.request()
        // Playwright throws when the request is issued before its frame exists
        // (early navigations, interrupted redirects, downloads). This callback has no
        // caller above it: an escaped throw becomes an unhandled rejection, which the
        // host elevates to a fatal load failure and the whole process exits. A request
        // without a resolvable frame cannot be a judgeable main-frame navigation, so
        // let it through instead of taking the host down with it.
        let frame: Frame
        try {
          frame = request.frame()
        } catch {
          return route.continue()
        }
        if (request.isNavigationRequest() && frame.parentFrame() === null) {
          // A tab authorized under a prompt-free session carries its exemption
          // for its own lifetime only; it never becomes a standing grant other
          // sessions inherit.
          const owner = this.tracked.get(frame.page())
          if (owner?.unfenced) return route.continue()
          let url: URL
          try {
            url = new URL(request.url())
          } catch {
            return route.abort('blockedbyclient')
          }
          if ((url.protocol === 'http:' || url.protocol === 'https:') && !originAllowed(url)) {
            return route.abort('blockedbyclient')
          }
        }
        return route.continue()
      })
    }
    return context
  }
}
