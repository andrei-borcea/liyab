import Cocoa
import FlutterMacOS

/// The assistant on macOS: Option-Space, from any app, turns the app's window
/// into a floating command bar in "assist" mode, where Dart draws CommandBar
/// (lib/assist/command_bar.dart), in the manner of Spotlight. It is the same
/// window and the same Flutter engine, so the model is loaded once. Closing
/// (Esc, a click elsewhere, or Dart's close) puts the window back as it was
/// and returns to the app the user was in.
///
/// The "liyab/assist" protocol is Android's (android/.../AssistActivity.kt):
/// native -> Dart "opened" / "closed", Dart -> native "close"; plus, on the
/// desktop, Dart -> native "resize" {height} (the window follows the bar's
/// height, top edge fixed) and "openApp" (close the bar, show the app window).
final class AssistWindow {
  private struct Saved {
    let frame: NSRect
    let styleMask: NSWindow.StyleMask
    let level: NSWindow.Level
    let visible: Bool  // the window was on screen
    let active: Bool   // Liyab was the frontmost app
  }

  /// Match CommandBar in Dart: the bar's width and corner radius, and the
  /// transparent margin around it where the aura's glow is drawn.
  private static let barWidth: CGFloat = 720
  private static let cornerRadius: CGFloat = 18
  private static let margin: CGFloat = 30

  private let window: NSWindow
  private let view: FlutterViewController
  private let channel: FlutterMethodChannel
  private var saved: Saved?
  private var topEdge: CGFloat = 0
  private var material: NSVisualEffectView?
  private var closing = false

  var isOpen: Bool { saved != nil }

  init(window: NSWindow, view: FlutterViewController) {
    self.window = window
    self.view = view
    channel = FlutterMethodChannel(name: "liyab/assist", binaryMessenger: view.engine.binaryMessenger)
    channel.setMethodCallHandler { [weak self] call, result in
      switch call.method {
      case "close": self?.close()
      case "openApp": self?.openApp()
      case "resize":
        if let height = (call.arguments as? [String: Any])?["height"] as? Double { self?.resize(to: CGFloat(height)) }
      default: break
      }
      result(nil)
    }
  }

  func toggle() {
    if isOpen { close() } else { open() }
  }

  func open() {
    guard saved == nil else { return }
    saved = Saved(frame: window.frame, styleMask: window.styleMask, level: window.level,
                  visible: window.isVisible && !window.isMiniaturized, active: NSApp.isActive)
    channel.invokeMethod("opened", arguments: nil)  // Dart switches to the bar before the window shows

    // Where Spotlight sits: centred on the screen with the pointer, the bar's
    // top a fifth of the way down. The height starts at the bare bar and
    // follows Dart.
    let screen = NSScreen.screens.first { NSMouseInRect(NSEvent.mouseLocation, $0.frame, false) } ?? NSScreen.main
    let area = screen?.visibleFrame ?? window.frame
    topEdge = area.maxY - area.height * 0.2 + Self.margin
    let width = Self.barWidth + 2 * Self.margin
    let height: CGFloat = 115 + 2 * Self.margin
    window.styleMask = [.borderless, .fullSizeContentView]
    window.isOpaque = false
    window.backgroundColor = .clear
    view.backgroundColor = .clear
    window.level = .floating
    window.hasShadow = false  // the aura's glow takes the shadow's place
    window.setFrame(NSRect(x: area.midX - width / 2, y: topEdge - height, width: width, height: height), display: true)
    showMaterial()
    window.alphaValue = 0
    NSApp.activate(ignoringOtherApps: true)
    window.makeKeyAndOrderFront(nil)
    NSAnimationContext.runAnimationGroup { context in
      context.duration = 0.18
      window.animator().alphaValue = 1
    }
  }

  /// Fades the bar out, then puts the window back as it was.
  func close() {
    guard isOpen, !closing else { return }
    closing = true
    NSAnimationContext.runAnimationGroup({ context in
      context.duration = 0.14
      window.animator().alphaValue = 0
    }, completionHandler: { [weak self] in self?.restore() })
  }

  private func restore() {
    closing = false
    guard let s = saved else { return }
    saved = nil
    window.orderOut(nil)
    window.alphaValue = 1
    material?.removeFromSuperview()
    material = nil
    window.styleMask = s.styleMask
    window.level = s.level
    window.isOpaque = true
    window.backgroundColor = MainFlutterWindow.kiln
    view.backgroundColor = MainFlutterWindow.kiln
    window.setFrame(s.frame, display: false)
    channel.invokeMethod("closed", arguments: nil)
    if s.visible { s.active ? window.makeKeyAndOrderFront(nil) : window.orderFront(nil) }
    if !s.active { NSApp.hide(nil) }  // back to the app the user was in
  }

  /// Leaves the bar for the app's window, frontmost, with the same conversation.
  func openApp() {
    guard let s = saved else { return }
    saved = Saved(frame: s.frame, styleMask: s.styleMask, level: s.level, visible: true, active: true)
    close()  // restores the window frontmost and key
  }

  private func resize(to height: CGFloat) {
    guard isOpen, height > 0 else { return }
    var frame = window.frame
    frame.size.height = ceil(height)
    frame.origin.y = topEdge - frame.height
    window.setFrame(frame, display: true)
  }

  /// The system's blurred material under the transparent Flutter view, with
  /// the bar's rounded corners: the desktop shows through, as in Spotlight.
  /// It sits in the window's frame view, below the content view.
  private func showMaterial() {
    guard material == nil, let content = window.contentView, let frameView = content.superview else { return }
    // Inset by the margin: the material is exactly the bar, the glow lies outside it.
    let effect = NSVisualEffectView(frame: frameView.bounds.insetBy(dx: Self.margin, dy: Self.margin))
    effect.autoresizingMask = [.width, .height]
    effect.material = .hudWindow
    effect.blendingMode = .behindWindow
    effect.state = .active
    effect.appearance = NSAppearance(named: .darkAqua)
    effect.maskImage = Self.roundedMask(radius: Self.cornerRadius)
    frameView.addSubview(effect, positioned: .below, relativeTo: content)
    material = effect
  }

  /// A stretchable mask: only the corners are drawn, the rest stretches.
  private static func roundedMask(radius: CGFloat) -> NSImage {
    let side = radius * 2 + 1
    let image = NSImage(size: NSSize(width: side, height: side), flipped: false) { rect in
      NSColor.black.setFill()
      NSBezierPath(roundedRect: rect, xRadius: radius, yRadius: radius).fill()
      return true
    }
    image.capInsets = NSEdgeInsets(top: radius, left: radius, bottom: radius, right: radius)
    image.resizingMode = .stretch
    return image
  }
}
