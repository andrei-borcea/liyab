import Carbon.HIToolbox
import Cocoa
import FlutterMacOS

/// The app window. Its content runs under a transparent title bar, so the
/// night ground (Kiln) fills the whole window and the traffic lights sit on it;
/// Dart pads its top bar by the title bar's height (lib/main.dart). The same
/// window becomes the assistant sheet on Option-Space (AssistWindow).
class MainFlutterWindow: NSWindow {
  static let kiln = NSColor(srgbRed: 0x1B / 255.0, green: 0x14 / 255.0, blue: 0x20 / 255.0, alpha: 1)

  private(set) var assist: AssistWindow!
  private var hotKey: HotKey?

  override func awakeFromNib() {
    let flutterViewController = FlutterViewController()
    flutterViewController.backgroundColor = Self.kiln
    let windowFrame = self.frame
    self.contentViewController = flutterViewController
    self.setFrame(windowFrame, display: true)

    titlebarAppearsTransparent = true
    titleVisibility = .hidden
    styleMask.insert(.fullSizeContentView)
    backgroundColor = Self.kiln
    appearance = NSAppearance(named: .darkAqua)  // Liyab lives at night (themeMode: dark)
    minSize = NSSize(width: 420, height: 560)
    isReleasedWhenClosed = false  // closing hides it; Option-Space and the Dock bring it back
    setFrameAutosaveName("Liyab")

    RegisterGeneratedPlugins(registry: flutterViewController)
    assist = AssistWindow(window: self, view: flutterViewController)
    hotKey = HotKey(keyCode: kVK_Space, modifiers: optionKey) { [weak self] in self?.assist.toggle() }
    DeviceChannel.attach(flutterViewController.engine.binaryMessenger) { [weak self] in self?.hotKey != nil }
    DataChannel.attach(flutterViewController.engine.binaryMessenger)

    super.awakeFromNib()
  }

  // Esc closes the assistant sheet (keys Flutter does not handle travel up here).
  override func cancelOperation(_ sender: Any?) {
    if assist.isOpen { assist.close() } else { super.cancelOperation(sender) }
  }

  // A click in another app closes the sheet, as Spotlight does.
  override func resignKey() {
    super.resignKey()
    if assist.isOpen { assist.close() }
  }

  override var canBecomeKey: Bool { true }  // also while borderless in assist mode
}
