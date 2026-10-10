import Carbon.HIToolbox

/// A system-wide keyboard shortcut through Carbon's RegisterEventHotKey: the
/// only public API that delivers a key combination to a background app without
/// the Accessibility permission an event tap would need.
final class HotKey {
  private var hotKey: EventHotKeyRef?
  private var handler: EventHandlerRef?
  private let action: () -> Void

  /// `keyCode`: a kVK_* virtual key; `modifiers`: Carbon masks (optionKey, cmdKey, ...).
  /// `action` runs on the main thread. Fails (nil) when another app owns the combination.
  init?(keyCode: Int, modifiers: Int, action: @escaping () -> Void) {
    self.action = action
    var type = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
    let installed = InstallEventHandler(GetApplicationEventTarget(), { _, _, user in
      guard let user else { return OSStatus(eventNotHandledErr) }
      Unmanaged<HotKey>.fromOpaque(user).takeUnretainedValue().action()
      return noErr
    }, 1, &type, Unmanaged.passUnretained(self).toOpaque(), &handler)
    let id = EventHotKeyID(signature: OSType(0x4C59_4142), id: 1)  // 'LYAB'
    guard installed == noErr,
      RegisterEventHotKey(UInt32(keyCode), UInt32(modifiers), id, GetApplicationEventTarget(), 0, &hotKey) == noErr
    else {
      if let handler { RemoveEventHandler(handler) }
      return nil
    }
  }

  deinit {
    if let hotKey { UnregisterEventHotKey(hotKey) }
    if let handler { RemoveEventHandler(handler) }
  }
}
