// Persistent settings: device-wide ones and per-model generation settings.
import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

const defaultSystemPrompt = 'You are Liyab, a helpful assistant running entirely on this phone, without internet. '
    'Answer clearly and concisely.';

/// Generation settings of one model, saved under its file name.
class ModelSettings {
  ModelSettings({
    this.thinking = false,
    this.temperature = 0.7,
    this.topP = 0.9,
    this.topK = 40,
    this.maxTokens = 1024,
    this.contextLength = 4096,
    this.systemPrompt = defaultSystemPrompt,
  });

  bool thinking; // used only when the model has a thinking mode
  double temperature;
  double topP;
  int topK;
  int maxTokens;
  int contextLength; // takes effect on the next load
  String systemPrompt;

  /// Defaults, overridden by the model file's own sampling metadata, then by saved values.
  static ModelSettings resolve(SharedPreferences prefs, String model, Map<String, String> metadata) {
    final s = ModelSettings();
    double? number(String key) => double.tryParse(metadata[key] ?? '');
    s.temperature = number('general.sampling.temp') ?? s.temperature;
    s.topP = number('general.sampling.top_p') ?? s.topP;
    s.topK = number('general.sampling.top_k')?.round() ?? s.topK;
    final saved = prefs.getString(_key(model));
    if (saved == null) return s;
    try {
      final o = jsonDecode(saved) as Map<String, Object?>;
      s.thinking = o['thinking'] as bool? ?? s.thinking;
      s.temperature = (o['temperature'] as num?)?.toDouble() ?? s.temperature;
      s.topP = (o['top_p'] as num?)?.toDouble() ?? s.topP;
      s.topK = (o['top_k'] as num?)?.toInt() ?? s.topK;
      s.maxTokens = (o['max_tokens'] as num?)?.toInt() ?? s.maxTokens;
      s.contextLength = (o['context'] as num?)?.toInt() ?? s.contextLength;
      s.systemPrompt = o['system'] as String? ?? s.systemPrompt;
    } on FormatException {
      // unreadable: defaults
    }
    return s;
  }

  /// The context length to load `model` with, before an engine exists.
  static int contextFor(SharedPreferences prefs, String model) {
    final saved = prefs.getString(_key(model));
    if (saved == null) return 4096;
    try {
      return ((jsonDecode(saved) as Map<String, Object?>)['context'] as num?)?.toInt() ?? 4096;
    } on FormatException {
      return 4096;
    }
  }

  Future<void> save(SharedPreferences prefs, String model) => prefs.setString(
      _key(model),
      jsonEncode({
        'thinking': thinking,
        'temperature': temperature,
        'top_p': topP,
        'top_k': topK,
        'max_tokens': maxTokens,
        'context': contextLength,
        'system': systemPrompt,
      }));

  static String _key(String model) => 'model:$model';
}

/// Device-wide settings.
class DeviceSettings {
  DeviceSettings(this._prefs);
  final SharedPreferences _prefs;

  /// Run on the GPU (Vulkan) instead of the CPU.
  bool get useGpu => _prefs.getBool('gpu') ?? false;
  set useGpu(bool v) => _prefs.setBool('gpu', v);

  /// Skin / board temperature (°C) above which the engine throttles. 50 by
  /// default: on some phones the only "skin" sensor is a board thermistor that
  /// reads 45+ while charging. The OS thermal status throttles regardless.
  double get thermalLimitC => _prefs.getDouble('thermal_limit') ?? 50;
  set thermalLimitC(double v) => _prefs.setDouble('thermal_limit', v);

  /// Memory (MiB) the engine may keep resident. 5500 by default because
  /// HyperOS / MIUI stop any app above 6 GiB of PSS, whatever RAM is free.
  int get memoryBudgetMb => _prefs.getInt('memory_budget_mb') ?? 5500;
  set memoryBudgetMb(int v) => _prefs.setInt('memory_budget_mb', v);

  /// The last loaded model file, reloaded at start.
  String? get lastModel => _prefs.getString('model');
  set lastModel(String? v) => v == null ? _prefs.remove('model') : _prefs.setString('model', v);
}
