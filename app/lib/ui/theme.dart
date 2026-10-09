// Liyab's visual tokens (docs/brand): a night ground lit by its own flame.
import 'package:flutter/material.dart';

abstract final class Palette {
  static const kiln = Color(0xFF1B1420); // night ground
  static const kilnRaised = Color(0xFF251C2B); // sheets, input
  static const hairline = Color(0xFF3A2E41);
  static const ash = Color(0xFFEDE6EE); // text on night
  static const smoke = Color(0xFFA99BAD); // secondary text
  static const flare = Color(0xFFFF3D6E);
  static const ember = Color(0xFFFF7A3D); // primary action
  static const gold = Color(0xFFFFC24D);
  static const core = Color(0xFF9ED8FF);

  // Day theme
  static const dayGround = Color(0xFFF7F2F5);
  static const dayRaised = Color(0xFFFFFFFF);
  static const dayHairline = Color(0xFFE3D8E2);
  static const dayText = Color(0xFF2A1F2E);
  static const daySmoke = Color(0xFF6E5F73);
}

const displayFont = 'Bricolage Grotesque';
const bodyFont = 'Figtree';

ThemeData liyabTheme(Brightness brightness) {
  final night = brightness == Brightness.dark;
  final ground = night ? Palette.kiln : Palette.dayGround;
  final text = night ? Palette.ash : Palette.dayText;
  final scheme = ColorScheme(
    brightness: brightness,
    primary: Palette.ember,
    onPrimary: Colors.white,
    secondary: Palette.flare,
    onSecondary: Colors.white,
    tertiary: Palette.gold,
    onTertiary: Palette.dayText,
    error: const Color(0xFFFF5A5F),
    onError: Colors.white,
    surface: ground,
    onSurface: text,
    surfaceContainerHighest: night ? Palette.kilnRaised : Palette.dayRaised,
    onSurfaceVariant: night ? Palette.smoke : Palette.daySmoke,
    outline: night ? Palette.hairline : Palette.dayHairline,
  );
  final base = ThemeData(useMaterial3: true, colorScheme: scheme, fontFamily: bodyFont);
  return base.copyWith(
    scaffoldBackgroundColor: ground,
    textTheme: base.textTheme.copyWith(
      headlineMedium: TextStyle(fontFamily: displayFont, fontWeight: FontWeight.w700, fontSize: 28, color: text, letterSpacing: -0.4),
      titleLarge: TextStyle(fontFamily: displayFont, fontWeight: FontWeight.w600, fontSize: 20, color: text),
      bodyLarge: TextStyle(fontSize: 16.5, height: 1.5, color: text),
      bodyMedium: TextStyle(fontSize: 15, height: 1.45, color: text),
      labelSmall: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant, fontFeatures: const [FontFeature.tabularFigures()]),
    ),
    appBarTheme: AppBarTheme(backgroundColor: ground, foregroundColor: text, elevation: 0, scrolledUnderElevation: 0),
    bottomSheetTheme: BottomSheetThemeData(
      backgroundColor: scheme.surfaceContainerHighest,
      shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(28))),
    ),
  );
}
