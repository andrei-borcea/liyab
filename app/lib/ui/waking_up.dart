// The flame in the middle of a view while the model loads (at start, or back
// from an idle unload), so a wait reads as one and not as a hang.
import 'package:flutter/material.dart';

import 'living_flame.dart';

class WakingUp extends StatelessWidget {
  const WakingUp({super.key, required this.status, required this.heat, this.size = 140});

  /// What is loading (the app's status line).
  final String status;
  final double heat;
  final double size;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            LivingFlame(size: size, state: FlameState.thinking, heat: heat),
            const SizedBox(height: 8),
            Text('Waking up', style: theme.textTheme.headlineSmall, textAlign: TextAlign.center),
            const SizedBox(height: 4),
            Text(
              status,
              style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant),
              textAlign: TextAlign.center,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
          ],
        ),
      ),
    );
  }
}
