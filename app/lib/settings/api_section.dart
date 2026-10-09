// Settings, Apps on this phone: the local API (lib/api/local_api.dart), which
// lets other apps on the device use the loaded model through the OpenAI,
// Anthropic or gRPC APIs, with a token.
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../api/local_api.dart';
import '../state/app_state.dart';

class ApiSection extends StatelessWidget {
  const ApiSection({super.key, required this.app});
  final AppState app;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final muted = theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant);
    final mono = theme.textTheme.bodySmall?.copyWith(fontFamily: 'monospace');
    return ListenableBuilder(
      listenable: app,
      builder: (context, _) {
        final on = app.device.apiEnabled;
        final token = app.device.apiToken;
        void copy(String label, String text) {
          Clipboard.setData(ClipboardData(text: text));
          ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$label copied')));
        }

        return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            secondary: const Icon(Icons.api_outlined),
            title: const Text('Local API'),
            subtitle: Text(
              app.apiError.isNotEmpty && on
                  ? app.apiError
                  : 'Other apps on this phone can use the loaded model as an OpenAI, Anthropic or gRPC server, '
                      'with the token below. Nothing outside the phone can connect.',
              style: muted,
            ),
            value: on,
            onChanged: app.setApiEnabled,
          ),
          if (on && token != null) ...[
            ListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('OpenAI and Anthropic base URL'),
              subtitle: Text('http://127.0.0.1:${LocalApi.httpPort}/v1', style: mono),
              trailing: const Icon(Icons.copy, size: 18),
              onTap: () => copy('URL', 'http://127.0.0.1:${LocalApi.httpPort}/v1'),
            ),
            ListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('gRPC (liyab.v1.Liyab)'),
              subtitle: Text('127.0.0.1:${LocalApi.grpcPort}', style: mono),
            ),
            ListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('Token'),
              subtitle: Text('${token.substring(0, 12)}…', style: mono),
              trailing: const Icon(Icons.copy, size: 18),
              onTap: () => copy('Token', token),
            ),
            Align(
              alignment: Alignment.centerLeft,
              child: TextButton(onPressed: app.newApiToken, child: const Text('New token (the old one stops working)')),
            ),
            Text(
              'Requests wait their turn with your own messages. Android may stop Liyab in the background: keep it '
              'open for long jobs.',
              style: muted,
            ),
          ],
        ]);
      },
    );
  }
}
