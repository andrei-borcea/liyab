# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Report it privately through GitHub: on the
repository's **Security** tab, choose **Report a vulnerability**. Include what is affected, how to reproduce it, and
the impact you expect. You will get an answer as soon as possible, and credit in the fix's notes if you wish.

## Scope

Areas where a report is especially useful:

* **Model files.** The GGUF loader parses files the user downloads from the internet. Malformed or malicious files
  must be rejected with an error, never cause out-of-bounds reads or writes.
* **The app's local API.** It listens on the loopback interface only and requires a token. A way to reach it from
  outside the device, to use it without the token, or to read another client's data is in scope.
* **Personal data in the app.** The assistant reads calendar, notifications, messages and calls only when the user
  enables each source, and nothing leaves the device. Any path by which that data could leave the device, be
  written where other apps can read it, or be logged is in scope.
* **Tool results as instructions.** Text from messages or notifications that makes the assistant act against the
  user's intent (prompt injection) is in scope.

## Supported versions

Fixes go into the `main` branch.
