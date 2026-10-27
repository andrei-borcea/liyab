# Visual design

[← README](../../README.md) · [Documentation index](README.md)

Commands and source paths in examples are relative to the repository root.

## On this page

- [Look and feel](#look-and-feel)

---

## Look and feel

<p align="center"><img src="../brand/liyab-states.svg" width="100%" alt="The living flame's four states: resting, listening, thinking, answering"></p>

The Liyab mark (two blades of flame around a core of light) is not a static logo in the app: it is drawn live as the
assistant's presence. Its motion says what the assistant is doing:
* **resting**: it breathes slowly;
* **listening**: it leans toward your voice and its core swells with it;
* **thinking**: sparks rise while the model reasons;
* **answering**: it flickers as words stream.

It redraws about 24 times a second at rest and 30 while the model works, never at the display's full rate, so the UI
leaves the cores to the engine.

<p align="center"><img src="../brand/liyab-heat.svg" width="100%" alt="The flame's colour follows the phone's thermal headroom, from cool violet to hot red"></p>

Its colour is the device's temperature, read from the same thermal-headroom forecast that steers the engine:
violet-cool while there is room, ember in normal use, red as throttling gets near. The work happening on the device
stays visible without a single number on screen. The same colours light the edge of the assistant sheet and the input
field while the model works. On a computer they turn as an aura around the command bar: it barely breathes at rest,
wakes as you type, turns fast while the model reasons and flows while the answer streams, at most 30 frames a
second like the flame.

<p align="center"><img src="../brand/liyab-palette.svg" width="100%" alt="Liyab colour palette and typography"></p>

* **Night first.** The app sits on a deep night ground (*Kiln*) lit by its own flame. A day theme mirrors it with
  warm off-whites.
* **Fire for meaning.** *Ember* marks primary actions, *Flare* and *Gold* accent, and *Core*, the cool light at the
  flame's heart, is kept for the mark itself.
* **Type.**
  * Bricolage Grotesque for display, headings and the wordmark;
  * Figtree for everything you read, with tabular figures for live numbers.

  Both fonts are SIL Open Font License and ship in `app/assets/fonts`.
* **Sources.**
  * The tokens are defined in [`app/lib/ui/theme.dart`](../../app/lib/ui/theme.dart) and the flame in
    [`app/lib/ui/living_flame.dart`](../../app/lib/ui/living_flame.dart).
  * The mark and the app icon are in [`docs/brand/`](../brand).
  * The animated images on this page come from `tools/gen_brand_svgs.py` (`pip install fonttools`), which translates
    the app's flame to SVG with the same paths, colour ramps and rhythms.

<p align="center">
  <img src="../brand/liyab-mark.svg" width="96" alt="The Liyab mark">
  &nbsp;&nbsp;&nbsp;
  <img src="../brand/liyab-icon.svg" width="96" alt="The Liyab app icon">
  &nbsp;&nbsp;&nbsp;
  <img src="../brand/liyab-flame.svg" width="96" alt="The Liyab flame, animated">
</p>

---
