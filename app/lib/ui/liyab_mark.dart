import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

/// Static flame silhouette from the brand palette. No idle animation or timer.
class LiyabMark extends StatelessWidget {
  const LiyabMark({super.key, this.size = 48});
  final double size;
  @override
  Widget build(BuildContext context) => SvgPicture.asset('assets/brand/mark.svg', width: size, height: size,
    semanticsLabel: 'Liyab');
}
