import 'package:flutter/material.dart';

import 'package:smart_health2/pages/heart_bpm.dart';
import 'package:smart_health2/pages/into_page.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: OnboardingPage(),
      routes: {
        '/home': (context) => HomePage(),
      },
    );
  }
}
