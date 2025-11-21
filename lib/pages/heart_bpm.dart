import 'package:flutter/material.dart';
import 'package:heart_bpm/heart_bpm.dart';
import 'package:smart_health2/pages/bmi_page.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  List<SensorValue> data = [];
  int? bpmValue;
  bool measurementComplete = false;

  void _onBPMComplete() {
    if (bpmValue != null && !measurementComplete) {
      setState(() {
        measurementComplete = true;
      });
      // Navigate to temperature measurement
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => BMICalculatorPage(heartRate: bpmValue!),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Measure Heart Rate'),
      ),
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              'Cover the camera and flash with your finger',
              style: Theme.of(context)
                  .textTheme
                  .headlineLarge
                  ?.copyWith(fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(
                  Icons.favorite,
                  size: 100,
                  color: Colors.red,
                ),
                const SizedBox(width: 20),
                HeartBPMDialog(
                  context: context,
                  onBPM: (value) {
                    setState(() {
                      bpmValue = value;
                    });
                    if (value > 0) {
                      Future.delayed(
                          const Duration(seconds: 5), _onBPMComplete);
                    }
                  },
                  onRawData: (value) {
                    setState(() {
                      if (data.length == 100) data.removeAt(0);
                      data.add(value);
                    });
                  },
                  child: Text(
                    bpmValue?.toString() ?? '0',
                    style: const TextStyle(fontSize: 60),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            if (bpmValue != null && bpmValue! > 0)
              const Text(
                'Keep your finger steady for 5 more seconds...',
                style: TextStyle(fontSize: 16),
              ),
          ],
        ),
      ),
    );
  }
}
