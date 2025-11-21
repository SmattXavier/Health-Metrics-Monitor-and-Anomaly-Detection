import 'package:flutter/material.dart';
import 'package:smart_health2/pages/symptoms_page.dart';

class BMICalculatorPage extends StatefulWidget {
  final int heartRate;
  const BMICalculatorPage({Key? key, required this.heartRate})
      : super(key: key);

  @override
  State<BMICalculatorPage> createState() => _BMICalculatorPageState();
}

class _BMICalculatorPageState extends State<BMICalculatorPage> {
  final TextEditingController heightController = TextEditingController();
  final TextEditingController weightController = TextEditingController();
  double? bmi;

  void calculateBMI() {
    final h = double.tryParse(heightController.text);
    final w = double.tryParse(weightController.text);
    if (h != null && w != null && h > 0 && w > 0) {
      setState(() {
        bmi = w / ((h / 100) * (h / 100));
      });
    }
  }

  void goToSymptomsPage() {
    if (bmi != null) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => SymptomsPage(
            heartRate: widget.heartRate,
            bmi: bmi!,
          ),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Calculate BMI')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TextField(
              controller: heightController,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(labelText: 'Height (cm)'),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: weightController,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(labelText: 'Weight (kg)'),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: calculateBMI,
              child: const Text('Calculate BMI'),
            ),
            if (bmi != null) ...[
              const SizedBox(height: 16),
              Text('Your BMI: ${bmi!.toStringAsFixed(2)}',
                  style: const TextStyle(fontSize: 24)),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: goToSymptomsPage,
                child: const Text('Next: Add Symptoms'),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
