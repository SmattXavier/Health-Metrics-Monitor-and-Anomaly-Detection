import 'package:flutter/material.dart';
import 'results_page.dart';

class SymptomsPage extends StatefulWidget {
  final int heartRate;
  final double bmi;
  const SymptomsPage({Key? key, required this.heartRate, required this.bmi})
      : super(key: key);

  @override
  State<SymptomsPage> createState() => _SymptomsPageState();
}

class _SymptomsPageState extends State<SymptomsPage> {
  final List<String> commonSymptoms = [
    'Fever',
    'Cough',
    'Runny nose',
    'Headache',
    'Sore throat',
    'Fatigue',
    'Shortness of breath',
    'Muscle pain',
    'Nausea',
    'Vomiting',
    'Diarrhea',
    'Loss of taste',
    'Loss of smell',
    'Chest pain',
    'Dizziness',
    'Rash',
    'Chills',
    'Sneezing',
    'Congestion',
    'Abdominal pain',
    'Back pain',
  ];
  final List<String> selectedSymptoms = [];
  final TextEditingController customSymptomController = TextEditingController();

  void addCustomSymptom() {
    final symptom = customSymptomController.text.trim();
    if (symptom.isNotEmpty && !selectedSymptoms.contains(symptom)) {
      setState(() {
        selectedSymptoms.add(symptom);
        customSymptomController.clear();
      });
    }
  }

  void toggleSymptom(String symptom) {
    setState(() {
      if (selectedSymptoms.contains(symptom)) {
        selectedSymptoms.remove(symptom);
      } else {
        selectedSymptoms.add(symptom);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Add Symptoms')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Select your symptoms:', style: TextStyle(fontSize: 18)),
            Wrap(
              spacing: 8.0,
              children: commonSymptoms.map((symptom) {
                final selected = selectedSymptoms.contains(symptom);
                return FilterChip(
                  label: Text(symptom),
                  selected: selected,
                  onSelected: (_) => toggleSymptom(symptom),
                );
              }).toList(),
            ),
            const SizedBox(height: 24),
            const Text('Add a custom symptom:', style: TextStyle(fontSize: 18)),
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: customSymptomController,
                    decoration:
                        const InputDecoration(hintText: 'Enter symptom'),
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.add),
                  onPressed: addCustomSymptom,
                ),
              ],
            ),
            const SizedBox(height: 24),
            const Text('Selected symptoms:', style: TextStyle(fontSize: 18)),
            Wrap(
              spacing: 8.0,
              children: selectedSymptoms
                  .map((symptom) => Chip(
                        label: Text(symptom),
                        onDeleted: () => toggleSymptom(symptom),
                      ))
                  .toList(),
            ),
            const SizedBox(height: 32),
            Center(
              child: ElevatedButton(
                onPressed: selectedSymptoms.isNotEmpty
                    ? () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => ResultsPage(
                              heartRate: widget.heartRate,
                              bmi: widget.bmi,
                              symptoms: selectedSymptoms,
                            ),
                          ),
                        );
                      }
                    : null,
                child: const Text('Submit & View Results'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
