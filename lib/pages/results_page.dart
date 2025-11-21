import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class ResultsPage extends StatefulWidget {
  final int heartRate;
  final double bmi;
  final List<String> symptoms;

  const ResultsPage({
    super.key,
    required this.heartRate,
    required this.bmi,
    required this.symptoms,
  });

  @override
  State<ResultsPage> createState() => _ResultsPageState();
}

class _ResultsPageState extends State<ResultsPage> {
  Map<String, dynamic>? predictionData;
  bool isLoading = false;
  String? errorMessage;

  @override
  void initState() {
    super.initState();
    getPrediction();
  }

  Future<void> getPrediction() async {
    setState(() {
      isLoading = true;
      errorMessage = null;
    });

    try {
      final response = await http.post(
        Uri.parse('http://localhost:5000/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'bmi': widget.bmi,
          'heart_rate': widget.heartRate,
          'symptoms': widget.symptoms,
        }),
      );

      if (response.statusCode == 200) {
        setState(() {
          predictionData = jsonDecode(response.body);
          isLoading = false;
        });
      } else {
        setState(() {
          errorMessage = 'Failed to get prediction. Please try again.';
          isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        errorMessage = 'Error connecting to the server. Please try again.';
        isLoading = false;
      });
    }
  }

  Color getRiskColor(String risk) {
    switch (risk) {
      case 'High Risk':
        return Colors.red;
      case 'Moderate Risk':
        return Colors.orange;
      default:
        return Colors.green;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Health Risk Assessment'),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Your Measurements',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 20),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Row(
                            children: [
                              Icon(Icons.favorite, color: Colors.red),
                              SizedBox(width: 8),
                              Text('Heart Rate:'),
                            ],
                          ),
                          Text(
                            '${widget.heartRate} BPM',
                            style: const TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Row(
                            children: [
                              Icon(Icons.monitor_weight, color: Colors.blue),
                              SizedBox(width: 8),
                              Text('BMI:'),
                            ],
                          ),
                          Text(
                            widget.bmi.toStringAsFixed(2),
                            style: const TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      const Text('Symptoms:',
                          style: TextStyle(fontWeight: FontWeight.bold)),
                      Wrap(
                        spacing: 8.0,
                        children: widget.symptoms
                            .map((s) => Chip(label: Text(s)))
                            .toList(),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 32),
              const Text(
                'Risk Assessment',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 20),
              if (isLoading)
                const Center(
                  child: Column(
                    children: [
                      CircularProgressIndicator(),
                      SizedBox(height: 16),
                      Text('Analyzing your health data...'),
                    ],
                  ),
                )
              else if (errorMessage != null)
                Center(
                  child: Column(
                    children: [
                      const Icon(
                        Icons.error_outline,
                        color: Colors.red,
                        size: 48,
                      ),
                      const SizedBox(height: 16),
                      Text(
                        errorMessage!,
                        textAlign: TextAlign.center,
                        style: const TextStyle(color: Colors.red),
                      ),
                      const SizedBox(height: 16),
                      ElevatedButton(
                        onPressed: getPrediction,
                        child: const Text('Try Again'),
                      ),
                    ],
                  ),
                )
              else if (predictionData != null) ...[
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(
                              Icons.health_and_safety,
                              color:
                                  getRiskColor(predictionData!['prediction']),
                              size: 32,
                            ),
                            const SizedBox(width: 8),
                            Text(
                              predictionData!['prediction'],
                              style: TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                                color:
                                    getRiskColor(predictionData!['prediction']),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        const Text(
                          'Risk Probabilities:',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 8),
                        ...predictionData!['probabilities']
                            .entries
                            .map<Widget>((entry) => LinearProgressIndicator(
                                  value: entry.value,
                                  backgroundColor: Colors.grey[200],
                                  color: getRiskColor(entry.key),
                                  minHeight: 10,
                                ))
                            .expand(
                                (widget) => [widget, const SizedBox(height: 8)])
                            .toList(),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'From the data you have provided, you are likely to have:',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 8),
                        ...?predictionData?['possible_diseases']
                            ?.map<Widget>((disease) => Padding(
                                  padding:
                                      const EdgeInsets.symmetric(vertical: 2.0),
                                  child: Row(
                                    children: [
                                      const Icon(Icons.warning,
                                          color: Colors.orange, size: 20),
                                      const SizedBox(width: 8),
                                      Expanded(child: Text(disease)),
                                    ],
                                  ),
                                )),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Recommendations:',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 8),
                        ...predictionData!['recommendations'].map<Widget>(
                          (recommendation) => Padding(
                            padding: const EdgeInsets.symmetric(vertical: 4.0),
                            child: Row(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Icon(Icons.check_circle,
                                    color: Colors.green, size: 20),
                                const SizedBox(width: 8),
                                Expanded(child: Text(recommendation)),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 32),
                Container(
                  padding: const EdgeInsets.all(16.0),
                  decoration: BoxDecoration(
                    color: Colors.grey[200],
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Row(
                    children: [
                      Icon(Icons.info_outline, color: Colors.grey),
                      SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          'Disclaimer: This is not a medical diagnosis. Please consult a healthcare professional if symptoms persist or worsen.',
                          style: TextStyle(
                            color: Colors.grey,
                            fontSize: 12,
                            fontStyle: FontStyle.italic,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
