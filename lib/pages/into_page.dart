import 'package:flutter/material.dart';
import 'heart_bpm.dart';

class OnboardingPage extends StatefulWidget {
  const OnboardingPage({Key? key}) : super(key: key);

  @override
  State<OnboardingPage> createState() => _OnboardingPageState();
}

class _OnboardingPageState extends State<OnboardingPage> {
  final PageController _controller = PageController();
  int _currentPage = 0;

  final List<Map<String, String>> onboardingData = [
    {
      'title': 'Welcome to Smart Health',
      'subtitle': 'Your personal health companion.',
      'image': '',
    },
    {
      'title': 'Measure Heart Rate',
      'subtitle': 'Easily measure your heart rate using your phone camera.',
      'image': '',
    },
    {
      'title': 'Calculate BMI',
      'subtitle': 'Track your body mass index for better health insights.',
      'image': '',
    },
    {
      'title': 'Log Symptoms',
      'subtitle': 'Select and add symptoms for a more accurate assessment.',
      'image': '',
    },
    {
      'title': 'Get Health Predictions',
      'subtitle': 'Receive possible diagnoses and recommendations instantly.',
      'image': '',
    },
  ];

  void _nextPage() {
    if (_currentPage < onboardingData.length - 1) {
      _controller.nextPage(
          duration: const Duration(milliseconds: 300), curve: Curves.easeInOut);
    }
  }

  void _goToMainApp() {
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (context) => const HomePage()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: PageView.builder(
                controller: _controller,
                itemCount: onboardingData.length,
                onPageChanged: (index) {
                  setState(() {
                    _currentPage = index;
                  });
                },
                itemBuilder: (context, index) {
                  final data = onboardingData[index];
                  return Padding(
                    padding: const EdgeInsets.all(32.0),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        if (data['image']!.isNotEmpty)
                          Image.asset(data['image']!, height: 200),
                        Text(
                          data['title']!,
                          style: const TextStyle(
                              fontSize: 28, fontWeight: FontWeight.bold),
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 20),
                        Text(
                          data['subtitle']!,
                          style: const TextStyle(fontSize: 18),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  );
                },
              ),
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: List.generate(
                onboardingData.length,
                (index) => Container(
                  margin:
                      const EdgeInsets.symmetric(horizontal: 4, vertical: 16),
                  width: _currentPage == index ? 16 : 8,
                  height: 8,
                  decoration: BoxDecoration(
                    color: _currentPage == index ? Colors.blue : Colors.grey,
                    borderRadius: BorderRadius.circular(4),
                  ),
                ),
              ),
            ),
            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 32.0, vertical: 24),
              child: SizedBox(
                width: double.infinity,
                child: _currentPage == onboardingData.length - 1
                    ? ElevatedButton(
                        onPressed: _goToMainApp,
                        child: const Text('Get Started'),
                      )
                    : ElevatedButton(
                        onPressed: _nextPage,
                        child: const Text('Next'),
                      ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
