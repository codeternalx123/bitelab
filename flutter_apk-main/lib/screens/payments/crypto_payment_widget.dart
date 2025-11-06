import 'package:flutter/material.dart';
import '../../services/api_service.dart';
import 'package:qr_flutter/qr_flutter.dart';

class CryptoPaymentWidget extends StatefulWidget {
  const CryptoPaymentWidget({super.key});

  @override
  State<CryptoPaymentWidget> createState() => _CryptoPaymentWidgetState();
}

class _CryptoPaymentWidgetState extends State<CryptoPaymentWidget> {
  final ApiService _api = ApiService();
  String paymentAddress = '';
  String paymentId = '';
  bool loading = true;

  @override
  void initState() {
    super.initState();
    _initPayment();
  }

  Future<void> _initPayment() async {
    final resp = await _api.post('/api/v1/payments/crypto', {'amount': 100.0, 'currency': 'USD', 'crypto': 'BTC'});
    setState(() {
      paymentAddress = resp.data['paymentAddress'] ?? '';
      paymentId = resp.data['paymentId'] ?? '';
      loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Crypto payment')),
      body: Center(
        child: loading ? const CircularProgressIndicator() : Column(mainAxisSize: MainAxisSize.min, children: [
          QrImageView(
            data: paymentAddress,
            version: QrVersions.auto,
            size: 200.0,
          ),
          const SizedBox(height: 8),
          SelectableText(paymentAddress)
        ])
      ),
    );
  }
}
