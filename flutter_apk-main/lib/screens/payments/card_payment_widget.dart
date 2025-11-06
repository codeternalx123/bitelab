import 'package:flutter/material.dart';
import '../../core/services/payment_service.dart';

class CardPaymentWidget extends StatefulWidget {
  final PaymentService paymentService;
  const CardPaymentWidget({super.key, required this.paymentService});
  @override
  State<CardPaymentWidget> createState() => _CardPaymentWidgetState();
}

class _CardPaymentWidgetState extends State<CardPaymentWidget> {
  bool _loading = false;
  Future<void> _pay() async {
    setState(() => _loading = true);
    try {
      await widget.paymentService.payWithCard(amountCents: 1000, currency: 'USD');
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Payment success')));
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Payment failed: $e')));
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Card payment')),
      body: Center(child: _loading ? const CircularProgressIndicator() : ElevatedButton(onPressed: _pay, child: const Text('Pay \$10'))),
    );
  }
}
