import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../services/payment_service.dart';
import '../../providers/payment_service_provider.dart';
import 'crypto_payment_widget.dart';
import 'card_payment_widget.dart';

class PaymentsScreen extends ConsumerWidget {
  const PaymentsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final paymentService = ref.watch(paymentServiceProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('Payments')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(children: [
          Card(child: ListTile(title: const Text('Pay with card'), subtitle: const Text('Stripe PaymentSheet'), onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => CardPaymentWidget(paymentService: paymentService))))),
          Card(child: ListTile(title: const Text('Pay with crypto'), subtitle: const Text('BTC / ETH'), onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const CryptoPaymentWidget())))),
        ]),
      ),
    );
  }
}
